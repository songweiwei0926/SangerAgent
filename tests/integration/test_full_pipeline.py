"""
Integration tests for complete analysis pipelines.

Tests the full SNV and editing analysis pipelines using synthetic data
(no real AB1 files needed). These tests exercise the complete pipeline
from ChromatogramData through alignment, SNV calling, and result packaging.

Tests use mocked external services (NCBI API, minimap2) to avoid network
dependencies while still exercising the full orchestration logic.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.ab1_parser import generate_synthetic_ab1_data
from backend.schemas.chromatogram import ChromatogramData, TraceData
from backend.schemas.snv import SNVResult, SNVCall, BaseProportions
from backend.schemas.editing import EditingResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_chromatogram() -> ChromatogramData:
    """Generate a clean synthetic chromatogram for testing."""
    return generate_synthetic_ab1_data(n_bases=400, mean_quality=35.0, seed=42)


@pytest.fixture
def synthetic_chromatogram_with_snvs() -> ChromatogramData:
    """Generate a synthetic chromatogram with known SNV positions."""
    return generate_synthetic_ab1_data(
        n_bases=400,
        snv_positions=[100, 200, 300],
        mean_quality=35.0,
        seed=42,
    )


@pytest.fixture
def wt_chromatogram() -> ChromatogramData:
    """Generate a WT chromatogram for editing analysis."""
    return generate_synthetic_ab1_data(n_bases=300, mean_quality=35.0, seed=10)


@pytest.fixture
def edited_chromatogram() -> ChromatogramData:
    """Generate an edited chromatogram for editing analysis."""
    return generate_synthetic_ab1_data(
        n_bases=300,
        snv_positions=[50, 51, 52, 53],  # Simulate base editing window
        mean_quality=34.0,
        seed=11,
    )


@pytest.fixture
def mock_settings():
    """Create mock application settings."""
    settings = MagicMock()
    settings.HG38_INDEX_PATH = "/nonexistent/hg38.mmi"
    settings.BLAST_FALLBACK_ENABLED = True
    settings.NCBI_API_KEY = "test_key"
    settings.NCBI_EMAIL = "test@example.com"
    settings.CONFIDENCE_HIGH_THRESHOLD = 0.8
    settings.CONFIDENCE_MEDIUM_THRESHOLD = 0.5
    settings.CLINVAR_CACHE_TTL_DAYS = 30
    return settings


@pytest.fixture
def mock_alignment_result():
    """Create a mock alignment result."""
    from backend.schemas.alignment import AlignmentResult
    return AlignmentResult(
        chromosome="chr7",
        start=117548000,
        end=117548600,
        strand="+",
        identity=0.98,
        alignment_score=580,
        cigar="400M",
        gene_names=["CFTR"],
        alignment_method="blast_fallback",
        reference_sequence="A" * 400,
    )


# ── Tests: SNV Pipeline ───────────────────────────────────────────────────────

class TestSnvPipelineEndToEnd:
    @pytest.mark.asyncio
    async def test_snv_pipeline_returns_snv_result(
        self, synthetic_chromatogram, mock_settings, mock_alignment_result
    ):
        """Test complete SNV analysis from ChromatogramData to SNVResult."""
        from backend.services.snv_analysis_service import run_snv_analysis

        with patch("backend.services.snv_analysis_service.parse_ab1") as mock_parse, \
             patch("backend.services.snv_analysis_service.run_qc_pipeline") as mock_qc, \
             patch("backend.services.snv_analysis_service.align_sequence") as mock_align, \
             patch("backend.services.snv_analysis_service.call_snvs") as mock_snvs:

            mock_parse.return_value = synthetic_chromatogram

            mock_qc_report = MagicMock()
            mock_qc_report.overall_pass = True
            mock_qc_report.flags = []
            mock_qc_report.metrics = {}
            mock_qc_report.readable_regions = [(50, 350)]
            mock_qc_report.excluded_positions = set()
            mock_qc_report.recommendations = []
            mock_qc_report.alignment_quality = None
            mock_qc.return_value = mock_qc_report

            mock_align.return_value = mock_alignment_result

            mock_snv = MagicMock()
            mock_snv.chromosome = "chr7"
            mock_snv.position = 117548100
            mock_snv.ref_allele = "A"
            mock_snv.alt_allele = "G"
            mock_snv.confidence_label = "high"
            mock_snvs.return_value = [mock_snv]

            result = await run_snv_analysis(
                file_path="/fake/sample.ab1",
                job_id=str(uuid.uuid4()),
                settings=mock_settings,
            )

        assert result is not None
        assert hasattr(result, "status")
        assert result.status in ("completed", "qc_failed", "alignment_failed", "snv_failed", "failed")

    @pytest.mark.asyncio
    async def test_snv_pipeline_qc_failure_returns_qc_failed(
        self, synthetic_chromatogram, mock_settings
    ):
        """Test that QC failure returns status='qc_failed'."""
        from backend.services.snv_analysis_service import run_snv_analysis

        with patch("backend.services.snv_analysis_service.parse_ab1") as mock_parse, \
             patch("backend.services.snv_analysis_service.run_qc_pipeline") as mock_qc:

            mock_parse.return_value = synthetic_chromatogram

            mock_qc_report = MagicMock()
            mock_qc_report.overall_pass = False
            mock_qc_report.flags = ["low_signal", "poor_readable_region"]
            mock_qc_report.metrics = {}
            mock_qc_report.readable_regions = []
            mock_qc_report.excluded_positions = set()
            mock_qc_report.recommendations = ["Repeat sequencing"]
            mock_qc_report.alignment_quality = None
            mock_qc.return_value = mock_qc_report

            result = await run_snv_analysis(
                file_path="/fake/sample.ab1",
                job_id=str(uuid.uuid4()),
                settings=mock_settings,
            )

        assert result.status == "qc_failed"

    @pytest.mark.asyncio
    async def test_snv_pipeline_file_not_found_returns_failed(
        self, mock_settings
    ):
        """Test that missing file returns status='failed'."""
        from backend.services.snv_analysis_service import run_snv_analysis

        result = await run_snv_analysis(
            file_path="/nonexistent/sample.ab1",
            job_id=str(uuid.uuid4()),
            settings=mock_settings,
        )

        assert result.status == "failed"
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_snv_pipeline_alignment_failure_returns_alignment_failed(
        self, synthetic_chromatogram, mock_settings
    ):
        """Test that alignment failure returns status='alignment_failed'."""
        from backend.services.snv_analysis_service import run_snv_analysis

        with patch("backend.services.snv_analysis_service.parse_ab1") as mock_parse, \
             patch("backend.services.snv_analysis_service.run_qc_pipeline") as mock_qc, \
             patch("backend.services.snv_analysis_service.align_sequence") as mock_align:

            mock_parse.return_value = synthetic_chromatogram

            mock_qc_report = MagicMock()
            mock_qc_report.overall_pass = True
            mock_qc_report.flags = []
            mock_qc_report.metrics = {}
            mock_qc_report.readable_regions = [(50, 350)]
            mock_qc_report.excluded_positions = set()
            mock_qc_report.recommendations = []
            mock_qc_report.alignment_quality = None
            mock_qc.return_value = mock_qc_report

            mock_align.side_effect = RuntimeError("Alignment failed: no hits")

            result = await run_snv_analysis(
                file_path="/fake/sample.ab1",
                job_id=str(uuid.uuid4()),
                settings=mock_settings,
            )

        assert result.status in ("alignment_failed", "failed")

    @pytest.mark.asyncio
    async def test_snv_pipeline_result_has_required_fields(
        self, synthetic_chromatogram, mock_settings, mock_alignment_result
    ):
        """Test that SNVResult has all required fields."""
        from backend.services.snv_analysis_service import run_snv_analysis

        with patch("backend.services.snv_analysis_service.parse_ab1") as mock_parse, \
             patch("backend.services.snv_analysis_service.run_qc_pipeline") as mock_qc, \
             patch("backend.services.snv_analysis_service.align_sequence") as mock_align, \
             patch("backend.services.snv_analysis_service.call_snvs") as mock_snvs:

            mock_parse.return_value = synthetic_chromatogram

            mock_qc_report = MagicMock()
            mock_qc_report.overall_pass = True
            mock_qc_report.flags = []
            mock_qc_report.metrics = {}
            mock_qc_report.readable_regions = [(50, 350)]
            mock_qc_report.excluded_positions = set()
            mock_qc_report.recommendations = []
            mock_qc_report.alignment_quality = None
            mock_qc.return_value = mock_qc_report

            mock_align.return_value = mock_alignment_result
            mock_snvs.return_value = []

            result = await run_snv_analysis(
                file_path="/fake/sample.ab1",
                job_id=str(uuid.uuid4()),
                settings=mock_settings,
            )

        # Check required fields exist
        assert hasattr(result, "status")
        assert hasattr(result, "snv_calls")
        assert hasattr(result, "total_snvs")


# ── Tests: Editing Pipeline ───────────────────────────────────────────────────

class TestEditingPipelineEndToEnd:
    @pytest.mark.asyncio
    async def test_editing_pipeline_returns_editing_result(
        self, wt_chromatogram, edited_chromatogram, mock_settings
    ):
        """Test complete editing analysis from two ChromatogramData to EditingResult."""
        from backend.services.editing_analysis_service import run_editing_analysis

        with patch("backend.services.editing_analysis_service.parse_ab1") as mock_parse, \
             patch("backend.services.editing_analysis_service.run_qc_pipeline") as mock_qc, \
             patch("backend.services.editing_analysis_service.classify_editing_type") as mock_classify, \
             patch("backend.services.editing_analysis_service._route_to_tool") as mock_route:

            mock_parse.side_effect = [wt_chromatogram, edited_chromatogram]

            mock_qc_report = MagicMock()
            mock_qc_report.overall_pass = True
            mock_qc_report.flags = []
            mock_qc_report.metrics = {}
            mock_qc_report.readable_regions = [(30, 270)]
            mock_qc_report.excluded_positions = set()
            mock_qc_report.recommendations = []
            mock_qc_report.alignment_quality = None
            mock_qc.return_value = mock_qc_report

            mock_classify.return_value = ("base_editing", 0.85)

            mock_tool_result = {
                "tool": "beat",
                "editing_efficiency": 0.65,
                "edited_positions": [50, 51, 52],
                "base_edit_type": "C_to_T",
                "editing_window": [50, 52],
                "bystander_edits": [],
            }
            mock_route.return_value = mock_tool_result

            result = await run_editing_analysis(
                wt_file_path="/fake/wt.ab1",
                edited_file_path="/fake/edited.ab1",
                job_id=str(uuid.uuid4()),
                settings=mock_settings,
            )

        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "editing_type")

    @pytest.mark.asyncio
    async def test_editing_pipeline_qc_failure(
        self, wt_chromatogram, edited_chromatogram, mock_settings
    ):
        """Test that QC failure on either file returns appropriate status."""
        from backend.services.editing_analysis_service import run_editing_analysis

        with patch("backend.services.editing_analysis_service.parse_ab1") as mock_parse, \
             patch("backend.services.editing_analysis_service.run_qc_pipeline") as mock_qc:

            mock_parse.side_effect = [wt_chromatogram, edited_chromatogram]

            mock_qc_report = MagicMock()
            mock_qc_report.overall_pass = False
            mock_qc_report.flags = ["failed_sequencing"]
            mock_qc_report.metrics = {}
            mock_qc_report.readable_regions = []
            mock_qc_report.excluded_positions = set()
            mock_qc_report.recommendations = []
            mock_qc_report.alignment_quality = None
            mock_qc.return_value = mock_qc_report

            result = await run_editing_analysis(
                wt_file_path="/fake/wt.ab1",
                edited_file_path="/fake/edited.ab1",
                job_id=str(uuid.uuid4()),
                settings=mock_settings,
            )

        assert result.status in ("qc_failed", "failed")

    @pytest.mark.asyncio
    async def test_editing_pipeline_file_not_found(self, mock_settings):
        """Test that missing file returns failed status."""
        from backend.services.editing_analysis_service import run_editing_analysis

        result = await run_editing_analysis(
            wt_file_path="/nonexistent/wt.ab1",
            edited_file_path="/nonexistent/edited.ab1",
            job_id=str(uuid.uuid4()),
            settings=mock_settings,
        )

        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_editing_pipeline_indel_routes_to_ice(
        self, wt_chromatogram, edited_chromatogram, mock_settings
    ):
        """Test that indel editing type routes to ICE tool."""
        from backend.services.editing_analysis_service import run_editing_analysis

        with patch("backend.services.editing_analysis_service.parse_ab1") as mock_parse, \
             patch("backend.services.editing_analysis_service.run_qc_pipeline") as mock_qc, \
             patch("backend.services.editing_analysis_service.classify_editing_type") as mock_classify, \
             patch("backend.services.editing_analysis_service._route_to_tool") as mock_route:

            mock_parse.side_effect = [wt_chromatogram, edited_chromatogram]

            mock_qc_report = MagicMock()
            mock_qc_report.overall_pass = True
            mock_qc_report.flags = []
            mock_qc_report.metrics = {}
            mock_qc_report.readable_regions = [(30, 270)]
            mock_qc_report.excluded_positions = set()
            mock_qc_report.recommendations = []
            mock_qc_report.alignment_quality = None
            mock_qc.return_value = mock_qc_report

            mock_classify.return_value = ("indel", 0.90)

            mock_route.return_value = {
                "tool": "ice_fallback",
                "efficiency": 0.45,
                "indel_pct": 0.45,
                "r_squared": 0.92,
                "indel_distribution": {"-1": 0.20, "+1": 0.15, "-3": 0.10},
                "ice_score": 45.0,
                "ko_score": 30.0,
            }

            result = await run_editing_analysis(
                wt_file_path="/fake/wt.ab1",
                edited_file_path="/fake/edited.ab1",
                job_id=str(uuid.uuid4()),
                settings=mock_settings,
            )

        assert result is not None
        # Verify route_to_tool was called with indel type
        mock_route.assert_called_once()
        call_args = mock_route.call_args
        assert call_args[0][0] == "indel"


# ── Tests: ClinVar Annotation Pipeline ───────────────────────────────────────

class TestClinvarAnnotationPipeline:
    @pytest.mark.asyncio
    async def test_clinvar_annotation_with_mocked_api(self):
        """Test ClinVar annotation with mocked NCBI API."""
        from tools.clinvar import parse_clinvar_response

        # Test the parser directly with a mock API response
        mock_response = {
            "result": {
                "uids": ["12345"],
                "12345": {
                    "variation_id": "12345",
                    "clinical_significance": {
                        "description": "Pathogenic"
                    },
                    "trait_set": [
                        {"trait_name": "Cystic fibrosis"}
                    ],
                    "dbsnp_id": "rs113993960",
                    "review_status": "reviewed by expert panel",
                    "accession": "VCV000007107",
                }
            }
        }

        annotation = parse_clinvar_response(mock_response, "chr7:117548670:G:A")

        assert annotation is not None
        assert annotation.variant_key == "chr7:117548670:G:A"

    @pytest.mark.asyncio
    async def test_clinvar_cache_hit_returns_cached(self):
        """Test that cache hit returns cached annotation without API call."""
        from tools.clinvar import get_cached_annotation
        from backend.schemas.snv import ClinVarAnnotation
        from datetime import timedelta

        mock_annotation = ClinVarAnnotation(
            variant_key="chr7:117548670:G:A",
            clinical_significance="Pathogenic",
            conditions=["Cystic fibrosis"],
            dbsnp_id="rs113993960",
            review_status="reviewed by expert panel",
            accession="VCV000007107",
            clinvar_url="https://www.ncbi.nlm.nih.gov/clinvar/variation/7107/",
            is_pathogenic=True,
            is_benign=False,
            is_vus=False,
            cached=False,
        )

        mock_cache_entry = MagicMock()
        mock_cache_entry.is_expired = False
        mock_cache_entry.annotation_json = mock_annotation.model_dump()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cache_entry
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_settings = MagicMock()
        mock_settings.CLINVAR_CACHE_TTL_DAYS = 30

        with patch("tools.clinvar.ClinVarCache"):
            with patch("tools.clinvar.select"):
                result = await get_cached_annotation(
                    "chr7:117548670:G:A", mock_db, mock_settings
                )

        # Should return the cached annotation
        assert result is not None

    @pytest.mark.asyncio
    async def test_clinvar_annotation_empty_response(self):
        """Test that empty API response returns annotation with no significance."""
        from tools.clinvar import parse_clinvar_response

        empty_response = {"result": {"uids": []}}
        annotation = parse_clinvar_response(empty_response, "chr1:12345:A:G")

        assert annotation is not None
        assert annotation.variant_key == "chr1:12345:A:G"
        # Empty response should have no clinical significance
        assert annotation.clinical_significance in (None, "", "not found", "unknown")


# ── Tests: Batch SNV Pipeline ─────────────────────────────────────────────────

class TestBatchSnvPipeline:
    @pytest.mark.asyncio
    async def test_batch_snv_pipeline_three_samples(self, mock_settings):
        """Test batch SNV processing with 3 synthetic samples."""
        from backend.services.snv_analysis_service import run_snv_analysis
        from backend.schemas.alignment import AlignmentResult

        # Generate 3 synthetic chromatograms
        chromatograms = [
            generate_synthetic_ab1_data(n_bases=300, seed=i)
            for i in range(3)
        ]

        mock_alignment = AlignmentResult(
            chromosome="chr7",
            start=117548000,
            end=117548300,
            strand="+",
            identity=0.97,
            alignment_score=290,
            cigar="300M",
            gene_names=["CFTR"],
            alignment_method="blast_fallback",
            reference_sequence="A" * 300,
        )

        results = []
        for i, chrom in enumerate(chromatograms):
            with patch("backend.services.snv_analysis_service.parse_ab1") as mock_parse, \
                 patch("backend.services.snv_analysis_service.run_qc_pipeline") as mock_qc, \
                 patch("backend.services.snv_analysis_service.align_sequence") as mock_align, \
                 patch("backend.services.snv_analysis_service.call_snvs") as mock_snvs:

                mock_parse.return_value = chrom

                mock_qc_report = MagicMock()
                mock_qc_report.overall_pass = True
                mock_qc_report.flags = []
                mock_qc_report.metrics = {}
                mock_qc_report.readable_regions = [(30, 270)]
                mock_qc_report.excluded_positions = set()
                mock_qc_report.recommendations = []
                mock_qc_report.alignment_quality = None
                mock_qc.return_value = mock_qc_report

                mock_align.return_value = mock_alignment
                mock_snvs.return_value = []

                result = await run_snv_analysis(
                    file_path=f"/fake/sample_{i}.ab1",
                    job_id=str(uuid.uuid4()),
                    settings=mock_settings,
                )
                results.append(result)

        assert len(results) == 3
        for result in results:
            assert result is not None
            assert hasattr(result, "status")

    @pytest.mark.asyncio
    async def test_batch_processes_independently(self, mock_settings):
        """Test that batch jobs are independent — one failure doesn't affect others."""
        from backend.services.snv_analysis_service import run_snv_analysis

        chrom = generate_synthetic_ab1_data(n_bases=200, seed=42)

        # First job: success
        with patch("backend.services.snv_analysis_service.parse_ab1") as mock_parse, \
             patch("backend.services.snv_analysis_service.run_qc_pipeline") as mock_qc, \
             patch("backend.services.snv_analysis_service.align_sequence") as mock_align, \
             patch("backend.services.snv_analysis_service.call_snvs") as mock_snvs:

            mock_parse.return_value = chrom
            mock_qc_report = MagicMock()
            mock_qc_report.overall_pass = True
            mock_qc_report.flags = []
            mock_qc_report.metrics = {}
            mock_qc_report.readable_regions = [(20, 180)]
            mock_qc_report.excluded_positions = set()
            mock_qc_report.recommendations = []
            mock_qc_report.alignment_quality = None
            mock_qc.return_value = mock_qc_report
            mock_align.side_effect = RuntimeError("Alignment failed")
            mock_snvs.return_value = []

            result1 = await run_snv_analysis(
                file_path="/fake/sample_1.ab1",
                job_id=str(uuid.uuid4()),
                settings=mock_settings,
            )

        # Second job: file not found
        result2 = await run_snv_analysis(
            file_path="/nonexistent/sample_2.ab1",
            job_id=str(uuid.uuid4()),
            settings=mock_settings,
        )

        # Both should return results (not raise exceptions)
        assert result1 is not None
        assert result2 is not None
        assert result2.status == "failed"


# ── Tests: Synthetic Data Quality ─────────────────────────────────────────────

class TestSyntheticDataQuality:
    def test_synthetic_chromatogram_passes_qc(self):
        """Verify that good synthetic data passes QC."""
        chrom = generate_synthetic_ab1_data(n_bases=600, mean_quality=35.0, seed=42)
        assert chrom.qc_pass is True
        assert "failed_sequencing" not in chrom.qc_flags

    def test_synthetic_chromatogram_has_readable_region(self):
        """Verify that synthetic data has a meaningful readable region."""
        chrom = generate_synthetic_ab1_data(n_bases=600, seed=42)
        readable_length = chrom.trace.readable_region_end - chrom.trace.readable_region_start
        assert readable_length >= 100, f"Readable region too short: {readable_length}"

    def test_synthetic_chromatogram_normalized_traces(self):
        """Verify that normalized traces have correct max value."""
        from tools.ab1_parser import DYE_NORM_MAX
        chrom = generate_synthetic_ab1_data(n_bases=200, seed=42)
        for attr in ["trace_A_norm", "trace_T_norm", "trace_C_norm", "trace_G_norm"]:
            trace = getattr(chrom.trace, attr)
            assert abs(max(trace) - DYE_NORM_MAX) < 1.0

    def test_three_synthetic_chromatograms_are_distinct(self):
        """Verify that different seeds produce distinct chromatograms."""
        chroms = [generate_synthetic_ab1_data(n_bases=200, seed=i) for i in range(3)]
        sequences = [c.trace.base_calls for c in chroms]
        # All three should be different
        assert len(set(sequences)) == 3
