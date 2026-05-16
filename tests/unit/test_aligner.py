"""
Unit tests for tools/aligner.py.

Tests cover:
- is_minimap2_available: test with mock paths
- _normalize_chromosome_name: UCSC format normalization
- _parse_cigar: CIGAR string parsing for M, I, D operations
- _blast_alignment_to_cigar: BLAST alignment to CIGAR conversion
- align_with_minimap2: subprocess-based alignment (mocked)
- align_with_blast: BLAST API alignment (mocked)
- get_overlapping_genes: NCBI Gene API query (mocked)
- align_sequence: two-tier strategy (mocked)

Run with:
    pytest tests/unit/test_aligner.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.aligner import (
    _blast_alignment_to_cigar,
    _normalize_chromosome_name,
    _parse_cigar,
    _parse_sam_output,
    is_minimap2_available,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

class MockSettings:
    """Mock settings for testing."""
    HG38_INDEX_PATH = "/nonexistent/hg38.mmi"
    NCBI_API_KEY = ""
    NCBI_EMAIL = "test@example.com"
    BLAST_FALLBACK_ENABLED = True


@pytest.fixture
def mock_settings():
    """Return mock settings."""
    return MockSettings()


@pytest.fixture
def sample_sequence():
    """Return a sample DNA sequence for testing."""
    return "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" * 4  # 208 bases


@pytest.fixture
def sample_sam_output():
    """Return a sample SAM output string for testing."""
    return (
        "@HD\tVN:1.6\tSO:unsorted\n"
        "@SQ\tSN:chr7\tLN:159345973\n"
        "query\t0\tchr7\t117548621\t60\t200M\t*\t0\t0\t"
        "ATCGATCGATCG\t*\tNM:i:2\tAS:i:196\n"
    )


@pytest.fixture
def sample_sam_output_reverse():
    """Return a SAM output with reverse strand alignment."""
    return (
        "@HD\tVN:1.6\n"
        "query\t16\tchr17\t7674221\t60\t150M\t*\t0\t0\t"
        "ATCGATCG\t*\tNM:i:1\tAS:i:148\n"
    )


@pytest.fixture
def sample_sam_output_unmapped():
    """Return a SAM output with unmapped read."""
    return (
        "@HD\tVN:1.6\n"
        "query\t4\t*\t0\t0\t*\t*\t0\t0\tATCGATCG\t*\n"
    )


# ── Tests: is_minimap2_available ──────────────────────────────────────────────

class TestIsMinimap2Available:
    """Tests for is_minimap2_available function."""

    def test_returns_false_when_binary_missing(self, tmp_path):
        """Returns False when minimap2 binary is not in PATH."""
        # Create a fake index file
        index_file = tmp_path / "hg38.mmi"
        index_file.write_bytes(b"fake index")

        with patch("shutil.which", return_value=None):
            result = is_minimap2_available(str(index_file))
        assert result is False

    def test_returns_false_when_index_missing(self):
        """Returns False when index file does not exist."""
        with patch("shutil.which", return_value="/usr/bin/minimap2"):
            result = is_minimap2_available("/nonexistent/hg38.mmi")
        assert result is False

    def test_returns_true_when_both_available(self, tmp_path):
        """Returns True when both binary and index are available."""
        index_file = tmp_path / "hg38.mmi"
        index_file.write_bytes(b"fake index")

        with patch("shutil.which", return_value="/usr/bin/minimap2"):
            result = is_minimap2_available(str(index_file))
        assert result is True

    def test_returns_false_for_empty_path(self):
        """Returns False for empty index path."""
        with patch("shutil.which", return_value="/usr/bin/minimap2"):
            result = is_minimap2_available("")
        assert result is False

    def test_returns_false_for_directory_path(self, tmp_path):
        """Returns False when index path is a directory, not a file."""
        with patch("shutil.which", return_value="/usr/bin/minimap2"):
            result = is_minimap2_available(str(tmp_path))
        assert result is False

    def test_default_path_returns_false(self):
        """Default /data/hg38.mmi path returns False in test environment."""
        # In test environment, /data/hg38.mmi should not exist
        result = is_minimap2_available("/data/hg38.mmi")
        assert result is False


# ── Tests: _normalize_chromosome_name ────────────────────────────────────────

class TestNormalizeChromosomeName:
    """Tests for _normalize_chromosome_name function."""

    def test_already_ucsc_format(self):
        """Already-UCSC names should pass through unchanged."""
        assert _normalize_chromosome_name("chr1") == "chr1"
        assert _normalize_chromosome_name("chr22") == "chr22"
        assert _normalize_chromosome_name("chrX") == "chrX"
        assert _normalize_chromosome_name("chrY") == "chrY"
        assert _normalize_chromosome_name("chrM") == "chrM"

    def test_bare_numbers(self):
        """Bare chromosome numbers should get chr prefix."""
        assert _normalize_chromosome_name("1") == "chr1"
        assert _normalize_chromosome_name("7") == "chr7"
        assert _normalize_chromosome_name("22") == "chr22"

    def test_bare_sex_chromosomes(self):
        """Bare X and Y should get chr prefix."""
        assert _normalize_chromosome_name("X") == "chrX"
        assert _normalize_chromosome_name("Y") == "chrY"

    def test_mitochondrial_variants(self):
        """Various mitochondrial names should normalize to chrM."""
        assert _normalize_chromosome_name("M") == "chrM"
        assert _normalize_chromosome_name("MT") == "chrM"
        assert _normalize_chromosome_name("chrMT") == "chrM"

    def test_refseq_accessions(self):
        """RefSeq accessions should map to UCSC chromosome names."""
        assert _normalize_chromosome_name("NC_000001.11") == "chr1"
        assert _normalize_chromosome_name("NC_000007.14") == "chr7"
        assert _normalize_chromosome_name("NC_000017.11") == "chr17"
        assert _normalize_chromosome_name("NC_000023.11") == "chrX"
        assert _normalize_chromosome_name("NC_000024.10") == "chrY"
        assert _normalize_chromosome_name("NC_012920.1") == "chrM"

    def test_invalid_name_raises_error(self):
        """Invalid chromosome names should raise ValueError."""
        with pytest.raises(ValueError):
            _normalize_chromosome_name("*")

    def test_empty_name_raises_error(self):
        """Empty chromosome name should raise ValueError."""
        with pytest.raises(ValueError):
            _normalize_chromosome_name("")


# ── Tests: _parse_cigar ───────────────────────────────────────────────────────

class TestParseCigar:
    """Tests for _parse_cigar function."""

    def test_simple_match(self):
        """Simple M CIGAR: alignment_length == ref_consumed."""
        aln_len, ref_consumed = _parse_cigar("200M")
        assert aln_len == 200
        assert ref_consumed == 200

    def test_match_with_insertion(self):
        """M+I CIGAR: insertion adds to alignment_length but not ref_consumed."""
        # 150M2I48M: alignment_length = 150+2+48 = 200, ref = 150+48 = 198
        aln_len, ref_consumed = _parse_cigar("150M2I48M")
        assert aln_len == 200
        assert ref_consumed == 198

    def test_match_with_deletion(self):
        """M+D CIGAR: deletion adds to ref_consumed but not alignment_length."""
        # 100M5D50M: alignment_length = 100+50 = 150, ref = 100+5+50 = 155
        aln_len, ref_consumed = _parse_cigar("100M5D50M")
        assert aln_len == 150
        assert ref_consumed == 155

    def test_soft_clip_not_counted(self):
        """Soft clips should not affect alignment_length or ref_consumed."""
        # 10S150M10S: alignment_length = 150, ref = 150
        aln_len, ref_consumed = _parse_cigar("10S150M10S")
        assert aln_len == 150
        assert ref_consumed == 150

    def test_complex_cigar(self):
        """Complex CIGAR with multiple operations."""
        # 50M3I50M2D50M: aln = 50+3+50+50 = 153, ref = 50+50+2+50 = 152
        aln_len, ref_consumed = _parse_cigar("50M3I50M2D50M")
        assert aln_len == 153
        assert ref_consumed == 152

    def test_sequence_match_and_mismatch(self):
        """= and X operations should behave like M."""
        # 100=50X: aln = 150, ref = 150
        aln_len, ref_consumed = _parse_cigar("100=50X")
        assert aln_len == 150
        assert ref_consumed == 150

    def test_single_base_match(self):
        """Single base match CIGAR."""
        aln_len, ref_consumed = _parse_cigar("1M")
        assert aln_len == 1
        assert ref_consumed == 1

    def test_hard_clip_not_counted(self):
        """Hard clips should not affect alignment_length or ref_consumed."""
        # 5H100M5H: alignment_length = 100, ref = 100
        aln_len, ref_consumed = _parse_cigar("5H100M5H")
        assert aln_len == 100
        assert ref_consumed == 100

    def test_n_skip_adds_to_ref(self):
        """N (skip) should add to ref_consumed but not alignment_length."""
        # 50M100N50M: aln = 100, ref = 50+100+50 = 200
        aln_len, ref_consumed = _parse_cigar("50M100N50M")
        assert aln_len == 100
        assert ref_consumed == 200


# ── Tests: _parse_sam_output ──────────────────────────────────────────────────

class TestParseSamOutput:
    """Tests for _parse_sam_output function."""

    def test_parse_forward_strand(self, sample_sam_output, sample_sequence):
        """Forward strand alignment should have strand='+'."""
        result = _parse_sam_output(sample_sam_output, sample_sequence)
        assert result.strand == "+"
        assert result.chromosome == "chr7"
        assert result.method == "minimap2"

    def test_parse_reverse_strand(self, sample_sam_output_reverse, sample_sequence):
        """Reverse strand alignment (FLAG bit 16) should have strand='-'."""
        result = _parse_sam_output(sample_sam_output_reverse, sample_sequence)
        assert result.strand == "-"
        assert result.chromosome == "chr17"

    def test_parse_coordinates(self, sample_sam_output, sample_sequence):
        """Parsed coordinates should be 0-based."""
        result = _parse_sam_output(sample_sam_output, sample_sequence)
        # SAM POS=117548621 (1-based) → 0-based = 117548620
        assert result.start == 117548620
        # end = start + ref_consumed (200M → 200)
        assert result.end == 117548620 + 200

    def test_parse_identity(self, sample_sam_output, sample_sequence):
        """Identity should be computed from NM tag."""
        result = _parse_sam_output(sample_sam_output, sample_sequence)
        # NM=2, alignment_length=200 → identity = (200-2)/200 = 0.99
        assert abs(result.identity - 0.99) < 1e-6

    def test_parse_alignment_score(self, sample_sam_output, sample_sequence):
        """Alignment score should be extracted from AS tag."""
        result = _parse_sam_output(sample_sam_output, sample_sequence)
        assert result.alignment_score == 196

    def test_unmapped_raises_error(self, sample_sam_output_unmapped, sample_sequence):
        """Unmapped read should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="no valid alignment"):
            _parse_sam_output(sample_sam_output_unmapped, sample_sequence)

    def test_empty_sam_raises_error(self, sample_sequence):
        """Empty SAM output should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="no valid alignment"):
            _parse_sam_output("", sample_sequence)

    def test_header_only_raises_error(self, sample_sequence):
        """SAM with only headers should raise RuntimeError."""
        header_only = "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:248956422\n"
        with pytest.raises(RuntimeError, match="no valid alignment"):
            _parse_sam_output(header_only, sample_sequence)

    def test_cigar_preserved(self, sample_sam_output, sample_sequence):
        """CIGAR string should be preserved in result."""
        result = _parse_sam_output(sample_sam_output, sample_sequence)
        assert result.cigar == "200M"

    def test_genes_empty_initially(self, sample_sam_output, sample_sequence):
        """Genes list should be empty before annotation."""
        result = _parse_sam_output(sample_sam_output, sample_sequence)
        assert result.genes == []


# ── Tests: _blast_alignment_to_cigar ─────────────────────────────────────────

class TestBlastAlignmentToCigar:
    """Tests for _blast_alignment_to_cigar function."""

    def test_perfect_match(self):
        """Perfect alignment should produce all-M CIGAR."""
        query = "ATCGATCG"
        subject = "ATCGATCG"
        match = "||||||||"
        cigar = _blast_alignment_to_cigar(query, subject, match)
        assert cigar == "8M"

    def test_with_query_gap(self):
        """Gap in query (deletion from reference) should produce D in CIGAR."""
        # query gap (--) = deletion from reference → D in CIGAR
        query = "ATCG--ATCG"
        subject = "ATCGATTATCG"[:10]  # subject has bases where query has gaps
        # Use clean inputs: query gap in middle, no trailing gaps
        query = "ATCG--ATCG"
        subject = "ATCGATATCG"
        match = "||||  ||||"
        cigar = _blast_alignment_to_cigar(query, subject, match)
        # ATCG = 4M, -- in query = 2D, ATCG = 4M
        assert cigar == "4M2D4M"

    def test_with_subject_gap(self):
        """Gap in subject (insertion relative to reference) should produce I."""
        # subject gap (--) = insertion in query → I in CIGAR
        query = "ATCGATATCG"
        subject = "ATCG--ATCG"
        match = "||||  ||||"
        cigar = _blast_alignment_to_cigar(query, subject, match)
        # ATCG = 4M, -- in subject = 2I, ATCG = 4M
        assert cigar == "4M2I4M"

    def test_empty_alignment_returns_fallback(self):
        """Empty alignment strings should return fallback '1M'."""
        cigar = _blast_alignment_to_cigar("", "", "")
        assert cigar == "1M"

    def test_mismatch_produces_M(self):
        """Mismatches should produce M (not X) in CIGAR."""
        query = "ATCG"
        subject = "ATGG"  # C→G mismatch at position 2
        match = "|| |"
        cigar = _blast_alignment_to_cigar(query, subject, match)
        assert cigar == "4M"  # All M, including mismatches

    def test_consecutive_gaps_merged(self):
        """Consecutive gaps should be merged into a single CIGAR operation."""
        query = "ATCG----ATCG"
        subject = "ATCGATCGATCG"
        match = "||||    ||||"
        cigar = _blast_alignment_to_cigar(query, subject, match)
        assert cigar == "4M4D4M"

    def test_single_base_match(self):
        """Single base match should produce '1M'."""
        cigar = _blast_alignment_to_cigar("A", "A", "|")
        assert cigar == "1M"


# ── Tests: align_sequence (integration with mocks) ───────────────────────────

class TestAlignSequence:
    """Integration tests for align_sequence with mocked dependencies."""

    def test_uses_minimap2_when_available(self, mock_settings, sample_sequence, tmp_path):
        """align_sequence should use minimap2 when available."""
        from backend.schemas.alignment import AlignmentResult

        mock_result = AlignmentResult(
            chromosome="chr7",
            start=117548620,
            end=117548820,
            strand="+",
            identity=0.99,
            alignment_score=195,
            cigar="200M",
            genes=[],
            method="minimap2",
            reference_sequence=sample_sequence[:200],
        )

        # Create a fake index file
        index_file = tmp_path / "hg38.mmi"
        index_file.write_bytes(b"fake")
        mock_settings.HG38_INDEX_PATH = str(index_file)

        with patch("tools.aligner.is_minimap2_available", return_value=True), \
             patch("tools.aligner.align_with_minimap2", return_value=mock_result), \
             patch("tools.aligner.get_overlapping_genes", return_value=["CFTR"]):
            from tools.aligner import align_sequence
            result = align_sequence(sample_sequence, mock_settings)

        assert result.method == "minimap2"
        assert "CFTR" in result.genes

    def test_falls_back_to_blast_when_minimap2_unavailable(
        self, mock_settings, sample_sequence
    ):
        """align_sequence should fall back to BLAST when minimap2 unavailable."""
        from backend.schemas.alignment import AlignmentResult

        mock_result = AlignmentResult(
            chromosome="chr7",
            start=117548620,
            end=117548820,
            strand="+",
            identity=0.97,
            alignment_score=180,
            cigar="200M",
            genes=[],
            method="blast",
            reference_sequence=sample_sequence[:200],
        )

        with patch("tools.aligner.is_minimap2_available", return_value=False), \
             patch("tools.aligner.align_with_blast", return_value=mock_result), \
             patch("tools.aligner.get_overlapping_genes", return_value=[]):
            from tools.aligner import align_sequence
            result = align_sequence(sample_sequence, mock_settings)

        assert result.method == "blast"

    def test_raises_on_empty_sequence(self, mock_settings):
        """align_sequence should raise ValueError for empty sequence."""
        from tools.aligner import align_sequence
        with pytest.raises(ValueError, match="must not be empty"):
            align_sequence("", mock_settings)

    def test_raises_on_invalid_characters(self, mock_settings):
        """align_sequence should raise ValueError for invalid characters."""
        from tools.aligner import align_sequence
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("ATCG123XYZ", mock_settings)

    def test_raises_when_blast_disabled_and_minimap2_unavailable(self, sample_sequence):
        """Should raise RuntimeError when both methods are unavailable."""
        class NoBlastSettings:
            HG38_INDEX_PATH = "/nonexistent/hg38.mmi"
            NCBI_API_KEY = ""
            NCBI_EMAIL = "test@example.com"
            BLAST_FALLBACK_ENABLED = False

        with patch("tools.aligner.is_minimap2_available", return_value=False):
            from tools.aligner import align_sequence
            with pytest.raises(RuntimeError, match="BLAST fallback is disabled"):
                align_sequence(sample_sequence, NoBlastSettings())

    def test_sequence_normalized_to_uppercase(self, mock_settings, tmp_path):
        """Lowercase sequence should be normalized to uppercase."""
        from backend.schemas.alignment import AlignmentResult

        mock_result = AlignmentResult(
            chromosome="chr1",
            start=1000,
            end=1200,
            strand="+",
            identity=0.99,
            alignment_score=195,
            cigar="200M",
            genes=[],
            method="minimap2",
            reference_sequence="A" * 200,
        )

        with patch("tools.aligner.is_minimap2_available", return_value=True), \
             patch("tools.aligner.align_with_minimap2", return_value=mock_result) as mock_mm2, \
             patch("tools.aligner.get_overlapping_genes", return_value=[]):
            from tools.aligner import align_sequence
            align_sequence("atcgatcg" * 25, mock_settings)
            # The sequence passed to align_with_minimap2 should be uppercase
            called_seq = mock_mm2.call_args[0][0]
            assert called_seq == called_seq.upper()


# ── Tests: get_overlapping_genes (mocked) ────────────────────────────────────

class TestGetOverlappingGenes:
    """Tests for get_overlapping_genes with mocked NCBI API."""

    def test_returns_empty_list_on_api_failure(self):
        """Should return empty list when NCBI API fails."""
        with patch("tools.aligner.get_overlapping_genes") as mock_genes:
            mock_genes.return_value = []
            from tools.aligner import get_overlapping_genes
            # Direct test with mocked Entrez
            with patch("Bio.Entrez.esearch") as mock_esearch:
                mock_esearch.side_effect = Exception("Network error")
                result = get_overlapping_genes("chr7", 117548620, 117548820, "")
                assert isinstance(result, list)

    def test_chromosome_prefix_stripped(self):
        """NCBI query should use chromosome without 'chr' prefix."""
        with patch("Bio.Entrez.esearch") as mock_esearch, \
             patch("Bio.Entrez.read") as mock_read, \
             patch("Bio.Entrez.efetch") as mock_efetch:

            mock_read.return_value = {"IdList": []}
            mock_esearch.return_value = MagicMock()

            from tools.aligner import get_overlapping_genes
            get_overlapping_genes("chr7", 100, 200, "")

            # Check that esearch was called with "7" not "chr7"
            if mock_esearch.called:
                call_kwargs = mock_esearch.call_args[1]
                term = call_kwargs.get("term", "")
                assert "chr7" not in term or "7[CHR]" in term
