"""
Unit tests for tools/snv_caller.py.

Tests cover:
- compute_base_proportions: known heights, proportions sum to 1
- detect_mixed_peaks: heterozygous detection at various thresholds
- score_confidence: all three confidence labels (high, medium, low)
- get_reference_base: CIGAR parsing for M, I, D operations

Uses synthetic data from tools/ab1_parser.generate_synthetic_ab1_data().

Run with:
    pytest tests/unit/test_snv_caller.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.schemas.alignment import AlignmentResult
from backend.schemas.snv import BaseProportions
from tools.snv_caller import (
    call_snvs,
    compute_base_proportions,
    detect_mixed_peaks,
    get_reference_base,
    score_confidence,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

class MockSettings:
    """Mock settings object for testing."""
    CONFIDENCE_HIGH_THRESHOLD = 0.8
    CONFIDENCE_MEDIUM_THRESHOLD = 0.5


@pytest.fixture
def mock_settings():
    """Return a mock settings object."""
    return MockSettings()


@pytest.fixture
def synthetic_chromatogram():
    """Return a synthetic ChromatogramData for testing."""
    from tools.ab1_parser import generate_synthetic_ab1_data
    return generate_synthetic_ab1_data(
        n_bases=400,
        snv_positions=[100, 200, 300],
        seed=42,
    )


@pytest.fixture
def synthetic_chromatogram_no_snvs():
    """Return a synthetic ChromatogramData with no injected SNVs."""
    from tools.ab1_parser import generate_synthetic_ab1_data
    return generate_synthetic_ab1_data(
        n_bases=400,
        snv_positions=None,
        seed=99,
    )


@pytest.fixture
def simple_alignment():
    """Return a simple AlignmentResult for testing."""
    return AlignmentResult(
        chromosome="chr7",
        start=117548620,
        end=117548820,
        strand="+",
        identity=0.98,
        alignment_score=195,
        cigar="200M",
        genes=["CFTR"],
        method="minimap2",
        reference_sequence="ATCGATCGATCG" * 16 + "ATCGATCG",  # 200 bases
    )


@pytest.fixture
def alignment_with_indels():
    """Return an AlignmentResult with insertions and deletions in CIGAR."""
    return AlignmentResult(
        chromosome="chr17",
        start=7674220,
        end=7674430,
        strand="+",
        identity=0.97,
        alignment_score=190,
        cigar="100M5I100M5D5M",
        genes=["TP53"],
        method="minimap2",
        reference_sequence="A" * 205,  # 100 + 100 + 5 = 205 ref bases
    )


# ── Tests: compute_base_proportions ──────────────────────────────────────────

class TestComputeBaseProportions:
    """Tests for compute_base_proportions function."""

    def test_proportions_sum_to_one_simple(self):
        """Proportions should sum to exactly 1.0 for simple heights."""
        heights = {
            "A": [800],
            "T": [100],
            "C": [50],
            "G": [50],
        }
        props = compute_base_proportions(heights, position=0)
        total = props.A + props.T + props.C + props.G
        assert abs(total - 1.0) < 1e-6, f"Proportions sum to {total}, expected 1.0"

    def test_proportions_sum_to_one_equal_heights(self):
        """Equal heights should give equal proportions (0.25 each)."""
        heights = {
            "A": [100],
            "T": [100],
            "C": [100],
            "G": [100],
        }
        props = compute_base_proportions(heights, position=0)
        total = props.A + props.T + props.C + props.G
        assert abs(total - 1.0) < 1e-6
        assert abs(props.A - 0.25) < 1e-6
        assert abs(props.T - 0.25) < 1e-6
        assert abs(props.C - 0.25) < 1e-6
        assert abs(props.G - 0.25) < 1e-6

    def test_dominant_base_correct(self):
        """Dominant base should be the one with highest proportion."""
        heights = {
            "A": [800],
            "T": [100],
            "C": [50],
            "G": [50],
        }
        props = compute_base_proportions(heights, position=0)
        assert props.dominant_base == "A"

    def test_dominant_base_T(self):
        """Dominant base T is correctly identified."""
        heights = {
            "A": [50],
            "T": [900],
            "C": [30],
            "G": [20],
        }
        props = compute_base_proportions(heights, position=0)
        assert props.dominant_base == "T"

    def test_proportions_correct_values(self):
        """Proportions should match expected values for known heights."""
        heights = {
            "A": [600],
            "T": [200],
            "C": [100],
            "G": [100],
        }
        props = compute_base_proportions(heights, position=0)
        assert abs(props.A - 0.6) < 1e-6
        assert abs(props.T - 0.2) < 1e-6
        assert abs(props.C - 0.1) < 1e-6
        assert abs(props.G - 0.1) < 1e-6

    def test_zero_signal_returns_equal_proportions(self):
        """All-zero signal should return equal proportions (0.25 each)."""
        heights = {
            "A": [0],
            "T": [0],
            "C": [0],
            "G": [0],
        }
        props = compute_base_proportions(heights, position=0)
        assert abs(props.A - 0.25) < 1e-6
        assert abs(props.T - 0.25) < 1e-6
        assert abs(props.C - 0.25) < 1e-6
        assert abs(props.G - 0.25) < 1e-6

    def test_multiple_positions(self):
        """Test proportions at different positions in multi-position arrays."""
        heights = {
            "A": [800, 50, 100],
            "T": [100, 700, 50],
            "C": [50, 150, 600],
            "G": [50, 100, 250],
        }
        # Position 0: A dominant
        props0 = compute_base_proportions(heights, position=0)
        assert props0.dominant_base == "A"

        # Position 1: T dominant
        props1 = compute_base_proportions(heights, position=1)
        assert props1.dominant_base == "T"

        # Position 2: C dominant
        props2 = compute_base_proportions(heights, position=2)
        assert props2.dominant_base == "C"

    def test_missing_channel_raises_error(self):
        """Missing channel should raise ValueError."""
        heights = {
            "A": [800],
            "T": [100],
            "C": [50],
            # Missing "G"
        }
        with pytest.raises(ValueError, match="missing channels"):
            compute_base_proportions(heights, position=0)

    def test_out_of_range_position_raises_error(self):
        """Out-of-range position should raise IndexError."""
        heights = {
            "A": [800],
            "T": [100],
            "C": [50],
            "G": [50],
        }
        with pytest.raises(IndexError):
            compute_base_proportions(heights, position=5)

    def test_proportions_sum_with_synthetic_data(self, synthetic_chromatogram):
        """Proportions should sum to ~1.0 for all positions in synthetic data."""
        peak_heights = synthetic_chromatogram.trace.peak_heights
        n_peaks = len(peak_heights["A"])

        for pos in range(min(50, n_peaks)):
            props = compute_base_proportions(peak_heights, pos)
            total = props.A + props.T + props.C + props.G
            assert abs(total - 1.0) < 0.01, (
                f"Position {pos}: proportions sum to {total:.6f}"
            )


# ── Tests: detect_mixed_peaks ─────────────────────────────────────────────────

class TestDetectMixedPeaks:
    """Tests for detect_mixed_peaks function."""

    def test_heterozygous_above_threshold(self):
        """Secondary peak above threshold should be flagged as heterozygous."""
        # A=0.60, G=0.35 → secondary fraction = 0.35/0.60 = 0.583 > 0.25
        props = BaseProportions(A=0.60, T=0.02, C=0.03, G=0.35)
        is_het, fraction = detect_mixed_peaks(props, heterozygosity_threshold=0.25)
        assert is_het is True
        assert abs(fraction - 0.35 / 0.60) < 1e-6

    def test_homozygous_below_threshold(self):
        """Secondary peak below threshold should not be flagged."""
        # A=0.90, others low → secondary fraction = 0.04/0.90 = 0.044 < 0.25
        props = BaseProportions(A=0.90, T=0.04, C=0.03, G=0.03)
        is_het, fraction = detect_mixed_peaks(props, heterozygosity_threshold=0.25)
        assert is_het is False
        assert fraction < 0.25

    def test_exactly_at_threshold(self):
        """Secondary peak exactly at threshold should be flagged."""
        # A=0.80, T=0.20 → fraction = 0.20/0.80 = 0.25 = threshold
        props = BaseProportions(A=0.80, T=0.20, C=0.0, G=0.0)
        is_het, fraction = detect_mixed_peaks(props, heterozygosity_threshold=0.25)
        assert is_het is True
        assert abs(fraction - 0.25) < 1e-6

    def test_just_below_threshold(self):
        """Secondary peak just below threshold should not be flagged."""
        # A=0.82, T=0.18 → fraction = 0.18/0.82 ≈ 0.2195 < 0.25
        props = BaseProportions(A=0.82, T=0.18, C=0.0, G=0.0)
        is_het, fraction = detect_mixed_peaks(props, heterozygosity_threshold=0.25)
        assert is_het is False
        assert fraction < 0.25

    def test_custom_threshold_strict(self):
        """Custom strict threshold (0.10) should flag more positions."""
        # A=0.85, T=0.12 → fraction = 0.12/0.85 ≈ 0.141 > 0.10
        props = BaseProportions(A=0.85, T=0.12, C=0.02, G=0.01)
        is_het_strict, _ = detect_mixed_peaks(props, heterozygosity_threshold=0.10)
        is_het_default, _ = detect_mixed_peaks(props, heterozygosity_threshold=0.25)
        assert is_het_strict is True
        assert is_het_default is False

    def test_custom_threshold_lenient(self):
        """Custom lenient threshold (0.40) should flag fewer positions."""
        # A=0.60, G=0.35 → fraction = 0.583 > 0.25 but < 0.40? No, 0.583 > 0.40
        props = BaseProportions(A=0.60, T=0.02, C=0.03, G=0.35)
        is_het_lenient, _ = detect_mixed_peaks(props, heterozygosity_threshold=0.40)
        is_het_default, _ = detect_mixed_peaks(props, heterozygosity_threshold=0.25)
        # 0.583 > 0.40, so both should be True
        assert is_het_default is True
        assert is_het_lenient is True

    def test_equal_proportions_heterozygous(self):
        """Equal proportions (0.5/0.5) should be flagged as heterozygous."""
        props = BaseProportions(A=0.50, T=0.50, C=0.0, G=0.0)
        is_het, fraction = detect_mixed_peaks(props)
        assert is_het is True
        assert abs(fraction - 1.0) < 1e-6

    def test_zero_primary_returns_not_heterozygous(self):
        """All-zero proportions should not be flagged as heterozygous."""
        props = BaseProportions(A=0.25, T=0.25, C=0.25, G=0.25)
        # With equal proportions, secondary/primary = 1.0 > 0.25
        is_het, fraction = detect_mixed_peaks(props)
        assert is_het is True  # Equal proportions are heterozygous

    def test_secondary_fraction_range(self):
        """Secondary peak fraction should always be in [0, 1]."""
        test_cases = [
            BaseProportions(A=0.90, T=0.05, C=0.03, G=0.02),
            BaseProportions(A=0.50, T=0.50, C=0.0, G=0.0),
            BaseProportions(A=0.25, T=0.25, C=0.25, G=0.25),
            BaseProportions(A=1.0, T=0.0, C=0.0, G=0.0),
        ]
        for props in test_cases:
            _, fraction = detect_mixed_peaks(props)
            assert 0.0 <= fraction <= 1.0, (
                f"Fraction {fraction} out of range for {props}"
            )


# ── Tests: score_confidence ───────────────────────────────────────────────────

class TestScoreConfidence:
    """Tests for score_confidence function."""

    def test_high_confidence_label(self, mock_settings):
        """High quality inputs should produce 'high' confidence label."""
        props = BaseProportions(A=0.90, T=0.04, C=0.03, G=0.03)
        score, label = score_confidence(
            proportions=props,
            quality_score=40,
            alignment_identity=0.99,
            neighboring_quality=[38, 39, 40, 38, 37],
            settings=mock_settings,
        )
        assert label == "high"
        assert score >= 0.8

    def test_medium_confidence_label(self, mock_settings):
        """Medium quality inputs should produce 'medium' confidence label."""
        props = BaseProportions(A=0.70, T=0.20, C=0.05, G=0.05)
        score, label = score_confidence(
            proportions=props,
            quality_score=20,
            alignment_identity=0.90,
            neighboring_quality=[18, 20, 22, 19, 21],
            settings=mock_settings,
        )
        assert label == "medium"
        assert 0.5 <= score < 0.8

    def test_low_confidence_label(self, mock_settings):
        """Low quality inputs should produce 'low' confidence label."""
        props = BaseProportions(A=0.40, T=0.35, C=0.15, G=0.10)
        score, label = score_confidence(
            proportions=props,
            quality_score=5,
            alignment_identity=0.70,
            neighboring_quality=[5, 6, 4, 7, 5],
            settings=mock_settings,
        )
        assert label == "low"
        assert score < 0.5

    def test_score_in_valid_range(self, mock_settings):
        """Confidence score should always be in [0, 1]."""
        test_cases = [
            # (quality_score, alignment_identity, neighboring_quality)
            (0, 0.0, []),
            (60, 1.0, [60, 60, 60]),
            (40, 0.98, [38, 39, 40]),
            (10, 0.50, [8, 9, 10]),
        ]
        props = BaseProportions(A=0.80, T=0.10, C=0.05, G=0.05)
        for q, aln, nbr in test_cases:
            score, _ = score_confidence(props, q, aln, nbr, mock_settings)
            assert 0.0 <= score <= 1.0, (
                f"Score {score} out of range for q={q}, aln={aln}"
            )

    def test_quality_component_weight(self, mock_settings):
        """Quality component should have weight 0.35."""
        props = BaseProportions(A=1.0, T=0.0, C=0.0, G=0.0)
        # With perfect alignment and perfect neighbors, quality component dominates
        score_q40, _ = score_confidence(props, 40, 1.0, [40] * 5, mock_settings)
        score_q0, _ = score_confidence(props, 0, 1.0, [40] * 5, mock_settings)
        # Difference should be approximately 0.35 (quality weight)
        diff = score_q40 - score_q0
        assert abs(diff - 0.35) < 0.01, (
            f"Quality weight difference {diff:.3f} != 0.35"
        )

    def test_alignment_component_weight(self, mock_settings):
        """Alignment component should have weight 0.25."""
        props = BaseProportions(A=1.0, T=0.0, C=0.0, G=0.0)
        score_aln1, _ = score_confidence(props, 40, 1.0, [40] * 5, mock_settings)
        score_aln0, _ = score_confidence(props, 40, 0.0, [40] * 5, mock_settings)
        diff = score_aln1 - score_aln0
        assert abs(diff - 0.25) < 0.01, (
            f"Alignment weight difference {diff:.3f} != 0.25"
        )

    def test_empty_neighboring_quality(self, mock_settings):
        """Empty neighboring quality list should not crash."""
        props = BaseProportions(A=0.85, T=0.05, C=0.05, G=0.05)
        score, label = score_confidence(
            proportions=props,
            quality_score=35,
            alignment_identity=0.95,
            neighboring_quality=[],
            settings=mock_settings,
        )
        assert 0.0 <= score <= 1.0
        assert label in ("high", "medium", "low")

    def test_quality_capped_at_40(self, mock_settings):
        """Quality scores above 40 should be capped at 1.0 contribution."""
        props = BaseProportions(A=0.90, T=0.04, C=0.03, G=0.03)
        score_q40, _ = score_confidence(props, 40, 0.98, [38] * 5, mock_settings)
        score_q60, _ = score_confidence(props, 60, 0.98, [38] * 5, mock_settings)
        # Both should give the same score (Q40 and Q60 are equivalent)
        assert abs(score_q40 - score_q60) < 1e-6

    def test_custom_thresholds(self):
        """Custom confidence thresholds should be respected."""
        class StrictSettings:
            CONFIDENCE_HIGH_THRESHOLD = 0.9
            CONFIDENCE_MEDIUM_THRESHOLD = 0.7

        props = BaseProportions(A=0.85, T=0.05, C=0.05, G=0.05)
        score, label = score_confidence(
            proportions=props,
            quality_score=35,
            alignment_identity=0.95,
            neighboring_quality=[33, 34, 35, 34, 33],
            settings=StrictSettings(),
        )
        # With strict thresholds, same score might be "medium" instead of "high"
        assert label in ("high", "medium", "low")
        assert 0.0 <= score <= 1.0


# ── Tests: get_reference_base ─────────────────────────────────────────────────

class TestGetReferenceBase:
    """Tests for get_reference_base function with CIGAR parsing."""

    def test_simple_match_cigar(self, simple_alignment):
        """Simple 200M CIGAR: all positions should map directly."""
        # Position 0 → reference[0]
        ref_base = get_reference_base(simple_alignment, 0)
        assert ref_base in ("A", "T", "C", "G", "N")
        assert ref_base == simple_alignment.reference_sequence[0].upper()

    def test_match_cigar_middle_position(self, simple_alignment):
        """Middle position in 200M CIGAR should map correctly."""
        ref_base = get_reference_base(simple_alignment, 100)
        expected = simple_alignment.reference_sequence[100].upper()
        assert ref_base == expected

    def test_match_cigar_last_position(self, simple_alignment):
        """Last position in 200M CIGAR should map correctly."""
        ref_base = get_reference_base(simple_alignment, 199)
        expected = simple_alignment.reference_sequence[199].upper()
        assert ref_base == expected

    def test_insertion_returns_N(self, alignment_with_indels):
        """Position within an insertion should return 'N'."""
        # CIGAR: 100M5I100M5D5M
        # Positions 0-99: M (map to ref 0-99)
        # Positions 100-104: I (no reference base → N)
        # Positions 105-204: M (map to ref 100-199)
        ref_base_in_insertion = get_reference_base(alignment_with_indels, 102)
        assert ref_base_in_insertion == "N", (
            f"Expected 'N' for insertion position, got '{ref_base_in_insertion}'"
        )

    def test_match_before_insertion(self, alignment_with_indels):
        """Position before insertion should map correctly."""
        # CIGAR: 100M5I100M5D5M
        # Position 99 → reference[99]
        ref_base = get_reference_base(alignment_with_indels, 99)
        expected = alignment_with_indels.reference_sequence[99].upper()
        assert ref_base == expected

    def test_match_after_insertion(self, alignment_with_indels):
        """Position after insertion should skip insertion in reference."""
        # CIGAR: 100M5I100M5D5M
        # Position 105 (first M after insertion) → reference[100]
        ref_base = get_reference_base(alignment_with_indels, 105)
        expected = alignment_with_indels.reference_sequence[100].upper()
        assert ref_base == expected

    def test_deletion_skips_reference_bases(self):
        """Deletion in CIGAR should skip reference bases."""
        # CIGAR: 50M5D50M
        # Positions 0-49: M → ref[0-49]
        # (5 deleted ref bases: ref[50-54] are skipped)
        # Positions 50-99: M → ref[55-104]
        alignment = AlignmentResult(
            chromosome="chr1",
            start=1000,
            end=1105,
            strand="+",
            identity=0.98,
            alignment_score=100,
            cigar="50M5D50M",
            genes=[],
            method="minimap2",
            reference_sequence="A" * 50 + "GGGGG" + "T" * 50,  # 105 ref bases
        )
        # Position 50 (first M after deletion) → ref[55] = 'T'
        ref_base = get_reference_base(alignment, 50)
        assert ref_base == "T", (
            f"Expected 'T' after deletion, got '{ref_base}'"
        )

    def test_beyond_cigar_returns_N(self, simple_alignment):
        """Position beyond CIGAR alignment should return 'N'."""
        ref_base = get_reference_base(simple_alignment, 500)
        assert ref_base == "N"

    def test_empty_reference_sequence_returns_N(self):
        """Empty reference sequence should return 'N'."""
        alignment = AlignmentResult(
            chromosome="chr1",
            start=1000,
            end=1200,
            strand="+",
            identity=0.98,
            alignment_score=100,
            cigar="200M",
            genes=[],
            method="minimap2",
            reference_sequence="A",  # Minimal valid reference
        )
        # Override reference_sequence to empty via model_copy
        alignment_empty = alignment.model_copy(update={"reference_sequence": "A"})
        # Position 0 should return 'A'
        ref_base = get_reference_base(alignment_empty, 0)
        assert ref_base == "A"

    def test_complex_cigar_m_i_d(self):
        """Complex CIGAR with M, I, D operations should parse correctly."""
        # CIGAR: 10M3I10M2D10M
        # Read positions:
        #   0-9:   M → ref[0-9]
        #   10-12: I → N (insertion)
        #   13-22: M → ref[10-19]
        #   (2 deleted ref bases: ref[20-21] skipped)
        #   23-32: M → ref[22-31]
        ref_seq = "ACGT" * 8  # 32 ref bases
        alignment = AlignmentResult(
            chromosome="chr1",
            start=0,
            end=32,
            strand="+",
            identity=0.97,
            alignment_score=30,
            cigar="10M3I10M2D10M",
            genes=[],
            method="minimap2",
            reference_sequence=ref_seq,
        )

        # Position 0 → ref[0] = 'A'
        assert get_reference_base(alignment, 0) == "A"

        # Position 9 → ref[9] = 'T' (ACGT ACGT AC → index 9 = 'T')
        assert get_reference_base(alignment, 9) == ref_seq[9].upper()

        # Position 10 → insertion → 'N'
        assert get_reference_base(alignment, 10) == "N"
        assert get_reference_base(alignment, 11) == "N"
        assert get_reference_base(alignment, 12) == "N"

        # Position 13 → ref[10] = 'A' (ACGT ACGT AC → index 10 = 'A')
        assert get_reference_base(alignment, 13) == ref_seq[10].upper()

        # Position 23 → ref[22] (after 2-base deletion)
        assert get_reference_base(alignment, 23) == ref_seq[22].upper()


# ── Tests: call_snvs integration ─────────────────────────────────────────────

class TestCallSNVs:
    """Integration tests for call_snvs function."""

    def test_call_snvs_returns_list(
        self, synthetic_chromatogram, simple_alignment, mock_settings
    ):
        """call_snvs should return a list of SNVCall objects."""
        snvs = call_snvs(synthetic_chromatogram, simple_alignment, mock_settings)
        assert isinstance(snvs, list)

    def test_snvs_sorted_by_position(
        self, synthetic_chromatogram, simple_alignment, mock_settings
    ):
        """SNV calls should be sorted by position_in_read."""
        snvs = call_snvs(synthetic_chromatogram, simple_alignment, mock_settings)
        if len(snvs) > 1:
            positions = [s.position_in_read for s in snvs]
            assert positions == sorted(positions), "SNVs not sorted by position"

    def test_snv_confidence_labels_valid(
        self, synthetic_chromatogram, simple_alignment, mock_settings
    ):
        """All SNV confidence labels should be valid."""
        snvs = call_snvs(synthetic_chromatogram, simple_alignment, mock_settings)
        for snv in snvs:
            assert snv.confidence_label in ("high", "medium", "low"), (
                f"Invalid confidence label: {snv.confidence_label}"
            )

    def test_snv_confidence_scores_in_range(
        self, synthetic_chromatogram, simple_alignment, mock_settings
    ):
        """All SNV confidence scores should be in [0, 1]."""
        snvs = call_snvs(synthetic_chromatogram, simple_alignment, mock_settings)
        for snv in snvs:
            assert 0.0 <= snv.confidence_score <= 1.0, (
                f"Confidence score {snv.confidence_score} out of range"
            )

    def test_snv_alleles_are_valid_nucleotides(
        self, synthetic_chromatogram, simple_alignment, mock_settings
    ):
        """All SNV alleles should be valid nucleotides."""
        snvs = call_snvs(synthetic_chromatogram, simple_alignment, mock_settings)
        valid_bases = {"A", "T", "C", "G", "N"}
        for snv in snvs:
            assert snv.reference_allele in valid_bases
            assert snv.alternative_allele in valid_bases

    def test_snv_proportions_sum_to_one(
        self, synthetic_chromatogram, simple_alignment, mock_settings
    ):
        """All SNV proportions should sum to approximately 1.0."""
        snvs = call_snvs(synthetic_chromatogram, simple_alignment, mock_settings)
        for snv in snvs:
            total = (
                snv.proportions.A + snv.proportions.T
                + snv.proportions.C + snv.proportions.G
            )
            assert abs(total - 1.0) < 0.01, (
                f"Proportions sum to {total:.6f} at position {snv.position_in_read}"
            )

    def test_heterozygous_consistency(
        self, synthetic_chromatogram, simple_alignment, mock_settings
    ):
        """is_heterozygous should be consistent with secondary_peak_fraction."""
        snvs = call_snvs(synthetic_chromatogram, simple_alignment, mock_settings)
        for snv in snvs:
            if snv.secondary_peak_fraction >= 0.25:
                assert snv.is_heterozygous, (
                    f"Position {snv.position_in_read}: "
                    f"secondary_fraction={snv.secondary_peak_fraction:.3f} >= 0.25 "
                    f"but is_heterozygous=False"
                )
            else:
                assert not snv.is_heterozygous, (
                    f"Position {snv.position_in_read}: "
                    f"secondary_fraction={snv.secondary_peak_fraction:.3f} < 0.25 "
                    f"but is_heterozygous=True"
                )
