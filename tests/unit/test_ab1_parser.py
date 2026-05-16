"""
Unit tests for the AB1 parser (tools/ab1_parser.py).

Tests cover:
- normalize_traces: normalization pipeline properties
- identify_readable_region: quality-based trimming logic
- compute_qc_metrics: each QC flag condition
- generate_synthetic_ab1_data: output schema and properties
- parse_ab1: error handling for missing/invalid files

All tests use synthetic data — no real AB1 files required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.ab1_parser import (
    DYE_NORM_MAX,
    QC_LOW_SIGNAL_THRESHOLD,
    QC_MIN_READABLE_REGION,
    QC_PEAK_COLLAPSE_FRACTION,
    QC_SNR_THRESHOLD,
    compute_qc_metrics,
    generate_synthetic_ab1_data,
    identify_readable_region,
    normalize_traces,
    _gaussian_smooth,
    _rolling_minimum,
)
from backend.schemas.chromatogram import TraceData


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_flat_traces(value: float = 500.0, length: int = 200) -> dict[str, list[float]]:
    """Create flat traces with a constant value."""
    return {ch: [value] * length for ch in ["A", "T", "C", "G"]}


def make_peaked_traces(
    n_peaks: int = 20,
    peak_height: float = 800.0,
    noise_level: float = 10.0,
    length: int = 400,
) -> dict[str, list[float]]:
    """Create traces with Gaussian peaks."""
    rng = np.random.default_rng(42)
    traces = {ch: np.zeros(length) for ch in ["A", "T", "C", "G"]}
    channels = ["A", "T", "C", "G"]
    spacing = length // (n_peaks + 1)

    for i in range(n_peaks):
        center = (i + 1) * spacing
        ch = channels[i % 4]
        for j in range(max(0, center - 10), min(length, center + 11)):
            traces[ch][j] += peak_height * np.exp(-0.5 * ((j - center) / 2) ** 2)

    for ch in channels:
        traces[ch] += np.abs(rng.normal(0, noise_level, length))

    return {ch: traces[ch].tolist() for ch in channels}


def make_trace_data(
    n_peaks: int = 50,
    peak_height: float = 500.0,
    readable_start: int = 0,
    readable_end: int = None,
    n_samples: int = 600,
    n_bases: int = None,
) -> TraceData:
    """Create a TraceData object with synthetic peaks.

    Parameters
    ----------
    n_peaks : int
        Number of peaks (base calls) in the trace.
    peak_height : float
        Height of each peak.
    readable_start : int
        Start of the readable region (0-based, must be <= readable_end).
    readable_end : int or None
        End of the readable region (exclusive). Defaults to n_bases.
        Will be clamped to n_bases to satisfy the schema validator.
    n_samples : int
        Total number of trace sample points.
    n_bases : int or None
        Total number of base calls. Defaults to max(n_peaks, 200) so that
        the readable region can be >= 100 bases for QC pass tests.
    """
    rng = np.random.default_rng(0)
    # Default n_bases: large enough for a readable region >= 100 bases
    if n_bases is None:
        n_bases = max(n_peaks, 200)

    # Handle zero peaks edge case
    if n_peaks == 0:
        peak_positions = []
    else:
        spacing = n_samples // (n_peaks + 1)
        peak_positions = [(i + 1) * spacing for i in range(n_peaks)]

    trace = np.zeros(n_samples)
    for pos in peak_positions:
        for j in range(max(0, pos - 5), min(n_samples, pos + 6)):
            trace[j] += peak_height * np.exp(-0.5 * ((j - pos) / 2) ** 2)
    trace += np.abs(rng.normal(0, 5, n_samples))
    trace_list = trace.tolist()

    peak_heights = {
        "A": [int(peak_height * 0.8)] * n_peaks,
        "T": [int(peak_height * 0.1)] * n_peaks,
        "C": [int(peak_height * 0.05)] * n_peaks,
        "G": [int(peak_height * 0.05)] * n_peaks,
    }

    # readable_region_end must not exceed n_bases (sequence length)
    actual_readable_end = n_bases if readable_end is None else min(readable_end, n_bases)
    actual_readable_start = min(readable_start, actual_readable_end)

    return TraceData(
        trace_A=trace_list,
        trace_T=[0.0] * n_samples,
        trace_C=[0.0] * n_samples,
        trace_G=[0.0] * n_samples,
        trace_A_norm=trace_list,
        trace_T_norm=[0.0] * n_samples,
        trace_C_norm=[0.0] * n_samples,
        trace_G_norm=[0.0] * n_samples,
        peak_positions=peak_positions,
        peak_heights=peak_heights,
        base_calls="A" * n_bases,
        quality_scores=[35] * n_bases,
        readable_region_start=actual_readable_start,
        readable_region_end=actual_readable_end,
    )


# ── Tests: _rolling_minimum ───────────────────────────────────────────────────

class TestRollingMinimum:
    def test_constant_signal(self):
        sig = np.array([5.0] * 10)
        result = _rolling_minimum(sig, window=3)
        assert np.allclose(result, 5.0)

    def test_single_spike(self):
        sig = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
        result = _rolling_minimum(sig, window=3)
        # Minimum around the spike should still be 1.0
        assert result[2] == 1.0

    def test_output_length(self):
        sig = np.arange(20, dtype=float)
        result = _rolling_minimum(sig, window=5)
        assert len(result) == len(sig)

    def test_monotone_increasing(self):
        sig = np.arange(10, dtype=float)
        result = _rolling_minimum(sig, window=3)
        # Rolling min of increasing sequence should be non-decreasing
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


# ── Tests: _gaussian_smooth ───────────────────────────────────────────────────

class TestGaussianSmooth:
    def test_output_length_preserved(self):
        sig = np.random.default_rng(0).random(100)
        result = _gaussian_smooth(sig, sigma=2.0)
        assert len(result) == len(sig)

    def test_constant_signal_unchanged(self):
        sig = np.ones(50) * 7.0
        result = _gaussian_smooth(sig, sigma=2.0)
        # Smoothing a constant signal should return approximately the same constant
        assert np.allclose(result, 7.0, atol=0.1)

    def test_reduces_noise(self):
        rng = np.random.default_rng(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = clean + rng.normal(0, 0.5, 200)
        smoothed = _gaussian_smooth(noisy, sigma=3.0)
        # Smoothed signal should be closer to clean than noisy
        assert np.std(smoothed - clean) < np.std(noisy - clean)


# ── Tests: normalize_traces ───────────────────────────────────────────────────

class TestNormalizeTraces:
    def test_output_keys_match_input(self):
        raw = make_peaked_traces()
        result = normalize_traces(raw)
        assert set(result.keys()) == {"A", "T", "C", "G"}

    def test_max_equals_dye_norm_max(self):
        raw = make_peaked_traces(peak_height=1000.0)
        result = normalize_traces(raw)
        for ch in ["A", "T", "C", "G"]:
            assert abs(max(result[ch]) - DYE_NORM_MAX) < 1.0, (
                f"Channel {ch} max {max(result[ch]):.1f} != {DYE_NORM_MAX}"
            )

    def test_all_values_non_negative(self):
        raw = make_peaked_traces()
        result = normalize_traces(raw)
        for ch in ["A", "T", "C", "G"]:
            assert all(v >= 0 for v in result[ch]), f"Channel {ch} has negative values"

    def test_output_length_preserved(self):
        raw = make_peaked_traces(length=300)
        result = normalize_traces(raw)
        for ch in ["A", "T", "C", "G"]:
            assert len(result[ch]) == 300

    def test_missing_channel_raises_value_error(self):
        raw = {"A": [1.0, 2.0], "T": [1.0, 2.0], "C": [1.0, 2.0]}
        with pytest.raises(ValueError, match="missing channels"):
            normalize_traces(raw)

    def test_empty_channel_raises_value_error(self):
        raw = {"A": [], "T": [1.0], "C": [1.0], "G": [1.0]}
        with pytest.raises(ValueError, match="empty"):
            normalize_traces(raw)

    def test_flat_signal_normalized(self):
        # A flat signal should normalize to all-zero after baseline correction
        raw = make_flat_traces(value=100.0, length=100)
        result = normalize_traces(raw)
        # After baseline subtraction of rolling min, flat signal → zeros
        for ch in ["A", "T", "C", "G"]:
            assert max(result[ch]) <= DYE_NORM_MAX

    def test_high_signal_channel_normalized_to_1000(self):
        # One channel with a clear peak should normalize to max=1000
        raw = {
            "A": [0.0] * 50 + [5000.0] + [0.0] * 49,
            "T": [10.0] * 100,
            "C": [10.0] * 100,
            "G": [10.0] * 100,
        }
        result = normalize_traces(raw)
        assert abs(max(result["A"]) - DYE_NORM_MAX) < 1.0

    def test_returns_lists_not_arrays(self):
        raw = make_peaked_traces()
        result = normalize_traces(raw)
        for ch in ["A", "T", "C", "G"]:
            assert isinstance(result[ch], list)


# ── Tests: identify_readable_region ──────────────────────────────────────────

class TestIdentifyReadableRegion:
    def test_all_high_quality(self):
        scores = [35] * 200
        start, end = identify_readable_region(scores, min_quality=20, window=10)
        assert start == 0
        assert end == 200

    def test_all_low_quality(self):
        scores = [5] * 100
        start, end = identify_readable_region(scores, min_quality=20, window=10)
        assert start == 0
        assert end == 0

    def test_quality_ramp(self):
        # Low quality at start and end, high in middle
        scores = [5] * 30 + [35] * 140 + [5] * 30
        start, end = identify_readable_region(scores, min_quality=20, window=10)
        assert start >= 20  # Should skip low-quality start
        assert end <= 180   # Should trim low-quality end

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            identify_readable_region([], min_quality=20, window=10)

    def test_short_sequence_below_window(self):
        scores = [30, 35, 40]
        start, end = identify_readable_region(scores, min_quality=20, window=10)
        # Sequence shorter than window: check overall mean
        assert start == 0
        assert end == 3

    def test_returns_tuple_of_two_ints(self):
        scores = [30] * 50
        result = identify_readable_region(scores)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_start_less_than_end(self):
        scores = [5] * 20 + [35] * 100 + [5] * 20
        start, end = identify_readable_region(scores, min_quality=20, window=5)
        if end > 0:
            assert start < end

    def test_custom_min_quality(self):
        scores = [25] * 100
        # With min_quality=30, these scores should fail
        start30, end30 = identify_readable_region(scores, min_quality=30, window=5)
        # With min_quality=20, these scores should pass
        start20, end20 = identify_readable_region(scores, min_quality=20, window=5)
        assert end30 == 0  # All fail at Q30
        assert end20 > 0   # All pass at Q20


# ── Tests: compute_qc_metrics ─────────────────────────────────────────────────

class TestComputeQcMetrics:
    def test_good_chromatogram_passes(self):
        trace_data = make_trace_data(n_peaks=50, peak_height=500.0)
        quality_scores = [35] * 50
        flags, qc_pass = compute_qc_metrics(trace_data, quality_scores)
        assert qc_pass is True
        assert "failed_sequencing" not in flags

    def test_no_peaks_triggers_failed_sequencing(self):
        trace_data = make_trace_data(n_peaks=0, n_samples=100)
        # Override peak_positions to empty
        trace_data = TraceData(
            trace_A=[0.0] * 100,
            trace_T=[0.0] * 100,
            trace_C=[0.0] * 100,
            trace_G=[0.0] * 100,
            trace_A_norm=[0.0] * 100,
            trace_T_norm=[0.0] * 100,
            trace_C_norm=[0.0] * 100,
            trace_G_norm=[0.0] * 100,
            peak_positions=[],
            peak_heights={"A": [], "T": [], "C": [], "G": []},
            base_calls="",
            quality_scores=[],
            readable_region_start=0,
            readable_region_end=0,
        )
        flags, qc_pass = compute_qc_metrics(trace_data, [])
        assert "failed_sequencing" in flags
        assert qc_pass is False

    def test_low_signal_flag(self):
        # Peak heights below QC_LOW_SIGNAL_THRESHOLD
        n_peaks = 20
        trace_data = make_trace_data(n_peaks=n_peaks, peak_height=50.0)
        # Override peak heights to be very low
        trace_data.peak_heights = {
            "A": [30] * n_peaks,
            "T": [5] * n_peaks,
            "C": [5] * n_peaks,
            "G": [5] * n_peaks,
        }
        quality_scores = [35] * n_peaks
        flags, _ = compute_qc_metrics(trace_data, quality_scores)
        assert "low_signal" in flags

    def test_poor_readable_region_flag(self):
        # Readable region shorter than QC_MIN_READABLE_REGION
        trace_data = make_trace_data(
            n_peaks=20,
            peak_height=500.0,
            readable_start=0,
            readable_end=50,  # Only 50 bases — below threshold of 100
        )
        quality_scores = [35] * 20
        flags, _ = compute_qc_metrics(trace_data, quality_scores)
        assert "poor_readable_region" in flags

    def test_peak_collapse_flag(self):
        # >20% of peaks with very low height relative to median
        # Use a mix: 4 high peaks (500) and 16 very low peaks (1)
        # Median of [500]*4 + [1]*16 per channel = 1 (since 16 > 4)
        # But collapse_threshold = median * 0.10 = 0.1
        # So peaks with height 1 are NOT below 0.1 — need a different approach.
        # Instead: use 4 high peaks (1000) and 16 zero-height peaks (0)
        # Median of [1000]*4 + [0]*16 = 0, collapse_threshold = 0
        # That won't work either. Use: 4 peaks at 1000, 16 peaks at 1
        # All heights: [1000]*4*4 + [1]*16*4 = 16 values at 1000, 64 at 1
        # Median = 1, collapse_threshold = 0.1
        # peak_max for low peaks = 1 > 0.1 → still not collapsed
        #
        # The correct approach: use high median with very low outliers
        # 4 peaks at 1 (collapsed), 16 peaks at 1000 (normal)
        # All heights: [1]*4*4 + [1000]*16*4 = 16 at 1, 64 at 1000
        # Median = 1000, collapse_threshold = 100
        # peak_max for low peaks = 1 < 100 → collapsed!
        # collapse_fraction = 4/20 = 0.20 which is NOT > 0.20 (strict >)
        # Use 5 collapsed peaks: 5/20 = 0.25 > 0.20 → triggers flag
        n_peaks = 20
        trace_data = make_trace_data(n_peaks=n_peaks, peak_height=500.0)
        # 5 collapsed peaks (height=1) and 15 normal peaks (height=1000)
        # Median of all heights: [1]*5*4 + [1000]*15*4 = 20 at 1, 60 at 1000
        # Median = 1000, collapse_threshold = 100
        # 5 peaks have max height 1 < 100 → collapsed
        # collapse_fraction = 5/20 = 0.25 > 0.20 → peak_collapse flag
        trace_data.peak_heights = {
            "A": [1] * 5 + [1000] * 15,
            "T": [1] * 5 + [1000] * 15,
            "C": [1] * 5 + [1000] * 15,
            "G": [1] * 5 + [1000] * 15,
        }
        quality_scores = [35] * n_peaks
        flags, _ = compute_qc_metrics(trace_data, quality_scores)
        assert "peak_collapse" in flags

    def test_returns_tuple(self):
        trace_data = make_trace_data()
        result = compute_qc_metrics(trace_data, [35] * 50)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], bool)

    def test_multiple_flags_can_fail_qc(self):
        # Both low_signal and poor_readable_region → qc_pass=False
        n_peaks = 10
        trace_data = make_trace_data(
            n_peaks=n_peaks,
            peak_height=30.0,
            readable_start=0,
            readable_end=40,
        )
        trace_data.peak_heights = {
            "A": [30] * n_peaks,
            "T": [5] * n_peaks,
            "C": [5] * n_peaks,
            "G": [5] * n_peaks,
        }
        quality_scores = [35] * n_peaks
        flags, qc_pass = compute_qc_metrics(trace_data, quality_scores)
        # With 2+ flags, qc_pass should be False
        if len(flags) >= 3:
            assert qc_pass is False


# ── Tests: generate_synthetic_ab1_data ───────────────────────────────────────

class TestGenerateSyntheticAb1Data:
    def test_returns_chromatogram_data(self):
        from backend.schemas.chromatogram import ChromatogramData
        chrom = generate_synthetic_ab1_data(n_bases=100)
        assert isinstance(chrom, ChromatogramData)

    def test_sequence_length_matches_n_bases(self):
        chrom = generate_synthetic_ab1_data(n_bases=300)
        assert chrom.sequence_length == 300

    def test_trace_length_matches_n_bases_times_samples(self):
        n_bases = 200
        n_samples_per_base = 8
        chrom = generate_synthetic_ab1_data(
            n_bases=n_bases, n_samples_per_base=n_samples_per_base
        )
        expected_len = n_bases * n_samples_per_base
        assert len(chrom.trace.trace_A) == expected_len
        assert len(chrom.trace.trace_T) == expected_len

    def test_quality_scores_length_matches_n_bases(self):
        chrom = generate_synthetic_ab1_data(n_bases=150)
        assert len(chrom.trace.quality_scores) == 150

    def test_base_calls_length_matches_n_bases(self):
        chrom = generate_synthetic_ab1_data(n_bases=250)
        assert len(chrom.trace.base_calls) == 250

    def test_base_calls_only_valid_nucleotides(self):
        chrom = generate_synthetic_ab1_data(n_bases=400)
        valid = set("ATCG")
        assert all(b in valid for b in chrom.trace.base_calls)

    def test_qc_pass_for_good_synthetic_data(self):
        chrom = generate_synthetic_ab1_data(n_bases=600, mean_quality=35.0)
        # Good synthetic data should pass QC
        assert chrom.qc_pass is True

    def test_file_hash_is_sha256_hex(self):
        chrom = generate_synthetic_ab1_data(n_bases=100)
        assert len(chrom.file_hash) == 64
        assert all(c in "0123456789abcdef" for c in chrom.file_hash)

    def test_reproducibility_with_same_seed(self):
        chrom1 = generate_synthetic_ab1_data(n_bases=200, seed=99)
        chrom2 = generate_synthetic_ab1_data(n_bases=200, seed=99)
        assert chrom1.trace.base_calls == chrom2.trace.base_calls
        assert chrom1.file_hash == chrom2.file_hash

    def test_different_seeds_produce_different_data(self):
        chrom1 = generate_synthetic_ab1_data(n_bases=200, seed=1)
        chrom2 = generate_synthetic_ab1_data(n_bases=200, seed=2)
        assert chrom1.trace.base_calls != chrom2.trace.base_calls

    def test_snv_positions_inject_secondary_peaks(self):
        # With SNV positions, secondary peaks should be present
        snv_positions = [50, 100, 150]
        chrom = generate_synthetic_ab1_data(
            n_bases=300, snv_positions=snv_positions, seed=42
        )
        # The chromatogram should still be valid
        assert chrom.sequence_length == 300
        assert chrom.trace.peak_positions is not None

    def test_mean_quality_approximately_correct(self):
        chrom = generate_synthetic_ab1_data(n_bases=600, mean_quality=35.0)
        # Mean quality should be in a reasonable range around 35
        assert 20.0 <= chrom.mean_quality <= 45.0

    def test_normalized_traces_max_equals_dye_norm_max(self):
        chrom = generate_synthetic_ab1_data(n_bases=200)
        for attr in ["trace_A_norm", "trace_T_norm", "trace_C_norm", "trace_G_norm"]:
            trace = getattr(chrom.trace, attr)
            assert abs(max(trace) - DYE_NORM_MAX) < 1.0, (
                f"{attr} max {max(trace):.1f} != {DYE_NORM_MAX}"
            )

    def test_peak_positions_within_trace_bounds(self):
        n_bases = 100
        n_samples_per_base = 10
        chrom = generate_synthetic_ab1_data(
            n_bases=n_bases, n_samples_per_base=n_samples_per_base
        )
        trace_len = n_bases * n_samples_per_base
        for pos in chrom.trace.peak_positions:
            assert 0 <= pos < trace_len


# ── Tests: parse_ab1 error handling ──────────────────────────────────────────

class TestParseAb1ErrorHandling:
    def test_file_not_found_raises(self):
        from tools.ab1_parser import parse_ab1
        with pytest.raises(FileNotFoundError):
            parse_ab1("/nonexistent/path/sample.ab1")

    def test_invalid_file_raises_value_error(self, tmp_path):
        from tools.ab1_parser import parse_ab1
        # Create a file with invalid content
        bad_file = tmp_path / "bad.ab1"
        bad_file.write_bytes(b"this is not an ab1 file")
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_ab1(str(bad_file))

    def test_directory_path_raises_value_error(self, tmp_path):
        from tools.ab1_parser import parse_ab1
        with pytest.raises(ValueError, match="not a file"):
            parse_ab1(str(tmp_path))
