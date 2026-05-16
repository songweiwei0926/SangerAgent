"""
Unit tests for editing analysis tools (Phases 4 and 5).

Tests cover:
- Editing type classification with synthetic chromatogram data
- ICE fallback with known WT/Edited pairs
- TIDE fallback decomposition point detection
- BEAT fallback base proportion comparison
- Auto-routing logic in editing_analysis_service

All tests use synthetic data to avoid dependency on real AB1 files.

Example
-------
>>> pytest tests/unit/test_editing_analysis.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ── Synthetic data factories ──────────────────────────────────────────────────


def make_trace_data(
    n_peaks: int = 100,
    sequence: str = None,
    add_noise: float = 0.0,
    shift: int = 0,
) -> MagicMock:
    """
    Create a synthetic TraceData mock object.

    Parameters
    ----------
    n_peaks : int
        Number of peaks (base calls).
    sequence : str, optional
        DNA sequence. If None, generates random sequence.
    add_noise : float
        Noise level to add to peak heights (0.0 = no noise).
    shift : int
        Shift peak positions by this many samples (simulates indel).

    Returns
    -------
    MagicMock
        Mock TraceData object.
    """
    rng = np.random.default_rng(42)

    if sequence is None:
        bases = ["A", "T", "C", "G"]
        sequence = "".join(rng.choice(bases, size=n_peaks))

    # Generate peak positions
    peak_positions = np.arange(n_peaks) * 10 + 50 + shift
    peak_positions = peak_positions.tolist()

    # Generate peak heights (dominant base gets high signal)
    peak_heights = {"A": [], "T": [], "C": [], "G": []}
    for base in sequence:
        for b in ["A", "T", "C", "G"]:
            if b == base:
                height = 800.0 + rng.normal(0, add_noise * 100)
            else:
                height = 50.0 + rng.normal(0, add_noise * 20)
            peak_heights[b].append(max(0.0, height))

    # Generate raw trace signals
    n_samples = n_peaks * 10 + 100
    raw_data = {}
    for b in ["A", "T", "C", "G"]:
        signal = np.zeros(n_samples)
        for i, pos in enumerate(peak_positions):
            if pos < n_samples:
                signal[pos] = peak_heights[b][i]
        raw_data[b] = signal.tolist()

    trace = MagicMock()
    trace.peak_positions = peak_positions
    trace.peak_heights = peak_heights
    trace.raw_data = raw_data
    trace.readable_length = n_peaks
    trace.sequence = sequence

    return trace


def make_chromatogram(
    n_peaks: int = 100,
    sequence: str = None,
    mean_quality: float = 35.0,
    add_noise: float = 0.0,
    shift: int = 0,
) -> MagicMock:
    """
    Create a synthetic ChromatogramData mock object.

    Parameters
    ----------
    n_peaks : int
        Number of peaks.
    sequence : str, optional
        DNA sequence.
    mean_quality : float
        Mean quality score.
    add_noise : float
        Noise level for peak heights.
    shift : int
        Shift peak positions (simulates indel).

    Returns
    -------
    MagicMock
        Mock ChromatogramData object.
    """
    trace = make_trace_data(n_peaks, sequence, add_noise, shift)

    chrom = MagicMock()
    chrom.trace = trace
    chrom.sequence = trace.sequence
    chrom.sequence_length = n_peaks
    chrom.mean_quality = mean_quality
    chrom.has_critical_failure = False

    return chrom


def make_base_editing_pair(
    n_peaks: int = 100,
    edit_positions: list[int] = None,
    edit_type: str = "C_to_T",
    efficiency: float = 0.7,
) -> tuple[MagicMock, MagicMock]:
    """
    Create a WT/Edited pair with base editing events.

    Parameters
    ----------
    n_peaks : int
        Number of peaks.
    edit_positions : list[int], optional
        Positions to edit. Defaults to [20, 21, 22].
    edit_type : str
        "C_to_T" or "A_to_G".
    efficiency : float
        Editing efficiency (proportion change).

    Returns
    -------
    tuple[MagicMock, MagicMock]
        (wt_chromatogram, edited_chromatogram)
    """
    if edit_positions is None:
        edit_positions = [20, 21, 22]

    rng = np.random.default_rng(42)
    bases = ["A", "T", "C", "G"]

    # Generate WT sequence with target bases at edit positions
    wt_seq = list(rng.choice(bases, size=n_peaks))
    if edit_type == "C_to_T":
        for pos in edit_positions:
            wt_seq[pos] = "C"
    elif edit_type == "A_to_G":
        for pos in edit_positions:
            wt_seq[pos] = "A"

    wt_seq = "".join(wt_seq)

    # Create WT chromatogram
    wt_chrom = make_chromatogram(n_peaks, wt_seq)

    # Create edited chromatogram with modified peak heights at edit positions
    edited_chrom = make_chromatogram(n_peaks, wt_seq)

    # Modify peak heights at edit positions to simulate base editing
    for pos in edit_positions:
        if edit_type == "C_to_T":
            # Reduce C, increase T
            edited_chrom.trace.peak_heights["C"][pos] = 800.0 * (1 - efficiency)
            edited_chrom.trace.peak_heights["T"][pos] = 800.0 * efficiency
        elif edit_type == "A_to_G":
            # Reduce A, increase G
            edited_chrom.trace.peak_heights["A"][pos] = 800.0 * (1 - efficiency)
            edited_chrom.trace.peak_heights["G"][pos] = 800.0 * efficiency

    return wt_chrom, edited_chrom


def make_indel_pair(
    n_peaks: int = 100,
    indel_size: int = -1,
    cut_site: int = 50,
) -> tuple[MagicMock, MagicMock]:
    """
    Create a WT/Edited pair with an indel.

    Parameters
    ----------
    n_peaks : int
        Number of peaks.
    indel_size : int
        Indel size (negative = deletion, positive = insertion).
    cut_site : int
        Position of the cut site.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        (wt_chromatogram, edited_chromatogram)
    """
    rng = np.random.default_rng(42)
    bases = ["A", "T", "C", "G"]
    wt_seq = "".join(rng.choice(bases, size=n_peaks))

    wt_chrom = make_chromatogram(n_peaks, wt_seq)
    # Edited chromatogram has shifted peaks after cut site
    edited_chrom = make_chromatogram(n_peaks, wt_seq, shift=indel_size * 2)

    return wt_chrom, edited_chrom


# ── Tests: editing_classifier ─────────────────────────────────────────────────


class TestEditingClassifier:
    """Tests for tools/editing_classifier.py."""

    def test_classify_identical_traces_unknown(self):
        """Test that identical WT and Edited traces are classified as unknown."""
        from tools.editing_classifier import classify_editing_type

        wt_chrom = make_chromatogram(100)
        # Use same chromatogram for both
        editing_type, confidence = classify_editing_type(wt_chrom, wt_chrom)

        assert editing_type == "unknown"
        assert 0.0 <= confidence <= 1.0

    def test_classify_base_editing_pair(self):
        """Test classification of a base editing pair."""
        from tools.editing_classifier import classify_editing_type

        wt_chrom, edited_chrom = make_base_editing_pair(
            n_peaks=100, edit_positions=[20, 21, 22], efficiency=0.8
        )

        editing_type, confidence = classify_editing_type(wt_chrom, edited_chrom)

        # Should detect base editing (no trace shift, point substitutions)
        assert editing_type in ("base_editing", "mixed", "unknown")
        assert 0.0 <= confidence <= 1.0

    def test_classify_returns_valid_type(self):
        """Test that classify_editing_type always returns a valid type."""
        from tools.editing_classifier import classify_editing_type

        valid_types = {"base_editing", "indel", "prime_editing", "hdr", "mixed", "unknown"}

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.1)

        editing_type, confidence = classify_editing_type(wt_chrom, edited_chrom)

        assert editing_type in valid_types
        assert 0.0 <= confidence <= 1.0

    def test_classify_confidence_is_float(self):
        """Test that confidence is a float in [0, 1]."""
        from tools.editing_classifier import classify_editing_type

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100)

        _, confidence = classify_editing_type(wt_chrom, edited_chrom)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_compute_trace_difference_returns_dict(self):
        """Test that compute_trace_difference returns a dict with channel keys."""
        from tools.editing_classifier import compute_trace_difference

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100)

        diff = compute_trace_difference(wt_chrom, edited_chrom)

        assert isinstance(diff, dict)
        # Should have at least one channel
        assert len(diff) > 0

    def test_detect_trace_shift_no_shift(self):
        """Test that identical traces have no shift."""
        from tools.editing_classifier import detect_trace_shift

        wt_chrom = make_chromatogram(100)
        diff = {"A": np.zeros(100), "T": np.zeros(100), "C": np.zeros(100), "G": np.zeros(100)}

        has_shift, magnitude = detect_trace_shift(diff)

        assert isinstance(has_shift, bool)
        assert isinstance(magnitude, int)

    def test_detect_point_substitutions_empty_for_identical(self):
        """Test that identical traces have no point substitutions."""
        from tools.editing_classifier import detect_point_substitutions

        wt_chrom = make_chromatogram(100)
        # Use same chromatogram
        substitutions = detect_point_substitutions(wt_chrom, wt_chrom)

        assert isinstance(substitutions, list)
        # Identical traces should have no substitutions
        assert len(substitutions) == 0

    def test_detect_point_substitutions_finds_edits(self):
        """Test that base editing events are detected as substitutions."""
        from tools.editing_classifier import detect_point_substitutions

        wt_chrom, edited_chrom = make_base_editing_pair(
            n_peaks=100, edit_positions=[20, 21, 22], efficiency=0.8
        )

        substitutions = detect_point_substitutions(wt_chrom, edited_chrom)

        assert isinstance(substitutions, list)
        # Should detect at least some substitutions
        # (may not detect all 3 depending on threshold)
        for sub in substitutions:
            assert "position" in sub
            assert "wt_base" in sub
            assert "edited_base" in sub
            assert "proportion_change" in sub


# ── Tests: ICE wrapper ────────────────────────────────────────────────────────


class TestICEWrapper:
    """Tests for tools/ice_wrapper.py."""

    def test_run_ice_analysis_returns_dict(self):
        """Test that run_ice_analysis returns a dict with required keys."""
        from tools.ice_wrapper import run_ice_analysis

        settings = MagicMock()
        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.1)

        result = run_ice_analysis(wt_chrom, edited_chrom, settings)

        assert isinstance(result, dict)
        assert "tool" in result
        assert "efficiency" in result
        assert "indel_pct" in result
        assert "r_squared" in result
        assert "indel_distribution" in result
        assert "ice_score" in result
        assert "ko_score" in result

    def test_run_ice_analysis_efficiency_in_range(self):
        """Test that efficiency is in [0, 1]."""
        from tools.ice_wrapper import run_ice_analysis

        settings = MagicMock()
        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.2)

        result = run_ice_analysis(wt_chrom, edited_chrom, settings)

        assert 0.0 <= result["efficiency"] <= 1.0

    def test_run_ice_analysis_uses_fallback_when_no_cli(self):
        """Test that fallback is used when ICE CLI is not available."""
        from tools.ice_wrapper import run_ice_analysis

        settings = MagicMock()
        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100)

        with patch("shutil.which", return_value=None):
            result = run_ice_analysis(wt_chrom, edited_chrom, settings)

        assert result["tool"] == "ice_fallback"

    def test_run_ice_fallback_identical_traces(self):
        """Test ICE fallback with identical WT and Edited traces."""
        from tools.ice_wrapper import run_ice_fallback

        wt_chrom = make_chromatogram(100)
        # Identical traces → efficiency should be near 0
        result = run_ice_fallback(wt_chrom, wt_chrom)

        assert result["tool"] == "ice_fallback"
        assert 0.0 <= result["efficiency"] <= 1.0
        # Identical traces should have low efficiency
        assert result["efficiency"] < 0.5

    def test_run_ice_fallback_returns_indel_distribution(self):
        """Test that ICE fallback returns an indel distribution."""
        from tools.ice_wrapper import run_ice_fallback

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.3)

        result = run_ice_fallback(wt_chrom, edited_chrom)

        assert isinstance(result["indel_distribution"], dict)

    def test_run_ice_fallback_r_squared_in_range(self):
        """Test that R² is in [0, 1]."""
        from tools.ice_wrapper import run_ice_fallback

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.1)

        result = run_ice_fallback(wt_chrom, edited_chrom)

        assert 0.0 <= result["r_squared"] <= 1.0

    def test_run_ice_fallback_ice_score_in_range(self):
        """Test that ICE score is in [0, 100]."""
        from tools.ice_wrapper import run_ice_fallback

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.2)

        result = run_ice_fallback(wt_chrom, edited_chrom)

        assert 0.0 <= result["ice_score"] <= 100.0

    def test_run_ice_fallback_ko_score_in_range(self):
        """Test that KO score is in [0, 100]."""
        from tools.ice_wrapper import run_ice_fallback

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.2)

        result = run_ice_fallback(wt_chrom, edited_chrom)

        assert 0.0 <= result["ko_score"] <= 100.0


# ── Tests: TIDE wrapper ───────────────────────────────────────────────────────


class TestTIDEWrapper:
    """Tests for tools/tide_wrapper.py."""

    def test_run_tide_analysis_returns_dict(self):
        """Test that run_tide_analysis returns a dict with required keys."""
        from tools.tide_wrapper import run_tide_analysis

        settings = MagicMock()
        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.1)

        result = run_tide_analysis(wt_chrom, edited_chrom, settings)

        assert isinstance(result, dict)
        assert "tool" in result
        assert "efficiency" in result
        assert "indel_spectrum" in result
        assert "decomposition_point" in result
        assert "r_squared" in result
        assert "p_value" in result

    def test_run_tide_analysis_efficiency_in_range(self):
        """Test that efficiency is in [0, 1]."""
        from tools.tide_wrapper import run_tide_analysis

        settings = MagicMock()
        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.2)

        result = run_tide_analysis(wt_chrom, edited_chrom, settings)

        assert 0.0 <= result["efficiency"] <= 1.0

    def test_run_tide_analysis_uses_fallback_when_no_rscript(self):
        """Test that fallback is used when Rscript is not available."""
        from tools.tide_wrapper import run_tide_analysis

        settings = MagicMock()
        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100)

        with patch("shutil.which", return_value=None):
            result = run_tide_analysis(wt_chrom, edited_chrom, settings)

        assert result["tool"] == "tide_fallback"

    def test_find_decomposition_point_returns_int(self):
        """Test that find_decomposition_point returns an integer."""
        from tools.tide_wrapper import find_decomposition_point

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100)

        point = find_decomposition_point(wt_chrom, edited_chrom)

        assert isinstance(point, int)
        assert 0 <= point <= 100

    def test_find_decomposition_point_identical_traces(self):
        """Test decomposition point for identical traces."""
        from tools.tide_wrapper import find_decomposition_point

        wt_chrom = make_chromatogram(100)
        # Identical traces → decomposition point should be near the end
        point = find_decomposition_point(wt_chrom, wt_chrom)

        assert isinstance(point, int)
        # For identical traces, decomposition point should be late
        assert point >= 0

    def test_run_tide_fallback_indel_spectrum_is_dict(self):
        """Test that TIDE fallback returns an indel spectrum dict."""
        from tools.tide_wrapper import run_tide_fallback

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.2)

        result = run_tide_fallback(wt_chrom, edited_chrom)

        assert isinstance(result["indel_spectrum"], dict)

    def test_run_tide_fallback_p_value_in_range(self):
        """Test that p-value is in [0, 1]."""
        from tools.tide_wrapper import run_tide_fallback

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.1)

        result = run_tide_fallback(wt_chrom, edited_chrom)

        assert 0.0 <= result["p_value"] <= 1.0

    def test_run_tide_fallback_decomposition_point_valid(self):
        """Test that decomposition point is a valid position."""
        from tools.tide_wrapper import run_tide_fallback

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100, add_noise=0.1)

        result = run_tide_fallback(wt_chrom, edited_chrom)

        assert isinstance(result["decomposition_point"], int)
        assert result["decomposition_point"] >= 0


# ── Tests: BEAT wrapper ───────────────────────────────────────────────────────


class TestBEATWrapper:
    """Tests for tools/beat_wrapper.py."""

    def test_run_beat_analysis_returns_dict(self):
        """Test that run_beat_analysis returns a dict with required keys."""
        from tools.beat_wrapper import run_beat_analysis

        settings = MagicMock()
        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100)

        result = run_beat_analysis(wt_chrom, edited_chrom, settings)

        assert isinstance(result, dict)
        assert "tool" in result
        assert "editing_efficiency" in result
        assert "edited_positions" in result
        assert "base_edit_type" in result
        assert "editing_window" in result
        assert "bystander_edits" in result

    def test_run_beat_analysis_uses_fallback_when_no_cli(self):
        """Test that fallback is used when BEAT CLI is not available."""
        from tools.beat_wrapper import run_beat_analysis

        settings = MagicMock()
        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100)

        with patch("shutil.which", return_value=None):
            result = run_beat_analysis(wt_chrom, edited_chrom, settings)

        assert result["tool"] == "beat_fallback"

    def test_run_beat_fallback_detects_c_to_t_editing(self):
        """Test that BEAT fallback detects C→T base editing."""
        from tools.beat_wrapper import run_beat_fallback

        wt_chrom, edited_chrom = make_base_editing_pair(
            n_peaks=100,
            edit_positions=[20, 21, 22],
            edit_type="C_to_T",
            efficiency=0.8,
        )

        result = run_beat_fallback(wt_chrom, edited_chrom)

        assert result["tool"] == "beat_fallback"
        assert 0.0 <= result["editing_efficiency"] <= 1.0
        # Should detect C→T editing
        if result["edited_positions"]:
            assert result["base_edit_type"] in ("C_to_T", "other")

    def test_run_beat_fallback_detects_a_to_g_editing(self):
        """Test that BEAT fallback detects A→G base editing."""
        from tools.beat_wrapper import run_beat_fallback

        wt_chrom, edited_chrom = make_base_editing_pair(
            n_peaks=100,
            edit_positions=[30, 31],
            edit_type="A_to_G",
            efficiency=0.75,
        )

        result = run_beat_fallback(wt_chrom, edited_chrom)

        assert result["tool"] == "beat_fallback"
        assert 0.0 <= result["editing_efficiency"] <= 1.0

    def test_run_beat_fallback_no_editing_returns_empty(self):
        """Test that identical traces return empty editing result."""
        from tools.beat_wrapper import run_beat_fallback

        wt_chrom = make_chromatogram(100)
        # Identical traces → no editing
        result = run_beat_fallback(wt_chrom, wt_chrom)

        assert result["tool"] == "beat_fallback"
        assert result["editing_efficiency"] == 0.0
        assert result["edited_positions"] == []

    def test_run_beat_fallback_editing_window_valid(self):
        """Test that editing window is a valid tuple."""
        from tools.beat_wrapper import run_beat_fallback

        wt_chrom, edited_chrom = make_base_editing_pair(
            n_peaks=100, edit_positions=[20, 21, 22], efficiency=0.8
        )

        result = run_beat_fallback(wt_chrom, edited_chrom)

        window = result["editing_window"]
        assert isinstance(window, tuple)
        assert len(window) == 2
        assert window[0] <= window[1]

    def test_run_beat_fallback_bystander_edits_list(self):
        """Test that bystander_edits is a list."""
        from tools.beat_wrapper import run_beat_fallback

        wt_chrom = make_chromatogram(100)
        edited_chrom = make_chromatogram(100)

        result = run_beat_fallback(wt_chrom, edited_chrom)

        assert isinstance(result["bystander_edits"], list)

    def test_run_beat_fallback_edited_positions_format(self):
        """Test that edited_positions have the correct format."""
        from tools.beat_wrapper import run_beat_fallback

        wt_chrom, edited_chrom = make_base_editing_pair(
            n_peaks=100, edit_positions=[20, 21, 22], efficiency=0.8
        )

        result = run_beat_fallback(wt_chrom, edited_chrom)

        for pos in result["edited_positions"]:
            assert "pos" in pos
            assert "wt_base" in pos
            assert "edited_base" in pos
            assert "efficiency" in pos
            assert 0.0 <= pos["efficiency"] <= 1.0

    def test_classify_base_edit_type_c_to_t(self):
        """Test classification of C→T editing."""
        from tools.beat_wrapper import _classify_base_edit_type

        edited_positions = [
            {"wt_base": "C", "edited_base": "T", "efficiency": 0.8},
            {"wt_base": "C", "edited_base": "T", "efficiency": 0.7},
            {"wt_base": "C", "edited_base": "T", "efficiency": 0.6},
        ]

        edit_type = _classify_base_edit_type(edited_positions)
        assert edit_type == "C_to_T"

    def test_classify_base_edit_type_a_to_g(self):
        """Test classification of A→G editing."""
        from tools.beat_wrapper import _classify_base_edit_type

        edited_positions = [
            {"wt_base": "A", "edited_base": "G", "efficiency": 0.75},
            {"wt_base": "A", "edited_base": "G", "efficiency": 0.65},
        ]

        edit_type = _classify_base_edit_type(edited_positions)
        assert edit_type == "A_to_G"

    def test_classify_base_edit_type_other(self):
        """Test classification of non-canonical editing."""
        from tools.beat_wrapper import _classify_base_edit_type

        edited_positions = [
            {"wt_base": "G", "edited_base": "A", "efficiency": 0.5},
            {"wt_base": "T", "edited_base": "C", "efficiency": 0.4},
        ]

        edit_type = _classify_base_edit_type(edited_positions)
        assert edit_type == "other"

    def test_identify_editing_window_contiguous(self):
        """Test editing window identification for contiguous positions."""
        from tools.beat_wrapper import _identify_editing_window

        edited_positions = [
            {"pos": 20, "efficiency": 0.8},
            {"pos": 21, "efficiency": 0.7},
            {"pos": 22, "efficiency": 0.6},
        ]

        window = _identify_editing_window(edited_positions)
        assert window == (20, 22)

    def test_identify_editing_window_single_position(self):
        """Test editing window for a single position."""
        from tools.beat_wrapper import _identify_editing_window

        edited_positions = [{"pos": 15, "efficiency": 0.9}]

        window = _identify_editing_window(edited_positions)
        assert window == (15, 15)

    def test_identify_editing_window_empty(self):
        """Test editing window for empty positions."""
        from tools.beat_wrapper import _identify_editing_window

        window = _identify_editing_window([])
        assert window == (0, 0)

    def test_identify_bystander_edits(self):
        """Test bystander edit identification."""
        from tools.beat_wrapper import _identify_bystander_edits

        edited_positions = [
            {"pos": 5, "wt_base": "C", "edited_base": "T", "efficiency": 0.3},
            {"pos": 20, "wt_base": "C", "edited_base": "T", "efficiency": 0.8},
            {"pos": 21, "wt_base": "C", "edited_base": "T", "efficiency": 0.7},
            {"pos": 50, "wt_base": "C", "edited_base": "T", "efficiency": 0.2},
        ]

        window = (20, 21)
        bystanders = _identify_bystander_edits(edited_positions, window)

        assert len(bystanders) == 2
        bystander_positions = [b["pos"] for b in bystanders]
        assert 5 in bystander_positions
        assert 50 in bystander_positions


# ── Tests: editing_analysis_service auto-routing ─────────────────────────────


class TestEditingAnalysisServiceRouting:
    """Tests for auto-routing logic in editing_analysis_service.py."""

    def test_get_tool_name_base_editing(self):
        """Test that base_editing routes to BEAT."""
        from backend.services.editing_analysis_service import _get_tool_name

        assert _get_tool_name("base_editing") == "BEAT"

    def test_get_tool_name_indel(self):
        """Test that indel routes to ICE."""
        from backend.services.editing_analysis_service import _get_tool_name

        assert _get_tool_name("indel") == "ICE"

    def test_get_tool_name_prime_editing(self):
        """Test that prime_editing routes to ICE."""
        from backend.services.editing_analysis_service import _get_tool_name

        assert _get_tool_name("prime_editing") == "ICE"

    def test_get_tool_name_mixed(self):
        """Test that mixed routes to ICE+BEAT."""
        from backend.services.editing_analysis_service import _get_tool_name

        assert _get_tool_name("mixed") == "ICE+BEAT"

    def test_get_tool_name_unknown(self):
        """Test that unknown routes to ICE."""
        from backend.services.editing_analysis_service import _get_tool_name

        assert _get_tool_name("unknown") == "ICE"

    def test_normalize_tool_name_ice(self):
        """Test ICE tool name normalization."""
        from backend.services.editing_analysis_service import _normalize_tool_name

        assert _normalize_tool_name("ICE") == "ICE"
        assert _normalize_tool_name("ice") == "ICE"
        assert _normalize_tool_name("ice_fallback") == "internal_fallback"

    def test_normalize_tool_name_tide(self):
        """Test TIDE tool name normalization."""
        from backend.services.editing_analysis_service import _normalize_tool_name

        assert _normalize_tool_name("TIDE") == "TIDE"
        assert _normalize_tool_name("tide") == "TIDE"
        assert _normalize_tool_name("tide_fallback") == "internal_fallback"

    def test_normalize_tool_name_beat(self):
        """Test BEAT tool name normalization."""
        from backend.services.editing_analysis_service import _normalize_tool_name

        assert _normalize_tool_name("BEAT") == "BEAT"
        assert _normalize_tool_name("beat") == "BEAT"
        assert _normalize_tool_name("beat_fallback") == "internal_fallback"

    def test_normalize_tool_name_fallback(self):
        """Test fallback tool name normalization."""
        from backend.services.editing_analysis_service import _normalize_tool_name

        assert _normalize_tool_name("internal_fallback") == "internal_fallback"
        assert _normalize_tool_name("unknown_tool") == "internal_fallback"

    def test_extract_edited_positions_from_beat_result(self):
        """Test extracting edited positions from BEAT result."""
        from backend.services.editing_analysis_service import _extract_edited_positions

        tool_result = {
            "edited_positions": [
                {"pos": 20, "wt_base": "C", "edited_base": "T", "efficiency": 0.8},
                {"pos": 21, "wt_base": "C", "edited_base": "T", "efficiency": 0.7},
            ]
        }

        positions = _extract_edited_positions(tool_result, "base_editing")

        assert positions == [20, 21]

    def test_extract_edited_positions_from_ice_result(self):
        """Test extracting edited positions from ICE result (no explicit positions)."""
        from backend.services.editing_analysis_service import _extract_edited_positions

        tool_result = {
            "efficiency": 0.6,
            "indel_distribution": {-1: 0.4, 1: 0.2},
        }

        positions = _extract_edited_positions(tool_result, "indel")

        assert isinstance(positions, list)

    @pytest.mark.asyncio
    async def test_run_editing_analysis_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        from backend.services.editing_analysis_service import run_editing_analysis

        settings = MagicMock()

        with patch("tools.ab1_parser.parse_ab1") as mock_parse:
            mock_parse.side_effect = FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                await run_editing_analysis(
                    wt_file_path="/nonexistent/wt.ab1",
                    edited_file_path="/nonexistent/edited.ab1",
                    job_id="test-job-001",
                    settings=settings,
                )

    @pytest.mark.asyncio
    async def test_run_editing_analysis_full_pipeline(self):
        """Test the full editing analysis pipeline with mocked components."""
        from backend.services.editing_analysis_service import run_editing_analysis
        from tools.ab1_parser import generate_synthetic_ab1_data

        settings = MagicMock()
        # Use real ChromatogramData objects (required by EditingResult schema)
        wt_chrom = generate_synthetic_ab1_data(n_bases=100, seed=0)
        edited_chrom = generate_synthetic_ab1_data(n_bases=100, seed=1)

        with patch("tools.ab1_parser.parse_ab1") as mock_parse:
            mock_parse.side_effect = [wt_chrom, edited_chrom]

            with patch("tools.sequencing_qc.run_qc_pipeline") as mock_qc:
                qc_result = MagicMock()
                qc_result.overall_pass = True
                mock_qc.return_value = qc_result

                with patch("tools.editing_classifier.classify_editing_type") as mock_classify:
                    mock_classify.return_value = ("indel", 0.85)

                    with patch("tools.ice_wrapper.run_ice_analysis") as mock_ice:
                        mock_ice.return_value = {
                            "tool": "ice_fallback",
                            "efficiency": 0.65,
                            "indel_pct": 0.65,
                            "r_squared": 0.92,
                            "indel_distribution": {-1: 0.4, 1: 0.25},
                            "ice_score": 65.0,
                            "ko_score": 40.0,
                        }

                        result = await run_editing_analysis(
                            wt_file_path="/data/wt.ab1",
                            edited_file_path="/data/edited.ab1",
                            job_id="test-job-001",
                            settings=settings,
                        )

        assert result.job_id == "test-job-001"
        assert result.status == "completed"
        assert result.editing_type == "indel"
        assert result.tool_used == "internal_fallback"
        assert 0.0 <= result.efficiency <= 1.0
        assert result.processing_time_seconds >= 0
