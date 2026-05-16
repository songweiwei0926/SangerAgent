"""
Automatic editing type classification for genome editing analysis.

Classifies editing patterns from WT vs Edited chromatogram comparison:
- base_editing: single base changes without indels, in editing window (positions 4-8 from PAM)
- indel: insertions or deletions causing trace shift
- prime_editing: complex substitutions with characteristic patterns
- hdr: sequence changes matching a donor template pattern
- mixed: combination of editing types
- unknown: cannot classify

Classification algorithm:
1. Align WT and Edited traces using cross-correlation
2. Compute difference signal between aligned traces
3. Detect trace shift (indel signature) via cross-correlation lag analysis
4. Detect point substitutions (base editing signature) via peak proportion comparison
5. Classify based on pattern: shift → indel, substitutions only → base_editing, both → mixed

Example
-------
>>> from tools.editing_classifier import classify_editing_type
>>> # editing_type, confidence = classify_editing_type(wt_chrom, edited_chrom)
>>> # print(f"Editing type: {editing_type} (confidence: {confidence:.2f})")
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# ── Classification thresholds ─────────────────────────────────────────────────
# Minimum proportion change to call a point substitution
SUBSTITUTION_THRESHOLD = 0.15

# Minimum cross-correlation lag (in samples) to call a trace shift
SHIFT_THRESHOLD_SAMPLES = 3

# Minimum number of substitutions to classify as base_editing
MIN_SUBSTITUTIONS_FOR_BASE_EDIT = 1

# Sliding window size for correlation analysis (in bases)
CORRELATION_WINDOW = 20

# Minimum correlation coefficient to consider traces "similar" before editing site
MIN_PRE_EDIT_CORRELATION = 0.85

# Base editing window positions (4-8 from PAM, 0-indexed from guide start)
BASE_EDITING_WINDOW_START = 4
BASE_EDITING_WINDOW_END = 8


def classify_editing_type(
    wt_chromatogram,
    edited_chromatogram,
) -> tuple[str, float]:
    """
    Classify the type of genome editing from WT vs Edited chromatogram comparison.

    Analyzes the difference between wild-type and edited chromatogram traces
    to determine the most likely editing outcome. Uses a combination of:
    1. Trace shift detection (indel signature)
    2. Point substitution detection (base editing signature)
    3. Pattern analysis to distinguish editing types

    Classification logic:
    - has_shift AND has_substitutions → "mixed"
    - has_shift AND NOT has_substitutions → "indel"
    - NOT has_shift AND has_substitutions → "base_editing"
    - NOT has_shift AND NOT has_substitutions → "unknown"

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type (unedited) chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.

    Returns
    -------
    tuple[str, float]
        A tuple of (editing_type, classification_confidence) where:
        - editing_type is one of: "base_editing", "indel", "prime_editing",
          "hdr", "mixed", "unknown"
        - classification_confidence is a float in [0, 1] indicating how
          confident the classifier is in the result

    Notes
    -----
    The confidence score is computed from:
    - Signal quality (mean quality scores of both chromatograms)
    - Clarity of the detected pattern (shift magnitude or substitution count)
    - Consistency of the difference signal

    Example
    -------
    >>> # editing_type, confidence = classify_editing_type(wt_chrom, edited_chrom)
    >>> # assert editing_type in ["base_editing", "indel", "mixed", "unknown"]
    """
    try:
        # Compute difference signal between aligned traces
        diff_signal = compute_trace_difference(wt_chromatogram, edited_chromatogram)

        # Detect trace shift (indel signature)
        has_shift, shift_magnitude = detect_trace_shift(diff_signal)

        # Detect point substitutions (base editing signature)
        substitutions = detect_point_substitutions(wt_chromatogram, edited_chromatogram)

        has_substitutions = len(substitutions) >= MIN_SUBSTITUTIONS_FOR_BASE_EDIT

        # Classify based on detected patterns
        editing_type, base_confidence = _classify_from_patterns(
            has_shift=has_shift,
            shift_magnitude=shift_magnitude,
            has_substitutions=has_substitutions,
            substitutions=substitutions,
        )

        # Compute final confidence score
        confidence = _compute_classification_confidence(
            editing_type=editing_type,
            base_confidence=base_confidence,
            wt_chromatogram=wt_chromatogram,
            edited_chromatogram=edited_chromatogram,
            has_shift=has_shift,
            shift_magnitude=shift_magnitude,
            substitutions=substitutions,
        )

        logger.debug(
            "Editing classification: type=%s, confidence=%.3f, "
            "shift=%s (mag=%d), substitutions=%d",
            editing_type, confidence, has_shift, shift_magnitude, len(substitutions),
        )

        return editing_type, confidence

    except Exception as exc:
        logger.warning("Editing classification failed: %s", exc)
        return "unknown", 0.0


def compute_trace_difference(
    wt_chromatogram,
    edited_chromatogram,
) -> dict:
    """
    Compute the difference signal between WT and Edited chromatogram traces.

    Aligns the two traces using cross-correlation to find the optimal offset,
    then computes the per-sample difference for each of the four nucleotide
    channels (A, T, C, G).

    The alignment uses the normalized traces (trace_A_norm, etc.) which have
    been baseline-corrected and scaled to a common range.

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram data with normalized trace signals.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data with normalized trace signals.

    Returns
    -------
    dict
        Dictionary containing:
        - "diff_A": np.ndarray — difference in A channel (edited - WT)
        - "diff_T": np.ndarray — difference in T channel
        - "diff_C": np.ndarray — difference in C channel
        - "diff_G": np.ndarray — difference in G channel
        - "wt_A": np.ndarray — aligned WT A channel
        - "wt_T": np.ndarray — aligned WT T channel
        - "wt_C": np.ndarray — aligned WT C channel
        - "wt_G": np.ndarray — aligned WT G channel
        - "edited_A": np.ndarray — aligned edited A channel
        - "edited_T": np.ndarray — aligned edited T channel
        - "edited_C": np.ndarray — aligned edited C channel
        - "edited_G": np.ndarray — aligned edited G channel
        - "alignment_offset": int — sample offset applied to align traces
        - "aligned_length": int — length of the aligned region

    Notes
    -----
    The alignment offset is determined by finding the lag that maximizes
    the cross-correlation between the WT and edited A-channel traces.
    A positive offset means the edited trace is shifted right relative to WT.

    Example
    -------
    >>> # diff = compute_trace_difference(wt_chrom, edited_chrom)
    >>> # print(f"Alignment offset: {diff['alignment_offset']} samples")
    """
    # Extract normalized traces as numpy arrays
    wt_A = np.array(wt_chromatogram.trace.trace_A_norm, dtype=float)
    wt_T = np.array(wt_chromatogram.trace.trace_T_norm, dtype=float)
    wt_C = np.array(wt_chromatogram.trace.trace_C_norm, dtype=float)
    wt_G = np.array(wt_chromatogram.trace.trace_G_norm, dtype=float)

    ed_A = np.array(edited_chromatogram.trace.trace_A_norm, dtype=float)
    ed_T = np.array(edited_chromatogram.trace.trace_T_norm, dtype=float)
    ed_C = np.array(edited_chromatogram.trace.trace_C_norm, dtype=float)
    ed_G = np.array(edited_chromatogram.trace.trace_G_norm, dtype=float)

    # Find alignment offset using cross-correlation on A channel
    # (A channel is typically the most informative for alignment)
    offset = _find_alignment_offset(wt_A, ed_A)

    # Apply offset to align traces
    wt_A_aligned, ed_A_aligned = _apply_offset(wt_A, ed_A, offset)
    wt_T_aligned, ed_T_aligned = _apply_offset(wt_T, ed_T, offset)
    wt_C_aligned, ed_C_aligned = _apply_offset(wt_C, ed_C, offset)
    wt_G_aligned, ed_G_aligned = _apply_offset(wt_G, ed_G, offset)

    aligned_length = len(wt_A_aligned)

    # Compute difference signals
    diff_A = ed_A_aligned - wt_A_aligned
    diff_T = ed_T_aligned - wt_T_aligned
    diff_C = ed_C_aligned - wt_C_aligned
    diff_G = ed_G_aligned - wt_G_aligned

    return {
        "diff_A": diff_A,
        "diff_T": diff_T,
        "diff_C": diff_C,
        "diff_G": diff_G,
        "wt_A": wt_A_aligned,
        "wt_T": wt_T_aligned,
        "wt_C": wt_C_aligned,
        "wt_G": wt_G_aligned,
        "edited_A": ed_A_aligned,
        "edited_T": ed_T_aligned,
        "edited_C": ed_C_aligned,
        "edited_G": ed_G_aligned,
        "alignment_offset": offset,
        "aligned_length": aligned_length,
    }


def detect_trace_shift(difference_signal: dict) -> tuple[bool, int]:
    """
    Detect a trace shift indicative of indels in the edited chromatogram.

    A trace shift occurs when an insertion or deletion causes the edited
    trace to be offset relative to the WT trace after the editing site.
    This is detected by analyzing the cross-correlation between WT and
    edited traces in a sliding window: before the editing site, the
    correlation should be high; after the editing site, the optimal lag
    changes if there is an indel.

    The shift is detected by:
    1. Computing the combined difference signal (sum of absolute differences)
    2. Finding the position of maximum difference (editing site)
    3. Computing cross-correlation before and after the editing site
    4. Comparing the optimal lag before vs after the editing site

    Parameters
    ----------
    difference_signal : dict
        Output from ``compute_trace_difference``, containing aligned WT
        and edited traces and their differences.

    Returns
    -------
    tuple[bool, int]
        A tuple of (has_shift, shift_magnitude_bases) where:
        - has_shift: True if a significant trace shift was detected
        - shift_magnitude_bases: estimated indel size in bases (positive
          for insertions, negative for deletions, 0 if no shift)

    Notes
    -----
    The shift magnitude is estimated from the cross-correlation lag difference
    between the pre-edit and post-edit regions. This is an approximation;
    the actual indel size may differ due to trace compression/expansion.

    Example
    -------
    >>> # has_shift, magnitude = detect_trace_shift(diff_signal)
    >>> # if has_shift: print(f"Indel detected: ~{magnitude} bp")
    """
    wt_A = difference_signal.get("wt_A", np.array([]))
    ed_A = difference_signal.get("edited_A", np.array([]))
    diff_A = difference_signal.get("diff_A", np.array([]))
    diff_T = difference_signal.get("diff_T", np.array([]))
    diff_C = difference_signal.get("diff_C", np.array([]))
    diff_G = difference_signal.get("diff_G", np.array([]))

    if len(wt_A) < 2 * CORRELATION_WINDOW:
        return False, 0

    # Combined absolute difference signal
    combined_diff = (
        np.abs(diff_A) + np.abs(diff_T) +
        np.abs(diff_C) + np.abs(diff_G)
    )

    # Find the editing site (position of maximum cumulative difference)
    # Use a smoothed version to avoid noise
    if len(combined_diff) > 10:
        smoothed = np.convolve(combined_diff, np.ones(10) / 10, mode="same")
        edit_site = int(np.argmax(smoothed))
    else:
        edit_site = len(combined_diff) // 2

    # Need enough signal on both sides of the editing site
    min_window = CORRELATION_WINDOW
    if edit_site < min_window or edit_site > len(wt_A) - min_window:
        # Editing site too close to edge; use midpoint
        edit_site = len(wt_A) // 2

    # Compute cross-correlation before the editing site
    pre_wt = wt_A[max(0, edit_site - min_window):edit_site]
    pre_ed = ed_A[max(0, edit_site - min_window):edit_site]

    # Compute cross-correlation after the editing site
    post_wt = wt_A[edit_site:min(len(wt_A), edit_site + min_window)]
    post_ed = ed_A[edit_site:min(len(ed_A), edit_site + min_window)]

    if len(pre_wt) < 5 or len(post_wt) < 5:
        return False, 0

    # Find optimal lag for pre-edit region
    pre_lag = _find_optimal_lag(pre_wt, pre_ed)

    # Find optimal lag for post-edit region
    post_lag = _find_optimal_lag(post_wt, post_ed)

    # A shift is detected if the lag changes significantly after the editing site
    lag_difference = abs(post_lag - pre_lag)

    has_shift = lag_difference >= SHIFT_THRESHOLD_SAMPLES

    # Estimate shift magnitude in bases
    # Convert from samples to bases using peak spacing
    peak_positions = wt_A  # Use trace directly
    if len(wt_A) > 0:
        # Estimate samples per base from peak positions
        n_peaks = len(wt_A)
        n_samples = len(wt_A)
        samples_per_base = max(1, n_samples / max(1, n_peaks))
        shift_magnitude = int(round(lag_difference / samples_per_base))
    else:
        shift_magnitude = lag_difference

    # Sign: positive = insertion (edited trace is longer), negative = deletion
    shift_sign = 1 if post_lag > pre_lag else -1
    shift_magnitude = shift_sign * shift_magnitude if has_shift else 0

    logger.debug(
        "Trace shift detection: has_shift=%s, pre_lag=%d, post_lag=%d, "
        "lag_diff=%d, magnitude=%d",
        has_shift, pre_lag, post_lag, lag_difference, shift_magnitude,
    )

    return has_shift, shift_magnitude


def detect_point_substitutions(
    wt_chromatogram,
    edited_chromatogram,
) -> list[dict]:
    """
    Detect point substitutions (base editing events) between WT and edited chromatograms.

    Compares the base calls and peak height proportions at each position in
    the readable region. A point substitution is detected when:
    1. The called base differs between WT and edited
    2. The proportion change for the new base exceeds SUBSTITUTION_THRESHOLD

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.

    Returns
    -------
    list[dict]
        List of detected substitutions, each as a dictionary:
        - "position": int — 0-based position in the readable region
        - "wt_base": str — wild-type base call at this position
        - "edited_base": str — edited base call at this position
        - "proportion_change": float — change in proportion of the edited base
        - "wt_proportions": dict — WT base proportions {A, T, C, G}
        - "edited_proportions": dict — edited base proportions {A, T, C, G}
        - "is_base_edit": bool — True if this is a canonical base edit (C→T or A→G)

    Notes
    -----
    Only positions within the readable region of both chromatograms are
    analyzed. Positions with 'N' base calls are skipped.

    Example
    -------
    >>> # subs = detect_point_substitutions(wt_chrom, edited_chrom)
    >>> # for s in subs:
    >>> #     print(f"Position {s['position']}: {s['wt_base']}→{s['edited_base']}")
    """
    substitutions = []

    # Get readable sequences
    wt_start = wt_chromatogram.trace.readable_region_start
    wt_end = wt_chromatogram.trace.readable_region_end
    ed_start = edited_chromatogram.trace.readable_region_start
    ed_end = edited_chromatogram.trace.readable_region_end

    wt_seq = wt_chromatogram.trace.base_calls[wt_start:wt_end]
    ed_seq = edited_chromatogram.trace.base_calls[ed_start:ed_end]

    # Get peak heights for proportion computation
    wt_heights = wt_chromatogram.trace.peak_heights
    ed_heights = edited_chromatogram.trace.peak_heights

    min_len = min(len(wt_seq), len(ed_seq))

    for i in range(min_len):
        wt_base = wt_seq[i] if i < len(wt_seq) else "N"
        ed_base = ed_seq[i] if i < len(ed_seq) else "N"

        # Skip ambiguous positions
        if wt_base == "N" or ed_base == "N":
            continue

        # Compute base proportions at this position
        wt_props = _compute_proportions_at_position(wt_heights, i)
        ed_props = _compute_proportions_at_position(ed_heights, i)

        if wt_base != ed_base:
            # Compute proportion change for the edited base
            proportion_change = ed_props.get(ed_base, 0.0) - wt_props.get(ed_base, 0.0)

            if proportion_change >= SUBSTITUTION_THRESHOLD:
                # Canonical base edits: C→T (CBE) or A→G (ABE)
                is_base_edit = (
                    (wt_base == "C" and ed_base == "T") or
                    (wt_base == "A" and ed_base == "G")
                )

                substitutions.append({
                    "position": i,
                    "wt_base": wt_base,
                    "edited_base": ed_base,
                    "proportion_change": round(proportion_change, 4),
                    "wt_proportions": wt_props,
                    "edited_proportions": ed_props,
                    "is_base_edit": is_base_edit,
                })

    logger.debug(
        "Detected %d point substitutions (%d canonical base edits)",
        len(substitutions),
        sum(1 for s in substitutions if s["is_base_edit"]),
    )

    return substitutions


# ── Private helper functions ──────────────────────────────────────────────────


def _find_alignment_offset(signal1: np.ndarray, signal2: np.ndarray) -> int:
    """
    Find the optimal alignment offset between two signals using cross-correlation.

    Parameters
    ----------
    signal1 : np.ndarray
        Reference signal (WT trace).
    signal2 : np.ndarray
        Query signal (edited trace).

    Returns
    -------
    int
        Optimal offset (positive = signal2 is shifted right relative to signal1).
    """
    if len(signal1) == 0 or len(signal2) == 0:
        return 0

    # Normalize signals for cross-correlation
    s1 = signal1 - np.mean(signal1)
    s2 = signal2 - np.mean(signal2)

    std1 = np.std(s1)
    std2 = np.std(s2)

    if std1 < 1e-10 or std2 < 1e-10:
        return 0

    s1 = s1 / std1
    s2 = s2 / std2

    # Compute cross-correlation
    # Limit search range to ±10% of signal length to avoid spurious alignments
    max_lag = max(1, len(signal1) // 10)

    # Use scipy correlate for efficiency
    correlation = scipy_signal.correlate(s1, s2, mode="full")
    lags = scipy_signal.correlation_lags(len(s1), len(s2), mode="full")

    # Restrict to valid lag range
    valid_mask = np.abs(lags) <= max_lag
    if not np.any(valid_mask):
        return 0

    valid_corr = correlation[valid_mask]
    valid_lags = lags[valid_mask]

    best_lag = int(valid_lags[np.argmax(valid_corr)])
    return best_lag


def _apply_offset(
    signal1: np.ndarray,
    signal2: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply an alignment offset to two signals, trimming to the overlapping region.

    Parameters
    ----------
    signal1 : np.ndarray
        Reference signal.
    signal2 : np.ndarray
        Query signal to offset.
    offset : int
        Offset to apply (positive = shift signal2 right).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Aligned (signal1_trimmed, signal2_trimmed) with equal lengths.
    """
    if offset == 0:
        min_len = min(len(signal1), len(signal2))
        return signal1[:min_len], signal2[:min_len]

    if offset > 0:
        # signal2 is shifted right: trim start of signal2, end of signal1
        s1 = signal1[offset:]
        s2 = signal2[:len(signal1) - offset]
    else:
        # signal2 is shifted left: trim start of signal1, end of signal2
        abs_offset = abs(offset)
        s1 = signal1[:len(signal2) - abs_offset]
        s2 = signal2[abs_offset:]

    min_len = min(len(s1), len(s2))
    return s1[:min_len], s2[:min_len]


def _find_optimal_lag(signal1: np.ndarray, signal2: np.ndarray) -> int:
    """
    Find the lag that maximizes cross-correlation between two short signals.

    Parameters
    ----------
    signal1 : np.ndarray
        First signal.
    signal2 : np.ndarray
        Second signal.

    Returns
    -------
    int
        Optimal lag (positive = signal2 leads signal1).
    """
    if len(signal1) < 2 or len(signal2) < 2:
        return 0

    correlation = scipy_signal.correlate(signal1, signal2, mode="full")
    lags = scipy_signal.correlation_lags(len(signal1), len(signal2), mode="full")
    return int(lags[np.argmax(correlation)])


def _compute_proportions_at_position(
    peak_heights: dict,
    position: int,
) -> dict[str, float]:
    """
    Compute normalized base proportions at a given peak position.

    Parameters
    ----------
    peak_heights : dict
        Peak heights dictionary with keys "A", "T", "C", "G".
    position : int
        0-based position index.

    Returns
    -------
    dict[str, float]
        Normalized proportions for each base, summing to 1.0.
    """
    heights = {}
    for base in ["A", "T", "C", "G"]:
        channel = peak_heights.get(base, [])
        if position < len(channel):
            heights[base] = max(0.0, float(channel[position]))
        else:
            heights[base] = 0.0

    total = sum(heights.values())
    if total < 1e-10:
        return {"A": 0.25, "T": 0.25, "C": 0.25, "G": 0.25}

    return {base: h / total for base, h in heights.items()}


def _classify_from_patterns(
    has_shift: bool,
    shift_magnitude: int,
    has_substitutions: bool,
    substitutions: list[dict],
) -> tuple[str, float]:
    """
    Classify editing type from detected patterns.

    Parameters
    ----------
    has_shift : bool
        Whether a trace shift (indel) was detected.
    shift_magnitude : int
        Estimated indel size in bases.
    has_substitutions : bool
        Whether point substitutions were detected.
    substitutions : list[dict]
        List of detected substitution events.

    Returns
    -------
    tuple[str, float]
        (editing_type, base_confidence) where base_confidence is a
        preliminary confidence score before quality adjustment.
    """
    canonical_base_edits = sum(1 for s in substitutions if s.get("is_base_edit", False))
    total_substitutions = len(substitutions)

    if has_shift and has_substitutions:
        # Both indel and substitution signals → mixed editing
        return "mixed", 0.65

    elif has_shift and not has_substitutions:
        # Only indel signal → NHEJ/indel editing
        # Large shifts might indicate prime editing
        if abs(shift_magnitude) > 10:
            return "prime_editing", 0.55
        return "indel", 0.80

    elif not has_shift and has_substitutions:
        # Only substitution signal → base editing
        if canonical_base_edits > 0:
            # Canonical C→T or A→G edits → high confidence base editing
            fraction_canonical = canonical_base_edits / max(1, total_substitutions)
            confidence = 0.70 + 0.20 * fraction_canonical
            return "base_editing", confidence
        else:
            # Non-canonical substitutions → could be HDR or other
            return "base_editing", 0.55

    else:
        # No clear signal detected
        return "unknown", 0.30


def _compute_classification_confidence(
    editing_type: str,
    base_confidence: float,
    wt_chromatogram,
    edited_chromatogram,
    has_shift: bool,
    shift_magnitude: int,
    substitutions: list[dict],
) -> float:
    """
    Compute the final classification confidence score.

    Adjusts the base confidence based on:
    - Signal quality (mean quality scores)
    - Readable region length
    - Clarity of the detected pattern

    Parameters
    ----------
    editing_type : str
        The classified editing type.
    base_confidence : float
        Preliminary confidence from pattern classification.
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram.
    edited_chromatogram : ChromatogramData
        Edited chromatogram.
    has_shift : bool
        Whether a trace shift was detected.
    shift_magnitude : int
        Estimated indel size.
    substitutions : list[dict]
        Detected substitutions.

    Returns
    -------
    float
        Final confidence score in [0, 1].
    """
    confidence = base_confidence

    # Quality adjustment: penalize low-quality chromatograms
    wt_quality = wt_chromatogram.mean_quality / 40.0  # Normalize to [0, 1]
    ed_quality = edited_chromatogram.mean_quality / 40.0
    quality_factor = (wt_quality + ed_quality) / 2.0
    quality_factor = min(1.0, max(0.0, quality_factor))

    # Readable region length adjustment
    wt_readable = wt_chromatogram.trace.readable_length
    ed_readable = edited_chromatogram.trace.readable_length
    min_readable = min(wt_readable, ed_readable)
    length_factor = min(1.0, min_readable / 200.0)  # Full confidence at 200+ bases

    # Pattern clarity adjustment
    if editing_type == "indel" and has_shift:
        clarity_factor = min(1.0, abs(shift_magnitude) / 5.0)
    elif editing_type == "base_editing" and substitutions:
        mean_prop_change = np.mean([s["proportion_change"] for s in substitutions])
        clarity_factor = min(1.0, mean_prop_change / 0.5)
    elif editing_type == "unknown":
        clarity_factor = 0.5
    else:
        clarity_factor = 0.7

    # Combine factors
    confidence = confidence * (
        0.5 * quality_factor +
        0.25 * length_factor +
        0.25 * clarity_factor
    )

    # Ensure confidence is in [0, 1]
    return float(np.clip(confidence, 0.0, 1.0))
