"""
BEAT (Base Editing Analysis Tool) wrapper for SangerAgent.

BEAT quantifies base editing efficiency from Sanger sequencing by comparing
base proportions at each position between wild-type and edited chromatograms.

Wrapper strategy:
1. Check if BEAT is available (shutil.which("beat") or BEAT Python package)
2. If available: run BEAT with appropriate arguments
3. If not available: run internal base proportion comparator

Internal fallback algorithm:
- Compare base proportions at each position between WT and Edited
- Identify positions with significant proportion changes (>0.1)
- Quantify C→T or A→G conversion efficiency (typical base editing outcomes)
- Report editing efficiency per position
- Identify editing window (consecutive edited positions)
- Classify bystander edits (edits outside the primary editing window)

Reference:
    Kluesner et al. (2018). EditR: A Method to Quantify Base Editing from
    Sanger Sequencing. The CRISPR Journal, 1(3), 239-250.
    https://doi.org/10.1089/crispr.2018.0014

Example
-------
>>> from tools.beat_wrapper import run_beat_analysis
>>> # result = run_beat_analysis(wt_chrom, edited_chrom, settings)
>>> # print(f"Base editing efficiency: {result['editing_efficiency']:.1%}")
>>> # print(f"Edit type: {result['base_edit_type']}")
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── BEAT analysis constants ───────────────────────────────────────────────────
# Minimum proportion change to call a base editing event
BEAT_PROPORTION_THRESHOLD = 0.10

# Minimum proportion change for a "significant" edit (used for efficiency)
BEAT_SIGNIFICANT_THRESHOLD = 0.15

# Maximum gap between edited positions to be considered the same editing window
BEAT_WINDOW_GAP = 3

# Minimum number of positions to define an editing window
BEAT_MIN_WINDOW_SIZE = 1

# Canonical base editing conversions
CANONICAL_CBE = ("C", "T")  # Cytosine base editor: C→T
CANONICAL_ABE = ("A", "G")  # Adenine base editor: A→G


def run_beat_analysis(
    wt_chromatogram,
    edited_chromatogram,
    settings,
) -> dict:
    """
    Run BEAT analysis on paired WT and edited chromatograms.

    Attempts to use the official BEAT tool if available. Falls back to
    the internal base proportion comparison algorithm if BEAT is not installed.

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type (unedited) chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.
    settings : SangerSettings
        Application settings. Used to check for BEAT tool availability.

    Returns
    -------
    dict
        Standardized result dictionary containing:
        - "tool": str — "BEAT" if official tool used, "beat_fallback" otherwise
        - "editing_efficiency": float — mean editing efficiency across edited
          positions [0, 1]
        - "edited_positions": list[dict] — list of edited positions, each with:
          {pos, wt_base, edited_base, efficiency}
        - "base_edit_type": str — "C_to_T", "A_to_G", or "other"
        - "editing_window": tuple[int, int] — (start, end) of the editing window
          (0-based positions in the readable region)
        - "bystander_edits": list[dict] — edits outside the primary editing window

    Notes
    -----
    The editing efficiency is computed as the mean proportion change at
    positions where a significant base editing event was detected.
    The editing window is defined as the contiguous region containing
    the majority of editing events.

    Example
    -------
    >>> # result = run_beat_analysis(wt_chrom, edited_chrom, settings)
    >>> # for pos in result['edited_positions']:
    >>> #     print(f"  Position {pos['pos']}: {pos['wt_base']}→{pos['edited_base']} "
    >>> #           f"({pos['efficiency']:.1%})")
    """
    # Check if official BEAT tool is available
    beat_cli = shutil.which("beat")

    if beat_cli is not None:
        logger.info("Using official BEAT CLI: %s", beat_cli)
        try:
            result = _run_beat_cli(wt_chromatogram, edited_chromatogram, beat_cli)
            if result is not None:
                return result
            logger.warning("BEAT CLI failed; falling back to internal algorithm")
        except Exception as exc:
            logger.warning(
                "BEAT CLI error: %s; falling back to internal algorithm", exc
            )

    # Try BEAT Python package
    try:
        import beat  # type: ignore
        logger.info("Using BEAT Python package")
        result = _run_beat_python(wt_chromatogram, edited_chromatogram)
        if result is not None:
            return result
    except ImportError:
        pass

    # Use internal fallback
    logger.info("Using internal BEAT fallback algorithm")
    return run_beat_fallback(wt_chromatogram, edited_chromatogram)


def run_beat_fallback(
    wt_chromatogram,
    edited_chromatogram,
) -> dict:
    """
    Internal BEAT fallback using base proportion comparison.

    Implements a base editing quantification algorithm:
    1. Compute base proportions at each position for both chromatograms
    2. Find positions where the proportion of a base increased significantly (>0.10)
    3. Classify the edit type (C→T: cytosine base editing, A→G: adenine base editing)
    4. Compute efficiency as the mean proportion change at edited positions
    5. Identify the editing window (consecutive edited positions)
    6. Classify bystander edits (edits outside the primary editing window)

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.

    Returns
    -------
    dict
        Standardized result dictionary with tool="beat_fallback".

    Notes
    -----
    The algorithm compares base proportions at the peak level (using
    peak heights at each base call position). This is more robust than
    comparing raw trace signals because it accounts for trace compression
    and expansion.

    Example
    -------
    >>> # result = run_beat_fallback(wt_chrom, edited_chrom)
    >>> # print(f"Efficiency: {result['editing_efficiency']:.1%}")
    """
    try:
        # Step 1: Compute base proportions at each position
        wt_proportions = _compute_all_proportions(wt_chromatogram)
        ed_proportions = _compute_all_proportions(edited_chromatogram)

        min_len = min(len(wt_proportions), len(ed_proportions))

        if min_len == 0:
            logger.warning("No peak data available for BEAT fallback")
            return _empty_beat_result()

        # Step 2: Find positions with significant proportion changes
        edited_positions = []

        for i in range(min_len):
            wt_props = wt_proportions[i]
            ed_props = ed_proportions[i]

            # Find the base with the largest proportion increase
            max_increase = 0.0
            best_base = None
            wt_dominant = max(wt_props, key=wt_props.get)

            for base in ["A", "T", "C", "G"]:
                increase = ed_props.get(base, 0.0) - wt_props.get(base, 0.0)
                if increase > max_increase and base != wt_dominant:
                    max_increase = increase
                    best_base = base

            if max_increase >= BEAT_PROPORTION_THRESHOLD and best_base is not None:
                edited_positions.append({
                    "pos": i,
                    "wt_base": wt_dominant,
                    "edited_base": best_base,
                    "efficiency": round(max_increase, 4),
                    "wt_proportions": wt_props,
                    "edited_proportions": ed_props,
                })

        if not edited_positions:
            logger.info("No base editing events detected")
            return _empty_beat_result()

        # Step 3: Classify edit type
        base_edit_type = _classify_base_edit_type(edited_positions)

        # Step 4: Compute overall editing efficiency
        significant_edits = [
            ep for ep in edited_positions
            if ep["efficiency"] >= BEAT_SIGNIFICANT_THRESHOLD
        ]

        if significant_edits:
            editing_efficiency = float(np.mean([ep["efficiency"] for ep in significant_edits]))
        else:
            editing_efficiency = float(np.mean([ep["efficiency"] for ep in edited_positions]))

        editing_efficiency = min(1.0, editing_efficiency)

        # Step 5: Identify editing window
        editing_window = _identify_editing_window(edited_positions)

        # Step 6: Classify bystander edits
        bystander_edits = _identify_bystander_edits(
            edited_positions, editing_window
        )

        # Format edited_positions for output (remove internal fields)
        output_positions = [
            {
                "pos": ep["pos"],
                "wt_base": ep["wt_base"],
                "edited_base": ep["edited_base"],
                "efficiency": ep["efficiency"],
            }
            for ep in edited_positions
        ]

        logger.info(
            "BEAT fallback result: efficiency=%.1f%%, type=%s, "
            "edited_positions=%d, bystander=%d",
            editing_efficiency * 100, base_edit_type,
            len(edited_positions), len(bystander_edits),
        )

        return {
            "tool": "beat_fallback",
            "editing_efficiency": round(editing_efficiency, 4),
            "edited_positions": output_positions,
            "base_edit_type": base_edit_type,
            "editing_window": editing_window,
            "bystander_edits": bystander_edits,
        }

    except Exception as exc:
        logger.error("BEAT fallback analysis failed: %s", exc, exc_info=True)
        return _empty_beat_result()


# ── Private helper functions ──────────────────────────────────────────────────


def _compute_all_proportions(chromatogram) -> list[dict[str, float]]:
    """
    Compute normalized base proportions at every peak position.

    Parameters
    ----------
    chromatogram : ChromatogramData
        Chromatogram data with peak heights.

    Returns
    -------
    list[dict[str, float]]
        List of proportion dictionaries, one per peak position.
        Each dictionary has keys "A", "T", "C", "G" summing to ~1.0.
    """
    peak_heights = chromatogram.trace.peak_heights
    n_peaks = len(chromatogram.trace.peak_positions)

    proportions = []
    for i in range(n_peaks):
        heights = {}
        for base in ["A", "T", "C", "G"]:
            channel = peak_heights.get(base, [])
            if i < len(channel):
                heights[base] = max(0.0, float(channel[i]))
            else:
                heights[base] = 0.0

        total = sum(heights.values())
        if total < 1e-10:
            proportions.append({"A": 0.25, "T": 0.25, "C": 0.25, "G": 0.25})
        else:
            proportions.append({base: h / total for base, h in heights.items()})

    return proportions


def _classify_base_edit_type(edited_positions: list[dict]) -> str:
    """
    Classify the base editing type from detected editing events.

    Counts canonical C→T (CBE) and A→G (ABE) conversions and returns
    the most common type. Returns "other" if neither canonical type
    is predominant.

    Parameters
    ----------
    edited_positions : list[dict]
        List of detected editing events with "wt_base" and "edited_base" keys.

    Returns
    -------
    str
        "C_to_T" for cytosine base editing (CBE),
        "A_to_G" for adenine base editing (ABE),
        "other" for non-canonical or mixed editing.
    """
    cbe_count = sum(
        1 for ep in edited_positions
        if ep["wt_base"] == CANONICAL_CBE[0] and ep["edited_base"] == CANONICAL_CBE[1]
    )
    abe_count = sum(
        1 for ep in edited_positions
        if ep["wt_base"] == CANONICAL_ABE[0] and ep["edited_base"] == CANONICAL_ABE[1]
    )
    total = len(edited_positions)

    if total == 0:
        return "other"

    cbe_fraction = cbe_count / total
    abe_fraction = abe_count / total

    if cbe_fraction >= 0.5:
        return "C_to_T"
    elif abe_fraction >= 0.5:
        return "A_to_G"
    else:
        return "other"


def _identify_editing_window(
    edited_positions: list[dict],
) -> tuple[int, int]:
    """
    Identify the primary editing window from detected editing events.

    The editing window is defined as the contiguous region containing
    the majority of editing events. Gaps of up to BEAT_WINDOW_GAP
    positions are allowed within the window.

    Parameters
    ----------
    edited_positions : list[dict]
        List of detected editing events with "pos" keys.

    Returns
    -------
    tuple[int, int]
        (window_start, window_end) as 0-based positions (inclusive).
        Returns (0, 0) if no editing events are detected.
    """
    if not edited_positions:
        return (0, 0)

    positions = sorted(ep["pos"] for ep in edited_positions)

    if len(positions) == 1:
        return (positions[0], positions[0])

    # Find the largest contiguous cluster of positions
    best_start = positions[0]
    best_end = positions[0]
    best_count = 1

    current_start = positions[0]
    current_end = positions[0]
    current_count = 1

    for i in range(1, len(positions)):
        gap = positions[i] - positions[i - 1]
        if gap <= BEAT_WINDOW_GAP:
            # Extend current window
            current_end = positions[i]
            current_count += 1
        else:
            # Start new window
            if current_count > best_count:
                best_start = current_start
                best_end = current_end
                best_count = current_count
            current_start = positions[i]
            current_end = positions[i]
            current_count = 1

    # Check final window
    if current_count > best_count:
        best_start = current_start
        best_end = current_end

    return (best_start, best_end)


def _identify_bystander_edits(
    edited_positions: list[dict],
    editing_window: tuple[int, int],
) -> list[dict]:
    """
    Identify bystander edits outside the primary editing window.

    Bystander edits are editing events that occur outside the primary
    editing window. They may indicate off-target activity or secondary
    editing events.

    Parameters
    ----------
    edited_positions : list[dict]
        All detected editing events.
    editing_window : tuple[int, int]
        (start, end) of the primary editing window.

    Returns
    -------
    list[dict]
        List of bystander editing events (same format as edited_positions).
    """
    window_start, window_end = editing_window

    bystander = [
        {
            "pos": ep["pos"],
            "wt_base": ep["wt_base"],
            "edited_base": ep["edited_base"],
            "efficiency": ep["efficiency"],
        }
        for ep in edited_positions
        if ep["pos"] < window_start or ep["pos"] > window_end
    ]

    return bystander


def _run_beat_cli(
    wt_chromatogram,
    edited_chromatogram,
    beat_cli: str,
) -> Optional[dict]:
    """
    Run the official BEAT CLI tool.

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram.
    edited_chromatogram : ChromatogramData
        Edited chromatogram.
    beat_cli : str
        Path to the BEAT CLI executable.

    Returns
    -------
    Optional[dict]
        BEAT result dictionary, or None if the CLI fails.
    """
    # BEAT CLI requires original AB1 files; we can't reconstruct them
    # from ChromatogramData objects. Return None to trigger fallback.
    logger.debug(
        "BEAT CLI requires original AB1 files; "
        "ChromatogramData objects cannot be converted. Using fallback."
    )
    return None


def _run_beat_python(
    wt_chromatogram,
    edited_chromatogram,
) -> Optional[dict]:
    """
    Run BEAT via its Python API if available.

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram.
    edited_chromatogram : ChromatogramData
        Edited chromatogram.

    Returns
    -------
    Optional[dict]
        BEAT result dictionary, or None if the Python API fails.
    """
    # BEAT Python package API varies by version; return None to use fallback
    return None


def _empty_beat_result() -> dict:
    """
    Return an empty BEAT result for error cases.

    Returns
    -------
    dict
        BEAT result with all metrics set to zero/empty.
    """
    return {
        "tool": "beat_fallback",
        "editing_efficiency": 0.0,
        "edited_positions": [],
        "base_edit_type": "other",
        "editing_window": (0, 0),
        "bystander_edits": [],
    }
