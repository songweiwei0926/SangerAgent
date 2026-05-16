"""
TIDE (Tracking of Indels by DEcomposition) wrapper for SangerAgent.

TIDE quantifies the frequency and size of indels from Sanger sequencing
by decomposing the edited chromatogram trace as a mixture of WT traces
shifted by different indel sizes.

Reference:
    Brinkman et al. (2014). Easy quantitative assessment of genome editing
    by sequence trace decomposition. Nucleic Acids Research, 42(22), e168.
    https://doi.org/10.1093/nar/gku936

Wrapper strategy:
1. Check if TIDE R script is available (shutil.which("Rscript") + TIDE script)
2. If available: run via Rscript subprocess and parse output
3. If not available: run internal Python fallback

Internal fallback algorithm:
- Align WT and Edited sequences using cross-correlation
- Find decomposition point (where traces diverge, r < 0.95)
- For each indel size (-10 to +10):
  a. Shift WT trace by indel size after the decomposition point
  b. Create basis vector for this indel component
- Use NNLS to find best-fit indel mixture
- Compute p-value from residual analysis
- Return indel spectrum

Example
-------
>>> from tools.tide_wrapper import run_tide_analysis
>>> # result = run_tide_analysis(wt_chrom, edited_chrom, settings)
>>> # print(f"TIDE efficiency: {result['efficiency']:.1%}")
>>> # print(f"Indel spectrum: {result['indel_spectrum']}")
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import nnls
from scipy import signal as scipy_signal
from scipy.stats import chi2

logger = logging.getLogger(__name__)

# ── TIDE analysis constants ───────────────────────────────────────────────────
# Indel sizes to test in the decomposition
TIDE_INDEL_SIZES = list(range(-10, 11))

# Minimum correlation to consider traces "similar" (pre-decomposition point)
TIDE_CORRELATION_THRESHOLD = 0.95

# Sliding window size for correlation analysis
TIDE_WINDOW_SIZE = 20

# Minimum frequency to include an indel in the spectrum
TIDE_MIN_FREQUENCY = 0.01

# Significance threshold for p-value
TIDE_SIGNIFICANCE_THRESHOLD = 0.05


def run_tide_analysis(
    wt_chromatogram,
    edited_chromatogram,
    settings,
) -> dict:
    """
    Run TIDE analysis on paired WT and edited chromatograms.

    Attempts to use the official TIDE R script if available. Falls back to
    the internal Python decomposition algorithm if TIDE is not installed.

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type (unedited) chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.
    settings : SangerSettings
        Application settings. Used to check for TIDE R script availability.

    Returns
    -------
    dict
        Standardized result dictionary containing:
        - "tool": str — "TIDE" if official tool used, "tide_fallback" otherwise
        - "efficiency": float — overall editing efficiency [0, 1]
        - "indel_spectrum": dict — {indel_size: frequency} mapping
          (e.g., {-1: 0.30, +1: 0.10, -3: 0.05})
        - "decomposition_point": int — 0-based position where traces diverge
        - "r_squared": float — goodness of fit [0, 1]
        - "p_value": float — statistical significance of the decomposition

    Notes
    -----
    The TIDE efficiency is the sum of all non-zero indel frequencies.
    The decomposition point is the last position where WT and edited
    traces are highly correlated (r > 0.95).

    Example
    -------
    >>> # result = run_tide_analysis(wt_chrom, edited_chrom, settings)
    >>> # for size, freq in result['indel_spectrum'].items():
    >>> #     print(f"  {size:+d} bp: {freq:.1%}")
    """
    # Check if TIDE R script is available
    rscript = shutil.which("Rscript")
    tide_script = _find_tide_script()

    if rscript is not None and tide_script is not None:
        logger.info("Using official TIDE R script: %s", tide_script)
        try:
            result = _run_tide_rscript(
                wt_chromatogram, edited_chromatogram, rscript, tide_script
            )
            if result is not None:
                return result
            logger.warning("TIDE R script failed; falling back to internal algorithm")
        except Exception as exc:
            logger.warning(
                "TIDE R script error: %s; falling back to internal algorithm", exc
            )

    # Use internal fallback
    logger.info("Using internal TIDE fallback algorithm")
    return run_tide_fallback(wt_chromatogram, edited_chromatogram)


def find_decomposition_point(
    wt_chromatogram,
    edited_chromatogram,
) -> int:
    """
    Find the position where WT and Edited traces diverge.

    Uses a sliding window Pearson correlation to find the last position
    where the WT and edited traces are highly correlated (r > 0.95).
    This position marks the start of the editing-induced divergence.

    The analysis is performed on the normalized traces at the base-call
    level (using peak heights at each peak position).

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.

    Returns
    -------
    int
        0-based position (in base calls) of the decomposition point.
        Returns the midpoint of the readable region if no clear
        divergence is detected.

    Notes
    -----
    The decomposition point is determined at the base-call level, not
    the sample level. This corresponds to the position in the sequence
    where the editing event occurred.

    Example
    -------
    >>> # decomp_point = find_decomposition_point(wt_chrom, edited_chrom)
    >>> # print(f"Traces diverge at position {decomp_point}")
    """
    # Extract peak heights at each base position
    wt_heights = _extract_peak_height_sequence(wt_chromatogram)
    ed_heights = _extract_peak_height_sequence(edited_chromatogram)

    min_len = min(len(wt_heights), len(ed_heights))

    if min_len < 2 * TIDE_WINDOW_SIZE:
        return min_len // 2

    # Slide window and compute correlation
    last_correlated_pos = TIDE_WINDOW_SIZE  # Default: start of readable region

    for i in range(TIDE_WINDOW_SIZE, min_len - TIDE_WINDOW_SIZE, 1):
        wt_window = wt_heights[i - TIDE_WINDOW_SIZE:i]
        ed_window = ed_heights[i - TIDE_WINDOW_SIZE:i]

        if np.std(wt_window) < 1e-10 or np.std(ed_window) < 1e-10:
            continue

        try:
            corr = float(np.corrcoef(wt_window, ed_window)[0, 1])
            if corr >= TIDE_CORRELATION_THRESHOLD:
                last_correlated_pos = i
            elif last_correlated_pos > TIDE_WINDOW_SIZE:
                # Correlation dropped below threshold after being high
                # This is the decomposition point
                break
        except Exception:
            continue

    logger.debug(
        "TIDE decomposition point: position %d (of %d)",
        last_correlated_pos, min_len,
    )

    return last_correlated_pos


def run_tide_fallback(
    wt_chromatogram,
    edited_chromatogram,
) -> dict:
    """
    Internal TIDE fallback using trace decomposition.

    Implements a simplified version of the TIDE algorithm:
    1. Find the decomposition point (where traces diverge)
    2. For each indel size (-10 to +10 bp):
       a. Shift the WT trace by the indel size after the decomposition point
       b. Create a basis vector for this indel component
    3. Use NNLS to find the best-fit mixture of indel components
    4. Compute the indel spectrum and overall efficiency
    5. Estimate p-value from residual analysis

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.

    Returns
    -------
    dict
        Standardized result dictionary with tool="tide_fallback".

    Notes
    -----
    The p-value is estimated using a chi-squared test on the residuals.
    A p-value < 0.05 indicates that the decomposition is statistically
    significant (i.e., the editing efficiency is non-zero).

    Example
    -------
    >>> # result = run_tide_fallback(wt_chrom, edited_chrom)
    >>> # print(f"Efficiency: {result['efficiency']:.1%}")
    """
    try:
        # Step 1: Find decomposition point
        decomp_point = find_decomposition_point(wt_chromatogram, edited_chromatogram)

        # Step 2: Extract peak height sequences
        wt_heights = _extract_peak_height_sequence(wt_chromatogram)
        ed_heights = _extract_peak_height_sequence(edited_chromatogram)

        min_len = min(len(wt_heights), len(ed_heights))

        if min_len < decomp_point + TIDE_WINDOW_SIZE:
            logger.warning("Not enough signal after decomposition point for TIDE")
            return _empty_tide_result(decomp_point)

        # Use only the post-decomposition region for analysis
        wt_post = wt_heights[decomp_point:min_len]
        ed_post = ed_heights[decomp_point:min_len]
        n_post = len(ed_post)

        if n_post < 10:
            return _empty_tide_result(decomp_point)

        # Step 3: Build basis matrix
        basis_matrix, valid_sizes = _build_tide_basis(wt_post, TIDE_INDEL_SIZES)

        if basis_matrix.shape[1] == 0:
            return _empty_tide_result(decomp_point)

        # Step 4: NNLS decomposition
        coefficients, residual_norm = nnls(basis_matrix, ed_post)

        # Normalize coefficients
        coeff_sum = np.sum(coefficients)
        if coeff_sum < 1e-10:
            return _empty_tide_result(decomp_point)

        normalized_coeffs = coefficients / coeff_sum

        # Step 5: Compute R²
        fitted = basis_matrix @ coefficients
        ss_res = np.sum((ed_post - fitted) ** 2)
        ss_tot = np.sum((ed_post - np.mean(ed_post)) ** 2)
        r_squared = float(1.0 - ss_res / max(ss_tot, 1e-10))
        r_squared = max(0.0, min(1.0, r_squared))

        # Step 6: Build indel spectrum
        # Index 0 corresponds to indel_size = 0 (unedited WT)
        indel_spectrum = {}
        wt_fraction = 0.0

        for i, indel_size in enumerate(valid_sizes):
            if i >= len(normalized_coeffs):
                break
            freq = float(normalized_coeffs[i])
            if indel_size == 0:
                wt_fraction = freq
            elif freq >= TIDE_MIN_FREQUENCY:
                indel_spectrum[indel_size] = round(freq, 4)

        # Overall efficiency = 1 - WT fraction
        efficiency = max(0.0, min(1.0, 1.0 - wt_fraction))

        # Step 7: Compute p-value using chi-squared test on residuals
        p_value = _compute_tide_pvalue(ed_post, fitted, len(valid_sizes))

        logger.info(
            "TIDE fallback result: efficiency=%.1f%%, decomp_point=%d, "
            "R²=%.3f, p=%.4f, indel_sizes=%d",
            efficiency * 100, decomp_point, r_squared, p_value,
            len(indel_spectrum),
        )

        return {
            "tool": "tide_fallback",
            "efficiency": round(efficiency, 4),
            "indel_spectrum": indel_spectrum,
            "decomposition_point": decomp_point,
            "r_squared": round(r_squared, 4),
            "p_value": round(p_value, 6),
        }

    except Exception as exc:
        logger.error("TIDE fallback analysis failed: %s", exc, exc_info=True)
        return _empty_tide_result(0)


# ── Private helper functions ──────────────────────────────────────────────────


def _extract_peak_height_sequence(chromatogram) -> np.ndarray:
    """
    Extract a combined peak height sequence from a chromatogram.

    Returns the maximum peak height across all four channels at each
    base position, normalized to [0, 1].

    Parameters
    ----------
    chromatogram : ChromatogramData
        Chromatogram data with peak heights.

    Returns
    -------
    np.ndarray
        Combined peak height sequence, normalized to [0, 1].
    """
    peak_heights = chromatogram.trace.peak_heights
    n_peaks = len(chromatogram.trace.peak_positions)

    if n_peaks == 0:
        return np.array([])

    heights = np.zeros(n_peaks)
    for base in ["A", "T", "C", "G"]:
        channel = peak_heights.get(base, [])
        for i in range(min(n_peaks, len(channel))):
            heights[i] = max(heights[i], float(channel[i]))

    # Normalize to [0, 1]
    max_h = np.max(heights)
    if max_h > 1e-10:
        heights = heights / max_h

    return heights


def _build_tide_basis(
    wt_post: np.ndarray,
    indel_sizes: list[int],
) -> tuple[np.ndarray, list[int]]:
    """
    Build the TIDE basis matrix for NNLS decomposition.

    Each column represents the WT trace shifted by a different indel size.
    The column for indel_size=0 is the unedited WT component.

    Parameters
    ----------
    wt_post : np.ndarray
        WT peak height sequence after the decomposition point.
    indel_sizes : list[int]
        List of indel sizes to test.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (basis_matrix, valid_indel_sizes) where basis_matrix has shape
        (n_samples, n_indels).
    """
    n_samples = len(wt_post)
    basis_columns = []
    valid_sizes = []

    for indel_size in indel_sizes:
        shifted = _shift_sequence(wt_post, indel_size, n_samples)
        if shifted is not None:
            basis_columns.append(shifted)
            valid_sizes.append(indel_size)

    if not basis_columns:
        return np.zeros((n_samples, 0)), []

    return np.column_stack(basis_columns), valid_sizes


def _shift_sequence(
    sequence: np.ndarray,
    shift: int,
    target_length: int,
) -> Optional[np.ndarray]:
    """
    Shift a sequence by a given number of positions.

    For positive shift (insertion): prepend zeros.
    For negative shift (deletion): remove elements from the start.
    For zero shift: return unchanged.

    Parameters
    ----------
    sequence : np.ndarray
        Input sequence.
    shift : int
        Number of positions to shift (positive = right, negative = left).
    target_length : int
        Desired output length.

    Returns
    -------
    Optional[np.ndarray]
        Shifted sequence of length target_length, or None if invalid.
    """
    if shift == 0:
        if len(sequence) >= target_length:
            return sequence[:target_length].copy()
        else:
            padded = np.zeros(target_length)
            padded[:len(sequence)] = sequence
            return padded

    if shift > 0:
        # Insertion: shift right (prepend zeros)
        if shift >= target_length:
            return None
        shifted = np.zeros(target_length)
        shifted[shift:] = sequence[:target_length - shift]
        return shifted
    else:
        # Deletion: shift left (remove from start)
        abs_shift = abs(shift)
        if abs_shift >= len(sequence):
            return None
        shifted_seq = sequence[abs_shift:]
        result = np.zeros(target_length)
        copy_len = min(len(shifted_seq), target_length)
        result[:copy_len] = shifted_seq[:copy_len]
        return result


def _compute_tide_pvalue(
    observed: np.ndarray,
    fitted: np.ndarray,
    n_params: int,
) -> float:
    """
    Compute a p-value for the TIDE decomposition using a chi-squared test.

    Tests the null hypothesis that the residuals are consistent with
    random noise (i.e., no editing occurred).

    Parameters
    ----------
    observed : np.ndarray
        Observed edited trace values.
    fitted : np.ndarray
        Fitted values from the NNLS decomposition.
    n_params : int
        Number of parameters in the model (degrees of freedom adjustment).

    Returns
    -------
    float
        P-value in [0, 1]. Small values indicate significant editing.
    """
    n = len(observed)
    if n <= n_params:
        return 1.0

    residuals = observed - fitted
    ss_res = np.sum(residuals ** 2)

    # Estimate noise variance from the pre-decomposition region
    # (use variance of residuals as proxy)
    noise_var = np.var(residuals)
    if noise_var < 1e-10:
        return 0.001  # Very good fit → significant

    # Chi-squared statistic
    df = n - n_params
    chi2_stat = ss_res / noise_var

    # P-value: probability of observing this chi-squared or larger under H0
    p_value = float(1.0 - chi2.cdf(chi2_stat, df=df))

    return max(0.0, min(1.0, p_value))


def _find_tide_script() -> Optional[str]:
    """
    Search for the TIDE R script in common locations.

    Returns
    -------
    Optional[str]
        Path to the TIDE R script, or None if not found.
    """
    # Common locations for TIDE R script
    search_paths = [
        "/usr/local/bin/tide.R",
        "/usr/bin/tide.R",
        os.path.expanduser("~/tide/tide.R"),
        os.path.expanduser("~/.local/bin/tide.R"),
    ]

    for path in search_paths:
        if os.path.isfile(path):
            return path

    # Check if TIDE is installed as an R package
    rscript = shutil.which("Rscript")
    if rscript:
        try:
            result = subprocess.run(
                [rscript, "-e", "library(TIDE); cat(system.file('scripts/tide.R', package='TIDE'))"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    return None


def _run_tide_rscript(
    wt_chromatogram,
    edited_chromatogram,
    rscript_path: str,
    tide_script_path: str,
) -> Optional[dict]:
    """
    Run the official TIDE R script and parse its output.

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram.
    edited_chromatogram : ChromatogramData
        Edited chromatogram.
    rscript_path : str
        Path to the Rscript executable.
    tide_script_path : str
        Path to the TIDE R script.

    Returns
    -------
    Optional[dict]
        TIDE result dictionary, or None if the script fails.
    """
    # TIDE R script requires AB1 files; we can't reconstruct them from
    # ChromatogramData objects. Return None to trigger fallback.
    logger.debug(
        "TIDE R script requires original AB1 files; "
        "ChromatogramData objects cannot be converted. Using fallback."
    )
    return None


def _empty_tide_result(decomp_point: int) -> dict:
    """
    Return an empty TIDE result for error cases.

    Parameters
    ----------
    decomp_point : int
        Decomposition point to include in the result.

    Returns
    -------
    dict
        TIDE result with all metrics set to zero.
    """
    return {
        "tool": "tide_fallback",
        "efficiency": 0.0,
        "indel_spectrum": {},
        "decomposition_point": decomp_point,
        "r_squared": 0.0,
        "p_value": 1.0,
    }
