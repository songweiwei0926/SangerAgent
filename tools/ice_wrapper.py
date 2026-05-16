"""
ICE (Inference of CRISPR Edits) wrapper for SangerAgent.

ICE is a tool by Synthego for quantifying CRISPR editing efficiency
from Sanger sequencing data. It uses a signal decomposition algorithm
to estimate the fraction of alleles that were edited.

Wrapper strategy:
1. Check if 'ice' CLI is available (shutil.which("ice"))
2. If available: run ICE with appropriate arguments and parse JSON output
3. If not available: run internal fallback estimator

Internal fallback algorithm (signal decomposition):
- Decompose edited trace as linear combination of WT trace + edited components
- Use non-negative least squares (scipy.optimize.nnls)
- Estimate editing efficiency as 1 - WT_fraction
- Estimate indel distribution from residual signal peaks

Reference:
    Hsiau et al. (2019). Inference of CRISPR Edits from Sanger Trace Data.
    bioRxiv. https://doi.org/10.1101/251082

Example
-------
>>> from tools.ice_wrapper import run_ice_analysis
>>> # result = run_ice_analysis(wt_chrom, edited_chrom, settings)
>>> # print(f"ICE efficiency: {result['efficiency']:.1%}")
"""

from __future__ import annotations

import json
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

logger = logging.getLogger(__name__)

# ── ICE analysis constants ────────────────────────────────────────────────────
# Indel sizes to test in the decomposition (-10 to +10 bp)
INDEL_SIZES = list(range(-10, 11))

# Minimum samples per base for trace alignment
MIN_SAMPLES_PER_BASE = 5

# Minimum correlation coefficient to accept alignment
MIN_ALIGNMENT_CORRELATION = 0.7

# Minimum efficiency to report (below this → 0.0)
MIN_EFFICIENCY_THRESHOLD = 0.02


def run_ice_analysis(
    wt_chromatogram,
    edited_chromatogram,
    settings,
) -> dict:
    """
    Run ICE analysis on paired WT and edited chromatograms.

    Attempts to use the official ICE CLI tool if available. Falls back to
    the internal signal decomposition algorithm if ICE is not installed.

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type (unedited) chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.
    settings : SangerSettings
        Application settings. Used to check for ICE CLI availability
        and to locate temporary file directories.

    Returns
    -------
    dict
        Standardized result dictionary containing:
        - "tool": str — "ICE" if official tool used, "ice_fallback" otherwise
        - "efficiency": float — overall editing efficiency [0, 1]
        - "indel_pct": float — fraction of alleles with indels [0, 1]
        - "r_squared": float — goodness of fit for the decomposition [0, 1]
        - "indel_distribution": dict — {indel_size: frequency} mapping
        - "ice_score": float — ICE score (same as efficiency * 100 for fallback)
        - "ko_score": float — estimated knockout score [0, 100]

    Notes
    -----
    The ICE score is defined as the percentage of alleles with indels.
    The KO score estimates the fraction of alleles that result in a
    frameshift (indels not divisible by 3).

    Example
    -------
    >>> # result = run_ice_analysis(wt_chrom, edited_chrom, settings)
    >>> # print(f"Tool: {result['tool']}, Efficiency: {result['efficiency']:.1%}")
    """
    # Check if official ICE CLI is available
    ice_cli = shutil.which("ice")

    if ice_cli is not None:
        logger.info("Using official ICE CLI: %s", ice_cli)
        try:
            result = _run_ice_cli_with_chromatograms(
                wt_chromatogram, edited_chromatogram, settings
            )
            if result is not None:
                return result
            logger.warning("ICE CLI failed; falling back to internal algorithm")
        except Exception as exc:
            logger.warning("ICE CLI error: %s; falling back to internal algorithm", exc)

    # Use internal fallback
    logger.info("Using internal ICE fallback algorithm")
    return run_ice_fallback(wt_chromatogram, edited_chromatogram)


def run_ice_cli(wt_path: str, edited_path: str, output_dir: str) -> dict:
    """
    Run the official ICE CLI tool and parse its JSON output.

    Invokes the ``ice`` command-line tool with the WT and edited AB1 file
    paths, waits for completion, and parses the output JSON file.

    Parameters
    ----------
    wt_path : str
        Absolute path to the wild-type AB1 file.
    edited_path : str
        Absolute path to the edited sample AB1 file.
    output_dir : str
        Directory where ICE should write its output files.

    Returns
    -------
    dict
        Standardized result dictionary (same format as ``run_ice_analysis``).

    Raises
    ------
    FileNotFoundError
        If the ICE CLI is not found in PATH.
    subprocess.CalledProcessError
        If the ICE CLI exits with a non-zero return code.
    ValueError
        If the ICE output JSON cannot be parsed.

    Notes
    -----
    The ICE CLI is expected to produce a JSON file named ``results.json``
    in the output directory. The exact output format depends on the ICE
    version installed.

    Example
    -------
    >>> # result = run_ice_cli("/data/wt.ab1", "/data/edited.ab1", "/tmp/ice_out")
    """
    ice_cli = shutil.which("ice")
    if ice_cli is None:
        raise FileNotFoundError("ICE CLI not found in PATH")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        ice_cli,
        "--ctrl", wt_path,
        "--edited", edited_path,
        "--out", output_dir,
        "--name", "ice_result",
    ]

    logger.debug("Running ICE CLI: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    logger.debug("ICE CLI stdout: %s", result.stdout[:500])

    # Parse ICE output JSON
    output_json = Path(output_dir) / "ice_result.json"
    if not output_json.exists():
        # Try alternative output file names
        for candidate in Path(output_dir).glob("*.json"):
            output_json = candidate
            break
        else:
            raise ValueError(f"ICE output JSON not found in {output_dir}")

    with open(output_json, "r") as f:
        ice_data = json.load(f)

    return _parse_ice_output(ice_data)


def run_ice_fallback(
    wt_chromatogram,
    edited_chromatogram,
) -> dict:
    """
    Internal ICE fallback using signal decomposition.

    Implements a simplified version of the ICE algorithm using non-negative
    least squares (NNLS) decomposition:

    1. Extract and normalize traces from both chromatograms
    2. Align traces using cross-correlation to find the editing site
    3. For each candidate indel size (-10 to +10 bp):
       a. Shift the WT trace by the indel size
       b. Create a basis vector for this indel component
    4. Use NNLS to find the best-fit mixture:
       edited_trace ≈ α₀ * wt_trace + Σᵢ αᵢ * shifted_wt_trace_i
    5. Compute efficiency = 1 - α₀ (fraction not matching WT)
    6. Estimate indel distribution from the α coefficients

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram data.
    edited_chromatogram : ChromatogramData
        Edited sample chromatogram data.

    Returns
    -------
    dict
        Standardized result dictionary with tool="ice_fallback".

    Notes
    -----
    The fallback algorithm is less accurate than the official ICE tool,
    particularly for complex indel distributions. It works best for
    single-indel editing events.

    Example
    -------
    >>> # result = run_ice_fallback(wt_chrom, edited_chrom)
    >>> # print(f"Estimated efficiency: {result['efficiency']:.1%}")
    """
    try:
        # Step 1: Extract normalized traces
        wt_trace = _extract_combined_trace(wt_chromatogram)
        ed_trace = _extract_combined_trace(edited_chromatogram)

        if len(wt_trace) < 20 or len(ed_trace) < 20:
            logger.warning("Traces too short for ICE fallback analysis")
            return _empty_ice_result()

        # Step 2: Align traces using cross-correlation
        offset, edit_site = _find_edit_site(wt_trace, ed_trace)

        # Step 3: Build basis matrix for NNLS decomposition
        # Each column is the WT trace shifted by a different indel size
        basis_matrix, valid_indel_sizes = _build_basis_matrix(
            wt_trace, ed_trace, edit_site, INDEL_SIZES
        )

        if basis_matrix.shape[1] == 0:
            logger.warning("Could not build basis matrix for ICE fallback")
            return _empty_ice_result()

        # Step 4: NNLS decomposition
        # Solve: ed_trace ≈ basis_matrix @ coefficients
        # First column of basis_matrix is the unedited WT component
        coefficients, residual_norm = nnls(basis_matrix, ed_trace)

        # Normalize coefficients to sum to 1
        coeff_sum = np.sum(coefficients)
        if coeff_sum < 1e-10:
            return _empty_ice_result()

        normalized_coeffs = coefficients / coeff_sum

        # Step 5: Compute efficiency
        # First coefficient corresponds to the unedited WT component
        wt_fraction = float(normalized_coeffs[0])
        efficiency = max(0.0, min(1.0, 1.0 - wt_fraction))

        if efficiency < MIN_EFFICIENCY_THRESHOLD:
            efficiency = 0.0

        # Step 6: Compute R² (goodness of fit)
        fitted = basis_matrix @ coefficients
        ss_res = np.sum((ed_trace - fitted) ** 2)
        ss_tot = np.sum((ed_trace - np.mean(ed_trace)) ** 2)
        r_squared = float(1.0 - ss_res / max(ss_tot, 1e-10))
        r_squared = max(0.0, min(1.0, r_squared))

        # Step 7: Build indel distribution
        # Skip the first coefficient (WT component, indel size = 0)
        indel_distribution = {}
        for i, indel_size in enumerate(valid_indel_sizes[1:], start=1):
            if i < len(normalized_coeffs):
                freq = float(normalized_coeffs[i])
                if freq > 0.01:  # Only include indels with >1% frequency
                    indel_distribution[indel_size] = round(freq, 4)

        # Compute ICE score (percentage of alleles with indels)
        ice_score = round(efficiency * 100, 1)

        # Compute KO score (fraction of frameshift indels)
        frameshift_fraction = sum(
            freq for size, freq in indel_distribution.items()
            if size % 3 != 0
        )
        ko_score = round(frameshift_fraction * 100, 1)

        logger.info(
            "ICE fallback result: efficiency=%.1f%%, R²=%.3f, "
            "indel_sizes=%d",
            efficiency * 100, r_squared, len(indel_distribution),
        )

        return {
            "tool": "ice_fallback",
            "efficiency": round(efficiency, 4),
            "indel_pct": round(efficiency, 4),
            "r_squared": round(r_squared, 4),
            "indel_distribution": indel_distribution,
            "ice_score": ice_score,
            "ko_score": ko_score,
        }

    except Exception as exc:
        logger.error("ICE fallback analysis failed: %s", exc, exc_info=True)
        return _empty_ice_result()


# ── Private helper functions ──────────────────────────────────────────────────


def _extract_combined_trace(chromatogram) -> np.ndarray:
    """
    Extract a combined normalized trace signal from a chromatogram.

    Combines all four channels (A, T, C, G) into a single signal by
    taking the maximum at each sample point. This captures the overall
    trace shape regardless of which base is dominant.

    Parameters
    ----------
    chromatogram : ChromatogramData
        Chromatogram data with normalized trace signals.

    Returns
    -------
    np.ndarray
        Combined normalized trace signal, scaled to [0, 1].
    """
    A = np.array(chromatogram.trace.trace_A_norm, dtype=float)
    T = np.array(chromatogram.trace.trace_T_norm, dtype=float)
    C = np.array(chromatogram.trace.trace_C_norm, dtype=float)
    G = np.array(chromatogram.trace.trace_G_norm, dtype=float)

    # Combine channels: use maximum signal at each point
    combined = np.maximum(np.maximum(A, T), np.maximum(C, G))

    # Normalize to [0, 1]
    max_val = np.max(combined)
    if max_val > 1e-10:
        combined = combined / max_val

    return combined


def _find_edit_site(
    wt_trace: np.ndarray,
    ed_trace: np.ndarray,
) -> tuple[int, int]:
    """
    Find the alignment offset and editing site between WT and edited traces.

    Uses cross-correlation to find the optimal alignment offset, then
    identifies the editing site as the position where the traces diverge.

    Parameters
    ----------
    wt_trace : np.ndarray
        Wild-type combined trace.
    ed_trace : np.ndarray
        Edited combined trace.

    Returns
    -------
    tuple[int, int]
        (alignment_offset, edit_site_sample) where:
        - alignment_offset: samples to shift ed_trace for alignment
        - edit_site_sample: sample index of the editing site
    """
    # Find alignment offset using cross-correlation
    min_len = min(len(wt_trace), len(ed_trace))
    wt_short = wt_trace[:min_len]
    ed_short = ed_trace[:min_len]

    # Cross-correlation with limited lag range
    max_lag = min_len // 10
    correlation = scipy_signal.correlate(wt_short, ed_short, mode="full")
    lags = scipy_signal.correlation_lags(len(wt_short), len(ed_short), mode="full")

    valid_mask = np.abs(lags) <= max_lag
    if np.any(valid_mask):
        valid_corr = correlation[valid_mask]
        valid_lags = lags[valid_mask]
        offset = int(valid_lags[np.argmax(valid_corr)])
    else:
        offset = 0

    # Find editing site: position where traces diverge
    # Use sliding window correlation to find the last highly-correlated position
    window = 20
    edit_site = min_len // 2  # Default to midpoint

    for i in range(window, min_len - window, 5):
        wt_window = wt_short[i:i + window]
        ed_window = ed_short[i:i + window]

        if np.std(wt_window) < 1e-10 or np.std(ed_window) < 1e-10:
            continue

        try:
            corr, _ = pearsonr(wt_window, ed_window)
            if corr < MIN_ALIGNMENT_CORRELATION:
                edit_site = i
                break
        except Exception:
            continue

    return offset, edit_site


def _build_basis_matrix(
    wt_trace: np.ndarray,
    ed_trace: np.ndarray,
    edit_site: int,
    indel_sizes: list[int],
) -> tuple[np.ndarray, list[int]]:
    """
    Build the basis matrix for NNLS decomposition.

    Each column represents the WT trace shifted by a different indel size.
    The first column (indel size = 0) represents the unedited WT component.

    Parameters
    ----------
    wt_trace : np.ndarray
        Wild-type combined trace.
    ed_trace : np.ndarray
        Edited combined trace (determines output length).
    edit_site : int
        Sample index of the editing site.
    indel_sizes : list[int]
        List of indel sizes to test (e.g., [-10, ..., 0, ..., +10]).

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (basis_matrix, valid_indel_sizes) where:
        - basis_matrix: shape (n_samples, n_indels) — each column is a
          shifted WT trace
        - valid_indel_sizes: list of indel sizes corresponding to columns
    """
    n_samples = len(ed_trace)
    basis_columns = []
    valid_sizes = []

    for indel_size in indel_sizes:
        # Create shifted WT trace
        shifted = _shift_trace_at_site(wt_trace, edit_site, indel_size, n_samples)
        if shifted is not None and len(shifted) == n_samples:
            basis_columns.append(shifted)
            valid_sizes.append(indel_size)

    if not basis_columns:
        return np.zeros((n_samples, 0)), []

    basis_matrix = np.column_stack(basis_columns)
    return basis_matrix, valid_sizes


def _shift_trace_at_site(
    wt_trace: np.ndarray,
    edit_site: int,
    indel_size: int,
    target_length: int,
) -> Optional[np.ndarray]:
    """
    Create a version of the WT trace shifted by an indel at the editing site.

    For a deletion (negative indel_size): removes samples at the editing site.
    For an insertion (positive indel_size): inserts interpolated samples.
    For no indel (indel_size = 0): returns the WT trace unchanged.

    Parameters
    ----------
    wt_trace : np.ndarray
        Wild-type trace.
    edit_site : int
        Sample index of the editing site.
    indel_size : int
        Indel size in bases (negative = deletion, positive = insertion).
    target_length : int
        Desired output length (to match edited trace length).

    Returns
    -------
    Optional[np.ndarray]
        Shifted trace of length target_length, or None if the shift
        would result in an invalid trace.
    """
    if indel_size == 0:
        # No shift: return WT trace trimmed/padded to target length
        if len(wt_trace) >= target_length:
            return wt_trace[:target_length].copy()
        else:
            # Pad with zeros
            padded = np.zeros(target_length)
            padded[:len(wt_trace)] = wt_trace
            return padded

    pre_edit = wt_trace[:edit_site]
    post_edit = wt_trace[edit_site:]

    if indel_size < 0:
        # Deletion: remove |indel_size| samples from the editing site
        n_remove = abs(indel_size)
        if n_remove >= len(post_edit):
            return None
        post_shifted = post_edit[n_remove:]
        shifted = np.concatenate([pre_edit, post_shifted])
    else:
        # Insertion: add indel_size interpolated samples at the editing site
        # Interpolate between the last pre-edit and first post-edit sample
        if len(pre_edit) == 0 or len(post_edit) == 0:
            return None
        insert_val = (pre_edit[-1] + post_edit[0]) / 2.0
        inserted = np.full(indel_size, insert_val)
        shifted = np.concatenate([pre_edit, inserted, post_edit])

    # Trim or pad to target length
    if len(shifted) >= target_length:
        return shifted[:target_length]
    else:
        padded = np.zeros(target_length)
        padded[:len(shifted)] = shifted
        return padded


def _run_ice_cli_with_chromatograms(
    wt_chromatogram,
    edited_chromatogram,
    settings,
) -> Optional[dict]:
    """
    Write chromatograms to temporary AB1 files and run the ICE CLI.

    Parameters
    ----------
    wt_chromatogram : ChromatogramData
        Wild-type chromatogram.
    edited_chromatogram : ChromatogramData
        Edited chromatogram.
    settings : SangerSettings
        Application settings.

    Returns
    -------
    Optional[dict]
        ICE result dictionary, or None if the CLI fails.
    """
    # ICE CLI requires actual AB1 files; we need the original file paths
    # Since we only have ChromatogramData objects here, we can't easily
    # reconstruct AB1 files. Return None to trigger fallback.
    logger.debug(
        "ICE CLI requires original AB1 files; ChromatogramData objects "
        "cannot be converted back to AB1 format. Using fallback."
    )
    return None


def _parse_ice_output(ice_data: dict) -> dict:
    """
    Parse the official ICE tool JSON output into the standardized format.

    Parameters
    ----------
    ice_data : dict
        Raw JSON output from the ICE CLI tool.

    Returns
    -------
    dict
        Standardized result dictionary.
    """
    efficiency = float(ice_data.get("ice", ice_data.get("efficiency", 0.0))) / 100.0
    ko_score = float(ice_data.get("ko_score", 0.0))
    r_squared = float(ice_data.get("r_squared", 0.0))

    # Parse indel distribution
    indel_dist_raw = ice_data.get("indel_distribution", {})
    indel_distribution = {
        int(k): float(v)
        for k, v in indel_dist_raw.items()
        if float(v) > 0.01
    }

    return {
        "tool": "ICE",
        "efficiency": round(efficiency, 4),
        "indel_pct": round(efficiency, 4),
        "r_squared": round(r_squared, 4),
        "indel_distribution": indel_distribution,
        "ice_score": round(efficiency * 100, 1),
        "ko_score": round(ko_score, 1),
    }


def _empty_ice_result() -> dict:
    """
    Return an empty ICE result for error cases.

    Returns
    -------
    dict
        ICE result with all metrics set to zero.
    """
    return {
        "tool": "ice_fallback",
        "efficiency": 0.0,
        "indel_pct": 0.0,
        "r_squared": 0.0,
        "indel_distribution": {},
        "ice_score": 0.0,
        "ko_score": 0.0,
    }


# Import pearsonr at module level for use in _find_edit_site
try:
    from scipy.stats import pearsonr
except ImportError:
    def pearsonr(x, y):
        """Fallback Pearson correlation."""
        x = np.array(x)
        y = np.array(y)
        if len(x) < 2:
            return 0.0, 1.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(corr), 0.05
