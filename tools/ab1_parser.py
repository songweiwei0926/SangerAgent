"""
AB1 Sanger sequencing file parser for SangerAgent.

This module provides a complete, production-grade parser for AB1 (Applied
Biosystems) Sanger sequencing files. It extracts all four fluorescence
channel traces, peak positions, base calls, and quality scores, then applies
a full normalization pipeline and QC assessment.

AB1 File Format
---------------
AB1 files use the ABIF (Applied Biosystems Information Format) binary format.
Key data tags extracted by this parser:

    DATA9  → Channel 1 raw trace (A - adenine, green dye)
    DATA10 → Channel 2 raw trace (C - cytosine, blue dye)
    DATA11 → Channel 3 raw trace (G - guanine, yellow/black dye)
    DATA12 → Channel 4 raw trace (T - thymine, red dye)
    PLOC2  → Peak locations (sample indices of called peaks)
    PBAS2  → Base calls (called sequence string)
    PCON2  → Quality values (Phred-like scores per base)

Note: Some AB1 files use PLOC1/PBAS1/PCON1 tags (older format). This parser
tries both variants.

Normalization Pipeline
----------------------
1. Baseline correction: subtract rolling minimum (window=50 samples)
2. Signal smoothing: Gaussian smoothing (sigma=2)
3. Local peak normalization: divide each peak region by local max
4. Dye intensity normalization: scale each channel so max=1000

QC Flags
--------
- "low_signal": mean peak height < 100 (weak sequencing signal)
- "noisy_trace": signal-to-noise ratio < 5 (high background noise)
- "peak_collapse": >20% of peaks below 10% of median height
- "poor_readable_region": readable region < 100 bases
- "failed_sequencing": no peaks detected (complete sequencing failure)

Example
-------
>>> from tools.ab1_parser import parse_ab1, generate_synthetic_ab1_data
>>> # Parse a real AB1 file
>>> chrom = parse_ab1("/path/to/sample.ab1")
>>> print(f"Sequence: {chrom.readable_sequence[:20]}...")
>>> print(f"Quality: {chrom.mean_quality:.1f}")
>>> print(f"QC pass: {chrom.qc_pass}")
>>>
>>> # Generate synthetic data for testing
>>> synthetic = generate_synthetic_ab1_data()
>>> print(f"Synthetic sequence length: {synthetic.sequence_length}")
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for schema imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.schemas.chromatogram import ChromatogramData, TraceData

logger = logging.getLogger(__name__)

# ── ABI Tag Constants ─────────────────────────────────────────────────────────
# DATA tags: channel order in AB1 files (standard ABI channel mapping)
ABI_CHANNEL_TAGS = {
    "A": "DATA9",   # Adenine  - green dye (FAM/dR110)
    "C": "DATA10",  # Cytosine - blue dye  (JOE/dR6G)
    "G": "DATA11",  # Guanine  - yellow/black dye (TAMRA/dTAMRA)
    "T": "DATA12",  # Thymine  - red dye   (ROX/ddROX)
}

# Peak location tag (try v2 first, fall back to v1)
PLOC_TAGS = ["PLOC2", "PLOC1"]
# Base call tag
PBAS_TAGS = ["PBAS2", "PBAS1"]
# Quality score tag
PCON_TAGS = ["PCON2", "PCON1"]

# ── Normalization Parameters ──────────────────────────────────────────────────
BASELINE_WINDOW = 50       # Rolling minimum window for baseline correction
GAUSSIAN_SIGMA = 2.0       # Sigma for Gaussian smoothing
DYE_NORM_MAX = 1000.0      # Target maximum after dye intensity normalization

# ── QC Thresholds ─────────────────────────────────────────────────────────────
QC_LOW_SIGNAL_THRESHOLD = 100       # Mean peak height below this = low_signal
QC_SNR_THRESHOLD = 5.0              # SNR below this = noisy_trace
QC_PEAK_COLLAPSE_FRACTION = 0.20    # >20% collapsed peaks = peak_collapse
QC_PEAK_COLLAPSE_HEIGHT_PCT = 0.10  # Peak height < 10% of median = collapsed
QC_MIN_READABLE_REGION = 100        # Readable region < 100 bases = poor_readable_region

# ── Readable Region Parameters ────────────────────────────────────────────────
DEFAULT_MIN_QUALITY = 20   # Phred Q20 = 99% accuracy
DEFAULT_WINDOW = 10        # Sliding window for quality trimming


def _rolling_minimum(signal: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling minimum of a 1D signal array.

    Uses a sliding window approach to find the local minimum at each position.
    Edge positions use the available window (no padding).

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values.
    window : int
        Window size for rolling minimum computation.

    Returns
    -------
    np.ndarray
        Array of rolling minimum values, same length as input.

    Example
    -------
    >>> import numpy as np
    >>> sig = np.array([5.0, 3.0, 8.0, 2.0, 6.0])
    >>> _rolling_minimum(sig, window=3)
    array([3., 3., 2., 2., 2.])
    """
    n = len(signal)
    result = np.empty(n, dtype=np.float64)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.min(signal[lo:hi])
    return result


def _gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Generate a normalized 1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian distribution.
    truncate : float
        Truncate the kernel at this many standard deviations.

    Returns
    -------
    np.ndarray
        Normalized 1D Gaussian kernel (sums to 1.0).

    Example
    -------
    >>> kernel = _gaussian_kernel(sigma=2.0)
    >>> abs(kernel.sum() - 1.0) < 1e-10
    True
    """
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_smooth(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to a 1D signal using convolution.

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values to smooth.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Smoothed signal array, same length as input.

    Example
    -------
    >>> import numpy as np
    >>> noisy = np.array([1.0, 5.0, 2.0, 4.0, 1.0, 5.0, 2.0])
    >>> smoothed = _gaussian_smooth(noisy, sigma=1.0)
    >>> smoothed.shape == noisy.shape
    True
    """
    kernel = _gaussian_kernel(sigma)
    # Use 'reflect' padding to avoid edge artifacts
    pad_width = len(kernel) // 2
    padded = np.pad(signal, pad_width, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    # Trim to original length
    return smoothed[:len(signal)]


def normalize_traces(raw_traces: dict[str, list[float]]) -> dict[str, list[float]]:
    """Apply the full normalization pipeline to raw trace signals.

    The normalization pipeline consists of four sequential steps:

    1. **Baseline correction**: Subtract the rolling minimum (window=50) to
       remove the fluorescence baseline drift common in capillary electrophoresis.

    2. **Signal smoothing**: Apply Gaussian smoothing (sigma=2) to reduce
       high-frequency noise while preserving peak shapes.

    3. **Local peak normalization**: For each 50-sample window, divide by the
       local maximum to normalize for signal intensity variations along the run.

    4. **Dye intensity normalization**: Scale each channel independently so
       that its global maximum equals 1000. This corrects for differences in
       dye quantum yield and detector sensitivity between channels.

    Parameters
    ----------
    raw_traces : dict[str, list[float]]
        Dictionary mapping channel names to raw signal lists.
        Expected keys: "A", "T", "C", "G".
        Each value is a list of raw fluorescence intensity values.

    Returns
    -------
    dict[str, list[float]]
        Dictionary mapping channel names to normalized signal lists.
        Same keys as input. Values are floats in [0, 1000].

    Raises
    ------
    ValueError
        If raw_traces is missing required channels (A, T, C, G).
    ValueError
        If any trace is empty.

    Example
    -------
    >>> import numpy as np
    >>> raw = {
    ...     "A": [100.0, 500.0, 200.0, 50.0],
    ...     "T": [50.0, 100.0, 800.0, 30.0],
    ...     "C": [30.0, 40.0, 60.0, 200.0],
    ...     "G": [20.0, 30.0, 25.0, 400.0],
    ... }
    >>> normalized = normalize_traces(raw)
    >>> max(normalized["A"])
    1000.0
    """
    required_channels = {"A", "T", "C", "G"}
    missing = required_channels - set(raw_traces.keys())
    if missing:
        raise ValueError(
            f"raw_traces is missing channels: {missing}. "
            f"Expected: {required_channels}"
        )

    normalized = {}

    for channel, raw_signal in raw_traces.items():
        if channel not in required_channels:
            continue

        if len(raw_signal) == 0:
            raise ValueError(f"Channel '{channel}' trace is empty")

        signal = np.array(raw_signal, dtype=np.float64)

        # Step 1: Baseline correction — subtract rolling minimum
        baseline = _rolling_minimum(signal, window=BASELINE_WINDOW)
        signal = np.maximum(signal - baseline, 0.0)
        logger.debug(
            "Channel %s: baseline correction applied (window=%d)",
            channel, BASELINE_WINDOW
        )

        # Step 2: Signal smoothing — Gaussian filter
        if len(signal) > 2 * GAUSSIAN_SIGMA:
            signal = _gaussian_smooth(signal, sigma=GAUSSIAN_SIGMA)
            signal = np.maximum(signal, 0.0)  # Clip negative values from smoothing
        logger.debug(
            "Channel %s: Gaussian smoothing applied (sigma=%.1f)",
            channel, GAUSSIAN_SIGMA
        )

        # Step 3: Local peak normalization — divide by local max in windows
        local_window = 50
        if len(signal) >= local_window:
            local_norm = np.empty_like(signal)
            for i in range(0, len(signal), local_window):
                chunk = signal[i:i + local_window]
                local_max = np.max(chunk)
                if local_max > 0:
                    local_norm[i:i + local_window] = chunk / local_max
                else:
                    local_norm[i:i + local_window] = chunk
            signal = local_norm
        logger.debug("Channel %s: local peak normalization applied", channel)

        # Step 4: Dye intensity normalization — scale so max = DYE_NORM_MAX
        global_max = np.max(signal)
        if global_max > 0:
            signal = (signal / global_max) * DYE_NORM_MAX
        else:
            # All-zero signal (failed channel)
            signal = np.zeros_like(signal)
        logger.debug(
            "Channel %s: dye normalization applied (max=%.1f)",
            channel, DYE_NORM_MAX
        )

        normalized[channel] = signal.tolist()

    return normalized


def identify_readable_region(
    quality_scores: list[int],
    min_quality: int = DEFAULT_MIN_QUALITY,
    window: int = DEFAULT_WINDOW,
) -> tuple[int, int]:
    """Find the readable region by trimming low-quality ends.

    Uses a sliding window approach to identify the contiguous high-quality
    region of the sequence. The readable region starts at the first window
    where the mean quality exceeds min_quality, and ends at the last such
    window.

    Parameters
    ----------
    quality_scores : list[int]
        Phred-like quality scores per base call. Range [0, 60].
    min_quality : int, optional
        Minimum quality threshold for a base to be considered readable.
        Default is 20 (Q20 = 99% accuracy).
    window : int, optional
        Sliding window size for quality assessment. A window is considered
        high-quality if its mean quality >= min_quality.
        Default is 10.

    Returns
    -------
    tuple[int, int]
        (start, end) indices of the readable region (0-based, end exclusive).
        Returns (0, 0) if no readable region is found.

    Raises
    ------
    ValueError
        If quality_scores is empty.

    Notes
    -----
    The algorithm:
    1. Slide a window of size `window` across the quality scores.
    2. Find the first window where mean quality >= min_quality (start).
    3. Find the last such window (end).
    4. Return (start, end + window) as the readable region bounds.

    Example
    -------
    >>> scores = [5, 8, 10, 25, 30, 35, 40, 38, 35, 30, 25, 20, 8, 5]
    >>> start, end = identify_readable_region(scores, min_quality=20, window=3)
    >>> start, end
    (3, 12)
    """
    if not quality_scores:
        raise ValueError("quality_scores must not be empty")

    n = len(quality_scores)
    scores = np.array(quality_scores, dtype=np.float64)

    if n < window:
        # Sequence shorter than window: check overall mean
        if np.mean(scores) >= min_quality:
            return (0, n)
        return (0, 0)

    # Compute sliding window means
    window_means = np.array([
        np.mean(scores[i:i + window])
        for i in range(n - window + 1)
    ])

    # Find windows that pass the quality threshold
    passing = np.where(window_means >= min_quality)[0]

    if len(passing) == 0:
        logger.warning(
            "No readable region found (all windows below Q%d threshold)",
            min_quality
        )
        return (0, 0)

    start = int(passing[0])
    end = int(passing[-1]) + window  # End is exclusive

    logger.debug(
        "Readable region identified: [%d, %d) (%d bases)",
        start, end, end - start
    )
    return (start, min(end, n))


def compute_qc_metrics(
    trace_data: TraceData,
    quality_scores: list[int],
) -> tuple[list[str], bool]:
    """Compute QC flags and pass/fail status for a chromatogram.

    Evaluates five QC criteria and returns a list of flags for any that fail.
    A chromatogram passes QC if it has no "failed_sequencing" flag and
    fewer than 3 other flags.

    Parameters
    ----------
    trace_data : TraceData
        The trace data object containing peak positions, heights, and signals.
    quality_scores : list[int]
        Phred-like quality scores per base call.

    Returns
    -------
    tuple[list[str], bool]
        A tuple of (qc_flags, qc_pass) where:
        - qc_flags: list of flag strings for failed QC criteria
        - qc_pass: True if the chromatogram passes QC

    QC Flags
    --------
    "failed_sequencing"
        No peaks detected. Critical failure — always causes qc_pass=False.
    "low_signal"
        Mean peak height across all channels < 100. Indicates weak signal.
    "noisy_trace"
        Signal-to-noise ratio < 5. High background noise.
    "peak_collapse"
        >20% of peaks have height < 10% of median peak height.
    "poor_readable_region"
        Readable region is < 100 bases.

    Example
    -------
    >>> flags, passed = compute_qc_metrics(trace_data, quality_scores)
    >>> if not passed:
    ...     print(f"QC failed: {flags}")
    """
    qc_flags: list[str] = []

    peak_positions = trace_data.peak_positions
    peak_heights = trace_data.peak_heights

    # ── Check 1: Failed sequencing (no peaks) ─────────────────────────────────
    if len(peak_positions) == 0:
        logger.warning("QC: failed_sequencing — no peaks detected")
        qc_flags.append("failed_sequencing")
        return qc_flags, False  # Critical failure, skip other checks

    # ── Check 2: Low signal ───────────────────────────────────────────────────
    all_heights = []
    for channel_heights in peak_heights.values():
        all_heights.extend([h for h in channel_heights if h is not None])

    if all_heights:
        mean_height = np.mean(all_heights)
        if mean_height < QC_LOW_SIGNAL_THRESHOLD:
            logger.warning(
                "QC: low_signal — mean peak height %.1f < %d",
                mean_height, QC_LOW_SIGNAL_THRESHOLD
            )
            qc_flags.append("low_signal")
    else:
        qc_flags.append("low_signal")

    # ── Check 3: Noisy trace (SNR) ────────────────────────────────────────────
    # SNR = mean peak height / std of baseline (non-peak) regions
    try:
        # Use the A channel as representative for SNR calculation
        trace_a = np.array(trace_data.trace_A, dtype=np.float64)
        if len(trace_a) > 0 and len(peak_positions) > 0:
            # Create a mask for non-peak regions (baseline)
            peak_mask = np.zeros(len(trace_a), dtype=bool)
            for pos in peak_positions:
                lo = max(0, pos - 5)
                hi = min(len(trace_a), pos + 6)
                peak_mask[lo:hi] = True

            baseline_signal = trace_a[~peak_mask]
            if len(baseline_signal) > 10:
                baseline_std = np.std(baseline_signal)
                if baseline_std > 0 and all_heights:
                    snr = np.mean(all_heights) / baseline_std
                    if snr < QC_SNR_THRESHOLD:
                        logger.warning(
                            "QC: noisy_trace — SNR %.2f < %.1f",
                            snr, QC_SNR_THRESHOLD
                        )
                        qc_flags.append("noisy_trace")
    except Exception as exc:
        logger.debug("SNR computation failed: %s", exc)

    # ── Check 4: Peak collapse ────────────────────────────────────────────────
    if all_heights:
        median_height = np.median(all_heights)
        collapse_threshold = median_height * QC_PEAK_COLLAPSE_HEIGHT_PCT
        n_peaks = len(peak_positions)

        # Count peaks where the dominant channel height is below threshold
        collapsed_count = 0
        for i in range(n_peaks):
            peak_max = max(
                peak_heights.get(ch, [0])[i] if i < len(peak_heights.get(ch, [])) else 0
                for ch in ["A", "T", "C", "G"]
            )
            if peak_max < collapse_threshold:
                collapsed_count += 1

        collapse_fraction = collapsed_count / n_peaks if n_peaks > 0 else 0
        if collapse_fraction > QC_PEAK_COLLAPSE_FRACTION:
            logger.warning(
                "QC: peak_collapse — %.1f%% of peaks collapsed (threshold: %.1f%%)",
                collapse_fraction * 100, QC_PEAK_COLLAPSE_FRACTION * 100
            )
            qc_flags.append("peak_collapse")

    # ── Check 5: Poor readable region ────────────────────────────────────────
    readable_length = trace_data.readable_region_end - trace_data.readable_region_start
    if readable_length < QC_MIN_READABLE_REGION:
        logger.warning(
            "QC: poor_readable_region — readable region %d bases < %d",
            readable_length, QC_MIN_READABLE_REGION
        )
        qc_flags.append("poor_readable_region")

    # ── Determine overall QC pass/fail ────────────────────────────────────────
    # Pass if: no failed_sequencing AND fewer than 3 other flags
    critical_flags = {"failed_sequencing"}
    has_critical = bool(set(qc_flags) & critical_flags)
    qc_pass = not has_critical and len(qc_flags) < 3

    logger.info(
        "QC assessment: flags=%s, pass=%s",
        qc_flags if qc_flags else "none", qc_pass
    )
    return qc_flags, qc_pass


def _extract_tag(annotations: dict, tag_names: list[str]) -> Optional[object]:
    """Extract a value from AB1 annotations, trying multiple tag names.

    Parameters
    ----------
    annotations : dict
        The annotations dictionary from a Biopython SeqRecord.
    tag_names : list[str]
        List of tag names to try in order.

    Returns
    -------
    Optional[object]
        The first found tag value, or None if none found.
    """
    abif_raw = annotations.get("abif_raw", {})
    for tag in tag_names:
        if tag in abif_raw:
            return abif_raw[tag]
    return None


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    str
        Lowercase SHA-256 hex digest.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_ab1(file_path: str) -> ChromatogramData:
    """
    Parse an AB1 Sanger sequencing file and extract all chromatogram data.

    Reads the binary AB1 file using Biopython's SeqIO with format "abi",
    extracts all four fluorescence channel traces, peak positions, base calls,
    and quality scores. Applies the full normalization pipeline and computes
    QC metrics.

    Parameters
    ----------
    file_path : str
        Absolute path to the .ab1 file.

    Returns
    -------
    ChromatogramData
        Complete chromatogram data including:
        - Raw traces for all four channels (A, T, C, G)
        - Normalized traces after baseline correction, smoothing, and scaling
        - Peak positions and heights
        - Base calls and quality scores
        - Readable region boundaries
        - QC flags and pass/fail status

    Raises
    ------
    FileNotFoundError
        If the file does not exist at file_path.
    ValueError
        If the file is not a valid AB1 file or cannot be parsed.
    ImportError
        If Biopython is not installed.

    Notes
    -----
    ABI channel mapping:
        DATA9  → A (adenine, green)
        DATA10 → C (cytosine, blue)
        DATA11 → G (guanine, yellow/black)
        DATA12 → T (thymine, red)

    The function tries both v2 (PLOC2, PBAS2, PCON2) and v1 (PLOC1, PBAS1,
    PCON1) tag variants for compatibility with different AB1 file versions.

    Example
    -------
    >>> chrom = parse_ab1("/data/samples/patient_001.ab1")
    >>> print(f"File: {chrom.file_name}")
    >>> print(f"Sequence ({chrom.sequence_length} bp): {chrom.readable_sequence[:30]}...")
    >>> print(f"Mean quality: {chrom.mean_quality:.1f}")
    >>> print(f"QC pass: {chrom.qc_pass}")
    >>> if chrom.qc_flags:
    ...     print(f"QC flags: {', '.join(chrom.qc_flags)}")
    """
    try:
        from Bio import SeqIO
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for AB1 parsing. "
            "Install with: pip install biopython"
        ) from exc

    # ── File validation ───────────────────────────────────────────────────────
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"AB1 file not found: '{file_path}'"
        )
    if not path.is_file():
        raise ValueError(
            f"Path is not a file: '{file_path}'"
        )

    file_name = path.name
    logger.info("Parsing AB1 file: %s", file_name)

    # ── Compute file hash ─────────────────────────────────────────────────────
    file_hash = _compute_file_hash(file_path)
    logger.debug("File SHA-256: %s", file_hash)

    # ── Parse AB1 file with Biopython ─────────────────────────────────────────
    try:
        record = SeqIO.read(file_path, "abi")
    except Exception as exc:
        raise ValueError(
            f"Failed to parse AB1 file '{file_name}': {exc}. "
            "Ensure the file is a valid AB1/ABIF format file."
        ) from exc

    annotations = record.annotations
    abif_raw = annotations.get("abif_raw", {})

    logger.debug(
        "AB1 file parsed: %d annotation tags found",
        len(abif_raw)
    )

    # ── Extract raw traces ────────────────────────────────────────────────────
    raw_traces: dict[str, list[float]] = {}
    for channel, tag in ABI_CHANNEL_TAGS.items():
        raw_data = abif_raw.get(tag)
        if raw_data is None:
            logger.warning(
                "Channel %s tag '%s' not found in AB1 file, using zeros",
                channel, tag
            )
            # Use zeros as fallback — will trigger low_signal QC flag
            raw_traces[channel] = [0.0]
        else:
            raw_traces[channel] = [float(v) for v in raw_data]
            logger.debug(
                "Channel %s: %d samples extracted from tag %s",
                channel, len(raw_traces[channel]), tag
            )

    # Ensure all traces have the same length (pad shorter ones with zeros)
    max_len = max(len(t) for t in raw_traces.values())
    for channel in raw_traces:
        if len(raw_traces[channel]) < max_len:
            raw_traces[channel].extend([0.0] * (max_len - len(raw_traces[channel])))

    # ── Extract peak positions ────────────────────────────────────────────────
    ploc_data = _extract_tag(annotations, PLOC_TAGS)
    if ploc_data is not None:
        peak_positions = [int(p) for p in ploc_data]
        logger.debug("Peak positions: %d peaks extracted", len(peak_positions))
    else:
        logger.warning("No peak location tag found (PLOC2/PLOC1), using empty list")
        peak_positions = []

    # ── Extract base calls ────────────────────────────────────────────────────
    pbas_data = _extract_tag(annotations, PBAS_TAGS)
    if pbas_data is not None:
        if isinstance(pbas_data, bytes):
            base_calls = pbas_data.decode("ascii", errors="replace").upper()
        else:
            base_calls = str(pbas_data).upper()
        logger.debug("Base calls: %d bases extracted", len(base_calls))
    else:
        # Fall back to Biopython's parsed sequence
        base_calls = str(record.seq).upper()
        logger.debug(
            "Using Biopython sequence as base calls: %d bases",
            len(base_calls)
        )

    # ── Extract quality scores ────────────────────────────────────────────────
    pcon_data = _extract_tag(annotations, PCON_TAGS)
    if pcon_data is not None:
        quality_scores = [int(q) for q in pcon_data]
        logger.debug("Quality scores: %d values extracted", len(quality_scores))
    else:
        # Fall back to Biopython's letter_annotations
        phred_quality = record.letter_annotations.get("phred_quality", [])
        if phred_quality:
            quality_scores = [int(q) for q in phred_quality]
        else:
            # Generate synthetic quality scores as fallback
            quality_scores = [20] * len(base_calls)
            logger.warning(
                "No quality scores found, using default Q20 for all bases"
            )

    # ── Align lengths: base_calls, quality_scores, peak_positions ─────────────
    n_bases = len(base_calls)

    # Trim or pad quality scores to match base calls
    if len(quality_scores) > n_bases:
        quality_scores = quality_scores[:n_bases]
    elif len(quality_scores) < n_bases:
        quality_scores.extend([0] * (n_bases - len(quality_scores)))

    # Filter peak positions to valid range
    peak_positions = [p for p in peak_positions if 0 <= p < max_len]

    # ── Extract peak heights at peak positions ────────────────────────────────
    peak_heights: dict[str, list[int]] = {"A": [], "T": [], "C": [], "G": []}
    for pos in peak_positions:
        for channel in ["A", "T", "C", "G"]:
            trace = raw_traces[channel]
            if pos < len(trace):
                peak_heights[channel].append(int(trace[pos]))
            else:
                peak_heights[channel].append(0)

    logger.debug(
        "Peak heights extracted: %d peaks × 4 channels",
        len(peak_positions)
    )

    # ── Apply normalization pipeline ──────────────────────────────────────────
    logger.info("Applying normalization pipeline...")
    normalized_traces = normalize_traces(raw_traces)

    # ── Identify readable region ──────────────────────────────────────────────
    if quality_scores:
        readable_start, readable_end = identify_readable_region(
            quality_scores,
            min_quality=DEFAULT_MIN_QUALITY,
            window=DEFAULT_WINDOW,
        )
    else:
        readable_start, readable_end = 0, n_bases

    # ── Compute mean quality ──────────────────────────────────────────────────
    mean_quality = float(np.mean(quality_scores)) if quality_scores else 0.0

    # ── Build TraceData object ────────────────────────────────────────────────
    trace_data = TraceData(
        trace_A=raw_traces["A"],
        trace_T=raw_traces["T"],
        trace_C=raw_traces["C"],
        trace_G=raw_traces["G"],
        trace_A_norm=normalized_traces["A"],
        trace_T_norm=normalized_traces["T"],
        trace_C_norm=normalized_traces["C"],
        trace_G_norm=normalized_traces["G"],
        peak_positions=peak_positions,
        peak_heights=peak_heights,
        base_calls=base_calls,
        quality_scores=quality_scores,
        readable_region_start=readable_start,
        readable_region_end=readable_end,
    )

    # ── Compute QC metrics ────────────────────────────────────────────────────
    qc_flags, qc_pass = compute_qc_metrics(trace_data, quality_scores)

    # ── Build and return ChromatogramData ─────────────────────────────────────
    chrom_data = ChromatogramData(
        file_name=file_name,
        file_hash=file_hash,
        trace=trace_data,
        sequence_length=n_bases,
        mean_quality=mean_quality,
        qc_flags=qc_flags,
        qc_pass=qc_pass,
    )

    logger.info(
        "AB1 parsing complete: %s | %d bases | Q%.1f | QC %s | flags: %s",
        file_name,
        n_bases,
        mean_quality,
        "PASS" if qc_pass else "FAIL",
        qc_flags if qc_flags else "none",
    )

    return chrom_data


def generate_synthetic_ab1_data(
    n_bases: int = 600,
    n_samples_per_base: int = 10,
    mean_quality: float = 35.0,
    quality_std: float = 5.0,
    snv_positions: Optional[list[int]] = None,
    seed: int = 42,
) -> ChromatogramData:
    """
    Generate synthetic chromatogram data for testing (no real AB1 file needed).

    Creates a realistic synthetic ChromatogramData object with Gaussian-shaped
    peaks, realistic quality scores, and optionally injected SNV positions.
    Useful for unit testing, integration testing, and example generation.

    Parameters
    ----------
    n_bases : int, optional
        Number of base calls to generate. Default is 600.
    n_samples_per_base : int, optional
        Number of trace samples per base call. Default is 10.
        Total trace length = n_bases * n_samples_per_base.
    mean_quality : float, optional
        Mean Phred quality score for generated bases. Default is 35.0.
    quality_std : float, optional
        Standard deviation of quality scores. Default is 5.0.
    snv_positions : Optional[list[int]], optional
        List of 0-based positions where SNVs should be injected.
        At these positions, a secondary peak is added to simulate
        heterozygous variants. Default is None (no SNVs).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    ChromatogramData
        Synthetic chromatogram data with:
        - Gaussian-shaped peaks for each base call
        - Realistic quality score distribution
        - Injected secondary peaks at snv_positions
        - Proper normalization applied
        - QC metrics computed

    Example
    -------
    >>> # Generate clean synthetic data
    >>> chrom = generate_synthetic_ab1_data(n_bases=400)
    >>> print(f"Sequence length: {chrom.sequence_length}")
    400
    >>> print(f"QC pass: {chrom.qc_pass}")
    True
    >>>
    >>> # Generate data with known SNV positions
    >>> chrom_snv = generate_synthetic_ab1_data(
    ...     n_bases=400,
    ...     snv_positions=[100, 200, 300],
    ...     seed=123,
    ... )
    """
    rng = np.random.default_rng(seed)

    # ── Generate base sequence ────────────────────────────────────────────────
    nucleotides = ["A", "T", "C", "G"]
    base_calls = "".join(rng.choice(nucleotides, size=n_bases))

    # ── Generate quality scores ───────────────────────────────────────────────
    # Low quality at ends (typical for Sanger sequencing)
    quality_scores_raw = rng.normal(mean_quality, quality_std, n_bases)

    # Apply quality ramp: low at start and end, high in middle
    ramp_length = min(50, n_bases // 6)
    ramp = np.linspace(5, mean_quality, ramp_length)
    quality_scores_raw[:ramp_length] = ramp + rng.normal(0, 2, ramp_length)
    quality_scores_raw[-ramp_length:] = ramp[::-1] + rng.normal(0, 2, ramp_length)

    quality_scores = np.clip(quality_scores_raw, 0, 60).astype(int).tolist()

    # ── Generate trace data ───────────────────────────────────────────────────
    n_samples = n_bases * n_samples_per_base
    peak_positions = [i * n_samples_per_base + n_samples_per_base // 2
                      for i in range(n_bases)]

    # Initialize traces
    raw_traces: dict[str, np.ndarray] = {
        "A": np.zeros(n_samples),
        "T": np.zeros(n_samples),
        "C": np.zeros(n_samples),
        "G": np.zeros(n_samples),
    }

    # Generate Gaussian peaks for each base call
    sigma_peak = 1.5  # Peak width in samples
    for i, base in enumerate(base_calls):
        peak_center = peak_positions[i]
        # Peak height proportional to quality score
        peak_height = 200.0 + quality_scores[i] * 10.0 + rng.normal(0, 20)
        peak_height = max(peak_height, 50.0)

        # Add Gaussian peak to the called base channel
        for j in range(
            max(0, peak_center - 15),
            min(n_samples, peak_center + 16)
        ):
            gaussian_val = peak_height * math.exp(
                -0.5 * ((j - peak_center) / sigma_peak) ** 2
            )
            raw_traces[base][j] += gaussian_val

        # Add background noise to all channels
        for ch in nucleotides:
            noise_level = peak_height * 0.03
            for j in range(
                max(0, peak_center - 15),
                min(n_samples, peak_center + 16)
            ):
                raw_traces[ch][j] += abs(rng.normal(0, noise_level))

        # Inject secondary peak for SNV positions
        if snv_positions and i in snv_positions:
            # Choose a different base for the secondary peak
            other_bases = [b for b in nucleotides if b != base]
            secondary_base = rng.choice(other_bases)
            secondary_height = peak_height * 0.45  # ~45% secondary peak

            for j in range(
                max(0, peak_center - 15),
                min(n_samples, peak_center + 16)
            ):
                gaussian_val = secondary_height * math.exp(
                    -0.5 * ((j - peak_center) / sigma_peak) ** 2
                )
                raw_traces[secondary_base][j] += gaussian_val

    # Add baseline drift (common in real sequencing)
    baseline_drift = np.linspace(50, 150, n_samples) + rng.normal(0, 10, n_samples)
    for ch in nucleotides:
        raw_traces[ch] += np.abs(baseline_drift)

    # Add random noise
    for ch in nucleotides:
        raw_traces[ch] += np.abs(rng.normal(0, 5, n_samples))

    # Convert to lists
    raw_traces_lists = {ch: raw_traces[ch].tolist() for ch in nucleotides}

    # ── Extract peak heights ──────────────────────────────────────────────────
    peak_heights: dict[str, list[int]] = {"A": [], "T": [], "C": [], "G": []}
    for pos in peak_positions:
        for ch in nucleotides:
            if pos < len(raw_traces_lists[ch]):
                peak_heights[ch].append(int(raw_traces_lists[ch][pos]))
            else:
                peak_heights[ch].append(0)

    # ── Apply normalization ───────────────────────────────────────────────────
    normalized_traces = normalize_traces(raw_traces_lists)

    # ── Identify readable region ──────────────────────────────────────────────
    readable_start, readable_end = identify_readable_region(
        quality_scores,
        min_quality=DEFAULT_MIN_QUALITY,
        window=DEFAULT_WINDOW,
    )

    # ── Build TraceData ───────────────────────────────────────────────────────
    trace_data = TraceData(
        trace_A=raw_traces_lists["A"],
        trace_T=raw_traces_lists["T"],
        trace_C=raw_traces_lists["C"],
        trace_G=raw_traces_lists["G"],
        trace_A_norm=normalized_traces["A"],
        trace_T_norm=normalized_traces["T"],
        trace_C_norm=normalized_traces["C"],
        trace_G_norm=normalized_traces["G"],
        peak_positions=peak_positions,
        peak_heights=peak_heights,
        base_calls=base_calls,
        quality_scores=quality_scores,
        readable_region_start=readable_start,
        readable_region_end=readable_end,
    )

    # ── Compute QC metrics ────────────────────────────────────────────────────
    qc_flags, qc_pass = compute_qc_metrics(trace_data, quality_scores)

    # ── Compute mean quality ──────────────────────────────────────────────────
    computed_mean_quality = float(np.mean(quality_scores))

    # ── Generate synthetic file hash ──────────────────────────────────────────
    hash_input = f"synthetic_{seed}_{n_bases}_{n_samples_per_base}".encode()
    synthetic_hash = hashlib.sha256(hash_input).hexdigest()

    # ── Build ChromatogramData ────────────────────────────────────────────────
    chrom_data = ChromatogramData(
        file_name=f"synthetic_seed{seed}_n{n_bases}.ab1",
        file_hash=synthetic_hash,
        trace=trace_data,
        sequence_length=n_bases,
        mean_quality=computed_mean_quality,
        qc_flags=qc_flags,
        qc_pass=qc_pass,
    )

    logger.info(
        "Synthetic AB1 data generated: %d bases, %d samples, Q%.1f, QC %s",
        n_bases,
        n_samples,
        computed_mean_quality,
        "PASS" if qc_pass else "FAIL",
    )

    return chrom_data
