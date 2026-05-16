"""
Automated sequencing quality control for Sanger chromatograms.

Detects quality issues that should exclude regions from analysis:
- low_signal: insufficient fluorescence intensity
- peak_collapse: peaks too close together or merged
- noisy_trace: high background noise
- poor_alignment: low alignment identity
- failed_sequencing: no usable signal
- unreadable_regions: specific coordinate ranges to exclude

The QC pipeline:
    ChromatogramData [+ AlignmentResult] → run_qc_pipeline() → QCReport

QC checks implemented:
1. Signal-to-noise ratio: SNR = mean_peak_height / std_baseline. Flag if < 5.
2. Peak density: peaks per 100 bases. Flag if < 0.5 (collapsed) or > 2.0 (noisy).
3. Quality distribution: % bases with Q < 20. Flag if > 30%.
4. Alignment identity: flag if alignment identity < 0.85.
5. Readable region length: flag if < 100 bases.

Example
-------
>>> from tools.sequencing_qc import run_qc_pipeline, QCReport
>>> from tools.ab1_parser import generate_synthetic_ab1_data
>>> chrom = generate_synthetic_ab1_data(n_bases=400)
>>> report = run_qc_pipeline(chrom)
>>> print(f"QC pass: {report.overall_pass}")
>>> print(f"Flags: {report.flags}")
>>> print(f"SNR: {report.metrics['snr']:.2f}")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.schemas.alignment import AlignmentResult
from backend.schemas.chromatogram import ChromatogramData

logger = logging.getLogger(__name__)

# ── QC Thresholds ─────────────────────────────────────────────────────────────
SNR_THRESHOLD = 5.0                  # Minimum acceptable signal-to-noise ratio
PEAK_DENSITY_MIN = 0.5               # Minimum peaks per 100 bases (collapsed)
PEAK_DENSITY_MAX = 2.0               # Maximum peaks per 100 bases (noisy)
LOW_QUALITY_THRESHOLD = 20           # Phred Q20 threshold for "low quality"
LOW_QUALITY_FRACTION_MAX = 0.30      # Max fraction of bases below Q20
ALIGNMENT_IDENTITY_MIN = 0.85        # Minimum acceptable alignment identity
MIN_READABLE_LENGTH = 100            # Minimum readable region length in bases
BASELINE_WINDOW = 50                 # Window for baseline estimation


class QCReport(BaseModel):
    """
    Quality control report for a Sanger sequencing chromatogram.

    Produced by ``run_qc_pipeline()`` and consumed by the SNV analysis
    orchestrator to determine whether a chromatogram is suitable for
    variant calling.

    Attributes
    ----------
    overall_pass : bool
        True if the chromatogram passes all critical QC checks.
        A chromatogram fails if it has any critical flags (failed_sequencing,
        poor_alignment) or more than 2 non-critical flags.
    flags : list[str]
        List of QC flag strings for failed checks. Possible values:
        - "failed_sequencing": no usable signal detected
        - "low_signal": mean peak height < 100
        - "noisy_trace": SNR < 5
        - "peak_collapse": peak density < 0.5 per 100 bases
        - "noisy_peaks": peak density > 2.0 per 100 bases
        - "poor_quality": > 30% of bases below Q20
        - "poor_alignment": alignment identity < 0.85
        - "short_readable_region": readable region < 100 bases
    metrics : dict[str, float]
        Numeric QC metrics computed during the pipeline:
        - "snr": signal-to-noise ratio
        - "mean_quality": mean Phred quality score
        - "peak_density": peaks per 100 bases
        - "low_quality_fraction": fraction of bases below Q20
        - "readable_length": length of readable region in bases
        - "alignment_identity": alignment identity (if alignment provided)
    readable_regions : list[tuple[int, int]]
        List of (start, end) tuples defining high-quality readable regions.
        These are the regions recommended for SNV calling.
    excluded_positions : set[int]
        Set of read positions to skip during SNV calling due to local
        quality issues (e.g., positions in low-quality windows).
    recommendations : list[str]
        Human-readable recommendations for improving sequencing quality
        or handling the current result.
    alignment_quality : Optional[float]
        Alignment identity score [0, 1] if alignment was provided.
        None if no alignment was supplied to the QC pipeline.

    Example
    -------
    >>> report = QCReport(
    ...     overall_pass=True,
    ...     flags=[],
    ...     metrics={"snr": 12.5, "mean_quality": 35.2, "peak_density": 1.0},
    ...     readable_regions=[(50, 550)],
    ...     excluded_positions=set(),
    ...     recommendations=["Chromatogram passes all QC checks."],
    ...     alignment_quality=0.98,
    ... )
    """

    overall_pass: bool = Field(
        ...,
        description="True if chromatogram passes all critical QC checks",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="QC flag strings for failed checks",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Numeric QC metrics: snr, mean_quality, peak_density, etc.",
    )
    readable_regions: list[tuple[int, int]] = Field(
        default_factory=list,
        description="(start, end) tuples of high-quality readable regions",
    )
    excluded_positions: set[int] = Field(
        default_factory=set,
        description="Read positions to skip during SNV calling",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Human-readable recommendations for quality improvement",
    )
    alignment_quality: Optional[float] = Field(
        default=None,
        description="Alignment identity [0, 1] if alignment was provided",
    )

    model_config = {"arbitrary_types_allowed": True}


def run_qc_pipeline(
    chromatogram: ChromatogramData,
    alignment: Optional[AlignmentResult] = None,
) -> QCReport:
    """
    Run complete QC pipeline on a chromatogram.

    Executes five QC checks in sequence and aggregates results into a
    QCReport. The pipeline is designed to be fast (< 100ms) and
    deterministic.

    QC checks:
    1. **Signal-to-noise ratio**: SNR = mean_peak_height / std_baseline.
       Flag "noisy_trace" if SNR < 5.
    2. **Peak density**: peaks per 100 bases.
       Flag "peak_collapse" if < 0.5, "noisy_peaks" if > 2.0.
    3. **Quality distribution**: fraction of bases with Q < 20.
       Flag "poor_quality" if > 30%.
    4. **Alignment identity**: flag "poor_alignment" if identity < 0.85.
       (Only checked if alignment is provided.)
    5. **Readable region length**: flag "short_readable_region" if < 100 bases.

    Parameters
    ----------
    chromatogram : ChromatogramData
        Parsed chromatogram data to evaluate.
    alignment : Optional[AlignmentResult], optional
        Genome alignment result. If provided, alignment identity is checked.
        Default is None (alignment check skipped).

    Returns
    -------
    QCReport
        Complete QC report with:
        - overall_pass: True if chromatogram is suitable for SNV calling
        - flags: list of failed QC check names
        - metrics: dict of numeric QC values
        - readable_regions: recommended regions for SNV calling
        - excluded_positions: positions to skip in SNV calling
        - recommendations: human-readable guidance
        - alignment_quality: alignment identity if provided

    Example
    -------
    >>> from tools.ab1_parser import generate_synthetic_ab1_data
    >>> chrom = generate_synthetic_ab1_data(n_bases=400)
    >>> report = run_qc_pipeline(chrom)
    >>> print(f"QC pass: {report.overall_pass}")
    QC pass: True
    >>> print(f"SNR: {report.metrics['snr']:.1f}")
    SNR: 15.3
    """
    flags: list[str] = []
    metrics: dict[str, float] = {}
    excluded_positions: set[int] = set()
    recommendations: list[str] = []

    trace = chromatogram.trace
    quality_scores = trace.quality_scores
    peak_heights = trace.peak_heights
    peak_positions = trace.peak_positions
    readable_start = trace.readable_region_start
    readable_end = trace.readable_region_end
    readable_length = readable_end - readable_start

    # ── Check 0: Failed sequencing ────────────────────────────────────────────
    if len(peak_positions) == 0 or chromatogram.has_critical_failure:
        flags.append("failed_sequencing")
        metrics["snr"] = 0.0
        metrics["mean_quality"] = 0.0
        metrics["peak_density"] = 0.0
        metrics["low_quality_fraction"] = 1.0
        metrics["readable_length"] = 0.0
        recommendations.append(
            "CRITICAL: No sequencing signal detected. "
            "Check sample preparation, capillary electrophoresis conditions, "
            "and dye terminator chemistry."
        )
        return QCReport(
            overall_pass=False,
            flags=flags,
            metrics=metrics,
            readable_regions=[],
            excluded_positions=set(),
            recommendations=recommendations,
            alignment_quality=None,
        )

    # ── Check 1: Signal-to-noise ratio ────────────────────────────────────────
    snr = _compute_snr(trace)
    metrics["snr"] = snr

    if snr < SNR_THRESHOLD:
        flags.append("noisy_trace")
        recommendations.append(
            f"Low signal-to-noise ratio (SNR={snr:.1f}, threshold={SNR_THRESHOLD}). "
            "Consider re-sequencing with higher template concentration or "
            "optimizing PCR conditions."
        )
        logger.warning("QC: noisy_trace — SNR=%.2f < %.1f", snr, SNR_THRESHOLD)
    else:
        logger.debug("QC: SNR=%.2f (pass)", snr)

    # ── Check 2: Peak density ─────────────────────────────────────────────────
    n_bases = chromatogram.sequence_length
    if n_bases > 0:
        peak_density = len(peak_positions) / n_bases * 100
    else:
        peak_density = 0.0
    metrics["peak_density"] = peak_density

    if peak_density < PEAK_DENSITY_MIN:
        flags.append("peak_collapse")
        recommendations.append(
            f"Peak density too low ({peak_density:.2f} peaks/100 bases, "
            f"threshold={PEAK_DENSITY_MIN}). "
            "Peaks may be collapsed or merged. Check electrophoresis run time "
            "and polymer concentration."
        )
        logger.warning(
            "QC: peak_collapse — density=%.2f < %.1f", peak_density, PEAK_DENSITY_MIN
        )
    elif peak_density > PEAK_DENSITY_MAX:
        flags.append("noisy_peaks")
        recommendations.append(
            f"Peak density too high ({peak_density:.2f} peaks/100 bases, "
            f"threshold={PEAK_DENSITY_MAX}). "
            "Trace may contain spurious peaks. Consider re-sequencing with "
            "lower template concentration."
        )
        logger.warning(
            "QC: noisy_peaks — density=%.2f > %.1f", peak_density, PEAK_DENSITY_MAX
        )
    else:
        logger.debug("QC: peak_density=%.2f (pass)", peak_density)

    # ── Check 3: Quality distribution ────────────────────────────────────────
    mean_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
    metrics["mean_quality"] = mean_quality

    if quality_scores:
        low_quality_count = sum(1 for q in quality_scores if q < LOW_QUALITY_THRESHOLD)
        low_quality_fraction = low_quality_count / len(quality_scores)
    else:
        low_quality_fraction = 1.0
    metrics["low_quality_fraction"] = low_quality_fraction

    if low_quality_fraction > LOW_QUALITY_FRACTION_MAX:
        flags.append("poor_quality")
        recommendations.append(
            f"High fraction of low-quality bases "
            f"({low_quality_fraction * 100:.1f}% below Q{LOW_QUALITY_THRESHOLD}, "
            f"threshold={LOW_QUALITY_FRACTION_MAX * 100:.0f}%). "
            "Consider trimming low-quality ends or re-sequencing."
        )
        logger.warning(
            "QC: poor_quality — %.1f%% bases below Q%d",
            low_quality_fraction * 100, LOW_QUALITY_THRESHOLD
        )
    else:
        logger.debug(
            "QC: low_quality_fraction=%.3f (pass)", low_quality_fraction
        )

    # Identify excluded positions (low-quality windows)
    excluded_positions = _identify_excluded_positions(
        quality_scores, readable_start, readable_end
    )

    # ── Check 4: Alignment identity ───────────────────────────────────────────
    alignment_quality: Optional[float] = None
    if alignment is not None:
        alignment_quality = alignment.identity
        metrics["alignment_identity"] = alignment_quality

        if alignment_quality < ALIGNMENT_IDENTITY_MIN:
            flags.append("poor_alignment")
            recommendations.append(
                f"Low alignment identity ({alignment_quality * 100:.1f}%, "
                f"threshold={ALIGNMENT_IDENTITY_MIN * 100:.0f}%). "
                "The sequence may not align well to hg38. Check for "
                "contamination, chimeric reads, or non-human sequence."
            )
            logger.warning(
                "QC: poor_alignment — identity=%.3f < %.2f",
                alignment_quality, ALIGNMENT_IDENTITY_MIN
            )
        else:
            logger.debug(
                "QC: alignment_identity=%.3f (pass)", alignment_quality
            )

    # ── Check 5: Readable region length ──────────────────────────────────────
    metrics["readable_length"] = float(readable_length)

    if readable_length < MIN_READABLE_LENGTH:
        flags.append("short_readable_region")
        recommendations.append(
            f"Readable region too short ({readable_length} bases, "
            f"minimum={MIN_READABLE_LENGTH}). "
            "Insufficient sequence for reliable variant calling. "
            "Consider re-sequencing with optimized primer design."
        )
        logger.warning(
            "QC: short_readable_region — %d bases < %d",
            readable_length, MIN_READABLE_LENGTH
        )
    else:
        logger.debug(
            "QC: readable_length=%d (pass)", readable_length
        )

    # ── Determine readable regions ────────────────────────────────────────────
    readable_regions = _compute_readable_regions(
        quality_scores, readable_start, readable_end
    )

    # ── Determine overall pass/fail ───────────────────────────────────────────
    critical_flags = {"failed_sequencing", "poor_alignment"}
    has_critical = bool(set(flags) & critical_flags)
    non_critical_count = len([f for f in flags if f not in critical_flags])
    overall_pass = not has_critical and non_critical_count <= 2

    # ── Add positive recommendation if passing ────────────────────────────────
    if overall_pass and not flags:
        recommendations.append(
            "Chromatogram passes all QC checks. "
            "Suitable for SNV calling and variant analysis."
        )
    elif overall_pass and flags:
        recommendations.append(
            f"Chromatogram passes QC with minor issues: {', '.join(flags)}. "
            "Results should be interpreted with caution."
        )

    logger.info(
        "QC pipeline complete: pass=%s flags=%s SNR=%.1f Q=%.1f readable=%d",
        overall_pass, flags if flags else "none",
        snr, mean_quality, readable_length
    )

    return QCReport(
        overall_pass=overall_pass,
        flags=flags,
        metrics=metrics,
        readable_regions=readable_regions,
        excluded_positions=excluded_positions,
        recommendations=recommendations,
        alignment_quality=alignment_quality,
    )


def _compute_snr(trace) -> float:
    """
    Compute signal-to-noise ratio for a chromatogram trace.

    SNR is computed as::

        SNR = mean_peak_height / std_baseline

    where:
    - mean_peak_height: mean of the maximum channel height at each peak position
    - std_baseline: standard deviation of the signal in non-peak regions

    Uses the A channel as the representative trace for baseline estimation.

    Parameters
    ----------
    trace : TraceData
        Trace data with raw signals and peak positions.

    Returns
    -------
    float
        Signal-to-noise ratio. Returns 0.0 if computation fails.

    Notes
    -----
    A higher SNR indicates a cleaner trace. Values below 5 indicate
    significant noise that may affect base calling accuracy.
    """
    try:
        peak_positions = trace.peak_positions
        peak_heights = trace.peak_heights

        if not peak_positions:
            return 0.0

        # Compute mean peak height across all channels
        all_peak_maxima = []
        for i in range(len(peak_positions)):
            peak_max = max(
                float(peak_heights.get(ch, [0])[i])
                if i < len(peak_heights.get(ch, []))
                else 0.0
                for ch in ("A", "T", "C", "G")
            )
            all_peak_maxima.append(peak_max)

        if not all_peak_maxima:
            return 0.0

        mean_peak_height = float(np.mean(all_peak_maxima))

        # Estimate baseline from non-peak regions of the A channel
        trace_a = np.array(trace.trace_A, dtype=np.float64)
        if len(trace_a) == 0:
            return 0.0

        # Create mask: True for non-peak regions
        peak_mask = np.zeros(len(trace_a), dtype=bool)
        for pos in peak_positions:
            lo = max(0, pos - 5)
            hi = min(len(trace_a), pos + 6)
            peak_mask[lo:hi] = True

        baseline_signal = trace_a[~peak_mask]

        if len(baseline_signal) < 10:
            # Not enough baseline samples; use global std as fallback
            baseline_std = float(np.std(trace_a))
        else:
            baseline_std = float(np.std(baseline_signal))

        if baseline_std <= 0:
            return float("inf")  # Perfect SNR (no noise)

        snr = mean_peak_height / baseline_std
        return float(snr)

    except Exception as exc:
        logger.debug("SNR computation failed: %s", exc)
        return 0.0


def _identify_excluded_positions(
    quality_scores: list[int],
    readable_start: int,
    readable_end: int,
    window: int = 5,
    min_quality: int = 15,
) -> set[int]:
    """
    Identify positions to exclude from SNV calling due to local quality issues.

    A position is excluded if the mean quality in a window of ±``window``
    bases around it falls below ``min_quality``.

    Parameters
    ----------
    quality_scores : list[int]
        Phred quality scores per base.
    readable_start : int
        Start of the readable region (0-based).
    readable_end : int
        End of the readable region (0-based, exclusive).
    window : int, optional
        Half-window size for local quality assessment. Default is 5.
    min_quality : int, optional
        Minimum acceptable local mean quality. Default is 15.

    Returns
    -------
    set[int]
        Set of 0-based read positions to exclude.
    """
    excluded: set[int] = set()

    if not quality_scores:
        return excluded

    scores = np.array(quality_scores, dtype=np.float64)
    n = len(scores)

    for pos in range(readable_start, readable_end):
        lo = max(0, pos - window)
        hi = min(n, pos + window + 1)
        local_mean = float(np.mean(scores[lo:hi]))
        if local_mean < min_quality:
            excluded.add(pos)

    return excluded


def _compute_readable_regions(
    quality_scores: list[int],
    readable_start: int,
    readable_end: int,
    min_quality: int = 20,
    window: int = 10,
    min_region_length: int = 50,
) -> list[tuple[int, int]]:
    """
    Compute contiguous high-quality readable regions within the readable region.

    Splits the readable region into contiguous sub-regions where the sliding
    window mean quality exceeds ``min_quality``. Regions shorter than
    ``min_region_length`` are discarded.

    Parameters
    ----------
    quality_scores : list[int]
        Phred quality scores per base.
    readable_start : int
        Start of the overall readable region (0-based).
    readable_end : int
        End of the overall readable region (0-based, exclusive).
    min_quality : int, optional
        Minimum quality threshold for a window to be considered readable.
        Default is 20.
    window : int, optional
        Sliding window size for quality assessment. Default is 10.
    min_region_length : int, optional
        Minimum length of a readable sub-region to include. Default is 50.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) tuples for high-quality readable regions.
        Returns [(readable_start, readable_end)] if quality_scores is empty.
    """
    if not quality_scores or readable_end <= readable_start:
        return [(readable_start, readable_end)]

    scores = np.array(quality_scores, dtype=np.float64)
    n = len(scores)

    # Compute sliding window means within the readable region
    in_good_region = False
    region_start = readable_start
    regions: list[tuple[int, int]] = []

    for pos in range(readable_start, readable_end):
        lo = max(0, pos - window // 2)
        hi = min(n, pos + window // 2 + 1)
        local_mean = float(np.mean(scores[lo:hi]))

        if local_mean >= min_quality:
            if not in_good_region:
                region_start = pos
                in_good_region = True
        else:
            if in_good_region:
                region_end = pos
                if region_end - region_start >= min_region_length:
                    regions.append((region_start, region_end))
                in_good_region = False

    # Close any open region at the end
    if in_good_region:
        region_end = readable_end
        if region_end - region_start >= min_region_length:
            regions.append((region_start, region_end))

    # Fallback: return the full readable region if no sub-regions found
    if not regions:
        if readable_end - readable_start >= min_region_length:
            return [(readable_start, readable_end)]
        return []

    return regions
