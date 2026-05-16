"""
SNV detection and mixed peak quantification for Sanger sequencing data.

Algorithm:
1. For each base position in the readable region:
   a. Extract peak heights for all 4 channels at that position
   b. Compute base proportions (normalize to sum=1)
   c. Detect secondary peaks (non-primary channel > threshold)
   d. Compare to reference sequence from alignment
   e. Call SNV if primary base differs from reference
   f. Flag heterozygosity if secondary peak > 25% of primary
2. Apply confidence scoring
3. Return list of SNVCall objects

The SNV calling pipeline:
    ChromatogramData + AlignmentResult → call_snvs() → list[SNVCall]

Confidence scoring formula:
    quality_component    = min(quality_score / 40, 1.0) * 0.35
    alignment_component  = alignment_identity * 0.25
    peak_clarity         = (primary_proportion - secondary_proportion) * 0.25
    neighbor_consistency = mean(neighboring_quality) / 40 * 0.15
    confidence_score     = sum of all components

Labels:
    high:   confidence_score >= CONFIDENCE_HIGH_THRESHOLD (default 0.8)
    medium: confidence_score >= CONFIDENCE_MEDIUM_THRESHOLD (default 0.5)
    low:    confidence_score < CONFIDENCE_MEDIUM_THRESHOLD

Example
-------
>>> from tools.snv_caller import call_snvs, compute_base_proportions
>>> from tools.ab1_parser import generate_synthetic_ab1_data
>>> from backend.config.settings import get_settings
>>> chrom = generate_synthetic_ab1_data(n_bases=400, snv_positions=[100, 200])
>>> # (alignment would be obtained from aligner.py in production)
>>> settings = get_settings()
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.schemas.alignment import AlignmentResult
from backend.schemas.chromatogram import ChromatogramData
from backend.schemas.snv import BaseProportions, SNVCall

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_HETEROZYGOSITY_THRESHOLD = 0.25
NEIGHBORING_WINDOW = 5          # ±5 bases for neighbor quality
MIN_TOTAL_SIGNAL = 1.0          # Minimum total signal to avoid division by zero
BASES = ("A", "T", "C", "G")


# ── Public API ────────────────────────────────────────────────────────────────

def call_snvs(
    chromatogram: ChromatogramData,
    alignment: AlignmentResult,
    settings,
) -> list[SNVCall]:
    """
    Detect SNVs and mixed peaks from chromatogram data.

    Iterates over each position in the readable region of the chromatogram.
    For each position:
    1. Extracts peak heights for all 4 channels (A, T, C, G).
    2. Computes base proportions (normalized to sum=1).
    3. Identifies the primary base (highest proportion).
    4. Retrieves the reference base at that position from the alignment.
    5. If primary base != reference base: records an SNV call.
    6. If secondary peak fraction > 0.25: flags as heterozygous.
    7. Computes a confidence score from quality, alignment identity, and peak clarity.

    Parameters
    ----------
    chromatogram : ChromatogramData
        Parsed chromatogram with normalized traces, peak heights, base calls,
        quality scores, and readable region boundaries.
    alignment : AlignmentResult
        Genome alignment result providing the reference sequence and CIGAR
        string for coordinate mapping.
    settings : SangerSettings
        Application settings providing:
        - CONFIDENCE_HIGH_THRESHOLD: minimum score for "high" label
        - CONFIDENCE_MEDIUM_THRESHOLD: minimum score for "medium" label

    Returns
    -------
    list[SNVCall]
        List of detected SNV calls, sorted by position_in_read.
        Each SNVCall includes position, alleles, proportions, heterozygosity
        status, secondary peak fraction, and confidence score.

    Raises
    ------
    ValueError
        If chromatogram or alignment data is inconsistent.

    Notes
    -----
    Only positions where the primary base differs from the reference are
    returned as SNV calls. Positions matching the reference are not included.
    Positions with reference base "N" are skipped.

    Example
    -------
    >>> from tools.ab1_parser import generate_synthetic_ab1_data
    >>> from backend.config.settings import get_settings
    >>> chrom = generate_synthetic_ab1_data(n_bases=400, snv_positions=[100])
    >>> # Assume alignment is available
    >>> settings = get_settings()
    >>> snvs = call_snvs(chrom, alignment, settings)
    >>> print(f"Found {len(snvs)} SNVs")
    """
    trace = chromatogram.trace
    peak_heights = trace.peak_heights
    quality_scores = trace.quality_scores
    base_calls = trace.base_calls

    readable_start = trace.readable_region_start
    readable_end = trace.readable_region_end

    alignment_identity = alignment.identity
    reference_sequence = alignment.reference_sequence

    high_threshold = getattr(settings, "CONFIDENCE_HIGH_THRESHOLD", 0.8)
    medium_threshold = getattr(settings, "CONFIDENCE_MEDIUM_THRESHOLD", 0.5)

    snv_calls: list[SNVCall] = []

    logger.info(
        "Calling SNVs in readable region [%d, %d) (%d bases)",
        readable_start, readable_end, readable_end - readable_start
    )

    for read_pos in range(readable_start, readable_end):
        # ── Step 1: Get peak heights at this position ─────────────────────────
        # peak_heights is indexed by peak index (not read position directly)
        # peak_positions[i] is the trace sample index for base i
        # We use read_pos as the index into peak_heights arrays
        if read_pos >= len(peak_heights.get("A", [])):
            logger.debug(
                "Position %d exceeds peak_heights length, stopping", read_pos
            )
            break

        # ── Step 2: Compute base proportions ─────────────────────────────────
        try:
            proportions = compute_base_proportions(peak_heights, read_pos)
        except Exception as exc:
            logger.debug(
                "Skipping position %d: proportion computation failed: %s",
                read_pos, exc
            )
            continue

        # ── Step 3: Identify primary base ─────────────────────────────────────
        primary_base = proportions.dominant_base

        # ── Step 4: Get reference base ────────────────────────────────────────
        ref_base = get_reference_base(alignment, read_pos - readable_start)

        if ref_base == "N":
            logger.debug(
                "Skipping position %d: reference base is N", read_pos
            )
            continue

        # ── Step 5: Check for SNV ─────────────────────────────────────────────
        if primary_base == ref_base:
            continue  # No variant at this position

        # ── Step 6: Detect heterozygosity ─────────────────────────────────────
        is_heterozygous, secondary_fraction = detect_mixed_peaks(
            proportions, DEFAULT_HETEROZYGOSITY_THRESHOLD
        )

        # ── Step 7: Compute confidence score ─────────────────────────────────
        quality_score = quality_scores[read_pos] if read_pos < len(quality_scores) else 0

        # Gather neighboring quality scores (±NEIGHBORING_WINDOW bases)
        neighbor_start = max(0, read_pos - NEIGHBORING_WINDOW)
        neighbor_end = min(len(quality_scores), read_pos + NEIGHBORING_WINDOW + 1)
        neighboring_quality = [
            quality_scores[i]
            for i in range(neighbor_start, neighbor_end)
            if i != read_pos
        ]

        confidence_score, confidence_label = score_confidence(
            proportions=proportions,
            quality_score=quality_score,
            alignment_identity=alignment_identity,
            neighboring_quality=neighboring_quality,
            settings=settings,
        )

        # ── Compute genomic position (1-based VCF convention) ─────────────────
        # Map read position to genomic position using alignment start
        read_offset = read_pos - readable_start
        genomic_pos = _read_pos_to_genomic(alignment, read_offset)

        snv_call = SNVCall(
            position_in_read=read_pos,
            genomic_position=genomic_pos,
            reference_allele=ref_base,
            alternative_allele=primary_base,
            proportions=proportions,
            is_heterozygous=is_heterozygous,
            secondary_peak_fraction=secondary_fraction,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            clinvar=None,
        )
        snv_calls.append(snv_call)

        logger.debug(
            "SNV at read pos %d (genomic %d): %s>%s het=%s conf=%.3f (%s)",
            read_pos, genomic_pos, ref_base, primary_base,
            is_heterozygous, confidence_score, confidence_label
        )

    logger.info(
        "SNV calling complete: %d SNVs found (%d heterozygous)",
        len(snv_calls),
        sum(1 for s in snv_calls if s.is_heterozygous)
    )

    return sorted(snv_calls, key=lambda s: s.position_in_read)


def compute_base_proportions(
    peak_heights: dict,
    position: int,
) -> BaseProportions:
    """
    Compute per-base proportions at a given peak position.

    Extracts the peak height for each of the four nucleotide channels (A, T, C, G)
    at the given position index, then normalizes so that the proportions sum to 1.0.

    If the total signal is zero (all channels are zero), returns equal proportions
    (0.25 each) to avoid division by zero.

    Parameters
    ----------
    peak_heights : dict
        Dictionary with keys "A", "T", "C", "G" containing lists of peak heights.
        Each list has one entry per base call position.
        Example: {"A": [100, 50, 200], "T": [50, 300, 80], "C": [30, 20, 40], "G": [20, 30, 60]}
    position : int
        0-based index into the peak_heights arrays.

    Returns
    -------
    BaseProportions
        Proportions for each base (A, T, C, G), summing to 1.0.
        Values are in [0, 1].

    Raises
    ------
    IndexError
        If position is out of range for any channel.
    ValueError
        If peak_heights is missing required channels.

    Example
    -------
    >>> heights = {"A": [800, 100], "T": [100, 700], "C": [50, 100], "G": [50, 100]}
    >>> props = compute_base_proportions(heights, position=0)
    >>> abs(props.A - 0.8) < 0.01
    True
    >>> abs(props.A + props.T + props.C + props.G - 1.0) < 1e-6
    True
    """
    required = {"A", "T", "C", "G"}
    missing = required - set(peak_heights.keys())
    if missing:
        raise ValueError(
            f"peak_heights missing channels: {missing}"
        )

    # Extract heights at this position
    heights: dict[str, float] = {}
    for base in BASES:
        channel = peak_heights[base]
        if position >= len(channel):
            raise IndexError(
                f"position {position} out of range for channel '{base}' "
                f"(length {len(channel)})"
            )
        h = float(channel[position])
        heights[base] = max(0.0, h)  # Clamp negative values to 0

    total = sum(heights.values())

    if total < MIN_TOTAL_SIGNAL:
        # All-zero signal: return equal proportions
        logger.debug(
            "Position %d: total signal %.2f < %.2f, using equal proportions",
            position, total, MIN_TOTAL_SIGNAL
        )
        return BaseProportions(A=0.25, T=0.25, C=0.25, G=0.25)

    # Normalize to sum=1
    proportions = {base: heights[base] / total for base in BASES}

    return BaseProportions(
        A=proportions["A"],
        T=proportions["T"],
        C=proportions["C"],
        G=proportions["G"],
    )


def score_confidence(
    proportions: BaseProportions,
    quality_score: int,
    alignment_identity: float,
    neighboring_quality: list[int],
    settings,
) -> tuple[float, str]:
    """
    Compute confidence score for an SNV call.

    The confidence score is a weighted sum of four components:

    1. **Quality component** (weight 0.35):
       ``min(quality_score / 40, 1.0) * 0.35``
       Phred Q40 = perfect quality. Scores above Q40 are capped at 1.0.

    2. **Alignment component** (weight 0.25):
       ``alignment_identity * 0.25``
       Higher alignment identity → more reliable reference comparison.

    3. **Peak clarity** (weight 0.25):
       ``(primary_proportion - secondary_proportion) * 0.25``
       Larger gap between primary and secondary peaks → cleaner call.

    4. **Neighbor consistency** (weight 0.15):
       ``mean(neighboring_quality) / 40 * 0.15``
       High quality in neighboring bases → more reliable local context.

    Total: ``confidence_score = sum of all four components`` (range [0, 1]).

    Labels:
    - "high":   confidence_score >= CONFIDENCE_HIGH_THRESHOLD (default 0.8)
    - "medium": confidence_score >= CONFIDENCE_MEDIUM_THRESHOLD (default 0.5)
    - "low":    confidence_score < CONFIDENCE_MEDIUM_THRESHOLD

    Parameters
    ----------
    proportions : BaseProportions
        Base proportions at the SNV position.
    quality_score : int
        Phred quality score at this position (range [0, 60]).
    alignment_identity : float
        Overall alignment identity fraction [0, 1].
    neighboring_quality : list[int]
        Quality scores of ±5 neighboring bases (may be empty).
    settings : SangerSettings
        Application settings with CONFIDENCE_HIGH_THRESHOLD and
        CONFIDENCE_MEDIUM_THRESHOLD.

    Returns
    -------
    tuple[float, str]
        (confidence_score, confidence_label) where:
        - confidence_score: float in [0, 1]
        - confidence_label: "high", "medium", or "low"

    Example
    -------
    >>> from backend.config.settings import get_settings
    >>> settings = get_settings()
    >>> props = BaseProportions(A=0.85, T=0.05, C=0.05, G=0.05)
    >>> score, label = score_confidence(props, 38, 0.98, [35, 36, 37, 38, 36], settings)
    >>> print(f"Score: {score:.3f}, Label: {label}")
    Score: 0.872, Label: high
    """
    high_threshold = getattr(settings, "CONFIDENCE_HIGH_THRESHOLD", 0.8)
    medium_threshold = getattr(settings, "CONFIDENCE_MEDIUM_THRESHOLD", 0.5)

    # Component 1: Quality score (weight 0.35)
    quality_component = min(quality_score / 40.0, 1.0) * 0.35

    # Component 2: Alignment identity (weight 0.25)
    alignment_component = float(alignment_identity) * 0.25

    # Component 3: Peak clarity (weight 0.25)
    # Sort proportions descending to get primary and secondary
    prop_values = sorted(
        [proportions.A, proportions.T, proportions.C, proportions.G],
        reverse=True
    )
    primary_proportion = prop_values[0]
    secondary_proportion = prop_values[1] if len(prop_values) > 1 else 0.0
    peak_clarity = max(0.0, primary_proportion - secondary_proportion) * 0.25

    # Component 4: Neighbor consistency (weight 0.15)
    if neighboring_quality:
        mean_neighbor_quality = float(np.mean(neighboring_quality))
    else:
        mean_neighbor_quality = float(quality_score)  # Fallback to self
    neighbor_consistency = min(mean_neighbor_quality / 40.0, 1.0) * 0.15

    # Total confidence score
    confidence_score = (
        quality_component
        + alignment_component
        + peak_clarity
        + neighbor_consistency
    )
    # Clamp to [0, 1]
    confidence_score = max(0.0, min(1.0, confidence_score))

    # Assign label
    if confidence_score >= high_threshold:
        confidence_label = "high"
    elif confidence_score >= medium_threshold:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    logger.debug(
        "Confidence: q=%.3f aln=%.3f clarity=%.3f nbr=%.3f total=%.3f (%s)",
        quality_component, alignment_component, peak_clarity,
        neighbor_consistency, confidence_score, confidence_label
    )

    return confidence_score, confidence_label


def detect_mixed_peaks(
    proportions: BaseProportions,
    heterozygosity_threshold: float = DEFAULT_HETEROZYGOSITY_THRESHOLD,
) -> tuple[bool, float]:
    """
    Detect heterozygous/mixed peaks at a position.

    A position is considered heterozygous if the secondary peak height
    exceeds ``heterozygosity_threshold`` fraction of the primary peak height.

    The secondary peak fraction is defined as::

        secondary_peak_fraction = second_highest_proportion / highest_proportion

    For example, if A=0.60 and G=0.35, the secondary peak fraction is
    0.35/0.60 = 0.583, which exceeds the default threshold of 0.25,
    indicating heterozygosity.

    Parameters
    ----------
    proportions : BaseProportions
        Base proportions at the position of interest.
    heterozygosity_threshold : float, optional
        Minimum secondary/primary ratio to call heterozygosity.
        Default is 0.25 (25% of primary peak).

    Returns
    -------
    tuple[bool, float]
        (is_heterozygous, secondary_peak_fraction) where:
        - is_heterozygous: True if secondary_peak_fraction >= threshold
        - secondary_peak_fraction: ratio of second-highest to highest proportion

    Example
    -------
    >>> # Heterozygous: A=0.60, G=0.35
    >>> props = BaseProportions(A=0.60, T=0.02, C=0.03, G=0.35)
    >>> is_het, fraction = detect_mixed_peaks(props, threshold=0.25)
    >>> is_het
    True
    >>> round(fraction, 3)
    0.583
    >>>
    >>> # Homozygous: A=0.90, others low
    >>> props2 = BaseProportions(A=0.90, T=0.04, C=0.03, G=0.03)
    >>> is_het2, fraction2 = detect_mixed_peaks(props2)
    >>> is_het2
    False
    """
    prop_values = sorted(
        [proportions.A, proportions.T, proportions.C, proportions.G],
        reverse=True
    )

    primary = prop_values[0]
    secondary = prop_values[1] if len(prop_values) > 1 else 0.0

    if primary <= 0.0:
        return False, 0.0

    secondary_fraction = secondary / primary
    is_heterozygous = secondary_fraction >= heterozygosity_threshold

    return is_heterozygous, secondary_fraction


def get_reference_base(
    alignment: AlignmentResult,
    read_position: int,
) -> str:
    """
    Get the reference base at a given read position.

    Uses the CIGAR string to map the read position to the corresponding
    reference position, then looks up the reference sequence from
    ``alignment.reference_sequence``.

    CIGAR operations handled:
    - M (match/mismatch): advances both read and reference
    - = (sequence match): advances both read and reference
    - X (sequence mismatch): advances both read and reference
    - I (insertion): advances read only (no reference base)
    - D (deletion): advances reference only (no read base)
    - N (skip): advances reference only
    - S (soft clip): advances read only (not counted in read_position)
    - H (hard clip): advances neither

    Parameters
    ----------
    alignment : AlignmentResult
        Genome alignment result with cigar string and reference_sequence.
    read_position : int
        0-based position within the aligned read (after soft-clipping).
        This is the offset from the start of the readable region.

    Returns
    -------
    str
        Single uppercase nucleotide character at the reference position.
        Returns "N" if:
        - The position falls in an insertion (no reference base)
        - The position is beyond the reference sequence length
        - The CIGAR string cannot be parsed
        - The reference sequence is empty

    Example
    -------
    >>> from backend.schemas.alignment import AlignmentResult
    >>> alignment = AlignmentResult(
    ...     chromosome="chr7",
    ...     start=100,
    ...     end=310,
    ...     strand="+",
    ...     identity=0.99,
    ...     alignment_score=200,
    ...     cigar="100M5I100M",
    ...     genes=[],
    ...     method="minimap2",
    ...     reference_sequence="A" * 200,
    ... )
    >>> get_reference_base(alignment, 0)
    'A'
    >>> get_reference_base(alignment, 100)  # In insertion region
    'N'
    >>> get_reference_base(alignment, 105)  # After insertion
    'A'
    """
    cigar = alignment.cigar
    reference_sequence = alignment.reference_sequence

    if not reference_sequence:
        logger.debug(
            "Reference sequence is empty; returning N for position %d",
            read_position
        )
        return "N"

    if not cigar:
        # No CIGAR: direct lookup
        if read_position < len(reference_sequence):
            return reference_sequence[read_position].upper()
        return "N"

    # Walk the CIGAR string to map read_position → reference_position
    read_consumed = 0
    ref_consumed = 0

    for match in re.finditer(r"(\d+)([MIDNSHP=X])", cigar):
        length = int(match.group(1))
        op = match.group(2)

        if op in ("M", "=", "X"):
            # Both read and reference advance
            if read_consumed + length > read_position:
                # Target position is within this M/=/X block
                offset_in_block = read_position - read_consumed
                ref_pos = ref_consumed + offset_in_block
                if ref_pos < len(reference_sequence):
                    return reference_sequence[ref_pos].upper()
                return "N"
            read_consumed += length
            ref_consumed += length

        elif op == "I":
            # Insertion: read advances, reference does not
            if read_consumed + length > read_position:
                # Position is within an insertion — no reference base
                return "N"
            read_consumed += length

        elif op in ("D", "N"):
            # Deletion/skip: reference advances, read does not
            ref_consumed += length

        elif op == "S":
            # Soft clip: read advances (but these bases are not in read_position
            # since read_position is relative to the aligned portion)
            # Soft clips at the start shift the read_position reference
            # We treat soft clips as consuming read positions
            if read_consumed + length > read_position:
                return "N"  # Position is in soft-clipped region
            read_consumed += length

        elif op in ("H", "P"):
            pass  # Hard clip and padding don't consume read or reference

    # Position is beyond the CIGAR alignment
    logger.debug(
        "Position %d is beyond CIGAR alignment (read_consumed=%d)",
        read_position, read_consumed
    )
    return "N"


def _read_pos_to_genomic(alignment: AlignmentResult, read_offset: int) -> int:
    """
    Convert a read offset (from alignment start) to a 1-based genomic position.

    Uses the CIGAR string to account for insertions and deletions when
    mapping from read coordinates to reference coordinates.

    Parameters
    ----------
    alignment : AlignmentResult
        Genome alignment result with start position and CIGAR string.
    read_offset : int
        0-based offset from the start of the aligned read.

    Returns
    -------
    int
        1-based genomic position (VCF convention).
        Returns alignment.start + 1 if mapping fails.

    Notes
    -----
    The returned position follows VCF convention (1-based, inclusive).
    The alignment.start is 0-based (BED/BAM convention), so we add 1.
    """
    cigar = alignment.cigar
    ref_start = alignment.start  # 0-based

    if not cigar:
        return ref_start + read_offset + 1  # Convert to 1-based

    read_consumed = 0
    ref_consumed = 0

    for match in re.finditer(r"(\d+)([MIDNSHP=X])", cigar):
        length = int(match.group(1))
        op = match.group(2)

        if op in ("M", "=", "X"):
            if read_consumed + length > read_offset:
                offset_in_block = read_offset - read_consumed
                genomic_0based = ref_start + ref_consumed + offset_in_block
                return genomic_0based + 1  # Convert to 1-based
            read_consumed += length
            ref_consumed += length

        elif op == "I":
            if read_consumed + length > read_offset:
                # In insertion: use the reference position just before
                genomic_0based = ref_start + ref_consumed
                return genomic_0based + 1
            read_consumed += length

        elif op in ("D", "N"):
            ref_consumed += length

        elif op == "S":
            if read_consumed + length > read_offset:
                genomic_0based = ref_start + ref_consumed
                return genomic_0based + 1
            read_consumed += length

        elif op in ("H", "P"):
            pass

    # Fallback: linear mapping
    return ref_start + read_offset + 1
