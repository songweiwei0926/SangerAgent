"""
Genome alignment service for SangerAgent.

Implements a two-tier alignment strategy:
1. Primary: local minimap2 alignment against hg38 index
2. Fallback: NCBI BLAST API when local index is unavailable

Both methods return a standardized AlignmentResult schema.

The alignment pipeline:
    readable_sequence (str) → align_sequence() → AlignmentResult

Minimap2 strategy:
    - Writes query to a temp FASTA file
    - Runs ``minimap2 -a -x sr {index_path} {query_fasta}`` via subprocess
    - Parses SAM output to extract chr, start, end, strand, CIGAR, NM tag
    - Computes identity = (alignment_length - NM) / alignment_length

BLAST fallback strategy:
    - Calls NCBI BLAST API via Biopython NCBIWWW.qblast()
    - Parses XML result to extract alignment coordinates
    - Converts BLAST accession to genomic coordinates via NCBI accession lookup
    - Applies exponential backoff for rate limiting (max 3 retries)
    - Caches results in memory (LRU cache, 100 entries)

Gene annotation:
    - Queries NCBI Gene API (E-utilities esearch) for overlapping genes
    - Returns list of HGNC gene symbols

Example
-------
>>> from tools.aligner import align_sequence, is_minimap2_available
>>> from backend.config.settings import get_settings
>>> settings = get_settings()
>>> # Check availability
>>> available = is_minimap2_available(settings.HG38_INDEX_PATH)
>>> print(f"minimap2 available: {available}")
>>> # Align a sequence
>>> result = align_sequence("ATCGATCGATCG" * 20, settings)
>>> print(f"Aligned to: {result.genomic_range}")
>>> print(f"Genes: {result.gene_list}")
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from typing import Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.schemas.alignment import AlignmentResult

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MINIMAP2_TIMEOUT_SECONDS = 120
BLAST_MAX_RETRIES = 3
BLAST_RETRY_BASE_DELAY = 2.0   # seconds; doubles each retry
BLAST_HITLIST_SIZE = 1
BLAST_DATABASE = "nt"
BLAST_PROGRAM = "blastn"
BLAST_ENTREZ_QUERY = "Homo sapiens[Organism]"
NCBI_GENE_DB = "gene"
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Chromosome name normalization: BLAST accession prefixes → UCSC chr names
# NC_000001.11 → chr1, NC_000023.11 → chrX, NC_000024.10 → chrY, etc.
_REFSEQ_TO_CHR: dict[str, str] = {
    "NC_000001": "chr1",  "NC_000002": "chr2",  "NC_000003": "chr3",
    "NC_000004": "chr4",  "NC_000005": "chr5",  "NC_000006": "chr6",
    "NC_000007": "chr7",  "NC_000008": "chr8",  "NC_000009": "chr9",
    "NC_000010": "chr10", "NC_000011": "chr11", "NC_000012": "chr12",
    "NC_000013": "chr13", "NC_000014": "chr14", "NC_000015": "chr15",
    "NC_000016": "chr16", "NC_000017": "chr17", "NC_000018": "chr18",
    "NC_000019": "chr19", "NC_000020": "chr20", "NC_000021": "chr21",
    "NC_000022": "chr22", "NC_000023": "chrX",  "NC_000024": "chrY",
    "NC_012920": "chrM",
}


# ── Public API ────────────────────────────────────────────────────────────────

def align_sequence(sequence: str, settings) -> AlignmentResult:
    """
    Align a DNA sequence to hg38 using minimap2 or BLAST fallback.

    Implements a two-tier strategy:
    1. If minimap2 binary is available and the hg38 index file exists,
       run local minimap2 alignment (fast, ~1-5 seconds).
    2. Otherwise, fall back to NCBI BLAST API (slower, ~30-120 seconds).

    After alignment, queries the NCBI Gene API to identify genes overlapping
    the aligned region.

    Parameters
    ----------
    sequence : str
        DNA sequence to align (basecalls from AB1 parser).
        Should be the readable region sequence from ChromatogramData.
    settings : SangerSettings
        Application settings providing:
        - HG38_INDEX_PATH: path to minimap2 hg38 index
        - NCBI_API_KEY: NCBI API key (empty = unauthenticated)
        - NCBI_EMAIL: email for NCBI Entrez API
        - BLAST_FALLBACK_ENABLED: whether BLAST fallback is allowed

    Returns
    -------
    AlignmentResult
        Standardized alignment result with chromosome, coordinates, strand,
        CIGAR string, identity, alignment score, genes, and reference sequence.

    Raises
    ------
    ValueError
        If the sequence is empty or contains invalid characters.
    RuntimeError
        If both minimap2 and BLAST fail to produce an alignment.

    Example
    -------
    >>> from backend.config.settings import get_settings
    >>> settings = get_settings()
    >>> result = align_sequence("ATCGATCG" * 25, settings)
    >>> print(result.genomic_range)
    'chr7:117548621-117548820'
    """
    if not sequence:
        raise ValueError("sequence must not be empty")

    sequence = sequence.upper().replace(" ", "").replace("\n", "")
    valid_bases = set("ATCGN")
    invalid = set(sequence) - valid_bases
    if invalid:
        raise ValueError(
            f"sequence contains invalid characters: {invalid}. "
            "Only A, T, C, G, N are allowed."
        )

    logger.info(
        "Aligning sequence of length %d to hg38", len(sequence)
    )

    # ── Tier 1: minimap2 ──────────────────────────────────────────────────────
    index_path = getattr(settings, "HG38_INDEX_PATH", "/data/hg38.mmi")
    if is_minimap2_available(index_path):
        logger.info("Using minimap2 for alignment (index: %s)", index_path)
        try:
            result = align_with_minimap2(sequence, index_path)
            # Annotate with overlapping genes
            api_key = getattr(settings, "NCBI_API_KEY", "")
            genes = get_overlapping_genes(
                result.chromosome, result.start, result.end, api_key
            )
            result = result.model_copy(
                update={"genes": list(dict.fromkeys(result.genes + genes))}
            )
            logger.info(
                "minimap2 alignment complete: %s identity=%.3f genes=%s",
                result.genomic_range, result.identity, result.gene_list
            )
            return result
        except Exception as exc:
            logger.warning(
                "minimap2 alignment failed: %s. Falling back to BLAST.", exc
            )

    # ── Tier 2: BLAST fallback ────────────────────────────────────────────────
    blast_enabled = getattr(settings, "BLAST_FALLBACK_ENABLED", True)
    if not blast_enabled:
        raise RuntimeError(
            "minimap2 alignment failed and BLAST fallback is disabled. "
            "Set BLAST_FALLBACK_ENABLED=true or provide a valid HG38_INDEX_PATH."
        )

    logger.info("Using NCBI BLAST API for alignment (fallback)")
    api_key = getattr(settings, "NCBI_API_KEY", "")
    email = getattr(settings, "NCBI_EMAIL", "sanger@example.com")

    result = align_with_blast(sequence, api_key, email)

    # Annotate with overlapping genes
    genes = get_overlapping_genes(
        result.chromosome, result.start, result.end, api_key
    )
    result = result.model_copy(
        update={"genes": list(dict.fromkeys(result.genes + genes))}
    )

    logger.info(
        "BLAST alignment complete: %s identity=%.3f genes=%s",
        result.genomic_range, result.identity, result.gene_list
    )
    return result


def align_with_minimap2(sequence: str, index_path: str) -> AlignmentResult:
    """
    Run minimap2 alignment using subprocess.

    Writes the query sequence to a temporary FASTA file, then runs::

        minimap2 -a -x sr {index_path} {query_fasta}

    Parses the SAM output to extract chromosome (RNAME), start position (POS),
    CIGAR string, strand (FLAG bit 16), and NM tag (number of mismatches).

    Computes sequence identity as::

        identity = (alignment_length - NM) / alignment_length

    where alignment_length is the number of reference bases consumed by the
    CIGAR string (M + D + N operations).

    Parameters
    ----------
    sequence : str
        DNA sequence to align (uppercase, ATCGN only).
    index_path : str
        Path to the minimap2 hg38 index file (.mmi format).

    Returns
    -------
    AlignmentResult
        Alignment result with chromosome, start, end, strand, identity,
        alignment_score, cigar, method="minimap2", and reference_sequence.
        The reference_sequence field is populated with the query sequence
        as a placeholder (real reference fetched separately if needed).

    Raises
    ------
    FileNotFoundError
        If the index file does not exist.
    RuntimeError
        If minimap2 subprocess fails or returns no alignment.
    ValueError
        If the SAM output cannot be parsed.

    Example
    -------
    >>> result = align_with_minimap2("ATCGATCG" * 25, "/data/hg38.mmi")
    >>> print(result.chromosome, result.start, result.end)
    chr7 117548620 117548820
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"minimap2 index not found: '{index_path}'"
        )

    # Write query to a temporary FASTA file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False, prefix="sanger_query_"
    ) as fasta_file:
        fasta_path = fasta_file.name
        fasta_file.write(f">query\n{sequence}\n")

    try:
        logger.debug(
            "Running minimap2: index=%s query=%s", index_path, fasta_path
        )
        cmd = ["minimap2", "-a", "-x", "sr", index_path, fasta_path]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MINIMAP2_TIMEOUT_SECONDS,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"minimap2 exited with code {proc.returncode}. "
                f"stderr: {proc.stderr[:500]}"
            )

        sam_output = proc.stdout
        logger.debug("minimap2 SAM output (%d chars)", len(sam_output))

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"minimap2 timed out after {MINIMAP2_TIMEOUT_SECONDS}s"
        ) from exc
    finally:
        # Always clean up the temp file
        try:
            os.unlink(fasta_path)
        except OSError:
            pass

    return _parse_sam_output(sam_output, sequence)


def _parse_sam_output(sam_output: str, query_sequence: str) -> AlignmentResult:
    """
    Parse SAM output from minimap2 and return an AlignmentResult.

    Extracts the first non-header, non-unmapped alignment record from the
    SAM output. SAM format columns (tab-separated):
    QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, [tags...]

    Parameters
    ----------
    sam_output : str
        Raw SAM format output from minimap2.
    query_sequence : str
        Original query sequence (used for reference_sequence placeholder).

    Returns
    -------
    AlignmentResult
        Parsed alignment result.

    Raises
    ------
    RuntimeError
        If no valid alignment record is found in the SAM output.
    ValueError
        If the SAM record cannot be parsed.
    """
    lines = sam_output.strip().split("\n")
    alignment_record = None

    for line in lines:
        if line.startswith("@"):
            continue  # Skip SAM header lines
        fields = line.split("\t")
        if len(fields) < 11:
            continue

        flag = int(fields[1])
        # Bit 4 (0x4) = unmapped; skip unmapped reads
        if flag & 0x4:
            logger.debug("Skipping unmapped SAM record (FLAG=%d)", flag)
            continue

        alignment_record = fields
        break

    if alignment_record is None:
        raise RuntimeError(
            "minimap2 produced no valid alignment. "
            "The sequence may not map to hg38 or the index may be incomplete."
        )

    # ── Parse SAM fields ──────────────────────────────────────────────────────
    flag = int(alignment_record[1])
    rname = alignment_record[2]   # Reference chromosome name
    pos = int(alignment_record[3]) - 1  # Convert 1-based SAM to 0-based
    cigar = alignment_record[5]
    optional_tags = alignment_record[11:]

    # Normalize chromosome name (e.g., "1" → "chr1", "chrM" stays "chrM")
    chromosome = _normalize_chromosome_name(rname)

    # Strand: FLAG bit 16 (0x10) = reverse strand
    strand = "-" if (flag & 0x10) else "+"

    # Parse CIGAR to compute alignment length and end position
    alignment_length, ref_consumed = _parse_cigar(cigar)

    end = pos + ref_consumed

    # Extract NM tag (number of mismatches/edits) and AS tag (alignment score)
    nm = 0
    as_score = 0
    for tag in optional_tags:
        if tag.startswith("NM:i:"):
            nm = int(tag[5:])
        elif tag.startswith("AS:i:"):
            as_score = int(tag[5:])

    # Compute identity
    if alignment_length > 0:
        identity = max(0.0, min(1.0, (alignment_length - nm) / alignment_length))
    else:
        identity = 0.0

    # Use alignment_length as score if AS tag not present
    if as_score == 0:
        as_score = alignment_length

    logger.debug(
        "SAM parsed: %s:%d-%d %s CIGAR=%s NM=%d identity=%.3f",
        chromosome, pos, end, strand, cigar, nm, identity
    )

    return AlignmentResult(
        chromosome=chromosome,
        start=pos,
        end=end,
        strand=strand,
        identity=identity,
        alignment_score=as_score,
        cigar=cigar,
        genes=[],
        method="minimap2",
        reference_sequence=query_sequence,
    )


def _normalize_chromosome_name(rname: str) -> str:
    """
    Normalize a chromosome name to UCSC format (chr1, chr2, ..., chrX, chrY, chrM).

    Handles:
    - Already-prefixed names: "chr1" → "chr1"
    - Bare numbers: "1" → "chr1"
    - RefSeq accessions: "NC_000001.11" → "chr1"
    - Mitochondrial: "MT" → "chrM", "chrMT" → "chrM"

    Parameters
    ----------
    rname : str
        Raw chromosome name from SAM RNAME field.

    Returns
    -------
    str
        UCSC-format chromosome name.

    Raises
    ------
    ValueError
        If the chromosome name cannot be normalized to a valid UCSC name.

    Example
    -------
    >>> _normalize_chromosome_name("7")
    'chr7'
    >>> _normalize_chromosome_name("NC_000007.14")
    'chr7'
    >>> _normalize_chromosome_name("chrX")
    'chrX'
    """
    if not rname or rname == "*":
        raise ValueError(f"Invalid chromosome name: '{rname}'")

    # Already in UCSC format
    if re.match(r"^chr[0-9XYM]([0-9]|_[A-Za-z0-9]+)?$", rname):
        # Handle chrMT → chrM
        if rname == "chrMT":
            return "chrM"
        return rname

    # RefSeq accession (NC_XXXXXX.XX)
    accession_base = rname.split(".")[0]
    if accession_base in _REFSEQ_TO_CHR:
        return _REFSEQ_TO_CHR[accession_base]

    # Bare number or letter
    if re.match(r"^[0-9]{1,2}$", rname):
        return f"chr{rname}"
    if rname in ("X", "Y"):
        return f"chr{rname}"
    if rname in ("M", "MT", "chrMT"):
        return "chrM"

    # Try adding chr prefix
    candidate = f"chr{rname}"
    if re.match(r"^chr[0-9XYM]([0-9]|_[A-Za-z0-9]+)?$", candidate):
        return candidate

    raise ValueError(
        f"Cannot normalize chromosome name '{rname}' to UCSC format. "
        "Expected format: chr1-chr22, chrX, chrY, chrM."
    )


def _parse_cigar(cigar: str) -> tuple[int, int]:
    """
    Parse a CIGAR string and return (alignment_length, ref_consumed).

    CIGAR operations:
    - M (match/mismatch): consumes both query and reference
    - I (insertion): consumes query only
    - D (deletion): consumes reference only
    - N (skip): consumes reference only
    - S (soft clip): consumes query only (not counted in alignment_length)
    - H (hard clip): consumes neither
    - P (padding): consumes neither
    - = (sequence match): consumes both
    - X (sequence mismatch): consumes both

    alignment_length counts M, I, =, X operations (query-consuming aligned bases).
    ref_consumed counts M, D, N, =, X operations (reference-consuming operations).

    Parameters
    ----------
    cigar : str
        CIGAR string (e.g., "150M2I48M", "100M5D50M").

    Returns
    -------
    tuple[int, int]
        (alignment_length, ref_consumed) where:
        - alignment_length: total aligned bases (for identity calculation)
        - ref_consumed: reference bases consumed (for end position calculation)

    Example
    -------
    >>> _parse_cigar("150M2I48M")
    (200, 198)
    >>> _parse_cigar("100M5D50M")
    (150, 155)
    """
    alignment_length = 0
    ref_consumed = 0

    for match in re.finditer(r"(\d+)([MIDNSHP=X])", cigar):
        length = int(match.group(1))
        op = match.group(2)

        if op in ("M", "=", "X"):
            alignment_length += length
            ref_consumed += length
        elif op == "I":
            alignment_length += length
            # Insertions do not consume reference
        elif op in ("D", "N"):
            ref_consumed += length
            # Deletions/skips do not consume query
        elif op in ("S", "H", "P"):
            pass  # Soft/hard clips and padding don't affect alignment length

    return alignment_length, ref_consumed


@lru_cache(maxsize=100)
def _blast_cached(sequence: str, api_key: str, email: str) -> AlignmentResult:
    """
    LRU-cached wrapper for BLAST alignment.

    Caches up to 100 unique (sequence, api_key, email) combinations in memory.
    This avoids redundant BLAST API calls for the same sequence within a
    single process lifetime.

    Parameters
    ----------
    sequence : str
        DNA sequence to align.
    api_key : str
        NCBI API key.
    email : str
        NCBI contact email.

    Returns
    -------
    AlignmentResult
        BLAST alignment result.

    Notes
    -----
    This function is decorated with ``@lru_cache(maxsize=100)``. The cache
    is keyed on all three parameters. To clear the cache (e.g., in tests),
    call ``_blast_cached.cache_clear()``.
    """
    return _blast_uncached(sequence, api_key, email)


def align_with_blast(sequence: str, api_key: str, email: str) -> AlignmentResult:
    """
    Align sequence using NCBI BLAST API (blastn against nt database, human only).

    Uses Biopython NCBIWWW.qblast() with:
    - program="blastn"
    - database="nt"
    - entrez_query="Homo sapiens[Organism]"
    - hitlist_size=1

    Parses the XML result to extract alignment coordinates. Converts BLAST
    accession numbers to UCSC chromosome names via the RefSeq accession map.

    Applies exponential backoff for rate limiting (max 3 retries):
    - Retry 1: wait 2 seconds
    - Retry 2: wait 4 seconds
    - Retry 3: wait 8 seconds

    Results are cached in memory (LRU cache, 100 entries) to avoid redundant
    API calls for the same sequence.

    Parameters
    ----------
    sequence : str
        DNA sequence to align (uppercase, ATCGN only).
    api_key : str
        NCBI Entrez API key. Empty string = unauthenticated (3 req/s limit).
    email : str
        Email address for NCBI Entrez API contact.

    Returns
    -------
    AlignmentResult
        Alignment result with chromosome, start, end, strand, identity,
        alignment_score, cigar, method="blast", and reference_sequence.

    Raises
    ------
    ImportError
        If Biopython is not installed.
    RuntimeError
        If BLAST fails after all retries or returns no hits.
    ValueError
        If the BLAST XML result cannot be parsed.

    Example
    -------
    >>> result = align_with_blast("ATCGATCG" * 25, api_key="", email="test@example.com")
    >>> print(result.chromosome, result.start, result.end)
    chr7 117548620 117548820
    """
    return _blast_cached(sequence, api_key, email)


def _blast_uncached(sequence: str, api_key: str, email: str) -> AlignmentResult:
    """
    Internal BLAST alignment implementation (not cached).

    Performs the actual NCBI BLAST API call with exponential backoff retry
    logic. Called by the LRU-cached wrapper ``_blast_cached``.

    Parameters
    ----------
    sequence : str
        DNA sequence to align.
    api_key : str
        NCBI API key.
    email : str
        NCBI contact email.

    Returns
    -------
    AlignmentResult
        BLAST alignment result.

    Raises
    ------
    ImportError
        If Biopython is not installed.
    RuntimeError
        If BLAST fails after all retries.
    """
    try:
        from Bio.Blast import NCBIWWW, NCBIXML
        from Bio import Entrez
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for BLAST alignment. "
            "Install with: pip install biopython"
        ) from exc

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    last_exception: Optional[Exception] = None

    for attempt in range(BLAST_MAX_RETRIES):
        try:
            logger.info(
                "BLAST API call (attempt %d/%d): sequence length=%d",
                attempt + 1, BLAST_MAX_RETRIES, len(sequence)
            )

            result_handle = NCBIWWW.qblast(
                program=BLAST_PROGRAM,
                database=BLAST_DATABASE,
                sequence=sequence,
                entrez_query=BLAST_ENTREZ_QUERY,
                hitlist_size=BLAST_HITLIST_SIZE,
                format_type="XML",
            )

            blast_records = list(NCBIXML.parse(result_handle))

            if not blast_records:
                raise RuntimeError("BLAST returned no records")

            blast_record = blast_records[0]

            if not blast_record.alignments:
                raise RuntimeError(
                    "BLAST returned no alignments for the query sequence. "
                    "The sequence may not match any human genomic region."
                )

            return _parse_blast_record(blast_record, sequence)

        except RuntimeError:
            raise  # Don't retry on "no alignments" — it's a real result
        except Exception as exc:
            last_exception = exc
            if attempt < BLAST_MAX_RETRIES - 1:
                delay = BLAST_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "BLAST attempt %d failed: %s. Retrying in %.1fs...",
                    attempt + 1, exc, delay
                )
                time.sleep(delay)
            else:
                logger.error(
                    "BLAST failed after %d attempts: %s",
                    BLAST_MAX_RETRIES, exc
                )

    raise RuntimeError(
        f"BLAST alignment failed after {BLAST_MAX_RETRIES} attempts. "
        f"Last error: {last_exception}"
    )


def _parse_blast_record(blast_record, query_sequence: str) -> AlignmentResult:
    """
    Parse a Biopython BLAST record into an AlignmentResult.

    Extracts the best HSP (High-Scoring Pair) from the first alignment hit.
    Converts BLAST coordinates (1-based, inclusive) to BED/BAM coordinates
    (0-based, half-open).

    Parameters
    ----------
    blast_record : Bio.Blast.Record.Blast
        Parsed BLAST record from NCBIXML.parse().
    query_sequence : str
        Original query sequence.

    Returns
    -------
    AlignmentResult
        Parsed alignment result.

    Raises
    ------
    ValueError
        If the BLAST record cannot be parsed or chromosome name is invalid.
    """
    alignment = blast_record.alignments[0]
    hsp = alignment.hsps[0]  # Best HSP

    # Extract accession from title (e.g., "NC_000007.14 Homo sapiens chromosome 7...")
    title = alignment.title
    accession = alignment.accession

    # Try to map accession to chromosome
    chromosome = _accession_to_chromosome(accession, title)

    # BLAST coordinates are 1-based inclusive; convert to 0-based half-open
    # hsp.sbjct_start and hsp.sbjct_end are 1-based
    if hsp.sbjct_start <= hsp.sbjct_end:
        start = hsp.sbjct_start - 1  # Convert to 0-based
        end = hsp.sbjct_end          # Already exclusive in 0-based
        strand = "+"
    else:
        # Reverse strand: sbjct_start > sbjct_end in BLAST
        start = hsp.sbjct_end - 1
        end = hsp.sbjct_start
        strand = "-"

    # Compute identity
    alignment_length = hsp.align_length
    if alignment_length > 0:
        identity = hsp.identities / alignment_length
    else:
        identity = 0.0

    # Build a simple CIGAR string from BLAST alignment
    cigar = _blast_alignment_to_cigar(hsp.query, hsp.sbjct, hsp.match)

    # Reference sequence from BLAST subject
    reference_sequence = hsp.sbjct.replace("-", "").upper()
    if not reference_sequence:
        reference_sequence = query_sequence

    logger.debug(
        "BLAST parsed: %s:%d-%d %s identity=%.3f score=%d",
        chromosome, start, end, strand, identity, hsp.score
    )

    return AlignmentResult(
        chromosome=chromosome,
        start=start,
        end=end,
        strand=strand,
        identity=identity,
        alignment_score=int(hsp.score),
        cigar=cigar,
        genes=[],
        method="blast",
        reference_sequence=reference_sequence,
    )


def _accession_to_chromosome(accession: str, title: str) -> str:
    """
    Convert a BLAST accession number to a UCSC chromosome name.

    Tries the RefSeq accession map first, then parses the title string
    for chromosome number information.

    Parameters
    ----------
    accession : str
        BLAST accession number (e.g., "NC_000007.14").
    title : str
        BLAST alignment title (e.g., "NC_000007.14 Homo sapiens chromosome 7...").

    Returns
    -------
    str
        UCSC chromosome name (e.g., "chr7").

    Raises
    ------
    ValueError
        If the chromosome cannot be determined from accession or title.

    Example
    -------
    >>> _accession_to_chromosome("NC_000007.14", "NC_000007.14 Homo sapiens chromosome 7")
    'chr7'
    """
    # Try RefSeq accession map
    accession_base = accession.split(".")[0]
    if accession_base in _REFSEQ_TO_CHR:
        return _REFSEQ_TO_CHR[accession_base]

    # Try to extract chromosome number from title
    # e.g., "chromosome 7", "chromosome X", "chromosome Y", "mitochondrion"
    chr_match = re.search(
        r"chromosome\s+([0-9]{1,2}|X|Y|MT?)\b",
        title,
        re.IGNORECASE
    )
    if chr_match:
        chr_name = chr_match.group(1).upper()
        if chr_name == "MT":
            return "chrM"
        return f"chr{chr_name}"

    mito_match = re.search(r"mitochondri", title, re.IGNORECASE)
    if mito_match:
        return "chrM"

    # Last resort: try to normalize the accession directly
    try:
        return _normalize_chromosome_name(accession)
    except ValueError:
        raise ValueError(
            f"Cannot determine chromosome from BLAST accession '{accession}' "
            f"and title '{title[:100]}'"
        )


def _blast_alignment_to_cigar(query: str, subject: str, match: str) -> str:
    """
    Convert a BLAST pairwise alignment to a CIGAR string.

    BLAST alignment strings use:
    - '-' in query = deletion from reference (D in CIGAR)
    - '-' in subject = insertion relative to reference (I in CIGAR)
    - Any other character = match/mismatch (M in CIGAR)

    Parameters
    ----------
    query : str
        Query sequence from BLAST HSP (may contain '-' for gaps).
    subject : str
        Subject sequence from BLAST HSP (may contain '-' for gaps).
    match : str
        Match string from BLAST HSP ('|' for match, ' ' for mismatch).

    Returns
    -------
    str
        CIGAR string (e.g., "150M2I48M").

    Example
    -------
    >>> _blast_alignment_to_cigar("ATCG--ATCG", "ATCGATCG--", "||||  ||||")
    '4M2D4M'
    """
    if not query or not subject:
        return "1M"  # Fallback

    cigar_ops: list[tuple[str, int]] = []
    current_op: Optional[str] = None
    current_count = 0

    for q_base, s_base in zip(query, subject):
        if q_base == "-":
            op = "D"  # Deletion from reference
        elif s_base == "-":
            op = "I"  # Insertion relative to reference
        else:
            op = "M"  # Match or mismatch

        if op == current_op:
            current_count += 1
        else:
            if current_op is not None:
                cigar_ops.append((current_op, current_count))
            current_op = op
            current_count = 1

    if current_op is not None:
        cigar_ops.append((current_op, current_count))

    if not cigar_ops:
        return "1M"

    return "".join(f"{count}{op}" for op, count in cigar_ops)


def get_overlapping_genes(
    chromosome: str,
    start: int,
    end: int,
    api_key: str,
) -> list[str]:
    """
    Query NCBI Gene API to find genes overlapping the aligned region.

    Uses NCBI E-utilities esearch with the query::

        "{chr}[CHR] AND {start}:{end}[CHRPOS] AND Homo sapiens[Organism]"

    Then fetches gene summaries via efetch to extract HGNC gene symbols.

    Parameters
    ----------
    chromosome : str
        UCSC chromosome name (e.g., "chr7"). The "chr" prefix is stripped
        before querying NCBI (NCBI uses "7" not "chr7").
    start : int
        0-based genomic start position.
    end : int
        0-based genomic end position (exclusive).
    api_key : str
        NCBI Entrez API key. Empty string = unauthenticated (3 req/s limit).

    Returns
    -------
    list[str]
        List of HGNC gene symbols overlapping the region.
        Empty list if no genes found or if the query fails.

    Notes
    -----
    NCBI Gene API uses 1-based coordinates for CHRPOS queries.
    The chromosome name must not have the "chr" prefix for NCBI queries.

    Example
    -------
    >>> genes = get_overlapping_genes("chr7", 117548620, 117548820, api_key="")
    >>> print(genes)
    ['CFTR']
    """
    try:
        from Bio import Entrez
    except ImportError:
        logger.warning(
            "Biopython not available; skipping gene annotation"
        )
        return []

    Entrez.email = "sanger@example.com"
    if api_key:
        Entrez.api_key = api_key

    # Strip "chr" prefix for NCBI queries
    ncbi_chr = chromosome.replace("chr", "")
    if ncbi_chr == "M":
        ncbi_chr = "MT"

    # Convert to 1-based coordinates for NCBI
    ncbi_start = start + 1
    ncbi_end = end

    query = (
        f"{ncbi_chr}[CHR] AND "
        f"{ncbi_start}:{ncbi_end}[CHRPOS] AND "
        f"Homo sapiens[Organism]"
    )

    logger.debug(
        "NCBI Gene query: %s (region: %s:%d-%d)",
        query, chromosome, start, end
    )

    try:
        # Search for gene IDs
        search_handle = Entrez.esearch(
            db=NCBI_GENE_DB,
            term=query,
            retmax=20,
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        gene_ids = search_results.get("IdList", [])
        if not gene_ids:
            logger.debug(
                "No genes found overlapping %s:%d-%d",
                chromosome, start, end
            )
            return []

        logger.debug(
            "Found %d gene IDs overlapping %s:%d-%d: %s",
            len(gene_ids), chromosome, start, end, gene_ids
        )

        # Fetch gene summaries to get symbols
        fetch_handle = Entrez.efetch(
            db=NCBI_GENE_DB,
            id=",".join(gene_ids),
            rettype="gene_table",
            retmode="text",
        )
        gene_text = fetch_handle.read()
        fetch_handle.close()

        # Parse gene symbols from the gene table output
        gene_symbols = _parse_gene_symbols(gene_text, gene_ids)

        logger.info(
            "Gene annotation: %s overlaps %s",
            gene_symbols if gene_symbols else "no genes",
            f"{chromosome}:{start}-{end}"
        )
        return gene_symbols

    except Exception as exc:
        logger.warning(
            "Gene annotation failed for %s:%d-%d: %s",
            chromosome, start, end, exc
        )
        return []


def _parse_gene_symbols(gene_text, gene_ids: list[str]) -> list[str]:
    """
    Parse gene symbols from NCBI gene table text output.

    The gene table format has tab-separated columns where the first column
    is the gene symbol. Lines starting with '#' are headers.

    Parameters
    ----------
    gene_text : str or bytes
        Raw text from NCBI efetch gene_table.
    gene_ids : list[str]
        List of gene IDs (used as fallback if parsing fails).

    Returns
    -------
    list[str]
        List of unique gene symbols in order of appearance.
    """
    if isinstance(gene_text, bytes):
        gene_text = gene_text.decode("utf-8", errors="replace")

    symbols = []
    for line in gene_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts:
            symbol = parts[0].strip()
            # Filter out non-gene-symbol lines (numbers, long strings, etc.)
            if symbol and re.match(r"^[A-Z][A-Z0-9\-\.]{0,19}$", symbol):
                symbols.append(symbol)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_symbols: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    return unique_symbols


def is_minimap2_available(index_path: str) -> bool:
    """
    Check if minimap2 binary and hg38 index are available.

    Verifies:
    1. The ``minimap2`` binary is in the system PATH (via ``shutil.which``).
    2. The hg38 index file exists at ``index_path``.

    Parameters
    ----------
    index_path : str
        Path to the minimap2 hg38 index file (.mmi format).

    Returns
    -------
    bool
        True if both minimap2 binary and index file are available.
        False if either is missing.

    Example
    -------
    >>> is_minimap2_available("/data/hg38.mmi")
    False  # Returns False if /data/hg38.mmi doesn't exist
    >>> is_minimap2_available("/nonexistent/path.mmi")
    False
    """
    # Check minimap2 binary
    minimap2_path = shutil.which("minimap2")
    if minimap2_path is None:
        logger.debug("minimap2 binary not found in PATH")
        return False

    # Check index file
    if not os.path.isfile(index_path):
        logger.debug(
            "minimap2 index not found at '%s'", index_path
        )
        return False

    logger.debug(
        "minimap2 available: binary=%s, index=%s",
        minimap2_path, index_path
    )
    return True
