"""
ClinVar annotation service for SangerAgent.

Architecture:
1. Check PostgreSQL cache for variant (chr:pos:ref:alt key)
2. If cached and not expired: return cached annotation
3. If not cached or expired: query NCBI E-utilities API
4. Parse response into ClinVarAnnotation schema
5. Store in PostgreSQL cache with TTL
6. Return annotation

NCBI E-utilities workflow:
- esearch: find ClinVar records for variant
  URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
  params: db=clinvar, term="{chr}[CHR] AND {pos}[CHRPOS] AND {ref}>{alt}[VARIANT]"
- efetch: retrieve full record
  URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
  params: db=clinvar, id={clinvar_id}, rettype=vcv, retmode=json

Rate limiting:
- Without API key: max 3 requests/second
- With API key: max 10 requests/second
- Implement token bucket rate limiter
- Exponential backoff on 429 errors (max 5 retries)

Example
-------
>>> import asyncio
>>> from tools.clinvar import annotate_variant, RateLimiter
>>> limiter = RateLimiter(rate=3.0)
>>> # asyncio.run(limiter.acquire())  # blocks until token available
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

import aiohttp
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ── NCBI E-utilities base URLs ────────────────────────────────────────────────
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Maximum concurrent ClinVar API calls (respects rate limits)
MAX_CONCURRENT_CLINVAR_CALLS = 3

# Global rate limiter instance (module-level singleton)
_rate_limiter: Optional["RateLimiter"] = None


def _get_rate_limiter(settings) -> "RateLimiter":
    """
    Return the module-level rate limiter singleton, creating it if needed.

    The rate is determined by whether an NCBI API key is configured:
    - With API key: 10 requests/second
    - Without API key: 3 requests/second

    Parameters
    ----------
    settings : SangerSettings
        Application settings containing NCBI_API_KEY.

    Returns
    -------
    RateLimiter
        The singleton rate limiter instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        rate = 10.0 if settings.NCBI_API_KEY else 3.0
        _rate_limiter = RateLimiter(rate=rate)
        logger.debug(
            "Created ClinVar rate limiter: %.1f req/s (API key: %s)",
            rate,
            bool(settings.NCBI_API_KEY),
        )
    return _rate_limiter


class RateLimiter:
    """
    Token bucket rate limiter for NCBI API calls.

    Implements the token bucket algorithm to enforce a maximum request rate.
    Tokens are added at the configured rate (tokens/second) up to a maximum
    of ``rate`` tokens. Each API call consumes one token. If no tokens are
    available, the caller waits until a token is replenished.

    Limits:
    - Without API key: max 3 requests/second
    - With API key: max 10 requests/second

    Thread-safe using asyncio.Lock for use in async contexts.

    Attributes
    ----------
    rate : float
        Maximum requests per second.
    tokens : float
        Current number of available tokens.
    last_update : float
        Monotonic timestamp of the last token replenishment.

    Example
    -------
    >>> import asyncio
    >>> limiter = RateLimiter(rate=3.0)
    >>> async def make_request():
    ...     await limiter.acquire()
    ...     # proceed with API call
    >>> asyncio.run(make_request())
    """

    def __init__(self, rate: float) -> None:
        """
        Initialize the token bucket rate limiter.

        Parameters
        ----------
        rate : float
            Maximum requests per second. Must be positive.

        Raises
        ------
        ValueError
            If rate is not positive.
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        self.rate = rate
        self.tokens = rate  # Start with a full bucket
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Wait until a token is available, then consume one token.

        Replenishes tokens based on elapsed time since the last call,
        then waits if no tokens are available. The wait time is calculated
        as (1 - tokens) / rate seconds.

        Returns
        -------
        None
            Returns when a token has been successfully acquired.

        Example
        -------
        >>> limiter = RateLimiter(rate=3.0)
        >>> import asyncio
        >>> asyncio.run(limiter.acquire())  # Immediate (bucket full)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            # Replenish tokens based on elapsed time
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                # Calculate wait time to get one token
                wait_time = (1.0 - self.tokens) / self.rate
                logger.debug(
                    "Rate limiter: waiting %.3fs for token (%.2f tokens available)",
                    wait_time,
                    self.tokens,
                )
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


async def annotate_variant(
    chromosome: str,
    position: int,
    ref_allele: str,
    alt_allele: str,
    db_session: AsyncSession,
    settings,
) -> "ClinVarAnnotation":
    """
    Annotate a genomic variant with ClinVar data.

    Checks the PostgreSQL cache first. If a valid (non-expired) cache entry
    exists, returns it immediately. Otherwise, queries the NCBI ClinVar API,
    parses the response, stores the result in the cache, and returns the
    annotation.

    If the variant is not found in ClinVar, returns an empty ClinVarAnnotation
    with all fields set to None.

    Parameters
    ----------
    chromosome : str
        Chromosome identifier (e.g., "chr1", "1", "chrX").
        The "chr" prefix is normalized internally.
    position : int
        Genomic position (1-based, hg38 coordinates, VCF convention).
    ref_allele : str
        Reference allele at this position (single nucleotide).
    alt_allele : str
        Alternative (variant) allele (single nucleotide).
    db_session : AsyncSession
        SQLAlchemy async database session for cache operations.
    settings : SangerSettings
        Application settings providing NCBI_API_KEY, NCBI_EMAIL,
        and CLINVAR_CACHE_TTL_DAYS.

    Returns
    -------
    ClinVarAnnotation
        Annotation object with clinical significance, conditions, dbSNP ID,
        review status, accession, and ClinVar URL.
        Returns an empty annotation (all None) if the variant is not in ClinVar.

    Raises
    ------
    Exception
        Propagates unexpected errors from the database or API. Callers
        should handle these gracefully.

    Example
    -------
    >>> import asyncio
    >>> from sqlalchemy.ext.asyncio import AsyncSession
    >>> # annotation = asyncio.run(annotate_variant(
    >>> #     "chr17", 7674220, "G", "A", db_session, settings
    >>> # ))
    >>> # print(annotation.clinical_significance)
    """
    from backend.schemas.snv import ClinVarAnnotation

    # Build canonical variant key
    variant_key = _build_variant_key(chromosome, position, ref_allele, alt_allele)
    logger.debug("Annotating variant: %s", variant_key)

    # Step 1: Check PostgreSQL cache
    cached = await get_cached_annotation(variant_key, db_session, settings)
    if cached is not None:
        logger.debug("Cache hit for variant %s", variant_key)
        return cached

    # Step 2: Query NCBI ClinVar API
    logger.debug("Cache miss for %s — querying NCBI ClinVar API", variant_key)
    api_response = await query_clinvar_api(
        chromosome=chromosome,
        position=position,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        settings=settings,
    )

    # Step 3: Parse API response
    if api_response is None:
        # Variant not found in ClinVar — return empty annotation
        annotation = ClinVarAnnotation(
            variant_id=None,
            clinical_significance=None,
            conditions=[],
            dbsnp_id=None,
            review_status=None,
            accession=None,
            clinvar_url=None,
            cached=False,
            annotation_timestamp=datetime.utcnow(),
        )
    else:
        annotation = parse_clinvar_response(api_response, variant_key)

    # Step 4: Store in cache (even empty annotations, to avoid repeated API calls)
    await cache_annotation(variant_key, annotation, db_session, settings)

    return annotation


async def query_clinvar_api(
    chromosome: str,
    position: int,
    ref_allele: str,
    alt_allele: str,
    settings,
) -> Optional[dict]:
    """
    Query NCBI ClinVar API for a variant using the E-utilities two-step workflow.

    Step 1 — esearch: Find ClinVar variation IDs matching the variant.
    Step 2 — efetch: Retrieve the full VCV (Variant-Condition-Variant) record
    for the first matching variation ID.

    Implements exponential backoff retry on HTTP 429 (rate limit exceeded)
    and transient network errors, up to 5 retries.

    Parameters
    ----------
    chromosome : str
        Chromosome identifier (e.g., "chr17", "17").
    position : int
        1-based genomic position (hg38).
    ref_allele : str
        Reference allele.
    alt_allele : str
        Alternative allele.
    settings : SangerSettings
        Application settings with NCBI_API_KEY and NCBI_EMAIL.

    Returns
    -------
    Optional[dict]
        Raw API response dictionary from the efetch call, or None if:
        - No ClinVar records found for this variant
        - API call fails after all retries
        - Response cannot be parsed

    Notes
    -----
    The NCBI E-utilities API requires an email address for all requests.
    With an API key, the rate limit is 10 req/s; without, it is 3 req/s.
    The rate limiter is applied before each API call.

    Example
    -------
    >>> # response = asyncio.run(query_clinvar_api("chr17", 7674220, "G", "A", settings))
    >>> # if response: print(response.get("variation_id"))
    """
    rate_limiter = _get_rate_limiter(settings)

    # Normalize chromosome (remove "chr" prefix for NCBI queries)
    chrom_clean = chromosome.replace("chr", "").replace("Chr", "")

    # Build esearch query
    # NCBI ClinVar search syntax: chromosome[CHR] AND position[CHRPOS]
    search_term = (
        f"{chrom_clean}[CHR] AND {position}[CHRPOS] AND "
        f"{ref_allele}>{alt_allele}[VARIANT]"
    )

    esearch_params = {
        "db": "clinvar",
        "term": search_term,
        "retmax": "5",
        "retmode": "json",
        "email": settings.NCBI_EMAIL,
    }
    if settings.NCBI_API_KEY:
        esearch_params["api_key"] = settings.NCBI_API_KEY

    max_retries = 5
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            # Acquire rate limiter token before each attempt
            await rate_limiter.acquire()

            async with aiohttp.ClientSession() as http_session:
                # Step 1: esearch to find variation IDs
                async with http_session.get(
                    ESEARCH_URL,
                    params=esearch_params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "ClinVar API rate limit hit (attempt %d/%d). "
                            "Waiting %.1fs before retry.",
                            attempt + 1, max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if resp.status != 200:
                        logger.warning(
                            "ClinVar esearch returned HTTP %d for %s:%d:%s>%s",
                            resp.status, chromosome, position, ref_allele, alt_allele,
                        )
                        return None

                    esearch_data = await resp.json(content_type=None)

                ids = esearch_data.get("esearchresult", {}).get("idlist", [])
                if not ids:
                    # Try a broader search without the variant filter
                    broad_term = f"{chrom_clean}[CHR] AND {position}[CHRPOS]"
                    broad_params = dict(esearch_params)
                    broad_params["term"] = broad_term

                    await rate_limiter.acquire()
                    async with http_session.get(
                        ESEARCH_URL,
                        params=broad_params,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as broad_resp:
                        if broad_resp.status == 200:
                            broad_data = await broad_resp.json(content_type=None)
                            ids = broad_data.get("esearchresult", {}).get("idlist", [])

                if not ids:
                    logger.debug(
                        "No ClinVar records found for %s:%d:%s>%s",
                        chromosome, position, ref_allele, alt_allele,
                    )
                    return None

                variation_id = ids[0]
                logger.debug(
                    "Found ClinVar variation ID %s for %s:%d:%s>%s",
                    variation_id, chromosome, position, ref_allele, alt_allele,
                )

                # Step 2: efetch to get full VCV record
                efetch_params = {
                    "db": "clinvar",
                    "id": variation_id,
                    "rettype": "vcv",
                    "retmode": "json",
                    "email": settings.NCBI_EMAIL,
                }
                if settings.NCBI_API_KEY:
                    efetch_params["api_key"] = settings.NCBI_API_KEY

                await rate_limiter.acquire()
                async with http_session.get(
                    EFETCH_URL,
                    params=efetch_params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as fetch_resp:
                    if fetch_resp.status == 429:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue

                    if fetch_resp.status != 200:
                        logger.warning(
                            "ClinVar efetch returned HTTP %d for variation %s",
                            fetch_resp.status, variation_id,
                        )
                        return None

                    # Try JSON first, fall back to text parsing
                    content_type = fetch_resp.headers.get("Content-Type", "")
                    if "json" in content_type:
                        fetch_data = await fetch_resp.json(content_type=None)
                    else:
                        # VCV records may return XML; parse as text
                        text = await fetch_resp.text()
                        fetch_data = {"raw_text": text, "variation_id": variation_id}

                    # Inject variation_id for downstream parsing
                    if isinstance(fetch_data, dict):
                        fetch_data["_variation_id"] = variation_id
                    return fetch_data

        except aiohttp.ClientError as exc:
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "ClinVar API network error (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt + 1, max_retries, exc, delay,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "ClinVar API failed after %d retries for %s:%d:%s>%s",
                    max_retries, chromosome, position, ref_allele, alt_allele,
                )
                return None

        except Exception as exc:
            logger.error(
                "Unexpected error querying ClinVar for %s:%d:%s>%s: %s",
                chromosome, position, ref_allele, alt_allele, exc,
            )
            return None

    return None


def parse_clinvar_response(api_response: dict, variant_key: str) -> "ClinVarAnnotation":
    """
    Parse NCBI ClinVar API response into a ClinVarAnnotation schema object.

    Handles both JSON VCV responses and XML-derived text responses.
    Extracts the following fields from the API response:

    - ``clinical_significance``: from ``classification.germlineClassification.description``
      or ``ClinicalSignificance.Description`` (XML path)
    - ``conditions``: from ``traitSet.trait[].name`` or ``TraitSet.Trait[].Name``
    - ``dbsnp_id``: from ``xref`` where ``db="dbSNP"``
    - ``review_status``: from ``classification.reviewStatus`` or ``ReviewStatus``
    - ``accession``: from the ``accession`` field or ``ClinVarAccession.Acc``
    - ``clinvar_url``: constructed as
      ``https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/``

    Parameters
    ----------
    api_response : dict
        Raw API response dictionary from ``query_clinvar_api``.
        May contain JSON VCV data or a ``raw_text`` key with XML content.
    variant_key : str
        Canonical variant key (e.g., "chr17:7674220:G:A") used for logging.

    Returns
    -------
    ClinVarAnnotation
        Populated annotation object. Fields that cannot be extracted are
        set to None (or empty list for conditions).

    Notes
    -----
    The ClinVar API response format varies between VCV JSON and XML.
    This function handles both formats gracefully, falling back to None
    for any field that cannot be extracted.

    Example
    -------
    >>> mock_response = {
    ...     "_variation_id": "12375",
    ...     "clinical_significance": {"description": "Pathogenic"},
    ...     "conditions": [{"name": "Cystic fibrosis"}],
    ... }
    >>> annotation = parse_clinvar_response(mock_response, "chr7:117548670:G:A")
    >>> annotation.clinical_significance
    'Pathogenic'
    """
    from backend.schemas.snv import ClinVarAnnotation

    variation_id = str(api_response.get("_variation_id", ""))
    clinvar_url = (
        f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/"
        if variation_id
        else None
    )

    # Handle XML raw text response
    if "raw_text" in api_response:
        return _parse_clinvar_xml_response(
            api_response["raw_text"],
            variation_id,
            clinvar_url,
        )

    # ── JSON VCV response parsing ─────────────────────────────────────────────
    clinical_significance: Optional[str] = None
    conditions: list[str] = []
    dbsnp_id: Optional[str] = None
    review_status: Optional[str] = None
    accession: Optional[str] = None

    # Navigate the nested JSON structure (ClinVar VCV JSON format)
    # Try multiple possible paths for each field

    # Clinical significance
    # Path 1: result.classifications.germlineClassification.description
    result = api_response.get("result", {})
    if isinstance(result, dict):
        classifications = result.get("classifications", {})
        germline = classifications.get("germlineClassification", {})
        clinical_significance = germline.get("description")

        if not clinical_significance:
            # Path 2: result.clinical_significance.description
            clin_sig = result.get("clinical_significance", {})
            if isinstance(clin_sig, dict):
                clinical_significance = clin_sig.get("description")

        # Review status
        review_status = germline.get("reviewStatus") or result.get("review_status")

        # Accession
        accession = result.get("accession") or result.get("rcv_accession")

        # Conditions from trait sets
        trait_sets = result.get("trait_set", []) or result.get("traitSet", [])
        if isinstance(trait_sets, list):
            for trait_set in trait_sets:
                traits = trait_set.get("trait", []) if isinstance(trait_set, dict) else []
                for trait in traits:
                    if isinstance(trait, dict):
                        name = trait.get("name") or trait.get("preferred_name")
                        if name:
                            conditions.append(str(name))

        # dbSNP ID from xrefs
        xrefs = result.get("xref", []) or result.get("xrefs", [])
        if isinstance(xrefs, list):
            for xref in xrefs:
                if isinstance(xref, dict) and xref.get("db", "").lower() == "dbsnp":
                    rs_id = str(xref.get("id", ""))
                    if rs_id:
                        dbsnp_id = f"rs{rs_id}" if not rs_id.startswith("rs") else rs_id
                        break

    # Fallback: try top-level keys (some API versions return flat structure)
    if not clinical_significance:
        clinical_significance = api_response.get("clinical_significance")
    if not review_status:
        review_status = api_response.get("review_status")
    if not accession:
        accession = api_response.get("accession")
    if not conditions:
        raw_conditions = api_response.get("conditions", [])
        if isinstance(raw_conditions, list):
            conditions = [str(c) for c in raw_conditions if c]

    # Validate dbsnp_id format (must match rs\d+)
    if dbsnp_id and not dbsnp_id.startswith("rs"):
        dbsnp_id = f"rs{dbsnp_id}"
    # Ensure it only contains digits after "rs"
    if dbsnp_id:
        rs_part = dbsnp_id[2:]
        if not rs_part.isdigit():
            dbsnp_id = None

    logger.debug(
        "Parsed ClinVar annotation for %s: significance=%s, conditions=%d",
        variant_key, clinical_significance, len(conditions),
    )

    return ClinVarAnnotation(
        variant_id=variation_id or None,
        clinical_significance=clinical_significance,
        conditions=conditions,
        dbsnp_id=dbsnp_id,
        review_status=review_status,
        accession=accession,
        clinvar_url=clinvar_url,
        cached=False,
        annotation_timestamp=datetime.utcnow(),
    )


def _parse_clinvar_xml_response(
    xml_text: str,
    variation_id: str,
    clinvar_url: Optional[str],
) -> "ClinVarAnnotation":
    """
    Parse a ClinVar XML response string into a ClinVarAnnotation.

    Used as a fallback when the API returns XML instead of JSON.
    Extracts clinical significance, conditions, dbSNP ID, review status,
    and accession from the XML structure.

    Parameters
    ----------
    xml_text : str
        Raw XML response text from the ClinVar efetch API.
    variation_id : str
        ClinVar variation ID (from esearch).
    clinvar_url : Optional[str]
        Pre-constructed ClinVar URL.

    Returns
    -------
    ClinVarAnnotation
        Parsed annotation. Fields that cannot be extracted are None.
    """
    import xml.etree.ElementTree as ET
    from backend.schemas.snv import ClinVarAnnotation

    clinical_significance: Optional[str] = None
    conditions: list[str] = []
    dbsnp_id: Optional[str] = None
    review_status: Optional[str] = None
    accession: Optional[str] = None

    try:
        root = ET.fromstring(xml_text)

        # Clinical significance
        for path in [
            ".//ClinicalSignificance/Description",
            ".//GermlineClassification/Description",
            ".//Classification/GermlineClassification/Description",
        ]:
            elem = root.find(path)
            if elem is not None and elem.text:
                clinical_significance = elem.text.strip()
                break

        # Review status
        for path in [
            ".//ClinicalSignificance/ReviewStatus",
            ".//ReviewStatus",
        ]:
            elem = root.find(path)
            if elem is not None and elem.text:
                review_status = elem.text.strip()
                break

        # Conditions from trait names
        for trait in root.findall(".//TraitSet/Trait"):
            for name_elem in trait.findall("Name/ElementValue"):
                if name_elem.get("Type") == "Preferred" and name_elem.text:
                    conditions.append(name_elem.text.strip())
                    break

        # dbSNP ID
        for xref in root.findall(".//XRef"):
            if xref.get("DB", "").lower() == "dbsnp":
                rs_id = xref.get("ID", "")
                if rs_id:
                    dbsnp_id = f"rs{rs_id}" if not rs_id.startswith("rs") else rs_id
                    break

        # Accession
        acc_elem = root.find(".//ClinVarAccession")
        if acc_elem is not None:
            accession = acc_elem.get("Acc")

    except ET.ParseError as exc:
        logger.warning("Failed to parse ClinVar XML response: %s", exc)

    return ClinVarAnnotation(
        variant_id=variation_id or None,
        clinical_significance=clinical_significance,
        conditions=conditions,
        dbsnp_id=dbsnp_id,
        review_status=review_status,
        accession=accession,
        clinvar_url=clinvar_url,
        cached=False,
        annotation_timestamp=datetime.utcnow(),
    )


async def get_cached_annotation(
    variant_key: str,
    db_session: AsyncSession,
    settings,
) -> Optional["ClinVarAnnotation"]:
    """
    Retrieve a ClinVar annotation from the PostgreSQL cache if not expired.

    Queries the ``clinvar_cache`` table for an entry matching the variant key.
    Returns None if no entry exists or if the entry has expired (expires_at
    is in the past).

    Parameters
    ----------
    variant_key : str
        Canonical variant identifier in "chr:pos:ref:alt" format
        (e.g., "chr17:7674220:G:A").
    db_session : AsyncSession
        SQLAlchemy async database session.
    settings : SangerSettings
        Application settings (not used directly but kept for API consistency).

    Returns
    -------
    Optional[ClinVarAnnotation]
        The cached annotation with ``cached=True`` and the original
        ``annotation_timestamp``, or None if not found or expired.

    Example
    -------
    >>> # cached = asyncio.run(get_cached_annotation(
    >>> #     "chr17:7674220:G:A", db_session, settings
    >>> # ))
    >>> # if cached: print(f"Cache hit: {cached.clinical_significance}")
    """
    from backend.models.models import ClinVarCache
    from backend.schemas.snv import ClinVarAnnotation

    try:
        result = await db_session.execute(
            select(ClinVarCache).where(ClinVarCache.variant_key == variant_key)
        )
        cache_entry = result.scalar_one_or_none()

        if cache_entry is None:
            logger.debug("Cache miss: no entry for %s", variant_key)
            return None

        if cache_entry.is_expired:
            logger.debug(
                "Cache expired for %s (expired at %s)",
                variant_key, cache_entry.expires_at,
            )
            return None

        # Deserialize the stored annotation
        annotation_data = cache_entry.annotation_json
        if not isinstance(annotation_data, dict):
            logger.warning(
                "Invalid cache entry for %s: annotation_json is not a dict",
                variant_key,
            )
            return None

        # Mark as cached and return
        annotation_data["cached"] = True
        annotation = ClinVarAnnotation.model_validate(annotation_data)
        logger.debug(
            "Cache hit for %s: significance=%s",
            variant_key, annotation.clinical_significance,
        )
        return annotation

    except Exception as exc:
        logger.warning(
            "Error reading ClinVar cache for %s: %s", variant_key, exc
        )
        return None


async def cache_annotation(
    variant_key: str,
    annotation: "ClinVarAnnotation",
    db_session: AsyncSession,
    settings,
) -> None:
    """
    Store a ClinVar annotation in the PostgreSQL cache with TTL expiry.

    Upserts the annotation into the ``clinvar_cache`` table. If an entry
    already exists for the variant key (e.g., an expired entry), it is
    updated with the new annotation and a fresh expiry timestamp.

    Parameters
    ----------
    variant_key : str
        Canonical variant identifier in "chr:pos:ref:alt" format.
    annotation : ClinVarAnnotation
        The annotation to cache. Serialized to JSON for storage.
    db_session : AsyncSession
        SQLAlchemy async database session.
    settings : SangerSettings
        Application settings providing CLINVAR_CACHE_TTL_DAYS.

    Returns
    -------
    None

    Notes
    -----
    The annotation is stored with ``cached=False`` in the JSON (the cached
    flag is set to True only when retrieved from cache). The expiry is set
    to ``datetime.utcnow() + timedelta(days=CLINVAR_CACHE_TTL_DAYS)``.

    Example
    -------
    >>> # asyncio.run(cache_annotation(
    >>> #     "chr17:7674220:G:A", annotation, db_session, settings
    >>> # ))
    """
    from backend.models.models import ClinVarCache

    try:
        now = datetime.utcnow()
        expires_at = now + timedelta(days=settings.CLINVAR_CACHE_TTL_DAYS)

        # Serialize annotation (store with cached=False; set True on retrieval)
        annotation_dict = annotation.model_dump(mode="json")
        annotation_dict["cached"] = False

        # Check if entry already exists
        result = await db_session.execute(
            select(ClinVarCache).where(ClinVarCache.variant_key == variant_key)
        )
        existing = result.scalar_one_or_none()

        if existing is not None:
            # Update existing entry
            existing.annotation_json = annotation_dict
            existing.cached_at = now
            existing.expires_at = expires_at
            logger.debug(
                "Updated ClinVar cache for %s (expires %s)", variant_key, expires_at
            )
        else:
            # Create new entry
            new_entry = ClinVarCache(
                id=uuid.uuid4(),
                variant_key=variant_key,
                annotation_json=annotation_dict,
                cached_at=now,
                expires_at=expires_at,
            )
            db_session.add(new_entry)
            logger.debug(
                "Cached ClinVar annotation for %s (expires %s)", variant_key, expires_at
            )

        await db_session.commit()

    except Exception as exc:
        logger.warning(
            "Failed to cache ClinVar annotation for %s: %s", variant_key, exc
        )
        await db_session.rollback()


async def annotate_snv_results(
    snv_result: "SNVResult",
    db_session: AsyncSession,
    settings,
) -> "SNVResult":
    """
    Annotate all SNVs in an SNVResult with ClinVar data.

    Processes all SNV calls concurrently, respecting the NCBI API rate limit
    by using a semaphore to cap concurrent API calls at MAX_CONCURRENT_CLINVAR_CALLS
    (3 by default). Updates each SNVCall's ``clinvar`` field in-place.

    Parameters
    ----------
    snv_result : SNVResult
        The SNV analysis result containing a list of SNVCall objects to annotate.
        The alignment result must be present to extract the chromosome.
    db_session : AsyncSession
        SQLAlchemy async database session for cache operations.
    settings : SangerSettings
        Application settings with NCBI credentials and cache TTL.

    Returns
    -------
    SNVResult
        The same SNVResult object with each SNVCall's ``clinvar`` field
        populated (or None if the variant is not in ClinVar).

    Notes
    -----
    ClinVar annotation failures for individual variants are logged as warnings
    but do not cause the overall annotation to fail. The SNVResult is returned
    with whatever annotations were successfully retrieved.

    Example
    -------
    >>> # annotated = asyncio.run(annotate_snv_results(snv_result, db_session, settings))
    >>> # for snv in annotated.snvs:
    >>> #     if snv.clinvar: print(snv.clinvar.clinical_significance)
    """
    if not snv_result.snvs:
        logger.debug("No SNVs to annotate in result for job %s", snv_result.job_id)
        return snv_result

    # Extract chromosome from alignment result
    chromosome = ""
    if snv_result.alignment is not None:
        chromosome = snv_result.alignment.chromosome

    logger.info(
        "Annotating %d SNVs with ClinVar data for job %s",
        len(snv_result.snvs), snv_result.job_id,
    )

    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLINVAR_CALLS)

    async def annotate_single_snv(snv, index: int) -> None:
        """Annotate a single SNV call with ClinVar data."""
        async with semaphore:
            try:
                annotation = await annotate_variant(
                    chromosome=chromosome,
                    position=snv.genomic_position,
                    ref_allele=snv.reference_allele,
                    alt_allele=snv.alternative_allele,
                    db_session=db_session,
                    settings=settings,
                )
                # Update the SNV call's clinvar field
                # SNVCall is a Pydantic model; we need to use model_copy
                snv_result.snvs[index] = snv.model_copy(
                    update={"clinvar": annotation}
                )
            except Exception as exc:
                logger.warning(
                    "Failed to annotate SNV at position %d: %s",
                    snv.genomic_position, exc,
                )

    # Run all annotations concurrently
    tasks = [
        annotate_single_snv(snv, i)
        for i, snv in enumerate(snv_result.snvs)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    annotated_count = sum(
        1 for snv in snv_result.snvs if snv.clinvar is not None
    )
    logger.info(
        "ClinVar annotation complete: %d/%d SNVs annotated",
        annotated_count, len(snv_result.snvs),
    )

    return snv_result


def _build_variant_key(
    chromosome: str,
    position: int,
    ref_allele: str,
    alt_allele: str,
) -> str:
    """
    Build a canonical variant key string for cache lookups.

    Normalizes the chromosome name to include the "chr" prefix and
    uppercases the alleles.

    Parameters
    ----------
    chromosome : str
        Chromosome identifier (e.g., "chr17", "17", "X").
    position : int
        1-based genomic position.
    ref_allele : str
        Reference allele.
    alt_allele : str
        Alternative allele.

    Returns
    -------
    str
        Canonical key in "chr{N}:{pos}:{REF}:{ALT}" format
        (e.g., "chr17:7674220:G:A").

    Example
    -------
    >>> _build_variant_key("17", 7674220, "g", "a")
    'chr17:7674220:G:A'
    >>> _build_variant_key("chr17", 7674220, "G", "A")
    'chr17:7674220:G:A'
    """
    # Normalize chromosome prefix
    chrom = chromosome.strip()
    if not chrom.lower().startswith("chr"):
        chrom = f"chr{chrom}"

    return f"{chrom}:{position}:{ref_allele.upper()}:{alt_allele.upper()}"
