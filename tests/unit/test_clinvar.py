"""
Unit tests for the ClinVar annotation service (tools/clinvar.py).

Tests cover:
- parse_clinvar_response: parsing mock API responses
- get_cached_annotation: cache hit and miss scenarios
- cache_annotation: storage and TTL
- RateLimiter: token bucket rate limiting behavior
- annotate_variant: full flow with mocked API and DB

All tests use mocking to avoid real NCBI API calls and database connections.

Example
-------
>>> pytest tests/unit/test_clinvar.py -v
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_settings():
    """Return mock application settings."""
    settings = MagicMock()
    settings.NCBI_API_KEY = ""
    settings.NCBI_EMAIL = "test@example.com"
    settings.CLINVAR_CACHE_TTL_DAYS = 30
    return settings


@pytest.fixture
def mock_settings_with_api_key():
    """Return mock settings with NCBI API key."""
    settings = MagicMock()
    settings.NCBI_API_KEY = "test_api_key_12345"
    settings.NCBI_EMAIL = "test@example.com"
    settings.CLINVAR_CACHE_TTL_DAYS = 30
    return settings


@pytest.fixture
def mock_db_session():
    """Return a mock async database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def sample_clinvar_json_response():
    """Return a sample ClinVar JSON API response."""
    return {
        "_variation_id": "12375",
        "result": {
            "classifications": {
                "germlineClassification": {
                    "description": "Pathogenic",
                    "reviewStatus": "reviewed by expert panel",
                }
            },
            "accession": "RCV000007535",
            "trait_set": [
                {
                    "trait": [
                        {"name": "Cystic fibrosis", "preferred_name": "Cystic fibrosis"}
                    ]
                }
            ],
            "xref": [
                {"db": "dbSNP", "id": "113993960"},
                {"db": "OMIM", "id": "219700"},
            ],
        },
    }


@pytest.fixture
def sample_clinvar_xml_response():
    """Return a sample ClinVar XML API response."""
    return {
        "_variation_id": "12375",
        "raw_text": """<?xml version="1.0" encoding="UTF-8"?>
<ClinVarResult>
  <ClinVarSet>
    <ReferenceClinVarAssertion>
      <ClinVarAccession Acc="RCV000007535" />
      <ClinicalSignificance>
        <ReviewStatus>reviewed by expert panel</ReviewStatus>
        <Description>Pathogenic</Description>
      </ClinicalSignificance>
      <TraitSet>
        <Trait>
          <Name>
            <ElementValue Type="Preferred">Cystic fibrosis</ElementValue>
          </Name>
        </Trait>
      </TraitSet>
      <XRef DB="dbSNP" ID="113993960" />
    </ReferenceClinVarAssertion>
  </ClinVarSet>
</ClinVarResult>""",
    }


@pytest.fixture
def sample_cached_annotation():
    """Return a sample ClinVarAnnotation as a dict (for cache storage)."""
    return {
        "variant_id": "12375",
        "clinical_significance": "Pathogenic",
        "conditions": ["Cystic fibrosis"],
        "dbsnp_id": "rs113993960",
        "review_status": "reviewed by expert panel",
        "accession": "RCV000007535",
        "clinvar_url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/12375/",
        "cached": False,
        "annotation_timestamp": datetime.utcnow().isoformat(),
    }


# ── Tests: parse_clinvar_response ─────────────────────────────────────────────


class TestParseClinvarResponse:
    """Tests for parse_clinvar_response function."""

    def test_parse_json_response_pathogenic(self, sample_clinvar_json_response):
        """Test parsing a JSON response with Pathogenic classification."""
        from tools.clinvar import parse_clinvar_response

        annotation = parse_clinvar_response(
            sample_clinvar_json_response, "chr7:117548670:G:A"
        )

        assert annotation.variant_id == "12375"
        assert annotation.clinical_significance == "Pathogenic"
        assert annotation.review_status == "reviewed by expert panel"
        assert annotation.accession == "RCV000007535"
        assert "Cystic fibrosis" in annotation.conditions
        assert annotation.dbsnp_id == "rs113993960"
        assert annotation.clinvar_url == "https://www.ncbi.nlm.nih.gov/clinvar/variation/12375/"
        assert annotation.cached is False

    def test_parse_xml_response(self, sample_clinvar_xml_response):
        """Test parsing an XML response."""
        from tools.clinvar import parse_clinvar_response

        annotation = parse_clinvar_response(
            sample_clinvar_xml_response, "chr7:117548670:G:A"
        )

        assert annotation.variant_id == "12375"
        assert annotation.clinical_significance == "Pathogenic"
        assert annotation.review_status == "reviewed by expert panel"
        assert "Cystic fibrosis" in annotation.conditions
        assert annotation.dbsnp_id == "rs113993960"

    def test_parse_response_missing_fields(self):
        """Test parsing a response with missing optional fields."""
        from tools.clinvar import parse_clinvar_response

        minimal_response = {
            "_variation_id": "99999",
            "result": {},
        }

        annotation = parse_clinvar_response(minimal_response, "chr1:100:A:T")

        assert annotation.variant_id == "99999"
        assert annotation.clinical_significance is None
        assert annotation.conditions == []
        assert annotation.dbsnp_id is None
        assert annotation.review_status is None

    def test_parse_response_benign_classification(self):
        """Test parsing a Benign classification."""
        from tools.clinvar import parse_clinvar_response

        response = {
            "_variation_id": "54321",
            "result": {
                "classifications": {
                    "germlineClassification": {
                        "description": "Benign",
                        "reviewStatus": "criteria provided, single submitter",
                    }
                },
            },
        }

        annotation = parse_clinvar_response(response, "chr1:100:A:T")

        assert annotation.clinical_significance == "Benign"
        assert annotation.is_benign is True
        assert annotation.is_pathogenic is False

    def test_parse_response_vus_classification(self):
        """Test parsing a Variant of Uncertain Significance."""
        from tools.clinvar import parse_clinvar_response

        response = {
            "_variation_id": "11111",
            "result": {
                "classifications": {
                    "germlineClassification": {
                        "description": "Uncertain significance",
                    }
                },
            },
        }

        annotation = parse_clinvar_response(response, "chr1:100:A:T")

        assert annotation.is_vus is True
        assert annotation.is_pathogenic is False
        assert annotation.is_benign is False

    def test_parse_response_multiple_conditions(self):
        """Test parsing a response with multiple conditions."""
        from tools.clinvar import parse_clinvar_response

        response = {
            "_variation_id": "22222",
            "result": {
                "classifications": {
                    "germlineClassification": {
                        "description": "Pathogenic",
                    }
                },
                "trait_set": [
                    {
                        "trait": [
                            {"name": "Condition A"},
                            {"name": "Condition B"},
                        ]
                    }
                ],
            },
        }

        annotation = parse_clinvar_response(response, "chr1:100:A:T")

        assert len(annotation.conditions) == 2
        assert "Condition A" in annotation.conditions
        assert "Condition B" in annotation.conditions

    def test_parse_response_dbsnp_id_normalization(self):
        """Test that dbSNP IDs are normalized to rs format."""
        from tools.clinvar import parse_clinvar_response

        response = {
            "_variation_id": "33333",
            "result": {
                "xref": [
                    {"db": "dbSNP", "id": "12345678"},  # Without rs prefix
                ]
            },
        }

        annotation = parse_clinvar_response(response, "chr1:100:A:T")

        assert annotation.dbsnp_id == "rs12345678"

    def test_parse_response_invalid_dbsnp_id_rejected(self):
        """Test that invalid dbSNP IDs are rejected."""
        from tools.clinvar import parse_clinvar_response

        response = {
            "_variation_id": "44444",
            "result": {
                "xref": [
                    {"db": "dbSNP", "id": "not_a_number"},
                ]
            },
        }

        annotation = parse_clinvar_response(response, "chr1:100:A:T")

        assert annotation.dbsnp_id is None

    def test_parse_response_clinvar_url_construction(self):
        """Test that ClinVar URL is correctly constructed."""
        from tools.clinvar import parse_clinvar_response

        response = {"_variation_id": "99999", "result": {}}
        annotation = parse_clinvar_response(response, "chr1:100:A:T")

        assert annotation.clinvar_url == "https://www.ncbi.nlm.nih.gov/clinvar/variation/99999/"

    def test_parse_response_annotation_timestamp_set(self):
        """Test that annotation_timestamp is set."""
        from tools.clinvar import parse_clinvar_response

        response = {"_variation_id": "12345", "result": {}}
        annotation = parse_clinvar_response(response, "chr1:100:A:T")

        assert annotation.annotation_timestamp is not None


# ── Tests: get_cached_annotation ─────────────────────────────────────────────


class TestGetCachedAnnotation:
    """Tests for get_cached_annotation function."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_annotation(
        self, mock_db_session, mock_settings, sample_cached_annotation
    ):
        """Test that a valid cache entry is returned."""
        from tools.clinvar import get_cached_annotation

        # Mock cache entry
        cache_entry = MagicMock()
        cache_entry.is_expired = False
        cache_entry.annotation_json = sample_cached_annotation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entry
        mock_db_session.execute.return_value = mock_result

        annotation = await get_cached_annotation(
            "chr7:117548670:G:A", mock_db_session, mock_settings
        )

        assert annotation is not None
        assert annotation.clinical_significance == "Pathogenic"
        assert annotation.cached is True

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, mock_db_session, mock_settings):
        """Test that a missing cache entry returns None."""
        from tools.clinvar import get_cached_annotation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        annotation = await get_cached_annotation(
            "chr1:100:A:T", mock_db_session, mock_settings
        )

        assert annotation is None

    @pytest.mark.asyncio
    async def test_expired_cache_returns_none(
        self, mock_db_session, mock_settings, sample_cached_annotation
    ):
        """Test that an expired cache entry returns None."""
        from tools.clinvar import get_cached_annotation

        # Mock expired cache entry
        cache_entry = MagicMock()
        cache_entry.is_expired = True
        cache_entry.annotation_json = sample_cached_annotation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entry
        mock_db_session.execute.return_value = mock_result

        annotation = await get_cached_annotation(
            "chr7:117548670:G:A", mock_db_session, mock_settings
        )

        assert annotation is None

    @pytest.mark.asyncio
    async def test_cache_db_error_returns_none(self, mock_db_session, mock_settings):
        """Test that a database error returns None gracefully."""
        from tools.clinvar import get_cached_annotation

        mock_db_session.execute.side_effect = Exception("DB connection error")

        annotation = await get_cached_annotation(
            "chr1:100:A:T", mock_db_session, mock_settings
        )

        assert annotation is None

    @pytest.mark.asyncio
    async def test_cache_hit_sets_cached_flag(
        self, mock_db_session, mock_settings, sample_cached_annotation
    ):
        """Test that cached=True is set on retrieved annotations."""
        from tools.clinvar import get_cached_annotation

        cache_entry = MagicMock()
        cache_entry.is_expired = False
        cache_entry.annotation_json = sample_cached_annotation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entry
        mock_db_session.execute.return_value = mock_result

        annotation = await get_cached_annotation(
            "chr7:117548670:G:A", mock_db_session, mock_settings
        )

        assert annotation.cached is True


# ── Tests: cache_annotation ───────────────────────────────────────────────────


class TestCacheAnnotation:
    """Tests for cache_annotation function."""

    @pytest.mark.asyncio
    async def test_cache_new_annotation(
        self, mock_db_session, mock_settings, sample_cached_annotation
    ):
        """Test storing a new annotation in the cache."""
        from tools.clinvar import cache_annotation
        from backend.schemas.snv import ClinVarAnnotation

        # No existing entry
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        annotation = ClinVarAnnotation(**sample_cached_annotation)

        await cache_annotation(
            "chr7:117548670:G:A", annotation, mock_db_session, mock_settings
        )

        # Verify add was called
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_update_existing_entry(
        self, mock_db_session, mock_settings, sample_cached_annotation
    ):
        """Test updating an existing cache entry."""
        from tools.clinvar import cache_annotation
        from backend.schemas.snv import ClinVarAnnotation

        # Existing entry
        existing_entry = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_entry
        mock_db_session.execute.return_value = mock_result

        annotation = ClinVarAnnotation(**sample_cached_annotation)

        await cache_annotation(
            "chr7:117548670:G:A", annotation, mock_db_session, mock_settings
        )

        # Verify existing entry was updated (not add)
        mock_db_session.add.assert_not_called()
        assert existing_entry.annotation_json is not None
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_ttl_is_set(
        self, mock_db_session, mock_settings, sample_cached_annotation
    ):
        """Test that the cache TTL is correctly set."""
        from tools.clinvar import cache_annotation
        from backend.schemas.snv import ClinVarAnnotation

        mock_settings.CLINVAR_CACHE_TTL_DAYS = 30

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        annotation = ClinVarAnnotation(**sample_cached_annotation)

        await cache_annotation(
            "chr7:117548670:G:A", annotation, mock_db_session, mock_settings
        )

        # Verify the added entry has expires_at set
        added_entry = mock_db_session.add.call_args[0][0]
        assert added_entry.expires_at is not None
        # TTL should be approximately 30 days from now
        expected_expiry = datetime.utcnow() + timedelta(days=30)
        diff = abs((added_entry.expires_at - expected_expiry).total_seconds())
        assert diff < 60  # Within 1 minute

    @pytest.mark.asyncio
    async def test_cache_db_error_handled_gracefully(
        self, mock_db_session, mock_settings, sample_cached_annotation
    ):
        """Test that database errors during caching are handled gracefully."""
        from tools.clinvar import cache_annotation
        from backend.schemas.snv import ClinVarAnnotation

        mock_db_session.execute.side_effect = Exception("DB error")

        annotation = ClinVarAnnotation(**sample_cached_annotation)

        # Should not raise
        await cache_annotation(
            "chr7:117548670:G:A", annotation, mock_db_session, mock_settings
        )

        mock_db_session.rollback.assert_called_once()


# ── Tests: RateLimiter ────────────────────────────────────────────────────────


class TestRateLimiter:
    """Tests for the RateLimiter token bucket implementation."""

    def test_rate_limiter_initialization(self):
        """Test that RateLimiter initializes with correct rate."""
        from tools.clinvar import RateLimiter

        limiter = RateLimiter(rate=3.0)
        assert limiter.rate == 3.0
        assert limiter.tokens == 3.0

    def test_rate_limiter_invalid_rate_raises(self):
        """Test that a non-positive rate raises ValueError."""
        from tools.clinvar import RateLimiter

        with pytest.raises(ValueError, match="Rate must be positive"):
            RateLimiter(rate=0.0)

        with pytest.raises(ValueError, match="Rate must be positive"):
            RateLimiter(rate=-1.0)

    @pytest.mark.asyncio
    async def test_rate_limiter_immediate_acquire_when_tokens_available(self):
        """Test that acquire() returns immediately when tokens are available."""
        from tools.clinvar import RateLimiter

        limiter = RateLimiter(rate=10.0)

        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should be nearly instantaneous (< 100ms)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_consumes_token(self):
        """Test that acquire() consumes a token."""
        from tools.clinvar import RateLimiter

        limiter = RateLimiter(rate=3.0)
        initial_tokens = limiter.tokens

        await limiter.acquire()

        # Tokens should decrease by 1 (approximately, accounting for replenishment)
        assert limiter.tokens < initial_tokens

    @pytest.mark.asyncio
    async def test_rate_limiter_multiple_acquires(self):
        """Test multiple sequential acquires."""
        from tools.clinvar import RateLimiter

        limiter = RateLimiter(rate=100.0)  # High rate for fast test

        # Should be able to acquire multiple times quickly
        for _ in range(5):
            await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiter_respects_rate(self):
        """Test that the rate limiter enforces the configured rate."""
        from tools.clinvar import RateLimiter

        # Use a very low rate to make the test measurable
        limiter = RateLimiter(rate=10.0)
        # Drain the bucket
        limiter.tokens = 0.0

        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should wait approximately 1/rate = 0.1 seconds
        # Allow generous tolerance for CI environments
        assert elapsed >= 0.05  # At least 50ms

    def test_rate_limiter_with_api_key_rate(self):
        """Test that API key rate is 10 req/s."""
        from tools.clinvar import RateLimiter

        limiter = RateLimiter(rate=10.0)
        assert limiter.rate == 10.0

    def test_rate_limiter_without_api_key_rate(self):
        """Test that no-API-key rate is 3 req/s."""
        from tools.clinvar import RateLimiter

        limiter = RateLimiter(rate=3.0)
        assert limiter.rate == 3.0


# ── Tests: annotate_variant ───────────────────────────────────────────────────


class TestAnnotateVariant:
    """Tests for the annotate_variant function (full flow)."""

    @pytest.mark.asyncio
    async def test_annotate_variant_cache_hit(
        self, mock_db_session, mock_settings, sample_cached_annotation
    ):
        """Test that a cached annotation is returned without API call."""
        from tools.clinvar import annotate_variant

        # Mock cache hit
        cache_entry = MagicMock()
        cache_entry.is_expired = False
        cache_entry.annotation_json = sample_cached_annotation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entry
        mock_db_session.execute.return_value = mock_result

        with patch("tools.clinvar.query_clinvar_api") as mock_api:
            annotation = await annotate_variant(
                chromosome="chr7",
                position=117548670,
                ref_allele="G",
                alt_allele="A",
                db_session=mock_db_session,
                settings=mock_settings,
            )

        # API should not be called on cache hit
        mock_api.assert_not_called()
        assert annotation.clinical_significance == "Pathogenic"
        assert annotation.cached is True

    @pytest.mark.asyncio
    async def test_annotate_variant_cache_miss_calls_api(
        self, mock_db_session, mock_settings, sample_clinvar_json_response
    ):
        """Test that the API is called on cache miss."""
        from tools.clinvar import annotate_variant

        # Mock cache miss
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("tools.clinvar.query_clinvar_api") as mock_api:
            mock_api.return_value = sample_clinvar_json_response

            annotation = await annotate_variant(
                chromosome="chr7",
                position=117548670,
                ref_allele="G",
                alt_allele="A",
                db_session=mock_db_session,
                settings=mock_settings,
            )

        mock_api.assert_called_once()
        assert annotation.clinical_significance == "Pathogenic"

    @pytest.mark.asyncio
    async def test_annotate_variant_not_in_clinvar(
        self, mock_db_session, mock_settings
    ):
        """Test that a variant not in ClinVar returns empty annotation."""
        from tools.clinvar import annotate_variant

        # Mock cache miss
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("tools.clinvar.query_clinvar_api") as mock_api:
            mock_api.return_value = None  # Not found in ClinVar

            annotation = await annotate_variant(
                chromosome="chr1",
                position=100,
                ref_allele="A",
                alt_allele="T",
                db_session=mock_db_session,
                settings=mock_settings,
            )

        assert annotation.variant_id is None
        assert annotation.clinical_significance is None
        assert annotation.conditions == []

    @pytest.mark.asyncio
    async def test_annotate_variant_chromosome_normalization(
        self, mock_db_session, mock_settings
    ):
        """Test that chromosome names are normalized (with/without chr prefix)."""
        from tools.clinvar import _build_variant_key

        # Test normalization
        key1 = _build_variant_key("17", 7674220, "G", "A")
        key2 = _build_variant_key("chr17", 7674220, "G", "A")

        assert key1 == key2
        assert key1 == "chr17:7674220:G:A"

    @pytest.mark.asyncio
    async def test_annotate_variant_caches_result(
        self, mock_db_session, mock_settings, sample_clinvar_json_response
    ):
        """Test that API results are cached after retrieval."""
        from tools.clinvar import annotate_variant

        # Mock cache miss
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("tools.clinvar.query_clinvar_api") as mock_api:
            with patch("tools.clinvar.cache_annotation") as mock_cache:
                mock_api.return_value = sample_clinvar_json_response
                mock_cache.return_value = None

                await annotate_variant(
                    chromosome="chr7",
                    position=117548670,
                    ref_allele="G",
                    alt_allele="A",
                    db_session=mock_db_session,
                    settings=mock_settings,
                )

        # cache_annotation should be called
        mock_cache.assert_called_once()


# ── Tests: build_variant_key ──────────────────────────────────────────────────


class TestBuildVariantKey:
    """Tests for the _build_variant_key helper function."""

    def test_key_with_chr_prefix(self):
        """Test key building with chr prefix."""
        from tools.clinvar import _build_variant_key

        key = _build_variant_key("chr17", 7674220, "G", "A")
        assert key == "chr17:7674220:G:A"

    def test_key_without_chr_prefix(self):
        """Test key building without chr prefix (adds it)."""
        from tools.clinvar import _build_variant_key

        key = _build_variant_key("17", 7674220, "G", "A")
        assert key == "chr17:7674220:G:A"

    def test_key_uppercase_alleles(self):
        """Test that alleles are uppercased."""
        from tools.clinvar import _build_variant_key

        key = _build_variant_key("chr1", 100, "a", "t")
        assert key == "chr1:100:A:T"

    def test_key_x_chromosome(self):
        """Test key building for X chromosome."""
        from tools.clinvar import _build_variant_key

        key = _build_variant_key("X", 1000, "C", "G")
        assert key == "chrX:1000:C:G"

    def test_key_format_consistency(self):
        """Test that key format is consistent."""
        from tools.clinvar import _build_variant_key

        key = _build_variant_key("chr7", 117548670, "G", "A")
        parts = key.split(":")
        assert len(parts) == 4
        assert parts[0].startswith("chr")
        assert parts[1].isdigit()
        assert parts[2] in "ATCG"
        assert parts[3] in "ATCG"
