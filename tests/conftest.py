"""
Pytest configuration and shared fixtures for SangerAgent tests.

Provides synthetic data fixtures and mock settings for unit and integration tests.
All fixtures use in-memory data — no real files, databases, or network calls required.
"""

from __future__ import annotations

import sys
import os

# Add project root to path so all modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def synthetic_chromatogram():
    """Provide synthetic ChromatogramData for testing."""
    from tools.ab1_parser import generate_synthetic_ab1_data
    return generate_synthetic_ab1_data(n_bases=200, seed=42)


@pytest.fixture
def synthetic_chromatogram_pair():
    """Provide a WT/Edited ChromatogramData pair for editing analysis tests."""
    from tools.ab1_parser import generate_synthetic_ab1_data
    wt = generate_synthetic_ab1_data(n_bases=200, seed=0)
    edited = generate_synthetic_ab1_data(n_bases=200, seed=1)
    return wt, edited


@pytest.fixture
def synthetic_snv_result(synthetic_chromatogram):
    """Provide a synthetic SNVResult for testing."""
    from tools.ab1_parser import generate_synthetic_ab1_data
    from backend.schemas.alignment import AlignmentResult
    from tools.snv_caller import call_snvs
    from unittest.mock import MagicMock

    chrom = generate_synthetic_ab1_data(n_bases=200, seed=42, snv_positions=[50, 100])
    alignment = AlignmentResult(
        chromosome="chr1",
        start=1000000,
        end=1000200,
        strand="+",
        cigar="200M",
        identity=0.99,
        alignment_score=200,
        genes=[],
    )
    settings = MagicMock()
    settings.CONFIDENCE_HIGH_THRESHOLD = 0.8
    settings.CONFIDENCE_MEDIUM_THRESHOLD = 0.5
    return call_snvs(chrom, alignment, settings)


@pytest.fixture
def mock_settings():
    """Provide mock Settings for testing."""
    from backend.config.settings import Settings
    return Settings(
        DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/testdb",
        REDIS_URL="redis://localhost:6379/0",
        HG38_INDEX_PATH="/nonexistent/hg38.mmi",
        BLAST_FALLBACK_ENABLED=True,
        NCBI_API_KEY="",
        NCBI_EMAIL="test@example.com",
    )


@pytest.fixture
def mock_db_session():
    """Provide a mock async database session for testing."""
    from unittest.mock import AsyncMock, MagicMock
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.close = AsyncMock()
    return session
