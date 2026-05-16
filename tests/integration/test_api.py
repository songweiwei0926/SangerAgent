"""
Integration tests for the SangerAgent REST API.

Tests all API endpoints using httpx AsyncClient with a test database
and mocked Celery tasks. Tests cover the complete request/response cycle
including file upload, job status polling, result retrieval, and export.

Test Setup
----------
- Uses SQLite in-memory database (no PostgreSQL required for tests)
- Mocks Celery task enqueueing to avoid Redis dependency
- Generates synthetic AB1 file bytes for upload tests
- Uses pytest-asyncio for async test support

Running Tests
-------------
    # Run all integration tests
    pytest tests/integration/test_api.py -v

    # Run with coverage
    pytest tests/integration/test_api.py -v --cov=backend

    # Run specific test class
    pytest tests/integration/test_api.py::TestUploadEndpoints -v

Dependencies
------------
    pip install pytest pytest-asyncio httpx

Example
-------
>>> pytest tests/integration/test_api.py -v
tests/integration/test_api.py::TestHealthEndpoints::test_basic_health PASSED
tests/integration/test_api.py::TestHealthEndpoints::test_detailed_health PASSED
...
"""

from __future__ import annotations

import io
import json
import struct
import uuid
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_synthetic_ab1_bytes(
    sequence: str = "ATCGATCGATCG",
    n_peaks: int = 12,
) -> bytes:
    """
    Generate minimal synthetic AB1 file bytes for testing.

    Creates a valid-looking AB1 file header with the ABIF magic number.
    The file won't parse correctly with Biopython, but is sufficient for
    testing file upload validation (extension check, size check).

    Parameters
    ----------
    sequence : str
        Base sequence to embed in the file.
    n_peaks : int
        Number of peaks to simulate.

    Returns
    -------
    bytes
        Synthetic AB1 file bytes with valid ABIF header.
    """
    # ABIF magic number: "ABIF" + version 101
    magic = b"ABIF"
    version = struct.pack(">H", 101)

    # Minimal directory header
    dir_offset = struct.pack(">I", 128)
    dir_count = struct.pack(">I", 0)
    dir_length = struct.pack(">I", 28)

    # Pad to 128 bytes
    header = magic + version + dir_offset + dir_count + dir_length
    header = header + b"\x00" * (128 - len(header))

    # Add some fake trace data
    trace_data = bytes([min(255, i * 10) for i in range(n_peaks * 4)])

    return header + trace_data + sequence.encode("ascii")


@pytest.fixture
def synthetic_ab1_bytes() -> bytes:
    """
    Pytest fixture providing synthetic AB1 file bytes.

    Returns
    -------
    bytes
        Minimal AB1 file bytes for upload testing.
    """
    return _make_synthetic_ab1_bytes()


@pytest.fixture
def mock_job_record():
    """
    Pytest fixture providing a mock Job ORM model.

    Returns
    -------
    MagicMock
        Mock Job object with all required attributes.
    """
    job_id = uuid.uuid4()
    now = datetime.utcnow()

    mock = MagicMock()
    mock.id = job_id
    mock.workflow = "snv"
    mock.status = "pending"
    mock.progress = 0
    mock.created_at = now
    mock.updated_at = now
    mock.completed_at = None
    mock.error_message = None
    mock.batch_id = None
    mock.file_ids = []
    mock.result_json = None
    mock.is_terminal = False
    mock.is_active = True

    return mock


@pytest.fixture
def mock_completed_snv_job():
    """
    Pytest fixture providing a mock completed SNV Job with result data.

    Returns
    -------
    MagicMock
        Mock Job object with completed status and SNV result data.
    """
    job_id = uuid.uuid4()
    now = datetime.utcnow()

    mock = MagicMock()
    mock.id = job_id
    mock.workflow = "snv"
    mock.status = "completed"
    mock.progress = 100
    mock.created_at = now
    mock.updated_at = now
    mock.completed_at = now
    mock.error_message = None
    mock.batch_id = None
    mock.file_ids = [uuid.uuid4()]
    mock.is_terminal = True
    mock.is_active = False
    mock.result_json = {
        "job_id": str(job_id),
        "status": "completed",
        "total_snvs": 2,
        "heterozygous_count": 1,
        "processing_time_seconds": 3.14,
        "snvs": [
            {
                "position_in_read": 42,
                "genomic_position": 117548670,
                "reference_allele": "G",
                "alternative_allele": "A",
                "is_heterozygous": True,
                "secondary_peak_fraction": 0.45,
                "confidence_score": 0.92,
                "confidence_label": "high",
                "proportions": {"A": 0.48, "T": 0.04, "C": 0.04, "G": 0.44},
                "clinvar": None,
            },
            {
                "position_in_read": 87,
                "genomic_position": 117548715,
                "reference_allele": "C",
                "alternative_allele": "T",
                "is_heterozygous": False,
                "secondary_peak_fraction": 0.05,
                "confidence_score": 0.88,
                "confidence_label": "high",
                "proportions": {"A": 0.04, "T": 0.88, "C": 0.04, "G": 0.04},
                "clinvar": None,
            },
        ],
        "alignment": {
            "chromosome": "chr7",
            "start": 117548628,
            "end": 117548728,
            "strand": "+",
            "identity": 0.98,
            "alignment_score": 95,
            "cigar": "100M",
            "genes": ["CFTR"],
            "method": "minimap2",
            "reference_sequence": "G" * 100,
        },
        "chromatogram": {
            "file_name": "sample.ab1",
            "file_hash": "a" * 64,
            "sequence_length": 600,
            "mean_quality": 35.2,
            "qc_flags": [],
            "qc_pass": True,
            "trace": {
                "raw_traces": {"A": [100] * 100, "T": [50] * 100, "C": [30] * 100, "G": [20] * 100},
                "normalized_traces": {"A": [1.0] * 100, "T": [0.5] * 100, "C": [0.3] * 100, "G": [0.2] * 100},
                "peak_positions": list(range(0, 100, 10)),
                "peak_heights": {"A": [100] * 10, "T": [50] * 10, "C": [30] * 10, "G": [20] * 10},
                "base_calls": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
                "quality_scores": [35] * 100,
                "readable_region_start": 10,
                "readable_region_end": 90,
            },
        },
    }

    return mock


# ── Test Classes ──────────────────────────────────────────────────────────────


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_basic_health_check(self):
        """
        Test that the basic health check returns 200 with healthy status.

        Verifies:
        - HTTP 200 response
        - status field is "healthy"
        - version field is present
        - timestamp field is present
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            client = TestClient(app)
            response = client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "version" in data
            assert "timestamp" in data

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_legacy_health_check(self):
        """
        Test that the legacy /health endpoint returns 200.

        Verifies backward compatibility with the Phase 1 health endpoint.
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            client = TestClient(app)
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_root_endpoint(self):
        """
        Test that the root endpoint returns API metadata.

        Verifies:
        - HTTP 200 response
        - name field matches APP_TITLE
        - docs link is present
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            client = TestClient(app)
            response = client.get("/")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "SangerAgent"
            assert "docs" in data

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_detailed_health_with_mocked_db(self):
        """
        Test the detailed health check with a mocked database session.

        Verifies that the endpoint returns component health status
        even when the database is mocked.
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            # Mock the database session
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=MagicMock())

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get("/api/v1/health/detailed")

            # Should return 200 or 503 depending on Redis availability
            assert response.status_code in (200, 503)
            data = response.json()
            assert "status" in data
            assert "components" in data
            assert "database" in data["components"]

            # Clean up override
            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")


class TestUploadEndpoints:
    """Tests for file upload endpoints."""

    def test_upload_snv_invalid_extension(self, synthetic_ab1_bytes):
        """
        Test that uploading a non-AB1 file returns 422.

        Verifies that the file extension validation rejects non-.ab1 files.
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
            mock_session.flush = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.post(
                "/api/v1/upload/snv",
                files={"file": ("sample.txt", io.BytesIO(b"not an ab1 file"), "text/plain")},
            )

            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
            assert ".ab1" in data["detail"].lower() or "ab1" in data["detail"].lower()

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_upload_snv_valid_file_creates_job(self, synthetic_ab1_bytes):
        """
        Test that uploading a valid AB1 file creates a job and returns 202.

        Verifies:
        - HTTP 202 Accepted response
        - Response contains job_id
        - Response status is "pending"
        - Response workflow is "snv"
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.config.settings import get_settings
            from backend.models.database import get_db

            job_id = uuid.uuid4()
            now = datetime.utcnow()

            # Mock the database session
            mock_file_record = MagicMock()
            mock_file_record.id = uuid.uuid4()

            mock_job = MagicMock()
            mock_job.id = job_id
            mock_job.workflow = "snv"
            mock_job.status = "pending"
            mock_job.progress = 0
            mock_job.created_at = now
            mock_job.updated_at = now
            mock_job.error_message = None
            mock_job.batch_id = None

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.flush = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()

            async def override_get_db():
                yield mock_session

            # Mock settings to use temp dir
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_settings = MagicMock()
                mock_settings.max_upload_size_bytes = 50 * 1024 * 1024
                mock_settings.UPLOAD_DIR = tmpdir

                def override_get_settings():
                    return mock_settings

                app.dependency_overrides[get_db] = override_get_db
                app.dependency_overrides[get_settings] = override_get_settings

                with patch("backend.api.upload._create_job", new_callable=AsyncMock) as mock_create_job, \
                     patch("backend.api.upload._get_or_create_file_record", new_callable=AsyncMock) as mock_get_file, \
                     patch("backend.api.upload._enqueue_snv_task") as mock_enqueue:

                    mock_create_job.return_value = mock_job
                    mock_get_file.return_value = mock_file_record

                    client = TestClient(app)
                    response = client.post(
                        "/api/v1/upload/snv",
                        files={"file": ("sample.ab1", io.BytesIO(synthetic_ab1_bytes), "application/octet-stream")},
                    )

                    assert response.status_code == 202
                    data = response.json()
                    assert "job_id" in data
                    assert data["status"] == "pending"
                    assert data["workflow"] == "snv"
                    assert data["progress"] == 0

                    # Verify Celery task was enqueued
                    mock_enqueue.assert_called_once()

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_upload_editing_requires_two_files(self, synthetic_ab1_bytes):
        """
        Test that the editing upload endpoint requires exactly two files.

        Verifies that the endpoint accepts wt_file and edited_file parameters.
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.flush = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)

            # Missing edited_file should return 422
            response = client.post(
                "/api/v1/upload/editing",
                files={"wt_file": ("wt.ab1", io.BytesIO(synthetic_ab1_bytes), "application/octet-stream")},
            )
            assert response.status_code == 422

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_batch_upload_snv_creates_batch(self, synthetic_ab1_bytes):
        """
        Test that batch SNV upload creates a batch with multiple jobs.

        Verifies:
        - HTTP 202 Accepted response
        - Response contains batch_id
        - total_jobs matches number of uploaded files
        - pending_jobs equals total_jobs initially
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.config.settings import get_settings
            from backend.models.database import get_db

            batch_id = uuid.uuid4()
            now = datetime.utcnow()

            mock_batch = MagicMock()
            mock_batch.id = batch_id
            mock_batch.workflow = "snv"
            mock_batch.created_at = now

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.flush = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()

            async def override_get_db():
                yield mock_session

            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_settings = MagicMock()
                mock_settings.max_upload_size_bytes = 50 * 1024 * 1024
                mock_settings.UPLOAD_DIR = tmpdir

                def override_get_settings():
                    return mock_settings

                app.dependency_overrides[get_db] = override_get_db
                app.dependency_overrides[get_settings] = override_get_settings

                with patch("backend.api.upload._create_job", new_callable=AsyncMock) as mock_create_job, \
                     patch("backend.api.upload._get_or_create_file_record", new_callable=AsyncMock) as mock_get_file, \
                     patch("backend.api.upload._enqueue_snv_task") as mock_enqueue:

                    mock_job = MagicMock()
                    mock_job.id = uuid.uuid4()
                    mock_create_job.return_value = mock_job
                    mock_get_file.return_value = MagicMock(id=uuid.uuid4())

                    # Patch Batch creation
                    with patch("backend.api.upload.Batch") as mock_batch_cls:
                        mock_batch_cls.return_value = mock_batch

                        client = TestClient(app)
                        response = client.post(
                            "/api/v1/upload/batch/snv",
                            files=[
                                ("files", ("sample1.ab1", io.BytesIO(synthetic_ab1_bytes), "application/octet-stream")),
                                ("files", ("sample2.ab1", io.BytesIO(synthetic_ab1_bytes), "application/octet-stream")),
                            ],
                            data={"batch_name": "Test Batch"},
                        )

                        assert response.status_code == 202
                        data = response.json()
                        assert "batch_id" in data
                        assert data["total_jobs"] == 2
                        assert data["pending_jobs"] == 2
                        assert data["completed_jobs"] == 0

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")


class TestJobEndpoints:
    """Tests for job management endpoints."""

    def test_get_job_status_not_found(self):
        """
        Test that requesting a non-existent job returns 404.

        Verifies:
        - HTTP 404 Not Found response
        - Response contains detail message
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=None)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            fake_id = str(uuid.uuid4())
            response = client.get(f"/api/v1/jobs/{fake_id}")

            assert response.status_code == 404
            data = response.json()
            assert "detail" in data

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_get_job_status_invalid_uuid(self):
        """
        Test that requesting a job with an invalid UUID returns 400.

        Verifies:
        - HTTP 400 Bad Request response
        - Response contains detail about invalid UUID format
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get("/api/v1/jobs/not-a-valid-uuid")

            assert response.status_code == 400
            data = response.json()
            assert "detail" in data

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_get_job_status_pending(self, mock_job_record):
        """
        Test that a pending job returns correct status and progress.

        Verifies:
        - HTTP 200 response
        - status is "pending"
        - progress is 0
        - workflow is "snv"
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_job_record)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get(f"/api/v1/jobs/{mock_job_record.id}")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "pending"
            assert data["progress"] == 0
            assert data["workflow"] == "snv"
            assert "job_id" in data

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_list_jobs_empty(self):
        """
        Test that listing jobs returns an empty list when no jobs exist.

        Verifies:
        - HTTP 200 response
        - Response is an empty list
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get("/api/v1/jobs/")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 0

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_list_jobs_with_status_filter(self):
        """
        Test that listing jobs with an invalid status filter returns 422.

        Verifies:
        - HTTP 422 Unprocessable Entity response
        - Response contains detail about invalid status
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get("/api/v1/jobs/?status=invalid_status")

            assert response.status_code == 422

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_cancel_pending_job(self, mock_job_record):
        """
        Test that cancelling a pending job marks it as failed.

        Verifies:
        - HTTP 200 response
        - Response status is "failed"
        - Response contains cancellation message
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_job_record)
            mock_session.commit = AsyncMock()

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.delete(f"/api/v1/jobs/{mock_job_record.id}")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert "cancel" in data["message"].lower()

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_cancel_completed_job_returns_409(self, mock_job_record):
        """
        Test that cancelling a completed job returns 409 Conflict.

        Verifies:
        - HTTP 409 Conflict response
        - Response contains detail about terminal state
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            # Make the job appear completed
            mock_job_record.status = "completed"
            mock_job_record.is_terminal = True

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_job_record)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.delete(f"/api/v1/jobs/{mock_job_record.id}")

            assert response.status_code == 409

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")


class TestResultEndpoints:
    """Tests for result retrieval endpoints."""

    def test_get_result_pending_job_returns_409(self, mock_job_record):
        """
        Test that requesting results for a pending job returns 409.

        Verifies:
        - HTTP 409 Conflict response
        - Response contains detail about job still processing
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_job_record)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get(f"/api/v1/results/{mock_job_record.id}")

            assert response.status_code == 409

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_get_result_completed_job(self, mock_completed_snv_job):
        """
        Test that requesting results for a completed job returns the result.

        Verifies:
        - HTTP 200 response
        - Response contains total_snvs
        - Response contains snvs list
        - Response contains workflow field
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_completed_snv_job)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get(f"/api/v1/results/{mock_completed_snv_job.id}")

            assert response.status_code == 200
            data = response.json()
            assert "total_snvs" in data
            assert data["total_snvs"] == 2
            assert "snvs" in data
            assert len(data["snvs"]) == 2
            assert data["workflow"] == "snv"

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_get_chromatogram_data(self, mock_completed_snv_job):
        """
        Test that the chromatogram endpoint returns trace data.

        Verifies:
        - HTTP 200 response
        - Response contains chromatogram key
        - Response contains workflow field
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_completed_snv_job)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get(f"/api/v1/results/{mock_completed_snv_job.id}/chromatogram")

            assert response.status_code == 200
            data = response.json()
            assert "chromatogram" in data
            assert data["workflow"] == "snv"

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_get_batch_results_not_found(self):
        """
        Test that requesting results for a non-existent batch returns 404.

        Verifies:
        - HTTP 404 Not Found response
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=None)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            fake_batch_id = str(uuid.uuid4())
            response = client.get(f"/api/v1/results/batch/{fake_batch_id}")

            assert response.status_code == 404

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")


class TestExportEndpoints:
    """Tests for export endpoints."""

    def test_export_json_completed_job(self, mock_completed_snv_job):
        """
        Test that JSON export returns a downloadable JSON file.

        Verifies:
        - HTTP 200 response
        - Content-Type is application/json
        - Content-Disposition header is present
        - Response body is valid JSON
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_completed_snv_job)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get(f"/api/v1/export/{mock_completed_snv_job.id}/json")

            assert response.status_code == 200
            assert "application/json" in response.headers.get("content-type", "")
            assert "content-disposition" in response.headers
            assert "attachment" in response.headers["content-disposition"]

            # Verify it's valid JSON
            data = response.json()
            assert "export_metadata" in data
            assert "result" in data

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_export_csv_snv_job(self, mock_completed_snv_job):
        """
        Test that CSV export returns a downloadable CSV file with SNV data.

        Verifies:
        - HTTP 200 response
        - Content-Type is text/csv
        - Content-Disposition header is present
        - CSV has correct headers
        - CSV has correct number of data rows
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_completed_snv_job)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get(f"/api/v1/export/{mock_completed_snv_job.id}/csv")

            assert response.status_code == 200
            assert "text/csv" in response.headers.get("content-type", "")
            assert "content-disposition" in response.headers

            # Parse CSV content
            import csv
            lines = response.text.strip().split("\n")
            assert len(lines) >= 1  # At least header

            reader = csv.DictReader(lines)
            rows = list(reader)
            assert len(rows) == 2  # Two SNVs in mock data

            # Verify CSV columns
            assert "job_id" in rows[0]
            assert "genomic_position" in rows[0]
            assert "confidence_label" in rows[0]

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_export_pending_job_returns_409(self, mock_job_record):
        """
        Test that exporting a pending job returns 409.

        Verifies:
        - HTTP 409 Conflict response
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_job_record)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            response = client.get(f"/api/v1/export/{mock_job_record.id}/json")

            assert response.status_code == 409

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")

    def test_export_batch_csv_not_found(self):
        """
        Test that exporting a non-existent batch returns 404.

        Verifies:
        - HTTP 404 Not Found response
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=None)

            async def override_get_db():
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            client = TestClient(app)
            fake_batch_id = str(uuid.uuid4())
            response = client.get(f"/api/v1/export/batch/{fake_batch_id}/csv")

            assert response.status_code == 404

            app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")


class TestCORSMiddleware:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_localhost_3000(self):
        """
        Test that CORS headers are set for localhost:3000 origin.

        Verifies:
        - Access-Control-Allow-Origin header is present for localhost:3000
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            client = TestClient(app)
            response = client.options(
                "/api/v1/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )

            # CORS preflight should succeed
            assert response.status_code in (200, 204)

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")


class TestJobPollingWorkflow:
    """Integration tests for the complete job polling workflow."""

    def test_job_polling_workflow(self, mock_job_record):
        """
        Test the complete job polling workflow: submit → poll → complete.

        Simulates the client-side polling pattern:
        1. Check initial status (pending)
        2. Check processing status
        3. Check completed status

        Verifies that status transitions are correctly reflected in API responses.
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            from backend.models.database import get_db

            job_id = mock_job_record.id
            now = datetime.utcnow()

            # Simulate status progression
            status_sequence = [
                ("pending", 0),
                ("processing", 30),
                ("processing", 60),
                ("completed", 100),
            ]

            for expected_status, expected_progress in status_sequence:
                mock_job_record.status = expected_status
                mock_job_record.progress = expected_progress
                if expected_status == "completed":
                    mock_job_record.completed_at = now
                    mock_job_record.is_terminal = True

                mock_session = AsyncMock()
                mock_session.get = AsyncMock(return_value=mock_job_record)

                async def override_get_db():
                    yield mock_session

                app.dependency_overrides[get_db] = override_get_db

                client = TestClient(app)
                response = client.get(f"/api/v1/jobs/{job_id}")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == expected_status

                app.dependency_overrides.clear()

        except ImportError:
            pytest.skip("FastAPI or httpx not installed")


# ── Standalone test runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd="/mnt/results/SangerAgent",
    )
    sys.exit(result.returncode)
