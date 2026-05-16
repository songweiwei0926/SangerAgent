"""
Unit tests for the batch processing service (backend/services/batch_service.py).

Tests cover:
- create_batch: batch creation and validation
- add_job_to_batch: job association
- get_batch_summary: aggregate statistics computation
- generate_batch_csv_report: CSV output format and content
- get_batch_qc_summary: QC aggregation

All tests use in-memory mocks — no real database required.
"""

from __future__ import annotations

import csv
import io
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(__file__).rsplit("/tests", 1)[0])

from backend.services.batch_service import (
    BatchSummaryData,
    add_job_to_batch,
    create_batch,
    generate_batch_csv_report,
    get_batch_qc_summary,
    get_batch_summary,
)


# ── Mock helpers ──────────────────────────────────────────────────────────────

def make_mock_batch(
    batch_id: uuid.UUID = None,
    name: str = "Test Batch",
    workflow: str = "snv",
    created_at: datetime = None,
) -> MagicMock:
    """Create a mock Batch ORM object."""
    batch = MagicMock()
    batch.id = batch_id or uuid.uuid4()
    batch.name = name
    batch.workflow = workflow
    batch.created_at = created_at or datetime.utcnow()
    return batch


def make_mock_job(
    job_id: uuid.UUID = None,
    status: str = "completed",
    result_json: dict = None,
    created_at: datetime = None,
    completed_at: datetime = None,
) -> MagicMock:
    """Create a mock Job ORM object."""
    job = MagicMock()
    job.id = job_id or uuid.uuid4()
    job.status = status
    job.result_json = result_json
    job.created_at = created_at or datetime.utcnow()
    job.completed_at = completed_at or (datetime.utcnow() + timedelta(seconds=30))
    return job


def make_snv_result(
    file_name: str = "sample.ab1",
    n_snvs: int = 3,
    qc_pass: bool = True,
    mean_quality: float = 35.0,
) -> dict:
    """Create a synthetic SNV result dict."""
    snv_calls = []
    for i in range(n_snvs):
        snv_calls.append({
            "chromosome": "chr7",
            "position": 1000 + i * 100,
            "ref_allele": "A",
            "alt_allele": "G",
            "base_proportions": {"A": 0.55, "T": 0.02, "C": 0.02, "G": 0.41},
            "is_heterozygous": True,
            "secondary_fraction": 0.43,
            "confidence_score": 0.85,
            "confidence_label": "high",
            "clinvar": {"clinical_significance": "Pathogenic", "accession": "VCV000001"},
            "transition_transversion": "transition",
            "quality_score": 35,
        })
    return {
        "file_name": file_name,
        "status": "completed",
        "snv_calls": snv_calls,
        "mean_quality": mean_quality,
        "qc_report": {"overall_pass": qc_pass, "flags": []},
    }


def make_editing_result(
    edited_file_name: str = "edited.ab1",
    efficiency: float = 0.65,
    editing_type: str = "base_editing",
    qc_pass: bool = True,
) -> dict:
    """Create a synthetic editing result dict."""
    return {
        "edited_file_name": edited_file_name,
        "status": "completed",
        "editing_type": editing_type,
        "efficiency": efficiency,
        "indel_pct": 0.0,
        "base_edit_pct": efficiency,
        "tool_used": "beat",
        "r_squared": 0.95,
        "qc_pass": qc_pass,
        "wt_mean_quality": 34.0,
        "edited_mean_quality": 33.5,
        "qc_report": {"overall_pass": qc_pass, "flags": []},
    }


def make_async_session(batch=None, jobs=None):
    """Create a mock AsyncSession."""
    session = AsyncMock()

    # Mock execute to return appropriate results
    async def mock_execute(stmt):
        result = MagicMock()
        # Determine what was queried based on call count
        if batch is not None and not hasattr(mock_execute, "_batch_returned"):
            mock_execute._batch_returned = True
            result.scalar_one_or_none.return_value = batch
        else:
            scalars_mock = MagicMock()
            scalars_mock.all.return_value = jobs or []
            result.scalars.return_value = scalars_mock
        return result

    session.execute = mock_execute
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


# ── Tests: create_batch ───────────────────────────────────────────────────────

class TestCreateBatch:
    @pytest.mark.asyncio
    async def test_creates_batch_with_snv_workflow(self):
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()

        async def mock_refresh(obj):
            obj.id = uuid.uuid4()
            obj.name = "Test"
            obj.workflow = "snv"
            obj.created_at = datetime.utcnow()

        session.refresh = mock_refresh

        with patch("backend.services.batch_service.Batch") as MockBatch:
            mock_batch = MagicMock()
            mock_batch.id = uuid.uuid4()
            mock_batch.name = "Test"
            mock_batch.workflow = "snv"
            mock_batch.created_at = datetime.utcnow()
            MockBatch.return_value = mock_batch

            batch = await create_batch("snv", "Test", session)
            assert batch.workflow == "snv"
            assert batch.name == "Test"

    @pytest.mark.asyncio
    async def test_creates_batch_with_editing_workflow(self):
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()

        async def mock_refresh(obj):
            pass

        session.refresh = mock_refresh

        with patch("backend.services.batch_service.Batch") as MockBatch:
            mock_batch = MagicMock()
            mock_batch.workflow = "editing"
            mock_batch.name = "Editing Batch"
            MockBatch.return_value = mock_batch

            batch = await create_batch("editing", "Editing Batch", session)
            assert batch.workflow == "editing"

    @pytest.mark.asyncio
    async def test_invalid_workflow_raises_value_error(self):
        session = AsyncMock()
        with pytest.raises(ValueError, match="Invalid workflow"):
            await create_batch("invalid_workflow", "Test", session)

    @pytest.mark.asyncio
    async def test_commits_to_database(self):
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        with patch("backend.services.batch_service.Batch") as MockBatch:
            MockBatch.return_value = MagicMock()
            await create_batch("snv", "Test", session)
            session.commit.assert_called_once()


# ── Tests: add_job_to_batch ───────────────────────────────────────────────────

class TestAddJobToBatch:
    @pytest.mark.asyncio
    async def test_invalid_batch_uuid_raises(self):
        session = AsyncMock()
        with pytest.raises(ValueError, match="Invalid UUID"):
            await add_job_to_batch("not-a-uuid", str(uuid.uuid4()), session)

    @pytest.mark.asyncio
    async def test_invalid_job_uuid_raises(self):
        session = AsyncMock()
        with pytest.raises(ValueError, match="Invalid UUID"):
            await add_job_to_batch(str(uuid.uuid4()), "not-a-uuid", session)

    @pytest.mark.asyncio
    async def test_job_not_found_raises(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()

        with patch("backend.services.batch_service.Job"):
            with patch("backend.services.batch_service.update"):
                with pytest.raises(ValueError, match="not found"):
                    await add_job_to_batch(
                        str(uuid.uuid4()), str(uuid.uuid4()), session
                    )

    @pytest.mark.asyncio
    async def test_successful_association(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()

        with patch("backend.services.batch_service.Job"):
            with patch("backend.services.batch_service.update"):
                # Should not raise
                await add_job_to_batch(
                    str(uuid.uuid4()), str(uuid.uuid4()), session
                )
                session.commit.assert_called_once()


# ── Tests: get_batch_summary ──────────────────────────────────────────────────

class TestGetBatchSummary:
    @pytest.mark.asyncio
    async def test_invalid_uuid_raises(self):
        session = AsyncMock()
        with pytest.raises(ValueError, match="Invalid batch_id"):
            await get_batch_summary("not-a-uuid", session)

    @pytest.mark.asyncio
    async def test_batch_not_found_raises(self):
        session = AsyncMock()
        batch_result = MagicMock()
        batch_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=batch_result)

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    with pytest.raises(ValueError, match="not found"):
                        await get_batch_summary(str(uuid.uuid4()), session)

    @pytest.mark.asyncio
    async def test_snv_batch_counts_snvs(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")
        jobs = [
            make_mock_job(result_json=make_snv_result(n_snvs=3)),
            make_mock_job(result_json=make_snv_result(n_snvs=5)),
        ]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    summary = await get_batch_summary(str(batch_id), session)

        assert summary.total_jobs == 2
        assert summary.total_snv_count == 8  # 3 + 5

    @pytest.mark.asyncio
    async def test_editing_batch_computes_mean_efficiency(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="editing")
        jobs = [
            make_mock_job(result_json=make_editing_result(efficiency=0.60)),
            make_mock_job(result_json=make_editing_result(efficiency=0.80)),
        ]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    summary = await get_batch_summary(str(batch_id), session)

        assert summary.mean_efficiency is not None
        assert abs(summary.mean_efficiency - 0.70) < 0.01

    @pytest.mark.asyncio
    async def test_empty_batch_returns_zeros(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = []
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    summary = await get_batch_summary(str(batch_id), session)

        assert summary.total_jobs == 0
        assert summary.completed_jobs == 0
        assert summary.total_snv_count == 0
        assert summary.qc_pass_rate == 0.0

    def test_batch_summary_data_to_dict(self):
        summary = BatchSummaryData(
            batch_id="abc",
            batch_name="Test",
            workflow="snv",
            total_jobs=5,
            completed_jobs=4,
            failed_jobs=1,
            pending_jobs=0,
            processing_jobs=0,
            mean_efficiency=None,
            total_snv_count=12,
            qc_pass_rate=0.8,
            mean_processing_seconds=45.0,
            created_at=datetime.utcnow(),
        )
        d = summary.to_dict()
        assert d["total_jobs"] == 5
        assert d["completed_jobs"] == 4
        assert d["total_snv_count"] == 12
        assert d["qc_pass_rate"] == 0.8


# ── Tests: generate_batch_csv_report ─────────────────────────────────────────

class TestGenerateBatchCsvReport:
    @pytest.mark.asyncio
    async def test_invalid_uuid_raises(self):
        session = AsyncMock()
        with pytest.raises(ValueError, match="Invalid batch_id"):
            await generate_batch_csv_report("not-a-uuid", session)

    @pytest.mark.asyncio
    async def test_batch_not_found_raises(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    with pytest.raises(ValueError, match="not found"):
                        await generate_batch_csv_report(str(uuid.uuid4()), session)

    @pytest.mark.asyncio
    async def test_snv_csv_has_correct_headers(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")
        jobs = [make_mock_job(result_json=make_snv_result(n_snvs=2))]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    csv_str = await generate_batch_csv_report(str(batch_id), session)

        reader = csv.DictReader(io.StringIO(csv_str))
        assert "chromosome" in reader.fieldnames
        assert "position" in reader.fieldnames
        assert "ref_allele" in reader.fieldnames
        assert "alt_allele" in reader.fieldnames
        assert "confidence_label" in reader.fieldnames

    @pytest.mark.asyncio
    async def test_snv_csv_row_count(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")
        jobs = [
            make_mock_job(result_json=make_snv_result(n_snvs=3)),
            make_mock_job(result_json=make_snv_result(n_snvs=2)),
        ]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    csv_str = await generate_batch_csv_report(str(batch_id), session)

        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 5  # 3 + 2 SNVs

    @pytest.mark.asyncio
    async def test_editing_csv_has_correct_headers(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="editing")
        jobs = [make_mock_job(result_json=make_editing_result())]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    csv_str = await generate_batch_csv_report(str(batch_id), session)

        reader = csv.DictReader(io.StringIO(csv_str))
        assert "editing_type" in reader.fieldnames
        assert "efficiency" in reader.fieldnames
        assert "tool_used" in reader.fieldnames

    @pytest.mark.asyncio
    async def test_editing_csv_one_row_per_job(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="editing")
        jobs = [
            make_mock_job(result_json=make_editing_result(efficiency=0.60)),
            make_mock_job(result_json=make_editing_result(efficiency=0.75)),
            make_mock_job(result_json=make_editing_result(efficiency=0.80)),
        ]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    csv_str = await generate_batch_csv_report(str(batch_id), session)

        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_returns_string(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = []
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    csv_str = await generate_batch_csv_report(str(batch_id), session)

        assert isinstance(csv_str, str)


# ── Tests: get_batch_qc_summary ───────────────────────────────────────────────

class TestGetBatchQcSummary:
    @pytest.mark.asyncio
    async def test_invalid_uuid_raises(self):
        session = AsyncMock()
        with pytest.raises(ValueError, match="Invalid batch_id"):
            await get_batch_qc_summary("not-a-uuid", session)

    @pytest.mark.asyncio
    async def test_batch_not_found_raises(self):
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    with pytest.raises(ValueError, match="not found"):
                        await get_batch_qc_summary(str(uuid.uuid4()), session)

    @pytest.mark.asyncio
    async def test_qc_pass_rate_calculation(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")
        jobs = [
            make_mock_job(result_json=make_snv_result(qc_pass=True)),
            make_mock_job(result_json=make_snv_result(qc_pass=True)),
            make_mock_job(result_json=make_snv_result(qc_pass=False)),
        ]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    qc = await get_batch_qc_summary(str(batch_id), session)

        assert abs(qc["qc_pass_rate"] - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_failed_samples_list(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")
        jobs = [
            make_mock_job(
                result_json={
                    "file_name": "good_sample.ab1",
                    "status": "completed",
                    "snv_calls": [],
                    "mean_quality": 35.0,
                    "qc_report": {"overall_pass": True, "flags": []},
                }
            ),
            make_mock_job(
                result_json={
                    "file_name": "bad_sample.ab1",
                    "status": "qc_failed",
                    "snv_calls": [],
                    "mean_quality": 10.0,
                    "qc_report": {"overall_pass": False, "flags": ["low_signal"]},
                }
            ),
        ]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    qc = await get_batch_qc_summary(str(batch_id), session)

        assert "bad_sample.ab1" in qc["failed_samples"]
        assert "good_sample.ab1" not in qc["failed_samples"]

    @pytest.mark.asyncio
    async def test_returns_required_keys(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = []
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    qc = await get_batch_qc_summary(str(batch_id), session)

        required_keys = {
            "batch_id", "total_samples", "passed_samples",
            "failed_samples", "qc_pass_rate", "mean_quality",
            "qc_flag_distribution",
        }
        assert required_keys.issubset(set(qc.keys()))

    @pytest.mark.asyncio
    async def test_mean_quality_computed(self):
        batch_id = uuid.uuid4()
        batch = make_mock_batch(batch_id=batch_id, workflow="snv")
        jobs = [
            make_mock_job(result_json=make_snv_result(mean_quality=30.0)),
            make_mock_job(result_json=make_snv_result(mean_quality=40.0)),
        ]

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = batch
            else:
                scalars = MagicMock()
                scalars.all.return_value = jobs
                result.scalars.return_value = scalars
            call_count[0] += 1
            return result

        session = AsyncMock()
        session.execute = mock_execute

        with patch("backend.services.batch_service.Batch"):
            with patch("backend.services.batch_service.Job"):
                with patch("backend.services.batch_service.select"):
                    qc = await get_batch_qc_summary(str(batch_id), session)

        assert abs(qc["mean_quality"] - 35.0) < 0.1
