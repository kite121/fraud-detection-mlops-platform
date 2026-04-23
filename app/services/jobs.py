from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select

from app.db import SessionLocal
from app.models import TrainingJob
from app.services.retry import retry_db


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_STARTED = "started"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"


@dataclass(slots=True)
class TrainingJobRecord:
    """Compact representation returned by the job service helpers."""

    job_id: str
    status: str
    batch_id: int | None
    dataset_version: str | None
    celery_task_id: str | None
    model_version: str | None


def _generate_job_id(prefix: str = "train") -> str:
    """Returns a stable, human-readable job identifier."""

    return f"{prefix}_{uuid4().hex[:12]}"


@retry_db
def create_training_job(
    *,
    batch_id: int | None,
    dataset_version: str | None,
    job_type: str = "training",
) -> TrainingJobRecord:
    """Creates a new queued job record before the Celery task is dispatched."""

    session = SessionLocal()

    try:
        entry = TrainingJob(
            job_id=_generate_job_id("train"),
            job_type=job_type,
            batch_id=batch_id,
            dataset_version=dataset_version,
            status=JOB_STATUS_QUEUED,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)

        return TrainingJobRecord(
            job_id=entry.job_id,
            status=entry.status,
            batch_id=entry.batch_id,
            dataset_version=entry.dataset_version,
            celery_task_id=entry.celery_task_id,
            model_version=entry.model_version,
        )
    finally:
        session.close()


def attach_celery_task_id(job_id: str, celery_task_id: str) -> None:
    """Stores the broker task id after a Celery task has been successfully enqueued."""

    session = SessionLocal()

    try:
        entry = _get_job_entry(session, job_id)
        entry.celery_task_id = celery_task_id
        session.commit()
    finally:
        session.close()


def mark_job_started(job_id: str) -> None:
    """Marks a queued job as started when the worker begins processing it."""

    session = SessionLocal()

    try:
        entry = _get_job_entry(session, job_id)
        entry.status = JOB_STATUS_STARTED
        entry.started_at = datetime.now(timezone.utc)
        entry.error_message = None
        session.commit()
    finally:
        session.close()


def mark_job_completed(job_id: str, *, model_version: str | None = None) -> None:
    """Marks a job as completed and stores the resulting model version when available."""

    session = SessionLocal()

    try:
        entry = _get_job_entry(session, job_id)
        entry.status = JOB_STATUS_COMPLETED
        entry.finished_at = datetime.now(timezone.utc)
        entry.error_message = None
        entry.model_version = model_version
        session.commit()
    finally:
        session.close()


def mark_job_failed(job_id: str, error_message: str) -> None:
    """Marks a job as failed and stores the error message for later inspection."""

    session = SessionLocal()

    try:
        entry = _get_job_entry(session, job_id)
        entry.status = JOB_STATUS_FAILED
        if entry.started_at is None:
            entry.started_at = datetime.now(timezone.utc)
        entry.finished_at = datetime.now(timezone.utc)
        entry.error_message = error_message
        session.commit()
    finally:
        session.close()


def get_job_by_id(job_id: str) -> TrainingJob:
    """Returns one training job by its job_id, or raises LookupError if not found."""

    session = SessionLocal()

    try:
        return _get_job_entry(session, job_id)
    finally:
        session.close()


def _get_job_entry(session, job_id: str) -> TrainingJob:
    """Loads one training job entry or raises a clear error if it does not exist."""

    statement = select(TrainingJob).where(TrainingJob.job_id == job_id)
    entry = session.execute(statement).scalar_one_or_none()

    if entry is None:
        raise LookupError(f"Training job with job_id={job_id!r} was not found.")

    return entry
