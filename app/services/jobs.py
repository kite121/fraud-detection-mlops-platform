from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select

from app.db import SessionLocal
from app.services.metrics import (
    observe_training_job_completed,
    observe_training_job_failed,
    observe_training_job_queued,
)
from app.models import TrainingJob


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"


@dataclass(slots=True)
class TrainingJobRecord:
    job_id: str
    status: str
    batch_id: int | None
    dataset_version: str | None
    celery_task_id: str | None
    model_version: str | None
    mlflow_run_id: str | None


def _generate_job_id(prefix: str = "train") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def create_training_job(
    *,
    batch_id: int | None,
    dataset_version: str | None,
    job_type: str = "training",
) -> TrainingJobRecord:
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
        observe_training_job_queued()

        return TrainingJobRecord(
            job_id=entry.job_id,
            status=entry.status,
            batch_id=entry.batch_id,
            dataset_version=entry.dataset_version,
            celery_task_id=entry.celery_task_id,
            model_version=entry.model_version,
            mlflow_run_id=entry.mlflow_run_id,
        )
    finally:
        session.close()


def create_job(
    *,
    job_type: str,
    dataset_version: str | None = None,
    batch_id: int | None = None,
) -> TrainingJob:
    session = SessionLocal()
    try:
        entry = TrainingJob(
            job_id=str(uuid4()),
            job_type=job_type,
            dataset_version=dataset_version or "unknown",
            batch_id=batch_id,
            status=JOB_STATUS_QUEUED,
            created_at=datetime.now(timezone.utc),
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry
    finally:
        session.close()


def attach_celery_task_id(job_id: str, celery_task_id: str) -> None:
    session = SessionLocal()
    try:
        entry = _get_job_entry(session, job_id)
        entry.celery_task_id = celery_task_id
        session.commit()
    finally:
        session.close()


def mark_job_started(job_id: str) -> None:
    session = SessionLocal()
    try:
        entry = _get_job_entry(session, job_id)
        entry.status = JOB_STATUS_RUNNING
        entry.started_at = datetime.now(timezone.utc)
        entry.error_message = None
        session.commit()
    finally:
        session.close()


def mark_job_running(job_id: str) -> TrainingJob | None:
    session = SessionLocal()
    try:
        job = session.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job is None:
            return None

        job.status = JOB_STATUS_RUNNING
        job.started_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(job)
        return job
    finally:
        session.close()


def mark_job_completed(
    job_id: str,
    *,
    model_version: str | None = None,
    mlflow_run_id: str | None = None,
) -> None:
    session = SessionLocal()
    try:
        entry = _get_job_entry(session, job_id)
        entry.status = JOB_STATUS_COMPLETED
        entry.finished_at = datetime.now(timezone.utc)
        entry.error_message = None
        entry.model_version = model_version
        entry.mlflow_run_id = mlflow_run_id
        session.commit()
        observe_training_job_completed()
    finally:
        session.close()


def mark_job_failed(job_id: str, error_message: str) -> None:
    session = SessionLocal()
    try:
        entry = _get_job_entry(session, job_id)
        if entry.started_at is None:
            entry.started_at = datetime.now(timezone.utc)
        entry.status = JOB_STATUS_FAILED
        entry.finished_at = datetime.now(timezone.utc)
        entry.error_message = error_message
        session.commit()
        observe_training_job_failed()
    finally:
        session.close()


def get_job_by_id(job_id: str) -> TrainingJob:
    session = SessionLocal()
    try:
        return _get_job_entry(session, job_id)
    finally:
        session.close()


def _get_job_entry(session, job_id: str) -> TrainingJob:
    statement = select(TrainingJob).where(TrainingJob.job_id == job_id)
    entry = session.execute(statement).scalar_one_or_none()

    if entry is None:
        raise LookupError(f"Training job with job_id={job_id!r} was not found.")

    return entry
