from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.db import SessionLocal
from app.models import TrainingJob


def create_job(
    *,
    job_type: str,
    dataset_version: str | None = None,
    batch_id: int | None = None,
) -> TrainingJob:
    job = TrainingJob(
        job_id=str(uuid.uuid4()),
        job_type=job_type,
        dataset_version=dataset_version or "unknown",
        batch_id=batch_id,
        status="queued",
        created_at=datetime.now(timezone.utc),
    )

    with SessionLocal() as db:
        db.add(job)
        db.commit()
        db.refresh(job)

    return job


def mark_job_running(job_id: str) -> TrainingJob | None:
    with SessionLocal() as db:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job is None:
            return None

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(job)
        return job


def mark_job_completed(
    job_id: str,
    *,
    model_version: str | None = None,
    mlflow_run_id: str | None = None,
) -> TrainingJob | None:
    with SessionLocal() as db:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job is None:
            return None

        job.status = "completed"
        job.finished_at = datetime.now(timezone.utc)
        job.model_version = model_version
        job.mlflow_run_id = mlflow_run_id
        db.commit()
        db.refresh(job)
        return job


def mark_job_failed(job_id: str, error_message: str) -> TrainingJob | None:
    with SessionLocal() as db:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job is None:
            return None

        job.status = "failed"
        job.error_message = error_message
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(job)
        return job
