from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.jobs import get_job_by_id

router = APIRouter()


class TrainingJobResponse(BaseModel):
    """Full status information about one training or retraining job."""

    job_id: str
    job_type: str
    status: str
    dataset_version: str | None
    batch_id: int | None
    celery_task_id: str | None
    model_version: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None

    model_config = {"from_attributes": True}


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_job(job_id: str) -> TrainingJobResponse:
    """
    Returns the current status and metadata of one training or retraining job.

    Possible status values:
        queued     —  job has been created and is waiting for a worker
        started    —  a worker is actively processing the job
        completed  —  training finished successfully; model_version is set
        failed     —  something went wrong; error_message contains the reason
    """
    try:
        job = get_job_by_id(job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return TrainingJobResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        dataset_version=job.dataset_version,
        batch_id=job.batch_id,
        celery_task_id=job.celery_task_id,
        model_version=job.model_version,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
    )
