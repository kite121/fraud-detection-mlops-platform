from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.events import publish_training_requested
from app.services.jobs import attach_celery_task_id, create_training_job, mark_job_failed
from app.services.metadata import get_batch_metadata, get_latest_batch_metadata
from app.workers.training_worker import run_training_job

router = APIRouter()


class TrainRequest(BaseModel):
    batch_id: int | None = Field(None, description="ID of ingested batch")
    dataset_version: str | None = Field(None, description="Dataset version tag")
    notes: str | None = Field(None, description="Optional notes")


class TrainResponse(BaseModel):
    """Acknowledgement returned after the training task has been queued."""

    status: str
    job_id: str
    message: str


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    """
    Validates the training target, creates a queued job in PostgreSQL,
    dispatches a Celery task, and returns the new job identifier.
    """
    if request.batch_id is None and request.dataset_version is None:
        raise HTTPException(400, "Either batch_id or dataset_version must be provided")

    try:
        if request.batch_id is not None:
            target_batch = get_batch_metadata(request.batch_id)
        else:
            target_batch = get_latest_batch_metadata(dataset_version=request.dataset_version)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        job = create_training_job(
            batch_id=target_batch.id,
            dataset_version=target_batch.dataset_version,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create training job: {exc}",
        ) from exc

    try:
        async_result = run_training_job.delay(
            job_id=job.job_id,
            batch_id=target_batch.id,
            dataset_version=target_batch.dataset_version,
            notes=request.notes,
        )
        attach_celery_task_id(job.job_id, async_result.id)

        publish_training_requested(
            batch_id=target_batch.id,
            dataset_version=target_batch.dataset_version,
            job_id=job.job_id,
        )
    except Exception as exc:
        mark_job_failed(job.job_id, f"Failed to enqueue training task: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue training task: {exc}",
        ) from exc

    return TrainResponse(
        status="queued",
        job_id=job.job_id,
        message="training task accepted",
    )
