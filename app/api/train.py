from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.jobs import attach_celery_task_id, create_training_job, mark_job_failed
from app.services.metadata import get_batch_metadata, get_latest_batch_metadata
from app.workers.training_worker import run_training_job


router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """
    Parameters for launching a training run.

    Provide either batch_id (specific ingestion batch) or dataset_version
    (the platform picks the latest uploaded batch for that version).
    At least one of the two must be set.
    """

    batch_id: int | None = Field(
        default=None,
        description="ID of the ingested batch to train on (from batch_metadata table).",
    )
    dataset_version: str | None = Field(
        default=None,
        description="Dataset version tag; uses the newest uploaded batch if batch_id is omitted.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional free-text notes attached to this training run.",
    )


class TrainResponse(BaseModel):
    """Acknowledgement returned after the training task has been queued."""

    status: str
    job_id: str
    message: str


# ---------------------------------------------------------------------------
# Endpoint — Task 7
# ---------------------------------------------------------------------------

@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    """
    Validates the training target, creates a queued job in PostgreSQL,
    dispatches a Celery task, and returns the new job identifier.
    """
    if request.batch_id is None and request.dataset_version is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: batch_id, dataset_version.",
        )

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
