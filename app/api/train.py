from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.celery_app import celery_app
from app.schemas import TrainEnqueueResponse
from app.services.events import publish_training_requested
from app.services.jobs import attach_celery_task_id, create_training_job, mark_job_failed
from app.services.metadata import get_batch_metadata, get_latest_batch_metadata
from app.services.tracing import get_tracer


TRAIN_MODEL_TASK_NAME = "workers.training_worker.train_model_task"

router = APIRouter()


class TrainRequest(BaseModel):
    batch_id: int | None = Field(None, description="ID of ingested batch")
    dataset_version: str | None = Field(None, description="Dataset version tag")
    notes: str | None = Field(None, description="Optional notes")


@router.post("/train", response_model=TrainEnqueueResponse)
def start_training(request: TrainRequest) -> TrainEnqueueResponse:
    tracer = get_tracer(__name__)

    if request.batch_id is None and request.dataset_version is None:
        raise HTTPException(
            status_code=400,
            detail="Either batch_id or dataset_version must be provided",
        )

    try:
        if request.batch_id is not None:
            target_batch = get_batch_metadata(request.batch_id)
        else:
            target_batch = get_latest_batch_metadata(
                dataset_version=request.dataset_version
            )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    with tracer.start_as_current_span("create_training_job") as span:
        try:
            job = create_training_job(
                batch_id=target_batch.id,
                dataset_version=target_batch.dataset_version,
                job_type="training",
            )
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create training job: {error}",
            ) from error

        job_id = job.job_id
        span.set_attribute("job.id", job_id)

    try:
        with tracer.start_as_current_span("enqueue_training_task") as span:
            span.set_attribute("job.id", job_id)

            async_result = celery_app.send_task(
                TRAIN_MODEL_TASK_NAME,
                args=[target_batch.id, target_batch.dataset_version, job_id],
            )
            attach_celery_task_id(job_id, async_result.id)

            publish_training_requested(
                batch_id=target_batch.id,
                dataset_version=target_batch.dataset_version,
                job_id=job_id,
            )
    except Exception as error:
        mark_job_failed(job_id, f"Failed to enqueue training job: {error}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue training job: {error}",
        ) from error

    return TrainEnqueueResponse(
        status="queued",
        job_id=job_id,
        message="training task accepted",
    )
