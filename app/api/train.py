from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.celery_app import celery_app
from app.schemas import TrainEnqueueResponse
from app.services.events import publish_training_requested
from app.services.jobs import create_job, mark_job_failed
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
        raise HTTPException(400, "Either batch_id or dataset_version must be provided")

    with tracer.start_as_current_span("create_training_job") as span:
        job = create_job(
            job_type="training",
            dataset_version=request.dataset_version,
            batch_id=request.batch_id,
        )
        job_id = job.job_id
        span.set_attribute("job.id", job_id)

    try:
        with tracer.start_as_current_span("enqueue_training_task") as span:
            span.set_attribute("job.id", job_id)
            celery_app.send_task(
                TRAIN_MODEL_TASK_NAME,
                args=[request.batch_id, request.dataset_version, job_id],
            )
            publish_training_requested(
                batch_id=request.batch_id,
                dataset_version=request.dataset_version,
                job_id=job_id,
            )
    except Exception as error:
        mark_job_failed(job_id, str(error))
        raise HTTPException(500, f"Failed to enqueue training job: {error}") from error

    return TrainEnqueueResponse(
        status="queued",
        job_id=job_id,
        message="training task accepted",
    )
