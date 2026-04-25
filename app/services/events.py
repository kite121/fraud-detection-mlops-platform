from __future__ import annotations

from typing import Any

from app.services.broker import (
    MODEL_DEPLOYED_QUEUE,
    REQUIRED_QUEUES,
    RETRAINING_REQUESTED_QUEUE,
    TRAINING_COMPLETED_QUEUE,
    TRAINING_REQUESTED_QUEUE,
    consume_queue,
    declare_queue,
    ensure_required_queues,
    get_connection,
    publish_message as _publish_message,
)


EVENT_TRAINING_REQUESTED = TRAINING_REQUESTED_QUEUE
EVENT_TRAINING_COMPLETED = TRAINING_COMPLETED_QUEUE
EVENT_MODEL_DEPLOYED = MODEL_DEPLOYED_QUEUE
EVENT_RETRAINING_REQUESTED = RETRAINING_REQUESTED_QUEUE


def publish_event(event_name: str, data: dict[str, Any]) -> bool:
    """
    Best-effort event publishing wrapper.

    Returns True on success and False on failure so callers can treat
    event delivery as a non-critical notification channel.
    """
    try:
        _publish_message(
            event_name,
            {
                "event_type": event_name,
                **data,
            },
        )
        return True
    except Exception:
        return False


def publish_training_requested(
    batch_id: int | None,
    dataset_version: str | None,
    job_id: str | None = None,
) -> bool:
    return publish_event(
        EVENT_TRAINING_REQUESTED,
        {
            "batch_id": batch_id,
            "dataset_version": dataset_version,
            "job_id": job_id,
        },
    )


def publish_training_completed(
    model_version: str,
    run_id: str | None = None,
    primary_metric: float | None = None,
    job_id: str | None = None,
) -> bool:
    return publish_event(
        EVENT_TRAINING_COMPLETED,
        {
            "model_version": model_version,
            "mlflow_run_id": run_id,
            "primary_metric": primary_metric,
            "job_id": job_id,
        },
    )


def publish_model_deployed(
    model_version: str,
    service: str = "inference-service",
    primary_metric: float | None = None,
    model_path: str | None = None,
    mlflow_run_id: str | None = None,
) -> bool:
    return publish_event(
        EVENT_MODEL_DEPLOYED,
        {
            "model_version": model_version,
            "service": service,
            "primary_metric": primary_metric,
            "model_path": model_path,
            "mlflow_run_id": mlflow_run_id,
        },
    )


def publish_retraining_requested(
    reason: str,
    dataset_version: str | None = None,
    batch_id: int | None = None,
    job_id: str | None = None,
) -> bool:
    return publish_event(
        EVENT_RETRAINING_REQUESTED,
        {
            "reason": reason,
            "dataset_version": dataset_version,
            "batch_id": batch_id,
            "job_id": job_id,
        },
    )
