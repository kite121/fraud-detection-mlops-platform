from __future__ import annotations

"""
Event publisher for model lifecycle events.

Publishes to RabbitMQ using the standard "pika" library.
If RabbitMQ is unavailable (e.g. during local development without the broker),
the publish call logs a warning and continues: it does NOT crash the training
pipeline. Events are best-effort notifications, not critical path operations.

Expected events:
    training_requested    —  published by "POST /train" when a job is queued
    training_completed    —  published by the worker after successful training
    model_deployed        —  published when a new best model is registered
    retraining_requested  —  published by the monitoring/drift service
"""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

EVENT_TRAINING_REQUESTED = "training_requested"
EVENT_TRAINING_COMPLETED = "training_completed"
EVENT_MODEL_DEPLOYED = "model_deployed"
EVENT_RETRAINING_REQUESTED = "retraining_requested"


def _get_rabbitmq_url() -> str:
    """Reads the broker URL from the environment."""
    return os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")


def _build_event_payload(event_name: str, data: dict[str, Any]) -> bytes:
    """Wraps event data in a standard envelope and serialises it to JSON bytes."""
    envelope = {
        "event": event_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "data": data,
    }
    return json.dumps(envelope, default=str).encode()


def publish_event(event_name: str, data: dict[str, Any]) -> bool:
    """
    Publishes one event to a RabbitMQ fanout exchange named after the event.

    Returns True on success, False if the broker is unavailable.
    Never raises: callers should treat failed publishes as non-fatal warnings.
    """
    try:
        import pika
    except ImportError:
        logger.warning(
            "[Events] pika is not installed: event %r not published. "
            "Add 'pika' to requirements.txt.",
            event_name,
        )
        return False

    url = _get_rabbitmq_url()
    payload = _build_event_payload(event_name, data)

    try:
        params = pika.URLParameters(url)
        params.socket_timeout = 3
        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        channel.exchange_declare(
            exchange=event_name,
            exchange_type="fanout",
            durable=True,
        )
        channel.basic_publish(
            exchange=event_name,
            routing_key="",
            body=payload,
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
        connection.close()

        logger.info("[Events] Published %r: %s", event_name, data)
        return True

    except Exception as exc:
        logger.warning(
            "[Events] Could not publish %r to RabbitMQ (%s: %s). "
            "The training pipeline continues without this notification.",
            event_name,
            type(exc).__name__,
            exc,
        )
        return False


def publish_training_requested(
    job_id: str,
    batch_id: int | None,
    dataset_version: str | None,
) -> bool:
    return publish_event(
        EVENT_TRAINING_REQUESTED,
        {
            "job_id": job_id,
            "batch_id": batch_id,
            "dataset_version": dataset_version,
        },
    )


def publish_training_completed(
    job_id: str,
    model_version: str,
    primary_metric: float,
) -> bool:
    return publish_event(
        EVENT_TRAINING_COMPLETED,
        {
            "job_id": job_id,
            "model_version": model_version,
            "primary_metric": primary_metric,
        },
    )


def publish_model_deployed(
    model_version: str,
    primary_metric: float,
    model_path: str | None,
    mlflow_run_id: str | None,
) -> bool:
    return publish_event(
        EVENT_MODEL_DEPLOYED,
        {
            "model_version": model_version,
            "primary_metric": primary_metric,
            "model_path": model_path,
            "mlflow_run_id": mlflow_run_id,
        },
    )


def publish_retraining_requested(
    reason: str,
    dataset_version: str | None = None,
) -> bool:
    return publish_event(
        EVENT_RETRAINING_REQUESTED,
        {
            "reason": reason,
            "dataset_version": dataset_version,
        },
    )
