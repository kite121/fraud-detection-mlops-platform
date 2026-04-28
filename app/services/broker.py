from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

import pika
from opentelemetry.trace import SpanKind

from app.config import settings
from app.services.tracing import extract_trace_context, get_tracer, inject_trace_headers


TRAINING_REQUESTED_QUEUE = "training_requested"
TRAINING_COMPLETED_QUEUE = "training_completed"
MODEL_DEPLOYED_QUEUE = "model_deployed"
RETRAINING_REQUESTED_QUEUE = "retraining_requested"
DATA_INGESTED_QUEUE = "data_ingested"

REQUIRED_QUEUES = (
    DATA_INGESTED_QUEUE,
    TRAINING_REQUESTED_QUEUE,
    TRAINING_COMPLETED_QUEUE,
    MODEL_DEPLOYED_QUEUE,
    RETRAINING_REQUESTED_QUEUE,
)


def get_connection() -> pika.BlockingConnection:
    return pika.BlockingConnection(pika.URLParameters(settings.rabbitmq_url))


def declare_queue(channel: pika.adapters.blocking_connection.BlockingChannel, queue_name: str) -> None:
    channel.queue_declare(queue=queue_name, durable=True)


def ensure_required_queues() -> None:
    connection = get_connection()
    try:
        channel = connection.channel()
        for queue_name in REQUIRED_QUEUES:
            declare_queue(channel, queue_name)
    finally:
        connection.close()


def _build_message(event_type: str, **payload: Any) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


def publish_message(
    queue_name: str,
    message: dict[str, Any],
    headers: dict[str, Any] | None = None,
) -> None:
    connection = get_connection()
    try:
        channel = connection.channel()
        declare_queue(channel, queue_name)
        tracer = get_tracer(__name__)
        trace_headers = inject_trace_headers(headers)
        with tracer.start_as_current_span(
            f"publish {queue_name}",
            kind=SpanKind.PRODUCER,
            attributes={"messaging.destination": queue_name},
        ):
            channel.basic_publish(
                exchange="",
                routing_key=queue_name,
                body=json.dumps(message).encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type="application/json",
                    headers=trace_headers,
                ),
            )
    finally:
        connection.close()


def consume_queue(
    queue_name: str,
    handler: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    reconnect_delay_seconds = 1.0

    while True:
        connection: pika.BlockingConnection | None = None

        try:
            connection = get_connection()
            channel = connection.channel()
            declare_queue(channel, queue_name)
            channel.basic_qos(prefetch_count=1)

            def callback(
                ch: pika.adapters.blocking_connection.BlockingChannel,
                method: Any,
                properties: pika.BasicProperties,
                body: bytes,
            ) -> None:
                message = json.loads(body.decode("utf-8"))
                tracer = get_tracer(__name__)
                context = extract_trace_context(properties.headers or {})
                try:
                    with tracer.start_as_current_span(
                        f"consume {queue_name}",
                        context=context,
                        kind=SpanKind.CONSUMER,
                        attributes={"messaging.destination": queue_name},
                    ):
                        handler(message, properties.headers or {})
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as error:
                    if ch.is_open:
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                    print(
                        f"[Broker] Handler error in queue {queue_name!r}: "
                        f"{type(error).__name__}: {error}"
                    )

            channel.basic_consume(queue=queue_name, on_message_callback=callback)
            print(f"[Broker] Consumer started for queue {queue_name!r}.")
            reconnect_delay_seconds = 1.0
            channel.start_consuming()
        except Exception as error:
            print(
                f"[Broker] Consumer for queue {queue_name!r} disconnected: "
                f"{type(error).__name__}: {error}. "
                f"Reconnecting in {reconnect_delay_seconds:.1f}s."
            )
            time.sleep(reconnect_delay_seconds)
            reconnect_delay_seconds = min(reconnect_delay_seconds * 2, 30.0)
        finally:
            if connection is not None and connection.is_open:
                connection.close()

def publish_training_requested(
    batch_id: int | None,
    dataset_version: str | None,
    job_id: str | None = None,
) -> None:
    publish_message(
        TRAINING_REQUESTED_QUEUE,
        _build_message(
            "training_requested",
            batch_id=batch_id,
            dataset_version=dataset_version,
            job_id=job_id,
        ),
    )


def publish_data_ingested(
    batch_id: int,
    dataset_version: str,
    client_id: str | None = None,
) -> None:
    publish_message(
        DATA_INGESTED_QUEUE,
        _build_message(
            "data_ingested",
            batch_id=batch_id,
            dataset_version=dataset_version,
            client_id=client_id,
        ),
    )


def publish_training_completed(
    model_version: str,
    run_id: str | None,
    primary_metric: float,
    job_id: str | None = None,
) -> None:
    publish_message(
        TRAINING_COMPLETED_QUEUE,
        _build_message(
            "training_completed",
            model_version=model_version,
            mlflow_run_id=run_id,
            primary_metric=primary_metric,
            job_id=job_id,
        ),
    )


def publish_model_deployed(model_version: str, service: str = "inference-service") -> None:
    publish_message(
        MODEL_DEPLOYED_QUEUE,
        _build_message(
            "model_deployed",
            model_version=model_version,
            service=service,
        ),
    )


def publish_retraining_requested(
    reason: str,
    dataset_version: str | None = None,
    batch_id: int | None = None,
    job_id: str | None = None,
) -> None:
    publish_message(
        RETRAINING_REQUESTED_QUEUE,
        _build_message(
            "retraining_requested",
            reason=reason,
            dataset_version=dataset_version,
            batch_id=batch_id,
            job_id=job_id,
        ),
    )
