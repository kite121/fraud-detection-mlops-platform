from __future__ import annotations

from celery import Celery

from app.config import settings
from app.services.tracing import instrument_celery_app


celery_app = Celery(
    "fraud_detection_mlops_platform",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["workers.training_worker", "workers.retraining_worker"],
)

celery_app.conf.update(
    task_default_queue="training",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone="UTC",
    task_track_started=True,
    task_always_eager=settings.celery_task_always_eager,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
)

instrument_celery_app()
