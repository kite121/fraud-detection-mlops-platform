from celery import Celery

from app.config import settings
from app.services.tracing import instrument_celery_app


celery_app = Celery(
    "fraud_detection",
    broker=settings.rabbitmq_url,
    backend="rpc://",
    include=["workers.training_worker", "workers.retraining_worker"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
)

instrument_celery_app()
