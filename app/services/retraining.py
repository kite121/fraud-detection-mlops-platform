from __future__ import annotations

from app.celery_app import celery_app
from app.services.events import publish_retraining_requested
from app.services.jobs import create_job, mark_job_failed


RETRAIN_MODEL_TASK_NAME = "workers.retraining_worker.retrain_model_task"


def trigger_retraining(
    *,
    reason: str,
    dataset_version: str | None = None,
    batch_id: int | None = None,
) -> str:
    job = create_job(
        job_type="retraining",
        dataset_version=dataset_version,
        batch_id=batch_id,
    )

    try:
        celery_app.send_task(
            RETRAIN_MODEL_TASK_NAME,
            args=[batch_id, dataset_version, job.job_id, reason],
        )
        publish_retraining_requested(
            reason=reason,
            dataset_version=dataset_version,
            batch_id=batch_id,
            job_id=job.job_id,
        )
    except Exception as error:
        mark_job_failed(job.job_id, str(error))
        raise

    return job.job_id
