from __future__ import annotations

from app.celery_app import celery_app
from app.services.dataset import prepare_training_dataset
from app.services.evaluation import evaluate_model
from app.services.events import publish_model_deployed, publish_training_completed
from app.services.jobs import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_STARTED,
    get_job_by_id,
    mark_job_completed,
    mark_job_failed,
    mark_job_started,
)
from app.services.model_storage import save_model_artifacts
from app.services.registry import register_model_version
from app.services.training import train_model


@celery_app.task(
    bind=True,
    name="app.workers.training_worker.run_training_job",
    # Sprint 3 Task 12: retry up to 3 times on unexpected errors, with exponential backoff.
    # Does NOT retry on LookupError or ValueError (those are data/config problems).
    autoretry_for=(Exception,),
    dont_autoretry_for=(LookupError, ValueError),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
def run_training_job(
    self,
    *,
    job_id: str,
    batch_id: int | None = None,
    dataset_version: str | None = None,
    notes: str | None = None,
) -> dict[str, str | float | None]:
    """
    Executes the full training pipeline in a background worker.

    Sprint 3 Task 12 -- Idempotency:
        Before doing any work the task checks the current job status.
        If the job is already "started", "completed", or "failed" it means
        Celery retried the task after a transient broker error. In that case
        the task returns early to avoid duplicate training runs.
    """

    # -+--  Sprint 3 Task 12 -- Idempotency guard: skip duplicate executions  --+-
    try:
        existing_job = get_job_by_id(job_id)
        if existing_job.status in (JOB_STATUS_STARTED, JOB_STATUS_COMPLETED, JOB_STATUS_FAILED):
            print(
                f"[Worker] Job {job_id!r} is already in status "
                f"{existing_job.status!r} — skipping duplicate execution."
            )
            return {
                "job_id": job_id,
                "status": existing_job.status,
                "model_version": existing_job.model_version,
                "skipped": True,
            }
    except LookupError:
        # Job record disappeared — nothing to protect against, proceed normally.
        pass

    mark_job_started(job_id)

    try:
        # Step 1: Prepare dataset
        prepared_dataset = prepare_training_dataset(
            batch_id=batch_id,
            dataset_version=dataset_version,
        )

        # Step 2: Train
        training_result = train_model(prepared_dataset)

        # Step 3: Evaluate
        evaluation_result = evaluate_model(training_result, prepared_dataset)

        # Step 4: Save artifacts under a preliminary name, then re-upload
        #         under the real versioned name once the registry assigns it
        artifact_paths = save_model_artifacts(
            model=training_result.model,
            metrics=evaluation_result.to_dict(),
            model_version="pending",
            preprocessor=prepared_dataset.preprocessor,
        )

        # Step 5: Register in PostgreSQL
        registered_model = register_model_version(
            dataset_version=prepared_dataset.metadata.dataset_version,
            training_batch_id=prepared_dataset.metadata.batch_id,
            primary_metric=evaluation_result.metrics.roc_auc,
            model_path=artifact_paths.model_path,
            metrics_path=artifact_paths.metrics_path,
            algorithm=training_result.metadata.model_name,
            hyperparameters=training_result.metadata.hyperparameters,
            notes=notes,
            mlflow_run_id=training_result.metadata.mlflow_run_id,
            model_uri=training_result.metadata.model_uri,
        )

        # Step 6: Re-upload artifacts under the real versioned path
        try:
            final_artifact_paths = save_model_artifacts(
                model=training_result.model,
                metrics=evaluation_result.to_dict(),
                model_version=registered_model.model_version,
                preprocessor=prepared_dataset.preprocessor,
            )
        except Exception as artifact_error:
            print(
                "[Worker] Warning: could not re-upload versioned artifacts "
                f"for job {job_id}: {artifact_error}"
            )
            final_artifact_paths = artifact_paths

        # Step 7: Mark job complete
        mark_job_completed(job_id, model_version=registered_model.model_version)

        # -+--  Sprint 3 Task 7:  Publish lifecycle events (best-effort, never fatal)  --+-
        publish_training_completed(
            job_id=job_id,
            model_version=registered_model.model_version,
            primary_metric=registered_model.primary_metric,
        )

        if registered_model.status == "best":
            # Notify inference-service that a better model is available.
            publish_model_deployed(
                model_version=registered_model.model_version,
                primary_metric=registered_model.primary_metric,
                model_path=final_artifact_paths.model_path,
                mlflow_run_id=registered_model.mlflow_run_id,
            )

        return {
            "job_id": job_id,
            "status": "completed",
            "model_version": registered_model.model_version,
            "model_status": registered_model.status,
            "primary_metric": registered_model.primary_metric,
            "model_path": final_artifact_paths.model_path,
            "metrics_path": final_artifact_paths.metrics_path,
            "mlflow_run_id": registered_model.mlflow_run_id,
            "model_uri": registered_model.model_uri,
        }

    except (LookupError, ValueError) as error:
        # Data/config errors — don't retry, mark failed immediately.
        mark_job_failed(job_id, str(error))
        raise

    except Exception as error:
        # Transient errors — Celery will retry (see task decorator above).
        # Only mark as failed on the last retry attempt.
        if self.request.retries >= self.max_retries:
            mark_job_failed(job_id, str(error))
        raise
