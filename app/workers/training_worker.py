from __future__ import annotations

from app.celery_app import celery_app
from app.services.dataset import prepare_training_dataset
from app.services.evaluation import evaluate_model
from app.services.jobs import mark_job_completed, mark_job_failed, mark_job_started
from app.services.model_storage import save_model_artifacts
from app.services.registry import register_model_version
from app.services.training import train_model


@celery_app.task(
    bind=True,
    name="app.workers.training_worker.run_training_job",
)
def run_training_job(
    self,
    *,
    job_id: str,
    batch_id: int | None = None,
    dataset_version: str | None = None,
    notes: str | None = None,
) -> dict[str, str | float | None]:
    """Executes the existing Sprint 2 training pipeline in a background worker."""

    mark_job_started(job_id)

    try:
        prepared_dataset = prepare_training_dataset(
            batch_id=batch_id,
            dataset_version=dataset_version,
        )
        training_result = train_model(prepared_dataset)
        evaluation_result = evaluate_model(training_result, prepared_dataset)

        artifact_paths = save_model_artifacts(
            model=training_result.model,
            metrics=evaluation_result.to_dict(),
            model_version="pending",
            preprocessor=prepared_dataset.preprocessor,
        )

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

        mark_job_completed(job_id, model_version=registered_model.model_version)

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
    except Exception as error:
        mark_job_failed(job_id, str(error))
        raise
