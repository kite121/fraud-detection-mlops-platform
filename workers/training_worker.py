from __future__ import annotations

import time

from app.celery_app import celery_app
from app.services.events import publish_training_completed
from app.services.dataset import prepare_training_dataset
from app.services.evaluation import evaluate_model
from app.services.jobs import mark_job_completed, mark_job_failed, mark_job_running
from app.services.metrics import observe_training_duration
from app.services.model_storage import save_model_artifacts
from app.services.registry import register_model_version, update_model_artifact_paths
from app.services.tracing import get_tracer
from app.services.training import train_model


@celery_app.task(bind=True, name="workers.training_worker.train_model_task")
def train_model_task(self, batch_id: int | None, dataset_version: str | None, job_id: str):
    tracer = get_tracer(__name__)
    started_at = time.perf_counter()

    try:
        with tracer.start_as_current_span("train_model_task") as span:
            span.set_attribute("job.id", job_id)
            mark_job_running(job_id)

            with tracer.start_as_current_span("prepare_training_dataset"):
                prepared = prepare_training_dataset(
                    batch_id=batch_id,
                    dataset_version=dataset_version,
                )

            with tracer.start_as_current_span("train_model"):
                training_result = train_model(prepared)

            with tracer.start_as_current_span("evaluate_model"):
                eval_result = evaluate_model(training_result, prepared)

            with tracer.start_as_current_span("save_pending_artifacts"):
                artifact_paths = save_model_artifacts(
                    model=training_result.model,
                    metrics=eval_result.to_dict(),
                    model_version="pending",
                    preprocessor=prepared.preprocessor,
                )

            registered = register_model_version(
                dataset_version=prepared.metadata.dataset_version,
                training_batch_id=prepared.metadata.batch_id,
                primary_metric=eval_result.metrics.roc_auc,
                model_path=artifact_paths.model_path,
                metrics_path=artifact_paths.metrics_path,
                algorithm=type(training_result.model).__name__,
                hyperparameters=training_result.metadata.hyperparameters,
                notes=f"Async job {job_id}",
                mlflow_run_id=training_result.metadata.mlflow_run_id,
                model_uri=training_result.metadata.model_uri,
            )

            with tracer.start_as_current_span("save_versioned_artifacts"):
                final_artifact_paths = save_model_artifacts(
                    model=training_result.model,
                    metrics=eval_result.to_dict(),
                    model_version=registered.model_version,
                    preprocessor=prepared.preprocessor,
                )
            update_model_artifact_paths(
                registered.model_version,
                model_path=final_artifact_paths.model_path,
                metrics_path=final_artifact_paths.metrics_path,
            )

            mark_job_completed(
                job_id,
                model_version=registered.model_version,
                mlflow_run_id=training_result.metadata.mlflow_run_id,
            )

            publish_training_completed(
                model_version=registered.model_version,
                run_id=training_result.metadata.mlflow_run_id,
                primary_metric=registered.primary_metric,
                job_id=job_id,
            )
            observe_training_duration(time.perf_counter() - started_at)
    except Exception as error:
        mark_job_failed(job_id, str(error))
        raise
