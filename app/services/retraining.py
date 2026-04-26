from __future__ import annotations

import threading
from dataclasses import dataclass

from app.celery_app import celery_app
from app.services.events import (
    DATA_INGESTED_QUEUE,
    consume_queue,
    publish_retraining_requested,
)
from app.services.jobs import attach_celery_task_id, create_training_job, mark_job_failed
from app.services.dataset import prepare_training_dataset
from app.services.evaluation import evaluate_model
from app.services.model_storage import save_model_artifacts
from app.services.monitoring import MonitoringRunResult, monitor_model
from app.services.registry import RegisteredModel, get_best_model, register_model_version
from app.services.tracing import get_tracer
from app.services.training import TrainedModelResult, train_model

RETRAIN_MODEL_TASK_NAME = "workers.retraining_worker.retrain_model_task"
_data_ingested_listener_started = False


@dataclass(slots=True)
class RetrainingResult:
    """Structured output returned by trigger_retraining()."""

    triggered: bool
    reason: str
    reference_batch_id: int
    current_batch_id: int
    drift_result_path: str
    reference_profile_path: str
    model_version: str | None = None
    model_status: str | None = None
    primary_metric: float | None = None
    model_path: str | None = None
    metrics_path: str | None = None
    mlflow_run_id: str | None = None
    model_uri: str | None = None


def _run_training_flow_for_batch(
    *,
    batch_id: int,
    notes: str | None = None,
) -> tuple[RegisteredModel, str, str]:
    """Runs the existing training pipeline end-to-end for one selected batch."""

    prepared_dataset = prepare_training_dataset(batch_id=batch_id)
    training_result: TrainedModelResult = train_model(prepared_dataset)
    evaluation_result = evaluate_model(training_result, prepared_dataset)

    pending_artifact_paths = save_model_artifacts(
        model=training_result.model,
        metrics=evaluation_result.to_dict(),
        model_version="pending",
        preprocessor=prepared_dataset.preprocessor,
    )

    registered_model = register_model_version(
        dataset_version=prepared_dataset.metadata.dataset_version,
        training_batch_id=prepared_dataset.metadata.batch_id,
        primary_metric=evaluation_result.metrics.roc_auc,
        model_path=pending_artifact_paths.model_path,
        metrics_path=pending_artifact_paths.metrics_path,
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
        model_path = final_artifact_paths.model_path
        metrics_path = final_artifact_paths.metrics_path
    except Exception as artifact_error:
        print(
            "[Retraining] Warning: could not re-upload versioned artifacts "
            f"for batch {batch_id}: {artifact_error}"
        )
        model_path = pending_artifact_paths.model_path
        metrics_path = pending_artifact_paths.metrics_path

    return registered_model, model_path, metrics_path


def _enqueue_retraining_job(
    *,
    batch_id: int,
    dataset_version: str | None,
    reason: str,
) -> str:
    """Creates one retraining job record and dispatches the Celery task."""

    job = create_training_job(
        batch_id=batch_id,
        dataset_version=dataset_version,
        job_type="retraining",
    )

    try:
        async_result = celery_app.send_task(
            RETRAIN_MODEL_TASK_NAME,
            args=[batch_id, dataset_version, job.job_id, reason],
        )
        attach_celery_task_id(job.job_id, async_result.id)
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


def _handle_data_ingested_event(message: dict, _headers: dict) -> None:
    """
    Reacts to a newly ingested batch:
      1. look up the current best model
      2. compare the new batch against the best model's training batch
      3. enqueue retraining if drift exceeds the threshold
    """

    tracer = get_tracer(__name__)
    batch_id = int(message["batch_id"])
    dataset_version = message.get("dataset_version")

    with tracer.start_as_current_span("handle_data_ingested_event") as span:
        span.set_attribute("batch.id", batch_id)
        if dataset_version is not None:
            span.set_attribute("dataset.version", dataset_version)

        try:
            current_best = get_best_model()
        except LookupError:
            print(
                "[Retraining] Skipping auto-retraining check: "
                "no best model exists yet."
            )
            return

        reference_batch_id = current_best.training_batch_id
        if reference_batch_id == batch_id:
            print(
                "[Retraining] Skipping auto-retraining check: "
                "new batch matches the current reference batch."
            )
            return

        monitoring_result: MonitoringRunResult = monitor_model(
            reference_batch_id=reference_batch_id,
            current_batch_id=batch_id,
            dataset_version=dataset_version,
        )

        if not monitoring_result.drift_result.degraded:
            print(
                "[Retraining] Drift below threshold; "
                f"no retraining needed for batch {batch_id}."
            )
            return

        reason = (
            "auto-retraining triggered after data_ingested event "
            f"(reference_batch_id={reference_batch_id}, current_batch_id={batch_id})"
        )
        job_id = _enqueue_retraining_job(
            batch_id=batch_id,
            dataset_version=dataset_version,
            reason=reason,
        )
        print(
            "[Retraining] Drift exceeded threshold; "
            f"queued retraining job {job_id!r} for batch {batch_id}."
        )


def start_data_ingested_listener() -> None:
    """Starts one background consumer that reacts to data_ingested events."""
    global _data_ingested_listener_started

    if _data_ingested_listener_started:
        return

    _data_ingested_listener_started = True
    listener_thread = threading.Thread(
        target=consume_queue,
        args=(DATA_INGESTED_QUEUE, _handle_data_ingested_event),
        name="data-ingested-listener",
        daemon=True,
    )
    listener_thread.start()
    print("[Retraining] Background listener started for data_ingested events.")

def trigger_retraining(
    *,
    reference_batch_id: int,
    current_batch_id: int | None = None,
    dataset_version: str | None = None,
    notes: str | None = None,
) -> RetrainingResult:
    """Runs monitoring first and automatically retrains when drift exceeds the threshold."""

    monitoring_result: MonitoringRunResult = monitor_model(
        reference_batch_id=reference_batch_id,
        current_batch_id=current_batch_id,
        dataset_version=dataset_version,
    )

    detected_current_batch_id = monitoring_result.drift_result.current_batch_id

    if not monitoring_result.drift_result.degraded:
        return RetrainingResult(
            triggered=False,
            reason="drift_threshold_not_exceeded",
            reference_batch_id=reference_batch_id,
            current_batch_id=detected_current_batch_id,
            drift_result_path=monitoring_result.drift_result_path,
            reference_profile_path=monitoring_result.reference_profile_path,
        )

    retraining_notes = notes or (
        "Auto-retraining triggered by drift detection "
        f"(reference_batch_id={reference_batch_id}, current_batch_id={detected_current_batch_id})."
    )
    registered_model, model_path, metrics_path = _run_training_flow_for_batch(
        batch_id=detected_current_batch_id,
        notes=retraining_notes,
    )

    return RetrainingResult(
        triggered=True,
        reason="drift_threshold_exceeded",
        reference_batch_id=reference_batch_id,
        current_batch_id=detected_current_batch_id,
        drift_result_path=monitoring_result.drift_result_path,
        reference_profile_path=monitoring_result.reference_profile_path,
        model_version=registered_model.model_version,
        model_status=registered_model.status,
        primary_metric=registered_model.primary_metric,
        model_path=model_path,
        metrics_path=metrics_path,
        mlflow_run_id=registered_model.mlflow_run_id,
        model_uri=registered_model.model_uri,
    )
