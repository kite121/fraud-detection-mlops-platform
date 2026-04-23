from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.dataset import prepare_training_dataset
from app.services.evaluation import evaluate_model
from app.services.model_storage import save_model_artifacts
from app.services.registry import register_model_version
from app.services.training import train_model


router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """
    Parameters for launching a training run.

    Provide either batch_id (specific ingestion batch) or dataset_version
    (the platform picks the latest uploaded batch for that version).
    At least one of the two must be set.
    """

    batch_id: int | None = Field(
        default=None,
        description="ID of the ingested batch to train on (from batch_metadata table).",
    )
    dataset_version: str | None = Field(
        default=None,
        description="Dataset version tag; uses the newest uploaded batch if batch_id is omitted.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional free-text notes attached to this training run.",
    )


class TrainResponse(BaseModel):
    """Full information about the completed training run."""

    status: str
    model_version: str
    model_status: str          # 'best' or 'validated'
    primary_metric: float      # ROC-AUC on validation set
    model_path: str | None
    metrics_path: str | None
    mlflow_run_id: str | None
    model_uri: str | None


# ---------------------------------------------------------------------------
# Endpoint — Task 7
# ---------------------------------------------------------------------------

@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    """
    Runs the full training pipeline:

        1. prepare_training_dataset()  — loads raw CSV from MinIO, preprocesses it
        2. train_model()               — trains a CatBoost classifier, logs to MLflow
        3. evaluate_model()            — computes ROC-AUC, Precision, Recall, F1, PR-AUC
        4. save_model_artifacts()      — uploads model + metrics to MinIO bucket 'models'
        5. register_model_version()    — writes a record to model_registry in PostgreSQL
                                         and promotes the model to 'best' if it outperforms
                                         the current champion

    Returns a JSON response with the model version, primary metric, MinIO paths,
    and MLflow run information.
    """
    if request.batch_id is None and request.dataset_version is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: batch_id, dataset_version.",
        )

    # ------------------------------------------------------------------
    # Step 1 — Prepare dataset
    # ------------------------------------------------------------------
    try:
        prepared = prepare_training_dataset(
            batch_id=request.batch_id,
            dataset_version=request.dataset_version,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset preparation failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Step 2 — Train model (MLflow run is started inside train_model)
    # ------------------------------------------------------------------
    try:
        training_result = train_model(prepared)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Step 3 — Evaluate model
    # ------------------------------------------------------------------
    try:
        eval_result = evaluate_model(
            model=training_result.model,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model evaluation failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Step 4 — Save artifacts to MinIO
    # Temp model_version is a placeholder; the real one comes from the registry.
    # We upload first so the registry record has valid paths.
    # ------------------------------------------------------------------
    try:
        # Generate a preliminary version to name MinIO objects.
        # The registry will assign the same sequential version.
        artifact_paths = save_model_artifacts(
            model=training_result.model,
            metrics=eval_result.to_dict(),
            model_version="pending",          # overwritten below after registration
            preprocessor=getattr(prepared, "preprocessor", None),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Artifact upload failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Step 5 — Register model version in PostgreSQL
    # ------------------------------------------------------------------
    try:
        registered = register_model_version(
            dataset_version=prepared.metadata.dataset_version,
            training_batch_id=prepared.metadata.batch_id,
            primary_metric=eval_result.metrics.roc_auc,
            model_path=artifact_paths.model_path,
            metrics_path=artifact_paths.metrics_path,
            algorithm=type(training_result.model).__name__,
            hyperparameters=training_result.hyperparameters,
            notes=request.notes,
            mlflow_run_id=training_result.mlflow_run_id,
            model_uri=(
                f"runs:/{training_result.mlflow_run_id}/catboost-model"
                if training_result.mlflow_run_id
                else None
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model registration failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Re-upload artifacts under the real versioned paths now that we have
    # the model_version assigned by the registry.
    # ------------------------------------------------------------------
    try:
        final_paths = save_model_artifacts(
            model=training_result.model,
            metrics=eval_result.to_dict(),
            model_version=registered.model_version,
            preprocessor=getattr(prepared, "preprocessor", None),
        )
    except Exception as exc:
        # Not fatal — the model is already registered; just log and continue.
        print(f"[Train] Warning: could not re-upload under versioned path: {exc}")
        final_paths = artifact_paths

    return TrainResponse(
        status="success",
        model_version=registered.model_version,
        model_status=registered.status,
        primary_metric=registered.primary_metric,
        model_path=final_paths.model_path,
        metrics_path=final_paths.metrics_path,
        mlflow_run_id=registered.mlflow_run_id,
        model_uri=registered.model_uri,
    )
