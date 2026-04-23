from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.catboost
import mlflow.sklearn
from catboost import CatBoostClassifier

from app.config import settings
from app.services.dataset import PreparedTrainingDataset


DEFAULT_RANDOM_STATE = 42
DEFAULT_RUN_NAME = "training_run"

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "LogLoss",
    "eval_metric": "AUC",
    "iterations": 300,
    "learning_rate": 0.1,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "random_strength": 1.0,
    "bootstrap_type": "Bernoulli",
    "verbose": False,
    "allow_writing_files": False,
    "random_seed": DEFAULT_RANDOM_STATE,
}


class ModelTrainingError(RuntimeError):
    """Raised when the baseline model cannot be trained successfully."""


@dataclass(slots=True)
class TrainingRunMetadata:
    """Stores the most important details of one training run."""

    model_name: str
    hyperparameters: dict[str, Any]
    train_rows_count: int
    validation_rows_count: int
    train_positive_rate: float
    validation_positive_rate: float
    training_duration_seconds: float
    best_iteration: int | None
    dataset_metadata: dict[str, Any]
    mlflow_run_id: str
    mlflow_experiment_name: str
    model_uri: str
    preprocessor_uri: str


@dataclass(slots=True)
class TrainedModelResult:
    """Returns the trained model together with reproducible training metadata."""

    model: CatBoostClassifier
    metadata: TrainingRunMetadata


def _merge_hyperparameters(
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Builds a reproducible CatBoost configuration with optional overrides."""

    resolved_params = dict(DEFAULT_CATBOOST_PARAMS)
    resolved_params["random_seed"] = random_state

    if hyperparameters:
        resolved_params.update(hyperparameters)

    return resolved_params


def _log_dataset_metadata(prepared_dataset: PreparedTrainingDataset) -> None:
    """Logs compact dataset metadata into MLflow."""

    logging_metadata = prepared_dataset.logging_metadata

    mlflow.log_param("batch_id", logging_metadata["batch_id"])
    mlflow.log_param("dataset_version", logging_metadata["dataset_version"])
    mlflow.log_param("train_rows_count", logging_metadata["train_rows_count"])
    mlflow.log_param("validation_rows_count", logging_metadata["validation_rows_count"])
    mlflow.log_param("train_positive_rate", logging_metadata["train_positive_rate"])
    mlflow.log_param(
        "validation_positive_rate",
        logging_metadata["validation_positive_rate"],
    )
    mlflow.log_dict(logging_metadata, "dataset_metadata.json")

    # Keep the feature list as a dedicated artifact so it is easy to inspect in MLflow UI.
    mlflow.log_text(
        json.dumps(logging_metadata["feature_columns"], ensure_ascii=False, indent=2),
        "feature_columns.json",
    )


def train_model(
    prepared_dataset: PreparedTrainingDataset,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    run_name: str = DEFAULT_RUN_NAME,
) -> TrainedModelResult:
    """Trains a reproducible CatBoost baseline and logs the run to MLflow."""

    params = _merge_hyperparameters(
        hyperparameters=hyperparameters,
        random_state=random_state,
    )
    model = CatBoostClassifier(**params)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(params)
        _log_dataset_metadata(prepared_dataset)

        training_start_time = time.perf_counter()
        try:
            model.fit(
                prepared_dataset.X_train,
                prepared_dataset.y_train,
                eval_set=(prepared_dataset.X_validation, prepared_dataset.y_validation),
                use_best_model=True,
                early_stopping_rounds=30,
            )
        except Exception as error:
            raise ModelTrainingError(f"CatBoost training failed: {error}") from error

        training_duration_seconds = time.perf_counter() - training_start_time
        best_iteration = model.get_best_iteration()
        if best_iteration is not None and best_iteration < 0:
            best_iteration = None

        mlflow.log_metric("training_duration_seconds", training_duration_seconds)
        if best_iteration is not None:
            mlflow.log_metric("best_iteration", best_iteration)

        model_name_in_mlflow = "catboost-model"
        preprocessor_name_in_mlflow = "preprocessor"

        mlflow.set_tags(
            {
                "model_name": "CatBoostClassifier",
                "framework": "catboost",
                "task": "fraud_detection",
                "dataset_version": str(prepared_dataset.logging_metadata["dataset_version"]),
            }
        )

        mlflow.catboost.log_model(model, name=model_name_in_mlflow)
        mlflow.sklearn.log_model(
            sk_model=prepared_dataset.preprocessor,
            name=preprocessor_name_in_mlflow,
        )
        model_uri = f"runs:/{active_run.info.run_id}/{model_name_in_mlflow}"
        preprocessor_uri = f"runs:/{active_run.info.run_id}/{preprocessor_name_in_mlflow}"

        metadata = TrainingRunMetadata(
            model_name="CatBoostClassifier",
            hyperparameters=params,
            train_rows_count=prepared_dataset.metadata.train_rows_count,
            validation_rows_count=prepared_dataset.metadata.validation_rows_count,
            train_positive_rate=prepared_dataset.metadata.train_positive_rate,
            validation_positive_rate=prepared_dataset.metadata.validation_positive_rate,
            training_duration_seconds=training_duration_seconds,
            best_iteration=best_iteration,
            dataset_metadata=prepared_dataset.logging_metadata.copy(),
            mlflow_run_id=active_run.info.run_id,
            mlflow_experiment_name=settings.mlflow_experiment_name,
            model_uri=model_uri,
            preprocessor_uri=preprocessor_uri,
        )

        return TrainedModelResult(
            model=model,
            metadata=metadata,
        )
