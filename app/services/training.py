from __future__ import annotations

import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier

from app.services.dataset import PreparedTrainingDataset


DEFAULT_RANDOM_STATE = 42

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 300,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "random_strength": 1.0,
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
    best_validation_score: float | None
    serialized_model_size_bytes: int
    dataset_metadata: dict[str, Any]


@dataclass(slots=True)
class TrainedModelResult:
    """Returns the trained model together with reproducible training metadata."""

    model: CatBoostClassifier
    metadata: TrainingRunMetadata
    serialized_model: bytes


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


def _serialize_model(model: CatBoostClassifier) -> bytes:
    """Serializes the trained model so it can be stored as an artifact later."""

    return pickle.dumps(model)


def save_serialized_model(serialized_model: bytes, output_path: str | Path) -> Path:
    """Writes a serialized model artifact to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialized_model)
    return path


def train_model(
    prepared_dataset: PreparedTrainingDataset,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TrainedModelResult:
    """Trains a reproducible CatBoost baseline on the prepared fraud dataset."""

    params = _merge_hyperparameters(
        hyperparameters=hyperparameters,
        random_state=random_state,
    )
    model = CatBoostClassifier(**params)

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
    serialized_model = _serialize_model(model)

    best_validation_score: float | None = None
    best_score = model.get_best_score()
    if best_score:
        validation_scores = best_score.get("validation")
        if validation_scores:
            best_validation_score = validation_scores.get("AUC")

    best_iteration = model.get_best_iteration()
    if best_iteration is not None and best_iteration < 0:
        best_iteration = None

    metadata = TrainingRunMetadata(
        model_name="CatBoostClassifier",
        hyperparameters=params,
        train_rows_count=prepared_dataset.metadata.train_rows_count,
        validation_rows_count=prepared_dataset.metadata.validation_rows_count,
        train_positive_rate=prepared_dataset.metadata.train_positive_rate,
        validation_positive_rate=prepared_dataset.metadata.validation_positive_rate,
        training_duration_seconds=training_duration_seconds,
        best_iteration=best_iteration,
        best_validation_score=best_validation_score,
        serialized_model_size_bytes=len(serialized_model),
        dataset_metadata=asdict(prepared_dataset.metadata),
    )

    return TrainedModelResult(
        model=model,
        metadata=metadata,
        serialized_model=serialized_model,
    )
