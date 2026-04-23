from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.config import settings
from app.services.dataset import PreparedTrainingDataset
from app.services.training import TrainedModelResult


DEFAULT_THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)


@dataclass(slots=True)
class ThresholdSelection:
    """Stores the best threshold found for one metric."""

    threshold: float
    score: float


@dataclass(slots=True)
class EvaluationMetrics:
    """Stores evaluation metrics and threshold tuning outputs."""

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    precision_threshold: float
    recall_threshold: float
    f1_threshold: float
    accuracy_threshold: float


@dataclass(slots=True)
class ModelEvaluationResult:
    """Structured output returned by evaluate_model()."""

    model_name: str
    mlflow_run_id: str | None
    metrics: EvaluationMetrics
    dataset_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Converts the result into a plain dictionary for artifact logging."""

        return asdict(self)


def _resolve_model_and_run_id(
    model_or_result: Any,
) -> tuple[Any, str | None]:
    """Accepts either a raw model instance or a TrainedModelResult."""

    if isinstance(model_or_result, TrainedModelResult):
        return model_or_result.model, model_or_result.metadata.mlflow_run_id

    return model_or_result, None


def _extract_positive_class_probabilities(model: Any, X_validation: Any) -> np.ndarray:
    """Returns fraud-class probabilities for the validation split."""

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The provided model does not implement predict_proba(), "
            "so evaluation metrics cannot be computed."
        )

    probabilities = model.predict_proba(X_validation)
    probabilities_array = np.asarray(probabilities)

    if probabilities_array.ndim == 1:
        return probabilities_array

    if probabilities_array.shape[1] < 2:
        raise ValueError(
            "predict_proba() must return two columns for binary classification."
        )

    return probabilities_array[:, 1]


def _select_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric_name: str,
    threshold_grid: np.ndarray,
) -> ThresholdSelection:
    """Finds the threshold that maximizes the requested classification metric."""

    best_threshold = float(threshold_grid[0])
    best_score = -1.0

    for threshold in threshold_grid:
        predictions = (probabilities >= threshold).astype(int)

        if metric_name == "precision":
            score = precision_score(y_true, predictions, zero_division=0)
        elif metric_name == "recall":
            score = recall_score(y_true, predictions, zero_division=0)
        elif metric_name == "f1_score":
            score = f1_score(y_true, predictions, zero_division=0)
        elif metric_name == "accuracy":
            score = accuracy_score(y_true, predictions)
        else:
            raise ValueError(f"Unsupported metric_name: {metric_name!r}")

        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return ThresholdSelection(
        threshold=best_threshold,
        score=best_score,
    )


def evaluate_model(
    model_or_result: Any,
    prepared_dataset: PreparedTrainingDataset,
    threshold_grid: np.ndarray = DEFAULT_THRESHOLD_GRID,
) -> ModelEvaluationResult:
    """Evaluates the model, tunes thresholds, and logs the results to MLflow."""

    model, mlflow_run_id = _resolve_model_and_run_id(model_or_result)
    y_true = prepared_dataset.y_validation.to_numpy()
    y_probabilities = _extract_positive_class_probabilities(
        model,
        prepared_dataset.X_validation,
    )

    roc_auc = float(roc_auc_score(y_true, y_probabilities))
    pr_auc = float(average_precision_score(y_true, y_probabilities))

    best_precision = _select_best_threshold(
        y_true=y_true,
        probabilities=y_probabilities,
        metric_name="precision",
        threshold_grid=threshold_grid,
    )
    best_recall = _select_best_threshold(
        y_true=y_true,
        probabilities=y_probabilities,
        metric_name="recall",
        threshold_grid=threshold_grid,
    )
    best_f1 = _select_best_threshold(
        y_true=y_true,
        probabilities=y_probabilities,
        metric_name="f1_score",
        threshold_grid=threshold_grid,
    )
    best_accuracy = _select_best_threshold(
        y_true=y_true,
        probabilities=y_probabilities,
        metric_name="accuracy",
        threshold_grid=threshold_grid,
    )

    metrics = EvaluationMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=best_precision.score,
        recall=best_recall.score,
        f1_score=best_f1.score,
        accuracy=best_accuracy.score,
        precision_threshold=best_precision.threshold,
        recall_threshold=best_recall.threshold,
        f1_threshold=best_f1.threshold,
        accuracy_threshold=best_accuracy.threshold,
    )

    evaluation_result = ModelEvaluationResult(
        model_name=type(model).__name__,
        mlflow_run_id=mlflow_run_id,
        metrics=metrics,
        dataset_metadata=prepared_dataset.logging_metadata.copy(),
    )

    if mlflow_run_id is not None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metrics({
                "roc_auc": evaluation_result.metrics.roc_auc,
                "precision": evaluation_result.metrics.precision,
                "recall": evaluation_result.metrics.recall,
                "f1_score": evaluation_result.metrics.f1_score,
                "pr_auc": evaluation_result.metrics.pr_auc,
                "accuracy": evaluation_result.metrics.accuracy,
                "precision_threshold": evaluation_result.metrics.precision_threshold,
                "recall_threshold": evaluation_result.metrics.recall_threshold,
                "f1_threshold": evaluation_result.metrics.f1_threshold,
                "accuracy_threshold": evaluation_result.metrics.accuracy_threshold,
            })
            mlflow.log_dict(evaluation_result.to_dict(), "evaluation.json")

    return evaluation_result