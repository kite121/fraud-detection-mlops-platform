from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.services.dataset import PreparedTrainingDataset
from app.services.training import TrainedModelResult


DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_PRIMARY_METRIC = "pr_auc"
SUPPORTED_PRIMARY_METRICS = {
    "roc_auc",
    "precision",
    "recall",
    "f1_score",
    "pr_auc",
}


@dataclass(slots=True)
class EvaluationMetrics:
    """Stores the metrics that will later be compared across model versions."""

    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    confusion_matrix_summary: dict[str, int]


@dataclass(slots=True)
class ModelEvaluationResult:
    """Structured output returned by evaluate_model()."""

    model_name: str
    primary_metric_name: str
    primary_metric_value: float
    classification_threshold: float
    validation_rows_count: int
    validation_positive_rate: float
    metrics: EvaluationMetrics
    dataset_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Converts the result into a plain dictionary for API or registry storage."""

        return asdict(self)


def _resolve_model(model_or_result: Any) -> Any:
    """Accepts either a raw model instance or TrainedModelResult for convenience."""

    if isinstance(model_or_result, TrainedModelResult):
        return model_or_result.model

    return model_or_result


def _extract_positive_class_probabilities(model: Any, X_validation: Any) -> np.ndarray:
    """Returns fraud-class probabilities needed for ROC-AUC and PR-AUC."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_validation)
        probabilities_array = np.asarray(probabilities)

        if probabilities_array.ndim == 1:
            return probabilities_array

        if probabilities_array.shape[1] < 2:
            raise ValueError(
                "predict_proba() must return at least two columns for binary classification."
            )

        return probabilities_array[:, 1]

    raise ValueError(
        "The provided model does not implement predict_proba(), "
        "so probability-based metrics cannot be computed."
    )


def _validate_primary_metric(primary_metric: str) -> None:
    """Ensures the chosen registry metric is supported by evaluate_model()."""

    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(
            "Unsupported primary_metric. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_PRIMARY_METRICS))}"
        )


def evaluate_model(
    model_or_result: Any,
    prepared_dataset: PreparedTrainingDataset,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> ModelEvaluationResult:
    """Evaluates a fraud model on the validation split and returns structured metrics."""

    if not 0 < classification_threshold < 1:
        raise ValueError("classification_threshold must be between 0 and 1.")

    _validate_primary_metric(primary_metric)

    model = _resolve_model(model_or_result)
    y_true = prepared_dataset.y_validation.to_numpy()
    y_probabilities = _extract_positive_class_probabilities(
        model,
        prepared_dataset.X_validation,
    )
    y_predictions = (y_probabilities >= classification_threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_probabilities))
    precision = float(precision_score(y_true, y_predictions, zero_division=0))
    recall = float(recall_score(y_true, y_predictions, zero_division=0))
    f1 = float(f1_score(y_true, y_predictions, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_probabilities))

    tn, fp, fn, tp = confusion_matrix(y_true, y_predictions, labels=[0, 1]).ravel()
    metrics = EvaluationMetrics(
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1_score=f1,
        pr_auc=pr_auc,
        confusion_matrix_summary={
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    )

    metrics_dict = asdict(metrics)
    primary_metric_value = float(metrics_dict[primary_metric])

    return ModelEvaluationResult(
        model_name=type(model).__name__,
        primary_metric_name=primary_metric,
        primary_metric_value=primary_metric_value,
        classification_threshold=classification_threshold,
        validation_rows_count=prepared_dataset.metadata.validation_rows_count,
        validation_positive_rate=prepared_dataset.metadata.validation_positive_rate,
        metrics=metrics,
        dataset_metadata=asdict(prepared_dataset.metadata),
    )
