from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app.services.dataset import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
)
from app.services.model_storage import DEFAULT_MODELS_BUCKET, load_artifact_from_storage
from app.services.registry import get_best_model


@dataclass(slots=True)
class LoadedInferenceArtifacts:
    """Holds the currently deployed model and the matching preprocessing pipeline."""

    model: Any
    preprocessor: Any | None
    model_version: str


@dataclass(slots=True)
class FraudPredictionResult:
    """Structured prediction output returned by predict_fraud()."""

    prediction: int
    fraud_score: float
    model_version: str


def _build_versioned_artifact_path(model_version: str, artifact_name: str) -> str:
    """Builds the standard MinIO path for one versioned model artifact."""

    return f"s3://{DEFAULT_MODELS_BUCKET}/models/{model_version}/{artifact_name}"


def _add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Recreates the balance-based features used during model training."""

    dataframe = dataframe.copy()
    dataframe["origin_balance_delta"] = (
        dataframe["oldbalanceOrg"] - dataframe["newbalanceOrig"]
    )
    dataframe["destination_balance_delta"] = (
        dataframe["newbalanceDest"] - dataframe["oldbalanceDest"]
    )
    dataframe["origin_balance_error"] = (
        dataframe["oldbalanceOrg"] - dataframe["amount"] - dataframe["newbalanceOrig"]
    )
    dataframe["destination_balance_error"] = (
        dataframe["oldbalanceDest"] + dataframe["amount"] - dataframe["newbalanceDest"]
    )
    return dataframe


def _prepare_inference_features(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Turns one transaction payload into the exact feature frame expected by training."""

    dataframe = pd.DataFrame([dict(payload)])
    dataframe = _add_engineered_features(dataframe)

    feature_columns = [
        *BASE_NUMERIC_FEATURES,
        *ENGINEERED_NUMERIC_FEATURES,
        *BASE_CATEGORICAL_FEATURES,
    ]
    return dataframe.loc[:, feature_columns].copy()


def load_best_model() -> LoadedInferenceArtifacts:
    """Loads the current best model and its compatible preprocessor from storage."""

    best_model = get_best_model()
    model_path = _build_versioned_artifact_path(best_model.model_version, "model.joblib")
    preprocessor_path = _build_versioned_artifact_path(
        best_model.model_version,
        "preprocessor.joblib",
    )

    model = load_artifact_from_storage(model_path)

    try:
        preprocessor = load_artifact_from_storage(preprocessor_path)
    except Exception:
        preprocessor = None

    return LoadedInferenceArtifacts(
        model=model,
        preprocessor=preprocessor,
        model_version=best_model.model_version,
    )


def predict_fraud(payload: Mapping[str, Any]) -> FraudPredictionResult:
    """Loads the best model, prepares inference features, and returns one prediction."""

    loaded_artifacts = load_best_model()
    feature_frame = _prepare_inference_features(payload)

    model_input = feature_frame
    if loaded_artifacts.preprocessor is not None:
        model_input = loaded_artifacts.preprocessor.transform(feature_frame)

    prediction = int(np.asarray(loaded_artifacts.model.predict(model_input)).ravel()[0])

    if not hasattr(loaded_artifacts.model, "predict_proba"):
        raise ValueError("The deployed model does not implement predict_proba().")

    probabilities = np.asarray(loaded_artifacts.model.predict_proba(model_input))
    if probabilities.ndim == 1:
        fraud_score = float(probabilities[0])
    else:
        fraud_score = float(probabilities[0, 1])

    return FraudPredictionResult(
        prediction=prediction,
        fraud_score=fraud_score,
        model_version=loaded_artifacts.model_version,
    )
