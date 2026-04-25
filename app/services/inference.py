from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app.services.dataset import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
)
from app.services.model_storage import load_model_from_storage
from app.services.registry import get_best_model
from app.services.tracing import get_tracer


@dataclass(slots=True)
class LoadedInferenceArtifacts:
    model: Any
    preprocessor: Any | None
    model_version: str


_model_cache: LoadedInferenceArtifacts | None = None
_cache_lock = threading.Lock()


def invalidate_model_cache() -> None:
    global _model_cache
    with _cache_lock:
        _model_cache = None


def _resolve_preprocessor_path(model_path: str, preprocessor_path: str | None = None) -> str:
    if preprocessor_path:
        return preprocessor_path

    if not model_path.endswith("/model.joblib"):
        raise ValueError("Unable to infer preprocessor path from model artifact path.")

    return model_path.removesuffix("/model.joblib") + "/preprocessor.joblib"


def _load_from_storage() -> LoadedInferenceArtifacts:
    registry_entry = get_best_model()

    if not registry_entry.model_path:
        raise LookupError(
            f"Best model {registry_entry.model_version!r} has no model artifact path."
        )

    model = load_model_from_storage(registry_entry.model_path)

    try:
        preprocessor = load_model_from_storage(
            _resolve_preprocessor_path(
                registry_entry.model_path,
                getattr(registry_entry, "preprocessor_path", None),
            )
        )
    except Exception:
        preprocessor = None

    return LoadedInferenceArtifacts(
        model=model,
        preprocessor=preprocessor,
        model_version=registry_entry.model_version,
    )


def load_best_model(force_reload: bool = False) -> LoadedInferenceArtifacts:
    global _model_cache
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("load_best_model"):
        with _cache_lock:
            if force_reload or _model_cache is None:
                _model_cache = _load_from_storage()
            return _model_cache


def _add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
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
    dataframe = pd.DataFrame([dict(payload)])
    dataframe = _add_engineered_features(dataframe)

    feature_columns = [
        *BASE_NUMERIC_FEATURES,
        *ENGINEERED_NUMERIC_FEATURES,
        *BASE_CATEGORICAL_FEATURES,
    ]
    return dataframe.loc[:, feature_columns].copy()


def predict_fraud(payload: Mapping[str, Any]) -> dict[str, Any]:
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("predict_fraud"):
        loaded_artifacts = load_best_model()
        feature_frame = _prepare_inference_features(payload)

        with tracer.start_as_current_span("preprocessing"):
            model_input = feature_frame
            if loaded_artifacts.preprocessor is not None:
                model_input = loaded_artifacts.preprocessor.transform(feature_frame)

        with tracer.start_as_current_span("model_inference"):
            prediction = int(np.asarray(loaded_artifacts.model.predict(model_input)).ravel()[0])

            if not hasattr(loaded_artifacts.model, "predict_proba"):
                raise ValueError("The deployed model does not implement predict_proba().")

            probabilities = np.asarray(loaded_artifacts.model.predict_proba(model_input))
            if probabilities.ndim == 1:
                fraud_score = float(probabilities[0])
            else:
                fraud_score = float(probabilities[0, 1])

        return {
            "prediction": prediction,
            "fraud_score": fraud_score,
            "model_version": loaded_artifacts.model_version,
        }
