from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.model_storage import load_model_from_storage
from app.services.registry import get_best_model
from app.services.tracing import get_tracer


@dataclass(slots=True)
class LoadedInferenceArtifacts:
    model: Any
    preprocessor: Any
    model_version: str


_loaded_artifacts: LoadedInferenceArtifacts | None = None


def _resolve_preprocessor_path(model_path: str, preprocessor_path: str | None) -> str:
    if preprocessor_path:
        return preprocessor_path

    if not model_path.endswith("/model.joblib"):
        raise ValueError("Unable to infer preprocessor path from model artifact path.")

    return model_path.removesuffix("/model.joblib") + "/preprocessor.joblib"


def load_best_model(force_reload: bool = False) -> LoadedInferenceArtifacts:
    global _loaded_artifacts
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("load_best_model"):
        registry_entry = get_best_model()

        if (
            not force_reload
            and _loaded_artifacts is not None
            and _loaded_artifacts.model_version == registry_entry.model_version
        ):
            return _loaded_artifacts

        if not registry_entry.model_path:
            raise LookupError(
                f"Best model {registry_entry.model_version!r} has no model artifact path."
            )

        model = load_model_from_storage(registry_entry.model_path)
        preprocessor = load_model_from_storage(
            _resolve_preprocessor_path(
                registry_entry.model_path,
                getattr(registry_entry, "preprocessor_path", None),
            )
        )

        _loaded_artifacts = LoadedInferenceArtifacts(
            model=model,
            preprocessor=preprocessor,
            model_version=registry_entry.model_version,
        )
        return _loaded_artifacts


def _build_feature_frame(transaction: dict[str, Any]) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        [
            {
                "step": transaction["step"],
                "type": transaction["type"],
                "amount": transaction["amount"],
                "oldbalanceOrg": transaction["oldbalanceOrg"],
                "newbalanceOrig": transaction["newbalanceOrig"],
                "oldbalanceDest": transaction["oldbalanceDest"],
                "newbalanceDest": transaction["newbalanceDest"],
                "isFlaggedFraud": transaction["isFlaggedFraud"],
            }
        ]
    )

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


def predict_fraud(transaction: dict[str, Any]) -> dict[str, Any]:
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("predict_fraud"):
        loaded = load_best_model()

        with tracer.start_as_current_span("preprocessing"):
            features = _build_feature_frame(transaction)
            transformed = loaded.preprocessor.transform(features)

        with tracer.start_as_current_span("model_inference"):
            probabilities = loaded.model.predict_proba(transformed)
            fraud_score = float(probabilities[0][1])
            prediction = int(fraud_score >= 0.5)

        return {
            "prediction": prediction,
            "fraud_score": fraud_score,
            "model_version": loaded.model_version,
        }
