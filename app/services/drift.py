from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from minio import Minio

from app.services.dataset import BASE_NUMERIC_FEATURES, ENGINEERED_NUMERIC_FEATURES


load_dotenv()

TARGET_COLUMN = "isFraud"
DEFAULT_DRIFT_BUCKET = "monitoring-artifacts"
DEFAULT_BIN_COUNT = 10
DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_POSITIVE_RATE_THRESHOLD = 0.05


@dataclass(slots=True)
class ReferenceProfile:
    """Reference statistics produced from one batch after training."""

    batch_id: int
    dataset_version: str
    rows_count: int
    positive_rate: float
    numeric_feature_means: dict[str, float]
    numeric_feature_bins: dict[str, list[float]]
    numeric_feature_distributions: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeatureDriftResult:
    """Drift summary for one feature."""

    feature_name: str
    psi: float
    reference_mean: float
    current_mean: float
    mean_delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DriftCheckResult:
    """Overall drift decision for one current batch against one reference profile."""

    reference_batch_id: int
    current_batch_id: int
    dataset_version: str
    current_rows_count: int
    reference_positive_rate: float
    current_positive_rate: float
    positive_rate_delta: float
    psi_threshold: float
    positive_rate_threshold: float
    max_feature_psi: float
    degraded: bool
    feature_results: list[FeatureDriftResult]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_results"] = [result.to_dict() for result in self.feature_results]
        return payload


def _create_minio_client() -> Minio:
    """Creates a MinIO client for storing reference profiles and monitoring outputs."""

    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    if not access_key or not secret_key:
        raise ValueError("MINIO_ROOT_USER and MINIO_ROOT_PASSWORD must be set.")

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def _ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    """Creates the monitoring bucket if it does not exist yet."""

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def _upload_json_artifact(
    *,
    payload: dict[str, Any],
    object_key: str,
    bucket_name: str = DEFAULT_DRIFT_BUCKET,
) -> str:
    """Uploads one JSON monitoring artifact to MinIO and returns its s3 path."""

    client = _create_minio_client()
    _ensure_bucket_exists(client, bucket_name)

    data = json.dumps(payload, indent=2, default=float).encode()
    buffer = BytesIO(data)
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_key,
        data=buffer,
        length=len(data),
        content_type="application/json",
    )

    return f"s3://{bucket_name}/{object_key}"


def _add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Adds the same engineered balance features used during training."""

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


def _prepare_monitoring_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Expands the raw monitoring dataset so drift checks use the training feature space."""

    dataframe = _add_engineered_features(dataframe)
    required_columns = [*BASE_NUMERIC_FEATURES, *ENGINEERED_NUMERIC_FEATURES, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            "The monitoring dataset is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    return dataframe


def _compute_bins(values: pd.Series, bin_count: int) -> np.ndarray:
    """Builds stable bins for PSI and handles the degenerate constant-value case."""

    numeric_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric_values.size == 0:
        raise ValueError("Cannot build drift bins from an empty numeric feature.")

    min_value = float(np.min(numeric_values))
    max_value = float(np.max(numeric_values))

    if np.isclose(min_value, max_value):
        padding = 1.0 if np.isclose(min_value, 0.0) else abs(min_value) * 0.1
        return np.array([min_value - padding, max_value + padding], dtype=float)

    return np.linspace(min_value, max_value, num=bin_count + 1, dtype=float)


def _compute_distribution(values: pd.Series, bins: np.ndarray) -> np.ndarray:
    """Computes a smoothed histogram distribution for PSI calculation."""

    numeric_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric_values.size == 0:
        return np.full(len(bins) - 1, 1.0 / max(len(bins) - 1, 1))

    counts, _ = np.histogram(numeric_values, bins=bins)
    distribution = counts.astype(float) / max(float(counts.sum()), 1.0)
    epsilon = 1e-6
    distribution = np.clip(distribution, epsilon, None)
    distribution = distribution / distribution.sum()
    return distribution


def _compute_psi(reference_distribution: np.ndarray, current_distribution: np.ndarray) -> float:
    """Computes Population Stability Index for one feature."""

    return float(np.sum((current_distribution - reference_distribution) * np.log(
        current_distribution / reference_distribution
    )))


def build_reference_profile(
    *,
    dataframe: pd.DataFrame,
    batch_id: int,
    dataset_version: str,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> ReferenceProfile:
    """Builds the reference monitoring profile from a training or historical batch."""

    prepared_dataframe = _prepare_monitoring_dataframe(dataframe)
    numeric_features = [*BASE_NUMERIC_FEATURES, *ENGINEERED_NUMERIC_FEATURES]

    numeric_feature_means: dict[str, float] = {}
    numeric_feature_bins: dict[str, list[float]] = {}
    numeric_feature_distributions: dict[str, list[float]] = {}

    for feature_name in numeric_features:
        bins = _compute_bins(prepared_dataframe[feature_name], bin_count=bin_count)
        distribution = _compute_distribution(prepared_dataframe[feature_name], bins)
        numeric_feature_means[feature_name] = float(prepared_dataframe[feature_name].mean())
        numeric_feature_bins[feature_name] = bins.tolist()
        numeric_feature_distributions[feature_name] = distribution.tolist()

    return ReferenceProfile(
        batch_id=batch_id,
        dataset_version=dataset_version,
        rows_count=len(prepared_dataframe),
        positive_rate=float(prepared_dataframe[TARGET_COLUMN].mean()),
        numeric_feature_means=numeric_feature_means,
        numeric_feature_bins=numeric_feature_bins,
        numeric_feature_distributions=numeric_feature_distributions,
    )


def check_drift(
    *,
    reference_profile: ReferenceProfile,
    current_dataframe: pd.DataFrame,
    current_batch_id: int,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    positive_rate_threshold: float = DEFAULT_POSITIVE_RATE_THRESHOLD,
) -> DriftCheckResult:
    """Compares one current batch against the reference profile and returns a drift verdict."""

    prepared_dataframe = _prepare_monitoring_dataframe(current_dataframe)

    feature_results: list[FeatureDriftResult] = []
    max_feature_psi = 0.0

    for feature_name, reference_mean in reference_profile.numeric_feature_means.items():
        bins = np.asarray(reference_profile.numeric_feature_bins[feature_name], dtype=float)
        reference_distribution = np.asarray(
            reference_profile.numeric_feature_distributions[feature_name],
            dtype=float,
        )
        current_distribution = _compute_distribution(prepared_dataframe[feature_name], bins)
        psi = _compute_psi(reference_distribution, current_distribution)
        current_mean = float(prepared_dataframe[feature_name].mean())
        mean_delta = current_mean - float(reference_mean)
        max_feature_psi = max(max_feature_psi, psi)

        feature_results.append(
            FeatureDriftResult(
                feature_name=feature_name,
                psi=psi,
                reference_mean=float(reference_mean),
                current_mean=current_mean,
                mean_delta=mean_delta,
            )
        )

    current_positive_rate = float(prepared_dataframe[TARGET_COLUMN].mean())
    positive_rate_delta = abs(current_positive_rate - reference_profile.positive_rate)
    degraded = max_feature_psi >= psi_threshold or positive_rate_delta >= positive_rate_threshold

    return DriftCheckResult(
        reference_batch_id=reference_profile.batch_id,
        current_batch_id=current_batch_id,
        dataset_version=reference_profile.dataset_version,
        current_rows_count=len(prepared_dataframe),
        reference_positive_rate=reference_profile.positive_rate,
        current_positive_rate=current_positive_rate,
        positive_rate_delta=positive_rate_delta,
        psi_threshold=psi_threshold,
        positive_rate_threshold=positive_rate_threshold,
        max_feature_psi=max_feature_psi,
        degraded=degraded,
        feature_results=feature_results,
    )


def save_reference_profile(
    reference_profile: ReferenceProfile,
    *,
    bucket_name: str = DEFAULT_DRIFT_BUCKET,
) -> str:
    """Stores the reference profile as a JSON artifact in MinIO."""

    created_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    object_key = (
        f"reference-profiles/{reference_profile.dataset_version}/"
        f"batch_{reference_profile.batch_id}_{created_at}.json"
    )
    return _upload_json_artifact(
        payload=reference_profile.to_dict(),
        object_key=object_key,
        bucket_name=bucket_name,
    )


def save_drift_result(
    drift_result: DriftCheckResult,
    *,
    bucket_name: str = DEFAULT_DRIFT_BUCKET,
) -> str:
    """Stores one drift check result as a JSON artifact in MinIO."""

    created_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    object_key = (
        f"drift-results/{drift_result.dataset_version}/"
        f"reference_{drift_result.reference_batch_id}/"
        f"current_{drift_result.current_batch_id}_{created_at}.json"
    )
    return _upload_json_artifact(
        payload=drift_result.to_dict(),
        object_key=object_key,
        bucket_name=bucket_name,
    )
