from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.services.drift import (
    DEFAULT_POSITIVE_RATE_THRESHOLD,
    DEFAULT_PSI_THRESHOLD,
    DriftCheckResult,
    ReferenceProfile,
    build_reference_profile,
    check_drift,
    save_drift_result,
    save_reference_profile,
)
from app.services.metadata import get_batch_metadata, get_latest_batch_metadata
from app.services.storage import download_object_to_buffer


@dataclass(slots=True)
class MonitoringRunResult:
    """Structured result returned by monitor_model()."""

    reference_profile: ReferenceProfile
    drift_result: DriftCheckResult
    reference_profile_path: str
    drift_result_path: str


def _load_batch_dataframe(batch_id: int) -> pd.DataFrame:
    """Loads one uploaded raw batch into a DataFrame for monitoring checks."""

    batch_metadata = get_batch_metadata(batch_id)
    buffer = download_object_to_buffer(batch_metadata.storage_path)
    return pd.read_csv(buffer)


def monitor_model(
    *,
    reference_batch_id: int,
    current_batch_id: int | None = None,
    dataset_version: str | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    positive_rate_threshold: float = DEFAULT_POSITIVE_RATE_THRESHOLD,
) -> MonitoringRunResult:
    """Builds a reference profile and compares it to the selected current batch."""

    reference_batch = get_batch_metadata(reference_batch_id)

    if current_batch_id is None:
        current_batch = get_latest_batch_metadata(dataset_version=dataset_version or reference_batch.dataset_version)
    else:
        current_batch = get_batch_metadata(current_batch_id)

    if current_batch.id == reference_batch.id:
        raise ValueError("reference_batch_id and current_batch_id must point to different batches.")

    reference_dataframe = _load_batch_dataframe(reference_batch.id)
    current_dataframe = _load_batch_dataframe(current_batch.id)

    reference_profile = build_reference_profile(
        dataframe=reference_dataframe,
        batch_id=reference_batch.id,
        dataset_version=reference_batch.dataset_version,
    )
    drift_result = check_drift(
        reference_profile=reference_profile,
        current_dataframe=current_dataframe,
        current_batch_id=current_batch.id,
        psi_threshold=psi_threshold,
        positive_rate_threshold=positive_rate_threshold,
    )

    reference_profile_path = save_reference_profile(reference_profile)
    drift_result_path = save_drift_result(drift_result)

    return MonitoringRunResult(
        reference_profile=reference_profile,
        drift_result=drift_result,
        reference_profile_path=reference_profile_path,
        drift_result_path=drift_result_path,
    )
