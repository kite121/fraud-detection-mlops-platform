from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import MonitorRequest, MonitorResponse
from app.services.monitoring import monitor_model


router = APIRouter()


@router.post("/monitor", response_model=MonitorResponse)
def run_monitoring(request: MonitorRequest) -> MonitorResponse:
    """
    Runs one drift check manually and returns the monitoring verdict.

    This complements the automatic `data_ingested -> monitoring` runtime flow by
    exposing the same logic through an API endpoint, which is useful for smoke
    tests, demos, debugging, and ad-hoc investigations.
    """

    try:
        result = monitor_model(
            reference_batch_id=request.reference_batch_id,
            current_batch_id=request.current_batch_id,
            dataset_version=request.dataset_version,
            psi_threshold=request.psi_threshold,
            positive_rate_threshold=request.positive_rate_threshold,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Monitoring failed: {error}",
        ) from error

    return MonitorResponse(
        status="completed",
        reference_batch_id=result.drift_result.reference_batch_id,
        current_batch_id=result.drift_result.current_batch_id,
        degraded=result.drift_result.degraded,
        max_feature_psi=result.drift_result.max_feature_psi,
        positive_rate_delta=result.drift_result.positive_rate_delta,
        reference_profile_path=result.reference_profile_path,
        drift_result_path=result.drift_result_path,
    )
