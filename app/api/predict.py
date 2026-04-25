from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.inference import invalidate_model_cache, load_best_model, predict_fraud

router = APIRouter()


class PredictRequest(BaseModel):
    """Incoming transaction payload used for online fraud prediction."""

    step: int = Field(..., ge=0)
    type: str = Field(..., min_length=1)
    amount: float = Field(..., ge=0)
    nameOrig: str = Field(..., min_length=1)
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str = Field(..., min_length=1)
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    """Prediction result returned by POST /predict."""

    prediction: int
    fraud_score: float
    model_version: str


class ReloadResponse(BaseModel):
    """Confirmation returned after a successful model reload."""

    status: str
    model_version: str
    message: str


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Runs online fraud prediction using the currently best registered model."""

    try:
        result = predict_fraud(request.model_dump())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    return PredictResponse(
        prediction=result.prediction,
        fraud_score=result.fraud_score,
        model_version=result.model_version,
    )


@router.post("/reload", response_model=ReloadResponse)
def reload_model() -> ReloadResponse:
    """
    Sprint 3 Task 7: Model deployment.

    Clears the in-memory model cache and immediately loads the current best
    model from the registry and MinIO. Call this endpoint after a new best
    model has been registered to make the inference-service serve the updated
    model without restarting the container.

    This endpoint is also called automatically when a model_deployed event
    is received from the broker (Sprint 3 Task 1 / events.py).
    """
    try:
        invalidate_model_cache()
        artifacts = load_best_model()
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {exc}",
        ) from exc

    return ReloadResponse(
        status="reloaded",
        model_version=artifacts.model_version,
        message=f"Now serving model {artifacts.model_version!r}.",
    )
