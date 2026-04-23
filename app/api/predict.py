from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.inference import predict_fraud


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
