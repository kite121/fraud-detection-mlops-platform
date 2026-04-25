from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import PredictRequest, PredictResponse
from app.services.inference import predict_fraud


router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict_transaction(request: PredictRequest) -> PredictResponse:
    try:
        result = predict_fraud(request.model_dump())
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(500, f"Inference failed: {error}") from error

    return PredictResponse(**result)
