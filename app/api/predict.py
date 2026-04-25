from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas import PredictRequest, PredictResponse
from app.services.inference import load_best_model, predict_fraud


router = APIRouter()


class ReloadResponse(BaseModel):
    status: str
    model_version: str
    message: str


@router.post("/predict", response_model=PredictResponse)
def predict_transaction(request: PredictRequest) -> PredictResponse:
    try:
        result = predict_fraud(request.model_dump())
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {error}",
        ) from error

    return PredictResponse(**result)


@router.post("/reload", response_model=ReloadResponse)
def reload_model() -> ReloadResponse:
    try:
        artifacts = load_best_model(force_reload=True)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {error}",
        ) from error

    return ReloadResponse(
        status="reloaded",
        model_version=artifacts.model_version,
        message=f"Now serving model {artifacts.model_version!r}.",
    )
