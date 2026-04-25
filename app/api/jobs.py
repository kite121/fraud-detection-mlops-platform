from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import TrainingJobResponse
from app.services.jobs import get_job_by_id


router = APIRouter()


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_job(job_id: str) -> TrainingJobResponse:
    try:
        job = get_job_by_id(job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return TrainingJobResponse.model_validate(job)
