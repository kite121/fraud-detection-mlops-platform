from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db import SessionLocal
from app.models import TrainingJob
from app.schemas import TrainingJobResponse


router = APIRouter()


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str) -> TrainingJobResponse:
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job is None:
            raise HTTPException(404, f"Training job {job_id!r} was not found.")

        return TrainingJobResponse.model_validate(job)
    finally:
        db.close()
