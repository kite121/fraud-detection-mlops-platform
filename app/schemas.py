# Pydantic-schemas are separate classes for data validation in APIs (don't confuse them with ORM models)

from datetime import datetime
from pydantic import BaseModel


class BatchMetadataCreate(BaseModel):
    """Data that is received when a record is created."""
    dataset_version: str
    file_name: str
    storage_path: str
    rows_count: int
    status: str = "uploaded"
    validation_message: str | None = None
    source_name: str | None = None
    checksum: str | None = None


class BatchMetadataResponse(BaseModel):
    """Data that is returned from the API."""
    id: int
    dataset_version: str
    file_name: str
    storage_path: str
    rows_count: int
    status: str
    created_at: datetime
    validation_message: str | None

    model_config = {"from_attributes": True}  # allows you to create an ORM object


class TrainEnqueueResponse(BaseModel):
    status: str
    job_id: str
    message: str


class TrainingJobResponse(BaseModel):
    job_id: str
    job_type: str
    dataset_version: str | None = None
    batch_id: int | None = None
    celery_task_id: str | None = None
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    model_version: str | None = None
    mlflow_run_id: str | None = None

    model_config = {"from_attributes": True}


class PredictRequest(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int


class PredictResponse(BaseModel):
    prediction: int
    fraud_score: float
    model_version: str
