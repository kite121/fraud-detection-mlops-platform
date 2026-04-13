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
