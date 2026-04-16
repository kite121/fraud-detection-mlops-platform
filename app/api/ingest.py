from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.metadata import save_batch_metadata
from app.services.storage import upload_to_storage
from app.services.validation import validate_csv


router = APIRouter()


class IngestResponse(BaseModel):
    """Successful response returned by POST /ingest."""

    status: str
    batch_id: int
    rows_count: int
    storage_path: str


def _save_upload_to_temp_file(file: UploadFile) -> Path:
    """Stores the uploaded CSV temporarily so the service functions can reuse it."""

    suffix = Path(file.filename or "batch.csv").suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.file.read())
        return Path(temp_file.name)


@router.post("/ingest", response_model=IngestResponse)
def ingest_file(
    client_id: str = Form(...),
    dataset_version: str = Form(...),
    file: UploadFile = File(...),
) -> IngestResponse:
    """Accepts a CSV file, validates it, uploads it to storage, saves metadata, and returns the result."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    temp_path = _save_upload_to_temp_file(file)

    try:
        validation_result = validate_csv(temp_path)
        if not validation_result.is_valid:
            raise HTTPException(status_code=400, detail=validation_result.message)

        storage_path = upload_to_storage(
            file_path=temp_path,
            client_id=client_id,
            dataset_version=dataset_version,
        )
        batch_id = save_batch_metadata(
            client_id=client_id,
            dataset_version=dataset_version,
            file_name=file.filename,
            storage_path=storage_path,
            rows_count=validation_result.rows_count,
            status="uploaded",
        )

        return IngestResponse(
            status="success",
            batch_id=batch_id,
            rows_count=validation_result.rows_count,
            storage_path=storage_path,
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        temp_path.unlink(missing_ok=True)
