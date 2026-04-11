from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio


load_dotenv()

DEFAULT_BUCKET_NAME = "raw-data"


def _create_minio_client() -> Minio:
    """Creates a MinIO client from env vars so uploads work locally and in Docker."""

    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    if not access_key or not secret_key:
        raise ValueError("MINIO_ROOT_USER and MINIO_ROOT_PASSWORD must be set.")

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def _ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    """Creates the bucket on demand so uploads do not fail in a fresh environment."""

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def _build_object_key(
    file_name: str,
    client_id: str,
    dataset_version: str | None = None,
) -> str:
    """Builds a client-specific object path inside one shared bucket."""

    upload_date = datetime.now(UTC).date().isoformat()
    safe_file_name = Path(file_name).name.replace(" ", "_")
    safe_client_id = client_id.strip().replace(" ", "_")

    if not safe_client_id:
        raise ValueError("client_id must not be empty.")

    if dataset_version:
        return f"clients/{safe_client_id}/raw/{dataset_version}/{upload_date}/{safe_file_name}"

    return f"clients/{safe_client_id}/raw/{upload_date}/{safe_file_name}"


def upload_to_storage(
    file_path: str | Path,
    client_id: str,
    dataset_version: str | None = None,
    bucket_name: str = DEFAULT_BUCKET_NAME,
) -> str:
    """Uploads a validated raw CSV to MinIO and returns the final storage path."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    client = _create_minio_client()
    _ensure_bucket_exists(client, bucket_name)

    object_key = _build_object_key(
        path.name,
        client_id=client_id,
        dataset_version=dataset_version,
    )
    client.fput_object(
        bucket_name=bucket_name,
        object_name=object_key,
        file_path=str(path),
    )

    return f"s3://{bucket_name}/{object_key}"
