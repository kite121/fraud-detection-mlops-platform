from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
from dotenv import load_dotenv
from minio import Minio


load_dotenv()

DEFAULT_MODELS_BUCKET = "models"


# ---------------------------------------------------------------------------
# MinIO client (same approach as storage.py)
# ---------------------------------------------------------------------------

def _create_minio_client() -> Minio:
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
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def _upload_bytes(
    client: Minio,
    bucket_name: str,
    object_key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Uploads raw bytes to MinIO and returns the s3:// path."""
    buffer = BytesIO(data)
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_key,
        data=buffer,
        length=len(data),
        content_type=content_type,
    )
    return f"s3://{bucket_name}/{object_key}"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_with_joblib(obj: Any) -> bytes:
    """Serialises any joblib-compatible object (sklearn models, pipelines, etc.)."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        joblib.dump(obj, tmp_path)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ModelArtifactPaths:
    """
    Paths of the artifacts that were uploaded to MinIO.
    All paths follow the s3://bucket/key convention used throughout the platform.
    """
    model_path: str
    metrics_path: str
    preprocessor_path: str | None  # None when no preprocessor was provided


# ---------------------------------------------------------------------------
# Main function — Task 6
# ---------------------------------------------------------------------------

def save_model_artifacts(
    *,
    model: Any,
    metrics: dict,
    model_version: str,
    preprocessor: Any | None = None,
    bucket_name: str = DEFAULT_MODELS_BUCKET,
) -> ModelArtifactPaths:
    """
    Serialises and uploads all model artifacts to MinIO under a versioned prefix.

    Stored files:
        models/<model_version>/model.joblib        — the trained model
        models/<model_version>/metrics.json        — evaluation metrics dict
        models/<model_version>/preprocessor.joblib — sklearn preprocessor (optional)

    Args:
        model:          Trained model object (CatBoost, sklearn-compatible).
        metrics:        Dict returned by evaluate_model() — must be JSON-serialisable.
        model_version:  Version string from the registry, e.g. 'v001'.
        preprocessor:   Fitted sklearn preprocessor / pipeline (None if not used).
        bucket_name:    Target MinIO bucket (default: 'models').

    Returns:
        ModelArtifactPaths with s3:// URIs for model, metrics, and preprocessor.
    """
    client = _create_minio_client()
    _ensure_bucket_exists(client, bucket_name)

    prefix = f"models/{model_version}"

    # --- model ---
    print(f"[Storage] Uploading model artifact → {prefix}/model.joblib")
    model_bytes = _serialise_with_joblib(model)
    model_path = _upload_bytes(
        client,
        bucket_name,
        f"{prefix}/model.joblib",
        model_bytes,
    )

    # --- metrics ---
    print(f"[Storage] Uploading metrics → {prefix}/metrics.json")
    metrics_bytes = json.dumps(metrics, indent=2, default=float).encode()
    metrics_path = _upload_bytes(
        client,
        bucket_name,
        f"{prefix}/metrics.json",
        metrics_bytes,
        content_type="application/json",
    )

    # --- preprocessor (optional) ---
    preprocessor_path: str | None = None
    if preprocessor is not None:
        print(f"[Storage] Uploading preprocessor → {prefix}/preprocessor.joblib")
        preprocessor_bytes = _serialise_with_joblib(preprocessor)
        preprocessor_path = _upload_bytes(
            client,
            bucket_name,
            f"{prefix}/preprocessor.joblib",
            preprocessor_bytes,
        )

    print(f"[Storage] All artifacts for {model_version} uploaded to bucket '{bucket_name}'.")

    return ModelArtifactPaths(
        model_path=model_path,
        metrics_path=metrics_path,
        preprocessor_path=preprocessor_path,
    )


def load_model_from_storage(model_path: str) -> Any:
    """
    Downloads and deserialises a model from MinIO.
    Useful for inference service or manual inspection.

    Args:
        model_path: s3:// path returned by save_model_artifacts().
    """
    if not model_path.startswith("s3://"):
        raise ValueError(f"Expected s3:// path, got: {model_path!r}")

    bucket_and_key = model_path.removeprefix("s3://")
    bucket_name, _, object_key = bucket_and_key.partition("/")

    client = _create_minio_client()
    response = client.get_object(bucket_name, object_key)

    try:
        raw = response.read()
    finally:
        response.close()
        response.release_conn()

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(raw)

    try:
        return joblib.load(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
