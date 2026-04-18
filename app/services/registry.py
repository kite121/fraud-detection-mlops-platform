from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from dotenv import load_dotenv
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


load_dotenv()


# ---------------------------------------------------------------------------
# ORM setup (self-contained, same pattern as metadata.py)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base class for training-service ORM models."""


class ModelRegistry(Base):
    """
    Stores every trained model version so the platform can track, compare,
    and promote models independently of MLflow.

    Statuses:
        training   — the training run is still in progress
        validated  — training finished and metrics were recorded
        best       — this is the current champion model (only one at a time)
        archived   — a previous best model that has been superseded
    """

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    dataset_version: Mapped[str] = mapped_column(String(100), nullable=False)
    training_batch_id: Mapped[int] = mapped_column(Integer, nullable=False)
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metrics_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    primary_metric: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="validated")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    algorithm: Mapped[str | None] = mapped_column(String(100), nullable=True)
    hyperparameters: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    feature_set_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    mlflow_run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry version={self.model_version!r} "
            f"metric={self.primary_metric:.4f} status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _build_database_url() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "fraud_platform")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    if not user or not password:
        raise ValueError("POSTGRES_USER and POSTGRES_PASSWORD must be set.")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def _create_session_factory() -> sessionmaker:
    engine = create_engine(_build_database_url(), pool_pre_ping=True)
    return sessionmaker(bind=engine)


def init_registry_table() -> None:
    """Creates the model_registry table on a fresh database."""
    engine = create_engine(_build_database_url())
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_model_version(session) -> str:
    """
    Generates the next sequential version string: v001, v002, v003 …
    Uses the total number of rows in model_registry (including failed ones)
    so version numbers are never reused even after deletions.
    """
    count = session.query(ModelRegistry).count()
    return f"v{count + 1:03d}"


def _demote_current_best(session) -> None:
    """
    Moves the current 'best' model to 'archived' so there is always at most
    one champion. Call this before promoting a new model.
    """
    current_best = session.execute(
        select(ModelRegistry).where(ModelRegistry.status == "best")
    ).scalar_one_or_none()

    if current_best is not None:
        current_best.status = "archived"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegisteredModel:
    """Return value of register_model_version() — all the info the endpoint needs."""
    id: int
    model_version: str
    status: str
    primary_metric: float
    model_path: str | None
    metrics_path: str | None
    mlflow_run_id: str | None
    model_uri: str | None


def register_model_version(
    *,
    dataset_version: str,
    training_batch_id: int,
    primary_metric: float,
    model_path: str | None = None,
    metrics_path: str | None = None,
    algorithm: str | None = None,
    hyperparameters: dict | None = None,
    feature_set_version: str | None = None,
    notes: str | None = None,
    mlflow_run_id: str | None = None,
    model_uri: str | None = None,
) -> RegisteredModel:
    """
    Saves a trained model version to the registry and promotes it to 'best'
    if it outperforms all previously registered models on primary_metric (ROC-AUC).

    Args:
        dataset_version:    Version tag of the raw dataset used for training.
        training_batch_id:  ID from the batch_metadata table (Sprint 1).
        primary_metric:     ROC-AUC score on the validation set.
        model_path:         s3:// path to the serialised model artifact in MinIO.
        metrics_path:       s3:// path to the metrics JSON file in MinIO.
        algorithm:          Human-readable algorithm name, e.g. 'CatBoostClassifier'.
        hyperparameters:    Dict of hyperparameters; stored as JSON text.
        feature_set_version: Optional tag for the feature engineering version.
        notes:              Free-text notes about this run.
        mlflow_run_id:      Run ID returned by MLflow after logging the model.
        model_uri:          MLflow model URI, e.g. 'runs:/<run_id>/catboost-model'.

    Returns:
        RegisteredModel dataclass with id, version, status, and paths.
    """
    session_factory = _create_session_factory()
    session = session_factory()

    try:
        model_version = _next_model_version(session)

        # Find the current best metric so we know whether to promote this model.
        current_best: ModelRegistry | None = session.execute(
            select(ModelRegistry).where(ModelRegistry.status == "best")
        ).scalar_one_or_none()

        is_new_best = (
            current_best is None
            or primary_metric > current_best.primary_metric
        )

        if is_new_best:
            _demote_current_best(session)

        new_status = "best" if is_new_best else "validated"

        entry = ModelRegistry(
            model_version=model_version,
            dataset_version=dataset_version,
            training_batch_id=training_batch_id,
            model_path=model_path,
            metrics_path=metrics_path,
            primary_metric=primary_metric,
            status=new_status,
            algorithm=algorithm,
            hyperparameters=json.dumps(hyperparameters) if hyperparameters else None,
            feature_set_version=feature_set_version,
            notes=notes,
            mlflow_run_id=mlflow_run_id,
            model_uri=model_uri,
        )

        session.add(entry)
        session.commit()
        session.refresh(entry)

        print(
            f"[Registry] Registered {model_version} "
            f"(ROC-AUC={primary_metric:.4f}, status={new_status!r})"
        )

        return RegisteredModel(
            id=entry.id,
            model_version=entry.model_version,
            status=entry.status,
            primary_metric=entry.primary_metric,
            model_path=entry.model_path,
            metrics_path=entry.metrics_path,
            mlflow_run_id=entry.mlflow_run_id,
            model_uri=entry.model_uri,
        )

    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()


def get_best_model() -> ModelRegistry:
    """Returns the current champion model, or raises LookupError if none exists."""
    session_factory = _create_session_factory()
    session = session_factory()

    try:
        model = session.execute(
            select(ModelRegistry).where(ModelRegistry.status == "best")
        ).scalar_one_or_none()

        if model is None:
            raise LookupError("No model with status='best' found in the registry.")

        return model
    finally:
        session.close()


def get_model_by_version(model_version: str) -> ModelRegistry:
    """Returns one model entry by its version string, e.g. 'v001'."""
    session_factory = _create_session_factory()
    session = session_factory()

    try:
        model = session.execute(
            select(ModelRegistry).where(ModelRegistry.model_version == model_version)
        ).scalar_one_or_none()

        if model is None:
            raise LookupError(f"Model version {model_version!r} not found.")

        return model
    finally:
        session.close()
