from __future__ import annotations

import os
from datetime import UTC, datetime

from dotenv import load_dotenv
from sqlalchemy import DateTime, Integer, String, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


load_dotenv()


class Base(DeclarativeBase):
    """Base class for ORM models used by the ingestion service."""


class BatchMetadata(Base):
    """Stores one uploaded batch so the platform can track files per client."""

    __tablename__ = "batch_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(100), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    rows_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


def _build_database_url() -> str:
    """Builds the PostgreSQL connection string from env vars."""

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "fraud_platform")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    if not user or not password:
        raise ValueError("POSTGRES_USER and POSTGRES_PASSWORD must be set.")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def _create_session_factory() -> sessionmaker:
    """Creates a SQLAlchemy session factory for metadata writes."""

    engine = create_engine(_build_database_url())
    return sessionmaker(bind=engine)


def init_metadata_table() -> None:
    """Creates the metadata table so the service can run on a fresh database."""

    engine = create_engine(_build_database_url())
    Base.metadata.create_all(engine)


def save_batch_metadata(
    client_id: str,
    dataset_version: str,
    file_name: str,
    storage_path: str,
    rows_count: int,
    status: str,
) -> int:
    """Saves one batch metadata record in PostgreSQL and returns its ID."""

    session_factory = _create_session_factory()
    session = session_factory()

    try:
        batch_metadata = BatchMetadata(
            client_id=client_id,
            dataset_version=dataset_version,
            file_name=file_name,
            storage_path=storage_path,
            rows_count=rows_count,
            status=status,
        )
        session.add(batch_metadata)
        session.commit()
        session.refresh(batch_metadata)
        return batch_metadata.id
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()


def get_batch_metadata(batch_id: int) -> BatchMetadata:
    """Returns one uploaded batch by ID so downstream services can reuse it."""

    session_factory = _create_session_factory()
    session = session_factory()

    try:
        statement = select(BatchMetadata).where(BatchMetadata.id == batch_id)
        batch_metadata = session.execute(statement).scalar_one_or_none()
        if batch_metadata is None:
            raise LookupError(f"Batch with id={batch_id} was not found.")

        return batch_metadata
    finally:
        session.close()


def get_latest_batch_metadata(
    dataset_version: str | None = None,
    status: str = "uploaded",
) -> BatchMetadata:
    """Returns the newest uploaded batch, optionally filtered by dataset version."""

    session_factory = _create_session_factory()
    session = session_factory()

    try:
        statement = select(BatchMetadata).where(BatchMetadata.status == status)

        if dataset_version:
            statement = statement.where(BatchMetadata.dataset_version == dataset_version)

        statement = statement.order_by(
            BatchMetadata.created_at.desc(),
            BatchMetadata.id.desc(),
        )

        batch_metadata = session.execute(statement).scalars().first()
        if batch_metadata is None:
            if dataset_version:
                raise LookupError(
                    "No uploaded batch was found for "
                    f"dataset_version={dataset_version!r}."
                )

            raise LookupError("No uploaded batches were found.")

        return batch_metadata
    finally:
        session.close()
