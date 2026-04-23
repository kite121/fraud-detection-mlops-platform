# An ORM model is a Python class that automatically turns into a PostgreSQL table
# About the "status" field:
#   "pending" — the file has been accepted but not yet uploaded
#   "uploaded" — the file has been successfully uploaded to MinIO and recorded in the database
#   "failed" — something went wrong


from datetime import datetime, timezone
from sqlalchemy import Integer, String, BigInteger, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class BatchMetadata(Base):
    __tablename__ = "batch_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(100), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    rows_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    validation_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<BatchMetadata id={self.id} file={self.file_name} "
            f"status={self.status} rows={self.rows_count}>"
        )
      
