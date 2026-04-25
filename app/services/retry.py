from __future__ import annotations

"""
Fault-tolerance utilities for external service calls.

Provides retry decorators built on tenacity for:
  - MinIO (object storage)
  - PostgreSQL (via SQLAlchemy)
  - MLflow tracking server
  - RabbitMQ broker

Usage:
    from app.services.retry import retry_minio, retry_db

    @retry_minio
    def upload_file(...):
        ...

    # Or as a direct call wrapper:
    result = retry_db(lambda: session.execute(stmt).scalar())
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# -+--  Exception type lists: tenacity needs concrete exception types  --+-

def _minio_exceptions() -> tuple[type[Exception], ...]:
    """Returns MinIO-related transient exceptions when the library is available."""
    try:
        from minio.error import S3Error
        return (S3Error, ConnectionError, TimeoutError, OSError)
    except ImportError:
        return (ConnectionError, TimeoutError, OSError)


def _db_exceptions() -> tuple[type[Exception], ...]:
    """Returns SQLAlchemy transient exceptions."""
    try:
        from sqlalchemy.exc import OperationalError, DisconnectionError
        return (OperationalError, DisconnectionError, ConnectionError)
    except ImportError:
        return (ConnectionError,)


def _broker_exceptions() -> tuple[type[Exception], ...]:
    """Returns RabbitMQ/pika transient exceptions."""
    try:
        import pika.exceptions
        return (
            pika.exceptions.AMQPConnectionError,
            pika.exceptions.ChannelClosedByBroker,
            ConnectionError,
            TimeoutError,
        )
    except ImportError:
        return (ConnectionError, TimeoutError)


def _mlflow_exceptions() -> tuple[type[Exception], ...]:
    """Returns MLflow transient exceptions."""
    try:
        from mlflow.exceptions import MlflowException
        return (MlflowException, ConnectionError, TimeoutError)
    except ImportError:
        return (ConnectionError, TimeoutError)


# -+--  Retry decorators  --+-

def retry_minio(func: F) -> F:
    """
    Retries a MinIO operation up to 5 times with exponential backoff (2s–30s).
    Use on any function that calls the MinIO SDK directly.
    """
    return retry(
        retry=retry_if_exception_type(_minio_exceptions()),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def retry_db(func: F) -> F:
    """
    Retries a database operation up to 4 times with exponential backoff (1s–20s).
    Use on session.execute() calls or any SQLAlchemy operation.
    """
    return retry(
        retry=retry_if_exception_type(_db_exceptions()),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def retry_broker(func: F) -> F:
    """
    Retries a RabbitMQ publish/consume operation up to 4 times.
    Uses random jitter to avoid thundering herd after broker restart.
    """
    return retry(
        retry=retry_if_exception_type(_broker_exceptions()),
        wait=wait_random_exponential(multiplier=1, min=1, max=15),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def retry_mlflow(func: F) -> F:
    """
    Retries an MLflow call up to 3 times with exponential backoff (2s–20s).
    MLflow tracking is non-critical: if all retries fail, the error is logged
    but callers are expected to handle it gracefully.
    """
    return retry(
        retry=retry_if_exception_type(_mlflow_exceptions()),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


# -+--  Inline call wrapper (for one-off calls without decorating a full function)  --+-

def with_retry(
    func: Callable[[], Any],
    *,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
) -> Any:
    """
    Wraps a zero-argument callable with retry logic inline.

    Example:
        result = with_retry(
            lambda: session.execute(stmt).scalar_one(),
            exceptions=_db_exceptions(),
            max_attempts=4,
        )
    """
    wrapped = retry(
        retry=retry_if_exception_type(exceptions),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)
    return wrapped()
