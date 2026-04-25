from __future__ import annotations

"""
Prometheus metrics for the MLOps platform.

Provides:
  - HTTP request counter and latency histogram (via ASGI middleware)
  - ML-specific metrics: training duration, failed jobs, inference latency
  - /metrics endpoint for Prometheus scraping

Usage in main.py:
    from app.services.metrics import instrument_app
    instrument_app(app)
"""

import time

from fastapi import APIRouter, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

router = APIRouter()

# -+--  HTTP metrics: updated by the ASGI middleware below  --+-

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests.",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

HTTP_ERRORS_TOTAL = Counter(
    "http_errors_total",
    "Total number of HTTP 4xx/5xx responses.",
    ["method", "endpoint", "status_code"],
)

# -+--  Training / worker metrics: updated by "training_worker.py" helpers  --+-

TRAINING_DURATION_SECONDS = Histogram(
    "training_duration_seconds",
    "Time taken to complete one training run (end-to-end, including dataset prep).",
    buckets=[10, 30, 60, 120, 300, 600, 1200, 3600],
)

TRAINING_JOBS_TOTAL = Counter(
    "training_jobs_total",
    "Total number of training jobs dispatched.",
    ["status"],   # queued | completed | failed
)

FAILED_JOBS_TOTAL = Counter(
    "training_jobs_failed_total",
    'Total number of training jobs that ended in the "failed" state.',
)

MODELS_REGISTERED_TOTAL = Counter(
    "models_registered_total",
    'Total number of model versions registered in the "model_registry" table.',
    ["status"],   # best | validated
)

# -+--  Inference metrics: updated inside "predict_fraud()"  --+-

INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of /predict calls.",
)

INFERENCE_DURATION_SECONDS = Histogram(
    "inference_duration_seconds",
    "Time taken to return one prediction.",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

INFERENCE_ERRORS_TOTAL = Counter(
    "inference_errors_total",
    "Total number of prediction requests that resulted in an error.",
)

ACTIVE_MODEL_VERSION = Gauge(
    "active_model_version_info",
    "Metadata about the currently loaded model (always 1, labels carry the info).",
    ["model_version"],
)

# -+--  /metrics endpoint  --+-

@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape endpoint: exposes all registered metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# -+--  ASGI middleware: records HTTP metrics for every request automatically  --+-

def _normalise_path(path: str) -> str:
    """
    Replaces dynamic path segments with placeholders so Prometheus doesn't
    create a new label value for every unique "job_id" or "model_version".

    e.g. /jobs/train_abc123 -> /jobs/{job_id}
    """
    import re
    path = re.sub(r"/jobs/[^/]+", "/jobs/{job_id}", path)
    path = re.sub(r"/models/[^/]+", "/models/{model_version}", path)
    return path


class PrometheusMiddleware:
    """
    Lightweight ASGI middleware that records per-request HTTP metrics.
    Attach to the FastAPI app via "instrument_app()".
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        method = request.method
        endpoint = _normalise_path(request.url.path)

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            status_str = str(status_code)

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_str,
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            if status_code >= 400:
                HTTP_ERRORS_TOTAL.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_str,
                ).inc()


def instrument_app(app) -> None:
    """Attaches the Prometheus middleware and registers the /metrics route."""
    app.add_middleware(PrometheusMiddleware)
    app.include_router(router)
