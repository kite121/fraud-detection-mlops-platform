from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


_TRACING_INITIALIZED = False
_CELERY_INSTRUMENTED = False


def resolve_service_name() -> str:
    app_role = os.getenv("APP_ROLE", "training")
    mapping = {
        "training": "ingestion-service",
        "inference": "inference-service",
        "worker": "worker-service",
    }
    return mapping.get(app_role, app_role)


def init_tracing(service_name: str | None = None) -> None:
    global _TRACING_INITIALIZED

    if _TRACING_INITIALIZED:
        return

    exporter = OTLPSpanExporter(
        endpoint=os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://jaeger:4318/v1/traces",
        )
    )
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": service_name or resolve_service_name()}
        )
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACING_INITIALIZED = True


def instrument_fastapi_app(app: FastAPI) -> None:
    init_tracing()
    FastAPIInstrumentor.instrument_app(app)


def instrument_celery_app() -> None:
    global _CELERY_INSTRUMENTED

    init_tracing()
    if _CELERY_INSTRUMENTED:
        return

    CeleryInstrumentor().instrument()
    _CELERY_INSTRUMENTED = True


def get_tracer(name: str):
    init_tracing()
    return trace.get_tracer(name)


def inject_trace_headers(headers: dict[str, Any] | None = None) -> dict[str, Any]:
    carrier: dict[str, Any] = dict(headers or {})
    propagate.inject(carrier)
    return carrier


def extract_trace_context(headers: dict[str, Any] | None = None):
    normalized_headers: dict[str, str] = {}
    for key, value in (headers or {}).items():
        if isinstance(value, bytes):
            normalized_headers[key] = value.decode("utf-8")
        else:
            normalized_headers[key] = str(value)

    return propagate.extract(normalized_headers)
