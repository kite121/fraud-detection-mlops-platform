import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import ingest, jobs, monitoring, predict, train
from app.db import Base, check_db_connection, engine
from app.services.events import ensure_required_queues
from app.services.inference import load_best_model, start_model_deploy_listener
from app.services.metrics import instrument_app
from app.services.registry import init_registry_table
from app.services.retraining import start_data_ingested_listener
from app.services.tracing import instrument_fastapi_app, init_tracing, resolve_service_name


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_role = os.getenv("APP_ROLE", "training")

    print("[DB] Creating tables...")
    Base.metadata.create_all(bind=engine)
    init_registry_table()

    if check_db_connection():
        print("[DB] Connecting to PostgreSQL: OK")
    else:
        raise RuntimeError("Couldn't connect to PostgreSQL!")

    if app_role == "training":
        ensure_required_queues()
        start_data_ingested_listener()
        print("[Broker] RabbitMQ queues are ready.")
    elif app_role == "inference":
        try:
            load_best_model()
            print("[Inference] Best model loaded.")
        except Exception as error:
            print(f"[Inference] Model is not ready yet: {error}")

        start_model_deploy_listener()
        print("[Inference] Model deploy listener started.")

    yield

    engine.dispose()
    print("[DB] The database connection is closed.")


init_tracing(resolve_service_name())

app = FastAPI(title="Fraud Detection MLOps Platform", lifespan=lifespan)
instrument_fastapi_app(app)
instrument_app(app)

if os.getenv("APP_ROLE", "training") == "inference":
    app.include_router(predict.router)
else:
    app.include_router(ingest.router)
    app.include_router(train.router)
    app.include_router(jobs.router)
    app.include_router(monitoring.router)


@app.get("/health")
def health():
    return {"status": "ok", "db": check_db_connection()}
