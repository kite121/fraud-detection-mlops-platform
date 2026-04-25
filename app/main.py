import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import ingest, jobs, predict, train
from app.db import Base, check_db_connection, engine
from app.services.events import ensure_required_queues
from app.services.inference import load_best_model
from app.services.metrics import instrument_app
from app.services.registry import init_registry_table


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

    if app_role != "inference":
        ensure_required_queues()
        print("[Broker] RabbitMQ queues are ready.")
    else:
        try:
            load_best_model()
            print("[Inference] Best model loaded.")
        except Exception as error:
            print(f"[Inference] Model is not ready yet: {error}")

    yield

    engine.dispose()
    print("[DB] The database connection is closed.")


app = FastAPI(title="Fraud Detection MLOps Platform", lifespan=lifespan)

# Sprint 3 Task 8: attach Prometheus middleware and /metrics route
instrument_app(app)

if os.getenv("APP_ROLE", "training") == "inference":
    app.include_router(predict.router)
else:
    app.include_router(ingest.router)
    app.include_router(train.router)
    app.include_router(jobs.router)
    app.include_router(predict.router)


@app.get("/health")
def health():
    return {"status": "ok", "db": check_db_connection()}
