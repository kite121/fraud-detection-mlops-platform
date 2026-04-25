from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db import Base, engine, check_db_connection
from app.api import ingest
from app.api import predict
from app.api import train
from app.api import jobs  # Sprint 3 Task 4
from app.services.registry import init_registry_table
from app.services.metrics import instrument_app  # Sprint 3 Task 8


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -+--  startup  --+-
    print("[DB] Creating tables...")
    Base.metadata.create_all(bind=engine)
    init_registry_table()

    if check_db_connection():
        print("[DB] Connecting to PostgreSQL: OK")
    else:
        raise RuntimeError("Couldn't connect to PostgreSQL!")

    yield

    # -+--  shutdown  --+-
    engine.dispose()
    print("[DB] The database connection is closed.")


app = FastAPI(title="Fraud Detection MLOps Platform", lifespan=lifespan)

# Sprint 3 Task 8: attach Prometheus middleware and /metrics route
instrument_app(app)

app.include_router(ingest.router)
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(jobs.router)  # Sprint 3 Task 4: GET /jobs/{job_id}


@app.get("/health")
def health():
    return {"status": "ok", "db": check_db_connection()}
