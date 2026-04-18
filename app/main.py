from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db import Base, engine, check_db_connection
from app.api import ingest
from app.api import train
from app.services.registry import init_registry_table


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    print("[DB] Creating tables...")
    Base.metadata.create_all(bind=engine)   # BatchMetadata and other shared models
    init_registry_table()                   # model_registry (Sprint 2)

    if check_db_connection():
        print("[DB] Connecting to PostgreSQL: OK")
    else:
        raise RuntimeError("Couldn't connect to PostgreSQL!")

    yield  # app is running

    # --- shutdown ---
    engine.dispose()
    print("[DB] The database connection is closed.")


app = FastAPI(title="Fraud Detection MLOps Platform", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(train.router)


@app.get("/health")
def health():
    return {"status": "ok", "db": check_db_connection()}
