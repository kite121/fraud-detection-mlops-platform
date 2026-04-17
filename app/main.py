# Add DATABASE initialization

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db import Base, engine, check_db_connection
from app.api import ingest  # роутер коллеги из задания 8
from app.models import BatchMetadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    # !!! Starting !!!
    print("[DB] Creating tables...")
    Base.metadata.create_all(bind=engine)  # creates all tables from models.py
    
    if check_db_connection():
        print("[DB] Connecting to PostgreSQL: OK")
    else:
        raise RuntimeError("Couldn't connect to PostgreSQL!")
    
    yield  # This is where the app works
    
    # !!! Ending !!!
    engine.dispose()
    print("[DB] The database connection is closed.")


app = FastAPI(title="Fraud Detection Ingestion Service", lifespan=lifespan)

app.include_router(ingest.router)


@app.get("/health")
def health():
    return {"status": "ok", "db": check_db_connection()}
