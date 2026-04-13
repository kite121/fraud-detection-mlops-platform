# This creates the SQLAlchemy engine and session

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import settings

# Database connection engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # checks the connection before each request
    pool_size=5,
    max_overflow=10,
)

# Session Factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# Base class for all ORM models (task 4 will inherit from it)
class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency for FastAPI — opens and closes a session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """Checks that the PostgreSQL connection is working."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[DB] Ошибка подключения: {e}")
        return False
      
