# This file reads environment variables and provides access to them through the convenient "settings" object
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # --- PostgreSQL ---
    postgres_host: str
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # --- MinIO ---
    minio_endpoint: str
    minio_root_user: str
    minio_root_password: str
    minio_secure: bool = False

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "fraud-detection-training"
    mlflow_artifacts_bucket: str = "mlflow-artifacts"

    # --- Celery / RabbitMQ ---
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672//"

    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "rpc://"
    celery_task_always_eager: bool = False

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"


settings = Settings()
