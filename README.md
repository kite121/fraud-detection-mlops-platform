# fraud-detection-mlops-platform

Distributed MLOps platform for fraud detection on tabular transaction data, covering raw data ingestion, training dataset preparation, model training and registry, experiment tracking with MLflow, asynchronous orchestration, inference serving, monitoring, drift detection, and auto-retraining.

## Overview

This project is designed as an end-to-end MLOps platform for a fraud detection use case. Instead of treating model training as a standalone notebook task, the repository organizes the full ML lifecycle into platform layers:

- data ingestion and raw storage;
- metadata tracking in PostgreSQL;
- training dataset preparation from versioned raw batches;
- model training and evaluation;
- model registry and artifact versioning;
- MLflow experiment tracking;
- asynchronous training orchestration;
- online inference service;
- monitoring, drift checks, and retraining triggers.

The platform is built around containerized services and is intended to be run locally through Docker Compose as a lightweight distributed system.

## Project goal

The main goal of the project is to provide a reproducible and extensible fraud detection platform that can:

- accept transaction datasets from clients;
- validate and persist raw data safely;
- prepare consistent train and validation datasets;
- train and evaluate baseline fraud models;
- version and register trained models;
- expose an inference API for online scoring;
- detect data drift and quality degradation;
- trigger retraining when conditions require it.

## What problem it solves

Fraud detection systems usually need more than just a model. In practice, teams need a platform that handles:

- raw data delivery;
- dataset lineage;
- reproducible training runs;
- artifact storage;
- model version control;
- inference deployment;
- operational monitoring.

This repository is structured exactly around that idea: a small but realistic MLOps platform for the fraud detection lifecycle.

## Architecture

```mermaid
flowchart TB
    U[Client / User]

    subgraph API[Application Layer]
        TA[Training API]
        IA[Inference API]
    end

    subgraph ORCH[Orchestration Layer]
        R[RabbitMQ]
        W[Worker Service]
    end

    subgraph DATA[Storage and Tracking Layer]
        DB[PostgreSQL]
        M[MinIO]
        ML[MLflow]
    end

    subgraph OBS[Observability Layer]
        P[Prometheus]
        J[Jaeger]
    end

    U -->|HTTP: ingest, train, jobs, monitor| TA
    U -->|HTTP: predict| IA

    TA -->|store metadata| DB
    TA -->|store raw batches| M
    TA -->|publish jobs / events| R

    R -->|training / retraining tasks| W

    W -->|update jobs / registry| DB
    W -->|store models, metrics, drift artifacts| M
    W -->|log runs, params, metrics| ML
    W -->|publish deployment / retraining events| R

    IA -->|read best model metadata| DB
    IA -->|load model artifacts| M
    R -.->|model_deployed event| IA

    P -.->|runtime metrics| TA
    P -.->|runtime metrics| IA
    P -.->|runtime metrics| W

    J -.->|distributed traces| TA
    J -.->|distributed traces| IA
    J -.->|distributed traces| W
```

## Platform layers

### 1. Ingestion layer

The ingestion layer is responsible for accepting raw CSV files and registering them as versioned batches.

Core responsibilities:

- accept CSV uploads through `POST /ingest`;
- validate file structure and required columns;
- store raw data in MinIO;
- save batch metadata in PostgreSQL;
- return batch identifiers and storage paths.

This layer intentionally does not perform heavy preprocessing. Raw data remains raw, while downstream services build training-ready datasets separately.

### 2. Training dataset layer

The dataset preparation layer loads one uploaded raw batch and transforms it into a training-ready fraud detection dataset.

Core responsibilities:

- load a selected raw CSV from MinIO;
- resolve the correct batch through metadata;
- select features and target;
- create engineered balance-based features;
- split into train and validation with stratification;
- return both processed data and metadata suitable for experiment logging.

This separation between raw layer and training layer is one of the key design decisions in the project.

### 3. Training and evaluation layer

The training layer is responsible for fitting a baseline fraud detection model and computing the right evaluation metrics.

Expected responsibilities across the training flow:

- baseline model training for binary classification;
- reproducible hyperparameter handling;
- calculation of fraud-relevant metrics such as:
  - `ROC-AUC`
  - `Precision`
  - `Recall`
  - `F1-score`
  - optionally `PR-AUC`
- serialization of model and preprocessing artifacts;
- registration of model versions.

For fraud detection, the project explicitly treats `accuracy` as insufficient and focuses on imbalance-aware metrics.

### 4. MLflow tracking layer

The project also includes an MLflow layer on top of the training stack.

MLflow is used for:

- logging hyperparameters;
- logging dataset metadata;
- logging training and validation metrics;
- storing model artifacts;
- comparing training runs;
- linking registry entries to experiment runs.

Important design choice:

- PostgreSQL and MinIO remain the operational storage backbone;
- MLflow is added as an experiment tracking and artifact logging layer, not as a replacement for ingestion or registry logic.

### 5. Orchestration layer

The final platform design introduces asynchronous orchestration for heavy ML tasks.

Expected responsibilities:

- receive training requests without blocking the API;
- enqueue training jobs;
- execute training and retraining through workers;
- track job state transitions;
- publish platform events through a broker.

This allows the platform to move from a synchronous demo service to a more realistic distributed MLOps system.

### 6. Inference layer

The inference layer is responsible for serving the currently best fraud model.

Core responsibilities:

- load the best registered model;
- load the compatible preprocessing artifact;
- validate incoming transaction payloads;
- return fraud prediction, probability/score, and model version.

This layer is designed as a standalone service, separated from the training environment.

### 7. Monitoring and retraining layer

The last platform layer closes the loop.

Expected responsibilities:

- expose service and ML metrics;
- track model behavior in production;
- detect drift and quality degradation;
- trigger retraining when defined rules are met;
- support observability through Prometheus and distributed tracing.

## Sprint progression

The project is naturally structured through three sprints plus an MLflow extension.

### Sprint 1 — Data ingestion and raw storage

Delivered layer:

- Dockerized ingestion setup;
- PostgreSQL + MinIO + ingestion service;
- CSV validation;
- raw file storage in object storage;
- batch metadata persistence;
- `POST /ingest` endpoint.

### Sprint 2 — Training and model registry

Delivered layer:

- training dataset preparation from raw batches;
- baseline fraud model training;
- evaluation metrics for imbalanced binary classification;
- model registry and artifact versioning;
- `POST /train` training flow;
- MLflow tracking integration.

### Sprint 3 — Distributed orchestration and inference

Delivered layer:

- message broker and async execution model;
- worker-based training jobs;
- job tracking;
- standalone inference service;
- `POST /predict` endpoint;
- monitoring, drift detection, and retraining triggers.

### MLflow extension

The MLflow addition complements Sprint 1 and Sprint 2 by adding:

- experiment tracking;
- metrics logging;
- artifact logging into MinIO-backed storage;
- links between registry entries and MLflow runs.

## Current repository structure

The current branch already contains the distributed Sprint 3 components.

```text
fraud-detection-mlops-platform/
├── README.md
├── Dockerfile
├── Dockerfile.mlflow
├── docker-compose.yml
├── .env
├── .env.example
├── monitoring/
│   └── prometheus.yml
├── app/
│   ├── main.py
│   ├── config.py
│   ├── db.py
│   ├── models.py
│   ├── schemas.py
│   ├── celery_app.py
│   ├── api/
│   │   ├── ingest.py
│   │   ├── train.py
│   │   ├── jobs.py
│   │   ├── predict.py
│   │   └── monitoring.py
│   └── services/
│       ├── broker.py
│       ├── events.py
│       ├── validation.py
│       ├── storage.py
│       ├── metadata.py
│       ├── dataset.py
│       ├── training.py
│       ├── evaluation.py
│       ├── model_storage.py
│       ├── registry.py
│       ├── inference.py
│       ├── jobs.py
│       ├── monitoring.py
│       ├── drift.py
│       ├── retraining.py
│       ├── metrics.py
│       ├── tracing.py
│       └── retry.py
└── workers/
    ├── training_worker.py
    └── retraining_worker.py
```

## Runtime flow

The current operational flow is:

1. `POST /ingest` validates and stores one CSV batch.
2. The service writes `batch_metadata` to PostgreSQL and publishes `data_ingested`.
3. The training service listens to `data_ingested` and runs drift monitoring in the background.
4. If drift is above threshold, a retraining job is created automatically and dispatched to Celery.
5. `POST /train` can also manually enqueue an explicit training job.
6. Worker processes run training or retraining asynchronously.
7. The best model is registered in PostgreSQL and artifacts are saved to MinIO.
8. If a newly trained model becomes `best`, the platform publishes `model_deployed`.
9. The inference service listens for `model_deployed` and reloads the active model automatically.
10. `POST /predict` serves online fraud predictions from the current best model.

## Dataset assumptions

The platform expects PaySim-style transaction data with at least these columns:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

## Technology stack

- `Python`
- `FastAPI`
- `Celery`
- `RabbitMQ`
- `SQLAlchemy`
- `PostgreSQL`
- `MinIO`
- `Pandas`
- `scikit-learn`
- `CatBoost`
- `MLflow`
- `Prometheus`
- `Jaeger`
- `Docker Compose`

## Service map

When the stack is up, the important endpoints are:

- training API: [http://localhost:8000](http://localhost:8000)
- inference API: [http://localhost:8001](http://localhost:8001)
- worker metrics endpoint: [http://localhost:8002/metrics](http://localhost:8002/metrics)
- MLflow UI: [http://localhost:5000](http://localhost:5000)
- MinIO Console: [http://localhost:9001](http://localhost:9001)
- RabbitMQ Management: [http://localhost:15672](http://localhost:15672)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Jaeger: [http://localhost:16686](http://localhost:16686)

## Runbook

This section is the recommended end-to-end demo flow for the current branch.

### 1. Prepare `.env`

Copy `.env.example` to `.env` and fill in the required values.

Minimal local example:

```env
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=mlops_db
POSTGRES_USER=mlops_user
POSTGRES_PASSWORD=mlops_password

MINIO_ENDPOINT=minio:9000
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=password
MINIO_SECURE=false

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=fraud-detection-training
MLFLOW_ARTIFACTS_BUCKET=mlflow-artifacts
AWS_ACCESS_KEY_ID=admin
AWS_SECRET_ACCESS_KEY=password
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/%2F

CELERY_BROKER_URL=amqp://guest:guest@rabbitmq:5672/%2F
CELERY_RESULT_BACKEND=rpc://
```

### 2. Start the full stack

```bash
docker compose up --build
```

Wait until:

- `postgres` becomes healthy
- `minio` becomes healthy
- `rabbitmq` becomes healthy
- `mlflow` starts
- `ingestion-service`, `inference-service`, and `worker-service` are running

### 3. Verify health

Training service:

```bash
curl http://localhost:8000/health
```

Inference service:

```bash
curl http://localhost:8001/health
```

### 4. Upload a dataset batch

Use any CSV file from the following dataset:

- [CiferAI/Cifer-Fraud-Detection-Dataset-AF](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF)

Any CSV file from that dataset can be used to verify the service locally. Replace
`<path-to-csv-file>` below with the actual path to the downloaded file.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "client_id=test-client" \
  -F "dataset_version=v1" \
  -F "file=@<path-to-csv-file>"
```

Expected response:

- `status`
- `batch_id`
- `rows_count`
- `storage_path`

Keep the returned `batch_id`.

### 5. Enqueue training manually

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": <batch_id-from-step-4>,
    "notes": "manual demo run"
  }'
```

Expected response:

- `status: queued`
- `job_id`
- `message`

Keep the returned `job_id`.

### 6. Check training job status

```bash
curl http://localhost:8000/jobs/<job_id>
```

Status lifecycle:

- `queued`
- `running`
- `completed`
- `failed`

When completed, the response should include at least:

- `model_version`
- `mlflow_run_id`

### 7. Inspect MLflow

Open [http://localhost:5000](http://localhost:5000).

You should see:

- a training run
- logged hyperparameters
- dataset metadata
- evaluation metrics such as `roc_auc`, `pr_auc`, `precision`, `recall`, `f1_score`
- logged model and preprocessor artifacts

### 8. Run online inference

Call the inference service directly:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "PAYMENT",
    "amount": 100.0,
    "oldbalanceOrg": 500.0,
    "newbalanceOrig": 400.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 100.0,
    "isFlaggedFraud": 0
  }'
```

Expected response:

- `prediction`
- `fraud_score`
- `model_version`

### 9. Run monitoring manually

You can run one explicit drift check through the training API. This requires two
already ingested batches: one reference batch and one current batch.

```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_batch_id": <reference_batch_id>,
    "current_batch_id": <current_batch_id>
  }'
```

Expected response:

- `degraded`
- `max_feature_psi`
- `positive_rate_delta`
- `reference_profile_path`
- `drift_result_path`

### 10. Understand auto-monitoring and auto-retraining

The branch also supports an automatic runtime flow:

- after every successful `POST /ingest`, the service publishes `data_ingested`
- the training service consumes `data_ingested`
- it uses the current `best` model's `training_batch_id` as the reference batch
- it runs `monitor_model(...)`
- if `degraded == true`, it creates a retraining job automatically
- the worker executes retraining through Celery
- if the retrained model becomes the new `best`, `model_deployed` is published
- the inference service listens to `model_deployed` and reloads the active model automatically

This means that monitoring is available both:

- manually through `POST /monitor`
- automatically after new data ingestion

### 11. Inspect raw data, models, and artifacts

Use MinIO Console at [http://localhost:9001](http://localhost:9001).

Expected buckets:

- `raw-data`
- `models`
- `mlflow-artifacts`
- `monitoring-artifacts` may appear after monitoring/drift runs

### 12. Inspect observability

Prometheus:

- open [http://localhost:9090](http://localhost:9090)
- query metrics such as:
  - `http_requests_total`
  - `http_request_duration_seconds`
  - `training_jobs_total`
  - `training_duration_seconds`
  - `models_registered_total`
  - `inference_requests_total`
  - `inference_duration_seconds`
  - `active_model_version_info`

Jaeger:

- open [http://localhost:16686](http://localhost:16686)
- inspect traces for:
  - `POST /train`
  - `train_model_task`
  - `predict_fraud`
  - broker publish/consume spans

## What to expect from a successful demo

The branch should demonstrate the following complete path:

1. upload data with `POST /ingest`
2. enqueue training with `POST /train`
3. track job progress with `GET /jobs/{job_id}`
4. inspect the run in MLflow
5. serve predictions through `POST /predict`
6. inspect metrics in Prometheus
7. inspect traces in Jaeger
8. run drift monitoring manually with `POST /monitor`
9. allow automatic monitoring and retraining after new data ingestion

## Short summary

`fraud-detection-mlops-platform` is a distributed MLOps demo platform for fraud detection. It covers ingestion, async training orchestration, model registry, experiment tracking, inference serving, monitoring, drift checks, event-driven retraining, and observability in one reproducible Docker Compose stack.
