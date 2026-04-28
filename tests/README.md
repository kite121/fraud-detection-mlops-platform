# Test Results Summary

This folder contains the collected evidence for validating the distributed MLOps platform for fraud detection. Each test case has its own subdirectory `T-01` through `T-20` with:

- `notes.txt` - test goal, steps, expected and actual result, key IDs, and final status
- `outputs.txt` - executed commands and raw outputs
- `screenshots/` - visual evidence used in the report and demo

## Overall Outcome

All 20 planned tests were executed and all 20 passed.

| Metric | Value |
|---|---:|
| Total tests | 20 |
| Passed | 20 |
| Failed | 0 |
| Coverage areas | Startup, ingest, async training, jobs, MLflow, inference, monitoring, retraining, deployment, observability, fault tolerance |

## Test Matrix

| ID | Test Name | What Was Verified | Result | Key Observed Outcome |
|---|---|---|---|---|
| T-01 | Platform startup verification | Full Docker stack startup and API health checks | PASS | All required services started successfully; both `/health` endpoints returned OK |
| T-02 | `POST /ingest` verification | CSV upload, MinIO raw storage, PostgreSQL metadata | PASS | Batch `8` uploaded successfully with matching MinIO object and `batch_metadata` row |
| T-03 | Asynchronous training orchestration verification | Non-blocking `POST /train` behavior | PASS | API returned immediately with queued job `train_dcc0e0ceb95b` |
| T-04 | Training job lifecycle verification | `GET /jobs/{job_id}` and terminal metadata | PASS | Job `train_dcc0e0ceb95b` completed with model `v002` and MLflow run ID |
| T-05 | Worker-based training execution verification | Separate worker-side model training | PASS | Worker logs showed full task execution independent from API |
| T-06 | MLflow tracking verification | Logged params, metrics, artifacts, and dataset metadata | PASS | Training run `15aeaca7c41a4318bde0727f99534cdc` fully recorded in MLflow |
| T-07 | Inference model reload verification | Manual reload of deployed model in inference-service | PASS | Inference reloaded and served model version `v001` successfully |
| T-08 | Online inference verification | `POST /predict` response correctness | PASS | Prediction returned `prediction`, `fraud_score`, and `model_version=v001` |
| T-09 | Manual monitoring / drift check | `POST /monitor` and monitoring artifact creation | PASS | Monitoring completed for batches `8` and `9`; drift verdict was clean |
| T-10 | Auto-monitoring trigger after ingest | Event-driven monitoring after new batch upload | PASS | New ingest triggered automatic monitoring; no retraining required |
| T-11 | Auto-retraining trigger for degraded batch | Drift-triggered retraining job creation and execution | PASS | Degraded batch `11` triggered retraining job `train_fcee91c3099e`, producing model `v003` |
| T-12 | Auto-deploy of new best model | Event-driven inference model update | PASS | Inference-service received `model_deployed` and switched from `v001` to `v003` |
| T-13 | Parallel execution of two training jobs | Multiple training jobs accepted and processed correctly | PASS | Jobs `train_81a636284e0f` and `train_53a17ed4424d` both completed successfully |
| T-14 | Prometheus metrics verification | Runtime metrics exposure after ingest, train, and predict | PASS | HTTP, training, inference, registry, and active-model metrics were present and updated |
| T-15 | Jaeger distributed tracing verification | Trace visibility for training and prediction flows | PASS | Jaeger captured both training and inference traces with expected spans |
| T-16 | RabbitMQ management verification | Queue presence and broker activity in RabbitMQ UI | PASS | Application queues and message activity were visible in management UI |
| T-17 | Artifact and metadata consistency verification | Cross-layer consistency of DB, API, MLflow, and MinIO | PASS | Batch `11`, job `train_fcee91c3099e`, model `v003`, and artifacts were fully consistent |
| T-18 | Fault tolerance - restart worker-service | Recovery after worker restart | PASS | Worker restarted, reconnected, and completed new training job `train_1fa42ea75f03` |
| T-19 | Fault tolerance - restart inference-service | Recovery after inference restart | PASS | Inference restarted cleanly and continued serving predictions with `v003` |
| T-20 | Fault tolerance - restart RabbitMQ | Recovery of Celery jobs and event listeners after broker restart | PASS | Worker and ingestion listeners reconnected after RabbitMQ restart; post-restart training and ingest flows succeeded |

## Evidence Organization

Each test directory contains the complete materials needed to support the report, demonstration, and audit trail of the validation process. For example:

- API-focused tests contain request/response captures
- workflow tests contain job IDs, model versions, and MLflow run IDs
- observability tests contain Prometheus and Jaeger screenshots
- fault-tolerance tests contain restart logs and post-recovery proof

## Key End-to-End Results

The executed tests confirm that the system:

- starts as a full distributed stack
- ingests and stores raw datasets correctly
- performs training asynchronously through a worker layer
- tracks experiments and artifacts in MLflow
- serves online predictions through a dedicated inference service
- runs manual and automatic monitoring
- triggers retraining when degradation is detected
- auto-deploys the new best model to inference
- exposes metrics through Prometheus
- exposes traces through Jaeger
- recovers from basic restarts of worker, inference, and RabbitMQ services

## Notes

- All results in this folder correspond to the tested branch recorded in the individual `notes.txt` files.
- The most important identifiers reused across tests are `batch_id`, `job_id`, `model_version`, and `mlflow_run_id`.
- For full procedural details, refer to the per-test `notes.txt` files.
