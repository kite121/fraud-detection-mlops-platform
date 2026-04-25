from __future__ import annotations

"""
Backward-compatible shim for the old app.workers import path.

The active Celery tasks live in the top-level ``workers`` package and are
registered via ``app.celery_app``. Keep this module thin so stale imports from
main do not break after the merge.
"""

from workers.training_worker import train_model_task as run_training_job
from workers.training_worker import train_model_task


__all__ = ["run_training_job", "train_model_task"]
