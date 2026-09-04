"""MLflow tracking for routing and scheduling runs.

Every training run logs its configuration and results, so a number in the README can be
traced back to the run that produced it. Optional: without MLflow installed the helpers
no-op and the scripts still run.
"""
from __future__ import annotations

import os
from contextlib import contextmanager


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def _noop(params=None, metrics=None, artifact=None) -> None:
    return None


def _log(params=None, metrics=None, artifact=None) -> None:
    import mlflow

    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if artifact and os.path.exists(artifact):
        mlflow.log_artifact(artifact)


@contextmanager
def track(experiment: str, run_name: str):
    """Yield a logger; a no-op when MLflow is unavailable."""
    if not mlflow_available():
        yield _noop
        return

    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield _log
