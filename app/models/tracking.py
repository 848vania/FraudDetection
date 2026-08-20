from pathlib import Path
from typing import Any

import mlflow 
import mlflow.sklearn
from sklearn.pipeline import Pipeline


def setup_mlflow(config: dict[str, Any]) -> None:
    """
    Configure MLflow tracking URI and experiment name

    For local development, tracking_uri can be 'sqlite:///mlflow.db'
    """
    experiment_config = config.get('experiment', {})

    tracking_uri = experiment_config.get('tracking_uri', 'sqlite:///mlflow.db')
    experiment_name = experiment_config.get('name', 'fraud-risk-experiments')

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def flatten_dict(
        data: dict[str, Any],
        parent_key: str = "",
        separator: str = ".",
    ) -> dict[str, Any]:
    """
    Flatten nested dictionary for MLflow parameter logging
    """
    items = {}

    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key 

        if isinstance(value, dict):
            items.update(
                flatten_dict(
                    value,
                    parent_key= new_key,
                    separator= separator,
                )
            )
        else:
            items[new_key] = value

    return items 


def log_config_params(config: dict[str, Any]) -> None:
    """
    Log config values as MLflow parameters
    """
    flattened_config = flatten_dict(config)

    for key, value in flattened_config.items():
        # MLflow params should be simple string-like values
        mlflow.log_param(key, value)


def log_training_metrics(metrics: dict[str, float]) -> None:
    """
    Log training metrics to MLflow
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_artifact_file(artifact_path: str | Path) -> None:
    """
    Log a file artifact to MLFlow if it exists
    """
    artifact_path = Path(artifact_path)

    if artifact_path.exists():
        mlflow.log_artifact(str(artifact_path))


def log_sklearn_model(
        model_pipeline: Pipeline,
        artifact_path: str = 'model',
    ) -> None:
    """
    Log fitted sklearn model pipeline to MLflow
    """
    mlflow.sklearn.log_model(
        sk_model= model_pipeline,
        artifact_path= artifact_path,
        skops_trusted_types= ['numpy.dtype'],
    )


def get_active_run_id() -> str | None:
    """
    REturn active MLflow run ID if one exists
    """
    active_run = mlflow.active_run()

    if active_run is None:
        return None 

    return active_run.info.run_id