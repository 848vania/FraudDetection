import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc

from app.models.tracking import setup_mlflow
from app.models.train import load_model_artifact


def load_json_file(path: str | Path)  -> dict[str, Any]:
    """
    Load a JSON file

    Parameters
    -----------
    path:
        Path to JSON file 

    Returns
    --------
    dict[str, Any]
        Parsed JSON content
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def save_json_file(
        data: dict[str, Any],
        path: str | Path,
    ) -> None:
    """
    Save dictionary as JSON
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def get_model_uri_from_run(
        run_id: str,
        artifact_path: str  = 'model',
    ) -> str:
    """
    Build MLflow model URI from run ID
    """
    if not run_id:
        raise ValueError('run_id is required to build model URI')

    return f'runs:/{run_id}/{artifact_path}'


def register_model_from_run(
        run_id: str,
        registered_model_name: str,
        artifact_path: str = "model",
    ):
    """
    Register a model from an MLflow run
    """
    model_uri = get_model_uri_from_run(
        run_id= run_id,
        artifact_path= artifact_path,
    )

    model_version = mlflow.register_model(
        model_uri= model_uri,
        name= registered_model_name,
    )

    return model_version


def set_model_alias(
        registered_model_name: str,
        version: str,
        alias: str,
    ) -> None:
    """
    Set alias for registered model version

    Example:
        fraud-risk-model version 3 -> alias champion
    """
    client = mlflow.tracking.MlflowClient()

    client.set_registered_model_alias(
        name = registered_model_name,
        alias= alias,
        version= version,
    )


def set_model_version_tags(
        registered_model_name: str,
        version: str, 
        tags: dict[str, Any],
    ) -> None:
    """
    Set tags on a registered model version
    """
    client = mlflow.tracking.MlflowClient()

    for key, value in tags.items():
        if value is None:
            continue 

        client.set_model_version_tag(
            name= registered_model_name,
            version= version,
            key= key,
            value= str(value),
        )


def build_registry_metadata(
        config: dict[str, Any],
        training_metrics: dict[str, Any],
        evaluation_metrics: dict[str, Any],
        threshold_report: dict[str, Any],
        model_version: str | None = None,
    ) -> dict[str, Any]:
    """
    Build metadata for registered model 
    """
    registry_config = config.get('registry', {})
    model_config = config.get('model', {})

    return {
        'registered_model_name': registry_config.get(
            'registered_model_name',
            'fraud-risk-model',
        ),
        'model_version': model_version,
        'alias': registry_config.get('alias', 'champion'),
        'model_name': model_config.get('name'),
        'model_type': model_config.get('type'),
        'mlflow_run_id': training_metrics.get('mlflow_run_id'),
        'artifact_path': training_metrics.get('artifact_path'),
        'selected_threshold': evaluation_metrics.get('selected_threshold'),
        'threshold_objective': threshold_report.get('objective'),
        'valid_best_threshold_metrics': threshold_report.get(
            'best_threshold_metrics',
            {},
        ),
        'test_roc_auc': evaluation_metrics.get('test_roc_auc'),
        'test_pr_auc': evaluation_metrics.get('test_pr_auc'),
        'test_precision': evaluation_metrics.get('precision'),
        'test_recall': evaluation_metrics.get('recall'),
        'test_f1': evaluation_metrics.get('f1'),
        'test_expected_cost': evaluation_metrics.get('expected_cost'),
        'test_review_rate': evaluation_metrics.get('review_rate'),
        'registered_at': datetime.now(timezone.utc).isoformat(),
    }


def register_model_version(config: dict[str, Any]) -> dict[str, Any]:
    """
    Register the trained model in MLflow Model Registry 

    Returns 
    --------
    dict[str, Any]
        Registered model metadata
    """
    setup_mlflow(config)

    training_metrics = load_json_file(
        config['artifacts']['metrics_output_path']
    )
    evaluation_metrics = load_json_file(
        config['artifacts']['evaluation_metrics_path']
    )
    threshold_report = load_json_file(
        config['artifacts']['threshold_report_path']
    )

    run_id = training_metrics.get('mlflow_run_id')

    if not run_id:
        raise ValueError(
            "Cannot register model because mlflow_run_id is missing. "
            "Train with MLflow enabled first."
        )

    registered_model_name = config['registry']['registered_model_name']
    alias = config['registry'].get('alias', 'champion')

    model_version = register_model_from_run(
        run_id= run_id,
        registered_model_name= registered_model_name,
        artifact_path= 'model',
    )

    version = str(model_version.version)

    set_model_alias(
        registered_model_name= registered_model_name,
        version= version,
        alias= alias,
    )

    metadata  = build_registry_metadata(
        config= config,
        training_metrics= training_metrics,
        evaluation_metrics= evaluation_metrics,
        threshold_report= threshold_report,
        model_version= version,
    )

    set_model_version_tags(
        registered_model_name= registered_model_name,
        version= version,
        tags= {
            'model_type': metadata['model_type'],
            'selected_threshold': metadata['selected_threshold'],
            'test_roc_auc': metadata['test_roc_auc'],
            'test_pr_auc': metadata['test_pr_auc'],
            'test_precision': metadata['test_precision'],
            'test_recall': metadata['test_recall'],
            'test_expected_cost': metadata['test_expected_cost'],
        },
    )

    save_json_file(
        metadata,
        config['artifacts']['registered_model_metadata_path'],
    )

    return metadata


def load_registered_model(
        registered_model_name: str,
        alias: str = "champion",
    ):
    """
    Load registered model by alias 

    Returns an MLflow pyfunc model
    """
    model_uri = f"models:/{registered_model_name}@{alias}"

    return mlflow.pyfunc.load_model(model_uri)


def load_model_for_inference(
        config: dict[str, Any],
        prefer_registry: bool = True
    ):
    """
    Load model for inference 

    Prefer MLflow registry if available
    Fall back to local joblib artifact
    """
    if prefer_registry:
        try: 
            setup_mlflow(config)

            return load_registered_model(
                registered_model_name= config['registry']['registered_model_name'],
                alias = config['registry'].get('alias', 'champion'),
            )
        except Exception as error:
            print(f"Failed to load registered model. Falling back to local artifact: {error}")

    return load_model_artifact(
        config['artifacts']['model_output_path']
    )


def load_registered_model_metadata(
        config: dict[str, Any],
    ) -> dict[str, Any]:
    """
    Load registered model metadata JSON
    """
    metadata_path = config['artifacts']['registered_model_metadata_path']

    return load_json_file(metadata_path)