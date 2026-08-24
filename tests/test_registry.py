import pytest
from unittest.mock import patch 

from app.models.registry import (
    build_registry_metadata,
    get_model_uri_from_run,
    set_model_version_tags,
)


def test_get_model_uri_from_run():
    uri = get_model_uri_from_run(
        run_id= 'abc123',
        artifact_path= 'model',
    )

    assert uri == 'runs:/abc123/model'


def test_get_model_uri_from_run_requires_run_id():
    with pytest.raises(ValueError, match="run_id is required"):
        get_model_uri_from_run(run_id="")


def test_build_registry_metadata():
    config = {
        'registry': {
            'registered_model_name': 'fraud-risk-model',
            'alias': 'champion',
        },
        'model': {
            'name': 'baseline_logistic',
            'type': 'logistic_regression',
        },
    }

    training_metrics = {
        'mlflow_run_id': 'abc123',
        'artifact_path': 'artifacts/models/baseline_logistic.joblib',
    }

    evaluation_metrics = {
        'selected_threshold': 0.64,
        'test_roc_auc': 0.91,
        'test_pr_auc': 0.44,
        'precision': 0.36,
        'recall': 0.76,
        'f1': 0.49,
        'expected_cost': 45450.0,
        'review_rate': 0.06,
    }

    threshold_report = {
        'objective': 'min_expected_cost',
        'best_threshold_metrics': {
            'threshold': 0.64,
            'expected_cost': 44000,
        },
    }

    metadata  = build_registry_metadata(
        config= config,
        training_metrics= training_metrics,
        evaluation_metrics= evaluation_metrics,
        threshold_report= threshold_report,
        model_version= '3',
    )

    assert metadata['registered_model_name'] == 'fraud-risk-model'
    assert metadata['model_version'] == '3'
    assert metadata['selected_threshold'] == 0.64
    assert metadata['test_pr_auc'] == 0.44


@patch("app.models.registry.mlflow.tracking.MlflowClient")
def test_set_model_version_tags(mock_client_class):
    mock_client = mock_client_class.return_value

    set_model_version_tags(
        registered_model_name= 'fruad-risk-model',
        version= '1',
        tags = {
            'selected_threshold': 0.64,
            'test_pr_auc': 0.44,
        },
    )

    assert mock_client.set_model_version_tag.call_count == 2