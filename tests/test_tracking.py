from pathlib import Path
from unittest.mock import patch

from app.models.tracking import (
    flatten_dict,
    log_config_params,
    log_training_metrics,
)


def test_flatten_dict():
    config = {
        "model": {
            "type": "logistic_regression",
            "max_iter": 1000,
        },
        "training": {
            "target_col": "is_fraud",
        },
    }

    result = flatten_dict(config)

    assert result['model.type'] == 'logistic_regression'
    assert result['model.max_iter'] == 1000
    assert result['training.target_col'] == 'is_fraud'


@patch("app.models.tracking.mlflow.log_param")
def test_log_config_params(mock_log_param):
    config = {
        'model': {
            'type': 'logistic_regression',
            'max_iter': 1000,
        }
    }

    log_config_params(config)

    assert mock_log_param.call_count == 2


@patch("app.models.tracking.mlflow.log_metric")
def test_log_training_metrics(mock_log_metric):
    metrics = {
        'valid_roc_auc': 0.91,
        'valid_pr_auc': 0.42,
    }

    log_training_metrics(metrics)

    assert mock_log_metric.call_count == 2