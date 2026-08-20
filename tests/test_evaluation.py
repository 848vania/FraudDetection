from pathlib import Path

import pandas as pd 

from app.models.evaluate import (
    build_evaluation_report,
    calculate_probability_metrics,
    predict_probabilities,
    save_json,
)


class FakeModel:
    def predict_proba(self, X):
        return [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]


def test_predict_probabilities():
    model = FakeModel()
    X = pd.DataFrame({'feature': [1, 2, 3, 4]})

    probabilities = predict_probabilities(model, X)

    assert probabilities == [0.1, 0.2, 0.8, 0.9]


def test_calculate_probability_metrics():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    metrics = calculate_probability_metrics(
        y_true = y_true,
        y_proba = y_proba,
        prefix = 'test',
    )

    assert metrics['test_roc_auc'] == 1.0
    assert metrics['test_pr_auc'] == 1.0


def test_build_evaluation_report():
    probability_metrics = {
        'test_roc_auc': 0.9,
        'test_pr_auc': 0.5,
    }

    threshold_metrics = {
        'precision': 0.4,
        'recall': 0.8,
        'expected_cost': 1000,
    }

    report = build_evaluation_report(
        model_name= 'baseline_logistic',
        model_path= 'artifacts/models/model.joblib',
        selected_threshold= 0.7,
        probability_metrics= probability_metrics,
        threshold_metrics= threshold_metrics,
    )

    assert report['model_name'] == 'baseline_logistic'
    assert report['selected_threshold'] == 0.7
    assert report['test_roc_auc'] == 0.9
    assert report['expected_roc'] == 1000


def test_save_json(tmp_path):
    output_path = tmp_path / 'report.json'
    data = {'metric': 0.9}

    save_json(data, output_path)

    assert output_path.exists()