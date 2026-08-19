import json 
from pathlib import Path

import pandas as pd 

from app.features.build_features import  build_feature_dataframe, split_features_and_target
from app.models.train import (
    build_model,
    build_training_pipeline,
    calculate_training_metrics,
    fit_model,
    load_model_artifact,
    predict_validation_probabilities,
    save_model_artifact,
    save_training_metrics,
    train_model,
)

import pytest


def make_training_df(num_rows: int  = 100) -> pd.DataFrame:
    rows = []

    for i in range(num_rows):
        is_fraud = 1 if i % 10 == 0 else 0

        rows.append(
            {
                "transaction_id": f"txn_{i:06d}",
                "customer_id": f"cust_{i % 20:04d}",
                "transaction_amount": 500.0 if is_fraud else 50.0,
                "transaction_hour": 2 if is_fraud else 14,
                "merchant_category": "electronics" if is_fraud else "groceries",
                "customer_tenure_days": 10 if is_fraud else 500,
                "num_transactions_24h": 10 if is_fraud else 1,
                "num_failed_transactions_24h": 4 if is_fraud else 0,
                "avg_transaction_amount_30d": 80.0,
                "is_foreign_transaction": True if is_fraud else False,
                "device_type": "mobile",
                "country": "US",
                "previous_chargebacks": 1 if is_fraud else 0,
                "account_age_days": 10 if is_fraud else 500,
                "risk_score_external": 0.9 if is_fraud else 0.1,
                "is_fraud": is_fraud,
            }
        )

    raw_df = pd.DataFrame(rows)
    return build_feature_dataframe(raw_df, include_target=True)


def make_test_config(tmp_path: Path) -> dict:
    return {
        'model': {
            'name': 'test_logistic',
            'type': 'logistic_regression',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
        },
        'training': {
            'train_path': str(tmp_path / 'train.csv'),
            'valid_path': str(tmp_path / 'valid.csv'),
            'target_col': 'is_fraud',
        },
        'artifacts': {
            'model_output_path': str(tmp_path / 'model.joblib'),
            'metrics_output_path': str(tmp_path / 'metrics.json'),
        },
    }


def test_build_model_logistic_regression():
    config = {
        'model': {
            'type': 'logistic_regression',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
        }
    }

    model = build_model(config)

    assert model.__class__.__name__ == "LogisticRegression"


def test_build_model_unsupported_type_fails():
    config = {
        'model': {
            'type': 'unsupported_model',
        }
    }

    with pytest.raises(ValueError, match= 'Unsupported model type'):
        build_model(config)


def  test_build_training_pipeline():
    config = {
        'model': {
            'type': 'logistic_regression',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
        }
    }

    pipeline = build_training_pipeline(config)

    assert 'preprocessor' in pipeline.named_steps
    assert 'model' in pipeline.named_steps


def test_fit_model_and_predict_probabilities():
    df = make_training_df(num_rows=100)

    X = df.drop(columns= ['is_fraud', 'transaction_id', 'customer_id'], errors= 'ignore')
    y = df['is_fraud']

    config = {
        'model': {
            'type': 'logistic_regression',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
        }
    }

    pipeline = build_training_pipeline(config)
    fitted_pipeline = fit_model(pipeline, X, y)

    probabilities = predict_validation_probabilities(fitted_pipeline, X)

    assert len(probabilities) == len(X)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_fit_model_and_predict_probabilities():
    df = make_training_df(num_rows=100)
    X, y = split_features_and_target(df)

    config = {
        'model': {
            'type': 'logistic_regression',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
        }
    }

    pipeline = build_training_pipeline(config)
    fitted_pipeline = fit_model(pipeline, X, y)

    probabilities = predict_validation_probabilities(fitted_pipeline, X)

    assert len(probabilities) == len(X)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_calculate_training_metrics():
    y_valid =  pd.Series([0, 0, 1, 1])
    y_valid_proba = [0.1, 0.2, 0.8, 0.9]

    metrics = calculate_training_metrics(y_valid, y_valid_proba)

    assert 'valid_roc_auc' in metrics
    assert 'valid_pr_auc' in metrics
    assert 'valid_log_loss' in metrics
    assert metrics['valid_roc_auc'] == 1.0


def test_save_and_load_model_artifact(tmp_path):
    df = make_training_df(num_rows=1100)
    X,y = split_features_and_target(df)

    config = {
        'model': {
            'type': 'logistic_regression',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
        }
    }

    pipeline = build_training_pipeline(config)
    fitted_pipeline = fit_model(pipeline, X, y)

    output_path = tmp_path / 'model.joblib'

    save_model_artifact(fitted_pipeline, output_path)

    loaded_model = load_model_artifact(output_path)

    probabilities = loaded_model.predict_proba(X)[:,1]

    assert output_path.exists()
    assert  len(probabilities) == len(X)


def test_save_training_metrics(tmp_path):
    result = {
        'model_name': 'test_model',
        'valid_roc_auc': 0.9,
        'valid_pr_auc': 0.4,
    }

    output_path = tmp_path / 'metrics.json'

    save_training_metrics(result, output_path)

    with output_path.open('r', encoding='utf-8') as file:
        loaded = json.load(file)

    assert loaded['model_name'] == 'test_model'
    assert loaded['valid_roc_auc'] == 0.9


def test_train_model_end_to_end(tmp_path):
    df = make_training_df(num_rows=200)

    train_df = df.iloc[:150].reset_index(drop= True)
    valid_df = df.iloc[150:].reset_index(drop= True)

    train_df.to_csv(tmp_path / 'train.csv', index=False)
    valid_df.to_csv(tmp_path / 'valid.csv', index=False)

    config = make_test_config(tmp_path)

    result = train_model(config)

    assert result['model_name'] == 'test_logistic'
    assert 'valid_roc_auc' in result
    assert 'valid_pr_auc' in result
    assert Path(config["artifacts"]["model_output_path"]).exists()
    assert Path(config["artifacts"]["metrics_output_path"]).exists()