from pathlib import Path 

import pandas as pd 

from app.models.batch_predict import (
    add_batch_metadata,
    build_batch_summary,
    dataframe_to_transactions,
    predictions_to_dataframe,
    save_batch_predictions,
    batch_predict,
)


def make_batch_df():
    return pd.DataFrame(
        {
            "transaction_id": ["txn_1", "txn_2"],
            "customer_id": ["cust_1", "cust_2"],
            "transaction_amount": [249.99, 32.5],
            "transaction_hour": [23, 14],
            "merchant_category": ["electronics", "groceries"],
            "customer_tenure_days": [45, 800],
            "num_transactions_24h": [9, 1],
            "num_failed_transactions_24h": [3, 0],
            "avg_transaction_amount_30d": [61.25, 45.0],
            "is_foreign_transaction": [True, False],
            "device_type": ["mobile", "desktop"],
            "country": ["US", "US"],
            "previous_chargebacks": [1, 0],
            "account_age_days": [60, 800],
            "risk_score_external": [0.71, 0.12],
        }
    )


def make_predictions():
    return [
        {
            "transaction_id": "txn_1",
            "fraud_probability": 0.87,
            "risk_band": "high",
            "decision": "review",
            "threshold_used": 0.72,
            "model_name": "fraud-risk-model",
            "model_version": "3",
            "model_alias": "champion",
        },
        {
            "transaction_id": "txn_2",
            "fraud_probability": 0.12,
            "risk_band": "low",
            "decision": "approve",
            "threshold_used": 0.72,
            "model_name": "fraud-risk-model",
            "model_version": "3",
            "model_alias": "champion",
        },
    ]


def test_dataframe_to_transactions():
    df = make_batch_df()

    transactions = dataframe_to_transactions(df)

    assert isinstance(transactions, list)
    assert len(transactions) == 2 
    assert transactions[0]['transaction_id'] == 'txn_1'


def test_add_batch_metadata():
    transactions = dataframe_to_transactions(make_batch_df())
    predictions = make_predictions()

    enriched_predictions = add_batch_metadata(
        transactions= transactions, 
        predictions= predictions,
        scored_at="2026-08-24T04:43:00+00:00"
    )

    assert len(enriched_predictions) == 2
    assert enriched_predictions[0]['customer_id'] == 'cust_1'
    assert enriched_predictions[0]['scored_at'] == "2026-08-24T04:43:00+00:00"


def test_predictions_to_dataframe():
    prediction_df = predictions_to_dataframe(make_predictions())

    assert len(prediction_df) == 2
    assert 'fraud_probability' in prediction_df.columns
    assert 'decision' in prediction_df.columns


def test_save_batch_predictions(tmp_path):
    prediction_df = predictions_to_dataframe(make_predictions())

    output_path = tmp_path / 'predictions.csv'

    save_batch_predictions(
        prediction_df= prediction_df,
        output_path= output_path,
    )

    assert output_path.exists()

    loaded_df = pd.read_csv(output_path)

    assert len(loaded_df) == 2


def test_build_batch_summary(tmp_path):
    prediction_df = predictions_to_dataframe(make_predictions())

    summary = build_batch_summary(
        prediction_df= prediction_df,
        input_path = 'input.csv',
        output_path = 'output.csv',
        started_at= "2026-08-24T04:43:00+00:00",
        finished_at="2026-08-24T04:43:10+00:00",
    )

    assert summary['total_rows'] == 2
    assert summary['review_count'] == 1 
    assert summary['review_rate'] == 0.5
    assert summary['high_risk_count'] == 1
    assert summary['high_risk_rate'] == 0.5


def test_batch_predict_end_to_end_with_mock(tmp_path, monkeypatch):
    input_df = make_batch_df()
    input_path  = tmp_path / 'current_transactions.csv'
    output_path = tmp_path / 'predictions.csv'
    summary_path = tmp_path / 'summary.json'

    input_df.to_csv(input_path, index=False)

    def fake_score_transactions(transactions):
        return make_predictions()

    monkeypatch.setattr(
        "app.models.batch_predict.score_transactions",
        fake_score_transactions,
    )

    summary = batch_predict(
        input_path= input_path,
        output_path= output_path,
        summary_path= summary_path,
    )

    assert output_path.exists()
    assert summary_path.exists()
    assert summary['total_rows'] == 2