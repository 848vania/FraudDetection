import pandas as pd

from app.models.predict import (
    assign_risk_band,
    build_prediction_response,
    get_model_info,
    get_threshold_from_metadata,
    make_decision,
    predict_probability,
    predict_transaction,
    prepare_features_for_prediction,
    transaction_to_dataframe,
)


class FakeModel():
    def predict_proba(self, X):
        return [[0.13, 0.87]]


def make_transaction():
    return {
        'transaction_id': 'txn_123',
        'customer_id': 'cust_456',
        'transaction_amount': 249.99,
        'transaction_hour': 23,
        'merchant_category': 'electronics',
        'customer_tenure_days': 45,
        'num_transactions_24h': 9,
        'num_failed_transactions_24h': 3,
        'avg_transaction_amount_30d': 61.25,
        'is_foreign_transaction': True,
        'device_type': 'mobile',
        'country': 'US',
        'previous_chargebacks': 1,
        'account_age_days': 60,
        'risk_score_external': 0.71,
    }


def make_metadata():
    return {
        'registered_model_name': 'fraud-risk-model',
        'model_version': '3',
        'alias': 'champion',
        'model_type': 'logistic_regression',
        'selected_threshold': 0.72,
        'test_roc_auc': 0.91,
        'test_pr_auc': 0.44,
        'test_precision': 0.36,
        'test_recall': 0.76,
        'test_expected_cost': 45450.0,
    }


def test_transaction_to_dataframe():
    transaction = make_transaction()

    df = transaction_to_dataframe(transaction)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.loc[0, 'transaction_id'] == 'txn_123'


def test_prepare_features_for_prediction_removes_id_columns():
    transaction = make_transaction()

    df = transaction_to_dataframe(transaction)

    features = prepare_features_for_prediction(df)

    assert 'transaction_id' not in features.columns
    assert 'customer_id' not in features.columns
    assert 'amount_to_customer_avg_ratio' in features.columns
    assert 'failed_transaction_rate_24h' in features.columns


def test_predict_probability():
    model = FakeModel()
    features = pd.DataFrame({'feature': [1]})

    probability = predict_probability(model, features)

    assert probability == 0.87


def test_assign_risk_band():
    assert assign_risk_band(0.10) == 'low'
    assert assign_risk_band(0.30) == 'medium'
    assert assign_risk_band(0.69) == 'medium'
    assert assign_risk_band(0.70) == 'high'


def test_make_decision():
    assert make_decision(probability=0.78, threshold=0.5) == 'review'
    assert make_decision(probability=0.30, threshold=0.5) == 'approve'


def test_get_threshold_from_metadata():
    metadata = make_metadata()

    threshold = get_threshold_from_metadata(metadata)

    assert threshold == 0.72


def test_get_threshold_from_metadata_uses_default():
    threshold = get_threshold_from_metadata({})

    assert threshold == 0.5


def test_build_prediction_response():
    transaction = make_transaction()
    metadata = make_metadata()

    response = build_prediction_response(
        transaction= transaction,
        probability= 0.87,
        threshold= 0.72,
        metadata= metadata,
    )

    assert response['transaction_id'] == 'txn_123'
    assert response['fraud_probability'] == 0.87
    assert response['risk_band'] == "high"
    assert response['decision'] == 'review'
    assert response['threshold_used'] == 0.72
    assert response['model_name'] == 'fraud-risk-model'
    assert response['model_version'] == '3'


def test_predict_transaction_with_fake_model():
    transaction = make_transaction()
    model = FakeModel()
    metadata = make_metadata()

    response = predict_transaction(
        transaction= transaction,
        model= model, 
        metadata= metadata,
    )

    assert response['fraud_probability'] == 0.87
    assert response['risk_band'] == 'high'
    assert response['decision'] == 'review'


def test_get_model_info():
    metadata = make_metadata()

    info = get_model_info(metadata)

    assert info['registered_model_name'] == 'fraud-risk-model'
    assert info['model_version'] == '3'
    assert info['selected_threshold'] == 0.72
    assert info['test_pr_auc'] == 0.44