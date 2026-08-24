from fastapi.testclient import TestClient

from app.api.main import app 


client = TestClient(app)


def make_transaction_payload():
    return {
        'transaction_id': 'txn_test_001',
        'customer_id': 'cust_test_001',
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


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200 
    assert response.json()['status'] == 'ok'


def test_predict_endpoint(monkeypatch):
    def fake_predict_transaction(transaction):
        return {
            'transaction_id': transaction.get('transaction_id'),
            'fraud_probability': 0.87,
            'risk_band': 'high',
            'decision': 'review',
            'threshold_used': 0.72,
            'model_name': 'fraud-risk-model',
            'model_version': '3',
            'model_alias': 'champion',
        }

    monkeypatch.setattr(
        'app.api.routes_predict.predict_transaction',
        fake_predict_transaction,
    )

    response = client.post(
        '/predict',
        json= make_transaction_payload(),
    )

    data = response.json()

    assert response.status_code == 200
    assert data['fraud_probability'] == 0.87
    assert data['risk_band'] == 'high'
    assert data['decision'] == 'review'
    assert 'latency_ms' in data


def test_predict_enpoint_rejects_invalid_input():
    payload = make_transaction_payload()
    payload['transaction_hour'] = 30

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_model_info_endpoint(monkeypatch):
    def fake_get_model_info():
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

    monkeypatch.setattr(
        'app.api.routes_model.get_model_info',
        fake_get_model_info,
    )

    response = client.get('/model/info')

    data = response.json() 

    assert response.status_code == 200
    assert data['registered_model_name'] == 'fraud-risk-model'
    assert data['selected_threshold'] == 0.72