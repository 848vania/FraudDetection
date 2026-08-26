from app.monitoring.logger import log_prediction


def test_log_prediction_does_not_raise(monkeypatch):
    def fake_create_prediction_log(*args, **kwargs):
        raise RuntimeError('database unavailable')

    monkeypatch.setattr(
        "app.monitoring.logger.create_prediction_log",
        fake_create_prediction_log,
    )

    log_prediction(
        transaction= {'transaction_id': 'txn_1'},
        prediction= {
            'fraud_probability': 0.1,
            'risk_band': 'low',
            'decision': 'approve',
            'threshold_used': 0.72,
            'model_name': 'fraud-risk-model',
        },
        source= 'api',
    )

    assert True