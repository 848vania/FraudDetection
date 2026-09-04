import json

from app.database.connection import Base, SessionLocal, engine
from app.database.crud import (
    create_prediction_log,
    get_recent_prediction_logs,
)


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def make_transaction():
    return {
        'transaction_id': 'txn_1',
        'customer_id': 'cust_1',
        'transaction_amount': 249.99,
    }


def make_prediction():
    return {
        'transaction_id': 'txn_1',
        'fraud_probability': 0.87,
        'risk_band': 'high',
        'decision': 'review',
        'threshold_used': 0.72,
        'model_name': 'fraud-risk-model',
        'model_version': '3',
        'model_alias': 'champion',
        'latency_ms': 25.5,
    }


def test_create_prediction_log():
    db = SessionLocal()

    try:
        log = create_prediction_log(
            db= db,
            transaction= make_transaction(),
            prediction= make_prediction(),
            source= 'api',
        )

        assert log.id is not None
        assert log.transaction_id == 'txn_1'
        assert log.fraud_probability == 0.87
        assert log.risk_band == 'high'
        assert log.decision == 'review'

        loaded_input = json.loads(log.input_json)
        assert loaded_input['transaction_id'] == 'txn_1'

    finally:
        db.close()


def test_get_recent_prediction_logs():
    db = SessionLocal()

    try:
        create_prediction_log(
            db = db,
            transaction= make_transaction(),
            prediction= make_prediction(),
            source= 'api',
        )

        logs = get_recent_prediction_logs(db, limit= 10)

        assert len(logs) == 1
        assert logs[0].transaction_id == 'txn_1'

    finally:
        db.close()