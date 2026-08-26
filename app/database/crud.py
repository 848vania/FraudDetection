import json 
from typing import Any

from sqlalchemy.orm import Session

from app.database.models import PredictionLog


def create_prediction_log(
        db: Session,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        source: str = "api",
        latency_ms: float | None = None,
    ) -> PredictionLog:
    """
    Save one prediction interaction
    """
    log = PredictionLog(
        transaction_id = transaction.get('transaction_id'),
        customer_id = transaction.get('customer_id'),
        input_json = json.dumps(transaction, ensure_ascii=False),
        prediction_json = json.dumps(prediction, ensure_ascii=False),
        fraud_probability = float(prediction['fraud_probability']),
        risk_band = prediction['risk_band'],
        decision = prediction['decision'],
        threshold_used = float(prediction['threshold_used']),
        model_name = prediction['model_name'],
        model_version = prediction.get('model_version'),
        model_alias = prediction.get('model_alias'),
        latency_ms = latency_ms if latency_ms is not None else prediction.get('latency_ms'),
        source = source,
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    return log 


def get_recent_prediction_logs(
        db: Session,
        limit: int = 50,
    ) -> list[PredictionLog]:
    """
    Return most recent prediction logs
    """
    return (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )


def get_all_prediction_logs(db: Session) -> list[PredictionLog]:
    """
    Return all prediction logs 

    Fine for local portfolio project 
    For production, use pagination
    """
    return db.query(PredictionLog).all()


def count_prediction_logs(db: Session) -> int:
    """
    Count prediction logs
    """
    return db.query(PredictionLog).count()


def get_prediction_logs_by_source(
        db: Session,
        source: str,
        limit: int = 100,
    ) -> list[PredictionLog]:
    """
    Return recent logs from a specific source
    """
    return (
        db.query(PredictionLog)
        .filter(PredictionLog.source == source)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )