import json 
from typing import Any

from app.database.connection import SessionLocal, init_db
from app.database.crud import get_all_prediction_logs, get_recent_prediction_logs


def prediction_log_to_dict(log) -> dict[str, Any]:
    """
    Convert PredictionLog SQLAlchemy object to dictionary
    """
    try:
        input_data = json.loads(log.input_json)
    except json.JSONDecodeError:
        input_data = {}

    try:
        prediction_data = json.loads(log.prediction_json)
    except json.JSONDecodeError:
        prediction_data = {}

    return {
        'id': log.id,
        'transaction_id': log.transaction_id,
        'customer_id': log.customer_id,
        'input': input_data,
        'prediction': prediction_data,
        'fraud_probability': log.fraud_probability,
        'risk_band': log.risk_band,
        'decision': log.decision,
        'threshold_used': log.threshold_used,
        'model_name': log.model_name,
        'model_version': log.model_version,
        'model_alias': log.model_alias,
        'latency_ms': log.latency_ms,
        'source': log.source,
        'created_at': log.created_at.isoformat()
        if log.created_at 
        else None
    }



def average_latency_ms(logs) -> float:
    """
    Calculate average latency 
    """
    latencies = [
        log.latency_ms
        for log in logs
        if log.latency_ms is not None
    ]

    if not latencies:
        return 0.0

    return float(sum(latencies) / len(latencies))


def average_fraud_probability(logs) -> float:
    """
    Calculate average predicted fraud probability
    """
    if not logs:
        return 0.0

    return float(
        sum(log.fraud_probability for log in logs) / len(logs)
    )


def review_rate(logs) -> float:
    """
    Calculate fraction of predictions sent to review 
    """
    if not logs:
        return 0.0 

    review_count = sum(1 for log in logs if log.decision == 'review')

    return review_count / len(logs)


def high_risk_rate(logs) -> float:
    """
    Calculate fraction of high-risk prediction
    """
    if not logs:
        return 0.0

    high_risk_count = sum(1 for log in logs if log.risk_band == 'high')

    return high_risk_count / len(logs)


def value_distribution(logs, attribute: str) -> dict[str, int]:
    """
    Count distribution of a given PredictionLog attribute
    """
    distribution = {}

    for log in logs:
        value = getattr(log, attribute, None)

        if value is None:
            value = "unknown"

        value = str(value)

        distribution[value] = distribution.get(value, 0) + 1

    return distribution


def prediction_volume_by_day(logs) -> dict[str, int]:
    """
    Count predictions by date
    """
    volume = {}

    for log in logs:
        if not log.created_at:
            continue 

        date_key  = log.created_at.date().isoformat()

        volume[date_key] = volume.get(date_key, 0 ) + 1

    return volume


def get_monitoring_summary(
        recent_limit: int = 50,
    ) -> dict[str, Any]:
    """
    Build monitoring summary from prediction logs
    """
    init_db()

    db = SessionLocal()

    try:
        all_logs = get_all_prediction_logs(db)
        recent_logs = get_recent_prediction_logs(
            db,
            limit= recent_limit,
        )

        return {
            'total_predicitions': len(all_logs),
            'average_latency_ms': average_latency_ms(all_logs),
            'average_fraud_probability': average_fraud_probability(all_logs),
            'review_rate': review_rate(all_logs),
            'high_risk_rate': high_risk_rate(all_logs),
            'risk_band_distribution': value_distribution(
                all_logs,
                'risk_band',
            ),
            'decision_distribution': value_distribution(
                all_logs,
                'decision',
            ),
            'model_version_distribution':  value_distribution(
                all_logs,
                'model_version',
            ),
            'source_distribution': value_distribution(
                all_logs,
                'source',
            ),
            'prediction_volume_by_day': prediction_volume_by_day(all_logs),
            'recent_predictions': [
                prediction_log_to_dict(log)
                for log in recent_logs
            ],
        }

    finally:
        db.close()