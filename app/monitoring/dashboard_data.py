from typing import Any

from app.monitoring.metrics import get_monitoring_summary


def get_dashboard_data() -> dict[str, Any]:
    """
    Return monitoring data formatted for dashboard use
    """
    summary = get_monitoring_summary()

    return {
        'cards': {
            'Total Predictions': summary['total_predictions'],
            'Average Latency': summary['average_latency_ms'],
            'Review Rate': summary['review_rate'],
            'High Risk Rate': summary['high_risk_rate'],
            'Average Fraud Probability': summary['average_fraud_probability'],
        },
        'charts': {
            'risk_band_distribution': summary['risk_band_distribution'],
            'decision_distribution': summary['decision_distribution'],
            'model_version_distribution': summary['model_version_distribution'],
            'source_distribution': summary['source_distribution'],
            'prediction_volume_by_day': summary['prediction_volume_by_day'],
        },
        'recent_predictions': summary['recent_predictions'],
    }