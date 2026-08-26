from datetime import  datetime, timezone

from app.monitoring.metrics import (
    average_fraud_probability,
    average_latency_ms,
    high_risk_rate,
    prediction_volume_by_day,
    review_rate,
    value_distribution,
)


class FakeLog:
    def __init__(
            self,
            fraud_probability= 0.1,
            risk_band= 'low',
            decision = 'approve',
            latency_ms= 10.0,
            model_version= '1',
            source= 'api',
            created_at= None,
        ):
        self.fraud_probability = fraud_probability
        self.risk_band = risk_band
        self.decision = decision
        self.latency_ms = latency_ms
        self.model_version = model_version
        self.source = source
        self.created_at = created_at or datetime.now(timezone.utc)


def test_average_latency_ms():
    logs = [
        FakeLog(latency_ms= 10.0),
        FakeLog(latency_ms= 30.0),
    ]

    assert average_latency_ms(logs) == 20.0


def test_average_fraud_probability():
    logs = [
        FakeLog(fraud_probability=0.1),
        FakeLog(fraud_probability=0.3),
    ]

    assert average_fraud_probability(logs) == 0.2


def test_review_rate():
    logs = [
        FakeLog(decision='review'),
        FakeLog(decision='approve'),
        FakeLog(decision='approve'),
    ]

    assert review_rate(logs) == 1 / 3


def test_high_risk_rate():
    logs = [
        FakeLog(risk_band='high'),
        FakeLog(risk_band='medium'),
        FakeLog(risk_band='low'),
    ]

    assert high_risk_rate(logs) == 1 / 3


def test_value_distribution():
    logs = [
        FakeLog(risk_band= 'high'),
        FakeLog(risk_band= 'high'),
        FakeLog(risk_band= 'low'),
    ]

    distribution = value_distribution(logs, 'risk_band')

    assert distribution['high'] == 2
    assert distribution['low'] == 1


def test_prediction_volume_by_day():
    logs = [
        FakeLog(created_at=datetime(2026, 8, 25, tzinfo=timezone.utc)),
        FakeLog(created_at=datetime(2026, 8, 25, tzinfo=timezone.utc)),
        FakeLog(created_at=datetime(2026, 8, 26, tzinfo=timezone.utc)),
    ]

    volume = prediction_volume_by_day(logs)

    assert volume['2026-08-25'] == 2
    assert volume['2026-08-26'] == 1