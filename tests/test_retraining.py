from app.monitoring.retraining import (
    build_retraining_decision,
    check_drift_trigger,
    check_monitoring_trigger,
    check_performance_trigger,
    should_trigger_retraining,
)


def test_check_drift_trigger_detects_overall_drift():
    drift_summary = {
        'overall_drift_detected': True,
        'num_drifted_features': 1,
    }

    reasons = check_drift_trigger(drift_summary)

    assert  'overall_drift_detected' in reasons


def test_check_drift_trigger_detectes_many_drifted_features():
    drift_summary = {
        'overall_drift_detected': True,
        'num_drifted_features': 4,
    }

    reasons = check_drift_trigger(
        drift_summary,
        max_drifted_features_before_retrain=3
    )

    assert 'too_many_drifted_features' in reasons


def test_check_monitoring_trigger_detects_high_review_rate():
    monitoring_summary = {
        'review_rate': 0.40,
        'high_risk_rate': 0.05,
        'average_fraud_probability': 0.10,
    }

    reasons = check_monitoring_trigger(
        monitoring_summary,
        max_review_rate= 0.25,
    )

    assert 'review_rate_too_high' in reasons


def test_check_performance_trigger_detects_low_pr_auc():
    performance_summary = {
        'test_pr_auc': 0.20,
        'recall': 0.75, 
    }

    reasons =  check_performance_trigger(
        performance_summary,
        min_pr_auc= 0.30,
    )

    assert 'pr_auc_below_threshold' in reasons


def test_build_retraining_decision_triggers_when_reasons_exist():
    decision = build_retraining_decision(
        drift_reasons= ['overall_drift_detected'],
        monitoring_reasons= [],
        performance_reasons= [],
    )

    assert decision['trigger_retraining'] is True
    assert decision['recommended_action'] == 'run_retraining_pipeline'


def test_build_retraining_decision_no_trigger_when_no_reasons():
    decision = build_retraining_decision(
        drift_reasons= [],
        monitoring_reasons= [],
        performance_reasons= [],
    )

    assert decision['trigger_retraining'] is False
    assert decision['recommended_action'] == 'no_retraining_needed'


def test_should_trigger_retraining_combines_reasons():
    drift_summary = {
        'overall_drift_detected': True,
        'num_drifted_features': 4,
    }

    monitoring_summary = {
        'review_rate': 0.40,
        'high_risk_rate': 0.05,
        'average_fraud_probability': 0.10,
    }

    performance_summary = {
        'test_pr_auc': 0.20,
        'recall': 0.75,
    }

    decision = should_trigger_retraining(
        drift_summary= drift_summary,
        monitoring_summary= monitoring_summary,
        performance_summary= performance_summary,
    )

    assert decision['trigger_retraining'] is True 
    assert 'overall_drift_detected' in decision['reasons']
    assert 'review_rate_too_high' in decision['reasons']
    assert 'pr_auc_below_threshold' in decision['reasons']