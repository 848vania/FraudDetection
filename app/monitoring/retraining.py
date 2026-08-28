import json  
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.monitoring.metrics import  get_monitoring_summary


def load_json_file(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON file
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def save_json_file(
        data: dict[str, Any],
        path: str | Path,
    ) -> None:
    """
    Save dictionary as JSON
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def check_drift_trigger(
        drift_summary: dict[str, Any],
        max_drifted_features_before_retrain: int = 3,
    ) -> list[str]:
    """
    Return retraining reasons based on drift summary 
    Possible retraining reasons:
        - Overall drift is detected 
        - number of drifted features >- threshold
        - prediction drift is detected 
        - risk-band drift is detected
    """
    reasons = []

    if drift_summary.get('overall_drift_detected'):
        reasons.append('overall_drift_detected')

    num_drifted_features = drift_summary.get('num_drifted_features', 0)

    if num_drifted_features >= max_drifted_features_before_retrain:
        reasons.append('too_many_drifted_features')

    prediction_drift = drift_summary.get('prediction_drift') or {}

    if prediction_drift.get('prediction_drift_detected'):
        reasons.append('prediction_drift_detected')

    risk_band_drift = drift_summary.get('risk_band_drift') or {}

    if risk_band_drift.get('risk_band_drift_detected'):
        reasons.append('risk_band_drift_detected')

    return reasons 


def check_monitoring_trigger(
        monitoring_summary: dict[str, Any],
        max_review_rate: float = 0.25,
        max_high_risk_rate: float = 0.15,
        max_average_fraud_probability: float = 0.35,
    ) -> list[str]:
    """
    Return retraining reasons based on operational monitoring metrics
    """
    reasons = []

    review_rate = monitoring_summary.get('review_rate', 0.0)
    high_risk_rate = monitoring_summary.get('high_risk_rate', 0.0)
    average_fraud_probability = monitoring_summary.get(
        'average_fraud_probability',
        0.0,
    )

    if review_rate > max_review_rate:
        reasons.append('review_rate_too_high')

    if high_risk_rate > max_high_risk_rate:
        reasons.append('high_risk_rate_too_high')

    if average_fraud_probability > max_average_fraud_probability:
        reasons.append('average_fraud_probability_too_high')

    return reasons


def check_performance_trigger(
        performance_summary: dict[str, Any] | None = None,
        min_pr_auc: float = 0.30,
        min_recall: float = 0.60,
        max_expected_cost: float | None = None,
    ) -> list[str]:
    """
    Return retraining reasons based on labeled model performances.
    Trigger if:
        - PR_AUC below minimum 
        - Recall below minimum
        - Expected cost above maximum
    """
    reasons = []

    if not performance_summary:
        return reasons

    pr_auc = performance_summary.get(
        'test_pr_auc',
        performance_summary.get('pr_auc'),
    )

    recall = performance_summary.get(
        'recall',
        performance_summary.get('test_recall')
    )

    expected_cost = performance_summary.get(
        'expected_cost',
        performance_summary.get('test_expected_cost'),
    )

    if pr_auc is not None and pr_auc < min_pr_auc:
        reasons.append('pr_auc_below_threshold')

    if recall is not None and recall < min_recall:
        reasons.append('recall_below_threshold')

    if (
        max_expected_cost is not None 
        and expected_cost is not None
        and expected_cost > max_expected_cost
    ):
        reasons.append('expected_cost_too_high')

    return reasons


def build_retraining_decision(
        drift_reasons: list[str],
        monitoring_reasons: list[str],
        performance_reasons: list[str],
    ) -> dict[str, Any]:
    """
    Build final retraining trigger decision
    """
    all_reasons = drift_reasons + monitoring_reasons + performance_reasons

    trigger_retraining = len(all_reasons) > 0

    if trigger_retraining:
        recommended_action = 'run_retraining_pipeline'
    else:
        recommended_action = 'no_retraining_needed'

    return {
        'trigger_retraining': trigger_retraining,
        'recommended_action': recommended_action,
        'reasons': all_reasons,
        'drift_reasons': drift_reasons,
        'monitoring_reasons': monitoring_reasons,
        'performance_reasons': performance_reasons,
        'checked_at': datetime.now(timezone.utc).isoformat(),
    }


def should_trigger_retraining(
        drift_summary: dict[str, Any],
        monitoring_summary: dict[str, Any],
        performance_summary: dict[str, Any] | None = None,
        max_drifted_features_before_retrain: int = 3,
        max_review_rate:float =  0.25,
        max_high_risk_rate: float = 0.25,
        max_average_fraud_probability: float = 0.35,
        min_pr_auc: float = 0.30,
        min_recall: float = 0.60,
        max_expected_cost: float | None = None,
    ) -> dict[str, Any]:
    """
    Decide whether retraining should be triggered
    """
    drift_reasons = check_drift_trigger(
        drift_summary= drift_summary,
        max_drifted_features_before_retrain= max_drifted_features_before_retrain,
    )

    monitoring_reasons = check_monitoring_trigger(
        monitoring_summary= monitoring_summary,
        max_review_rate= max_review_rate,
        max_high_risk_rate= max_high_risk_rate,
        max_average_fraud_probability= max_average_fraud_probability,
    )

    performance_reasons = check_performance_trigger(
        performance_summary= performance_summary,
        min_pr_auc= min_pr_auc,
        min_recall= min_recall,
        max_expected_cost= max_expected_cost,
    )

    return build_retraining_decision(
        drift_reasons= drift_reasons,
        monitoring_reasons= monitoring_reasons,
        performance_reasons= performance_reasons,
    )


def run_retraining_trigger_check(
        drift_summary_path: str | Path  = 'data/results/drift_summary.json',
        performance_summary_path: str | Path  | None = 'data/results/evaluation_metrics.json',
        output_path: str | Path = 'data/results/retraining_trigger_report.json',
    ) -> dict[str, Any]:
    """
    Run retraining trigger check using saved drift report and monitoring logs
    """
    drift_summary = load_json_file(drift_summary_path)
    monitoring_summary = get_monitoring_summary()

    performance_summary = None 

    if performance_summary_path is not None:
        performance_path = Path(performance_summary_path)

        if performance_path.exists():
            performance_summary = load_json_file(performance_path)

    decision = should_trigger_retraining(
        drift_summary= drift_summary,
        monitoring_summary= monitoring_summary,
        performance_summary= performance_summary,
    )

    save_json_file(decision,  output_path)

    return decision