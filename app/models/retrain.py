import json 
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.evaluate import evaluate_model
from app.models.registry import register_model_version
from app.models.train import train_model


def load_json_file(path: str | Path) -> dict[str, Any]:
    """
    Load JSON file
    """
    path = Path(path)

    if not path.exists():
        raise  FileNotFoundError(f"JSON file not found: {path}")

    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def save_json_file(
        data: dict[str, Any],
        path: str | Path,
    ) -> None:
    """
    Save JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def should_skip_retraining(
        trigger_report: dict[str, Any],
    ) -> bool:
    """
    Return True when retraining should be skipped
    """
    return not trigger_report.get('trigger_retraining', False)


def compare_candidate_to_champion(
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
        min_pr_auc_improvement: float = 0.01,
        max_expected_cost_increase: float = 0.0,
        require_recall_at_least: float = 0.70,
    ) -> dict[str, Any]:
    """
    Compare candidate  model against champion model
    Promote if:
        - candidate PR-AUC improves by at least min_pr_auc improvement 
        - candidate expected cost is not worse than allowed 
        - candidate recall is above minimum
    """
    candidate_pr_auc = candidate_metrics.get('test_pr_auc', 0.0)
    champion_pr_auc = champion_metrics.get('test_pr_auc', 0.0)

    candidate_expected_cost = candidate_metrics.get('expected_cost', float('inf'))
    champion_expected_cost = champion_metrics.get('expected_cost', float('inf'))

    candidate_recall = candidate_metrics.get('recall', 0.0)

    pr_auc_improvement  = candidate_pr_auc - champion_pr_auc
    expected_cost_change = candidate_expected_cost - champion_expected_cost

    pr_auc_passed = pr_auc_improvement >= min_pr_auc_improvement
    cost_passed = expected_cost_change <= max_expected_cost_increase
    recall_passed = candidate_recall >= require_recall_at_least

    promote  = pr_auc_passed and cost_passed and recall_passed

    return {
        'promote': promote,
        'candidate_pr_auc': candidate_pr_auc,
        'champion_pr_auc': champion_pr_auc,
        'pr_auc_improvement': pr_auc_improvement,
        'candidate_expected_cost': candidate_expected_cost,
        'champion_expected_cost': champion_expected_cost,
        'expected_cost_change': expected_cost_change,
        'candidate_recall': candidate_recall,
        'require_recall': require_recall_at_least,
        'checks': {
            'pr_auc_passed':  pr_auc_passed,
            'cost_passed': cost_passed,
            'recall_passed': recall_passed,
        },
    }


def build_retraining_report(
        trigger_report: dict[str, Any],
        training_result: dict[str, Any] | None,
        candidate_evaluation: dict[str, Any] | None,
        champion_evaluation: dict[str, Any] | None, 
        comparison: dict[str, Any] | None,
        registered_metadata: dict[str, Any] | None,
        skipped: bool,
    ) -> dict[str, Any]:
    """
    Build retraining pipeline report
    """
    return {
        'skipped': skipped,
        'trigger_report': trigger_report,
        'training_result': training_result,
        'candidate_evaluation': candidate_evaluation,
        'champion_evaluation': champion_evaluation,
        'comparison': comparison,
        'registered_metadata': registered_metadata,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


def run_retraining_pipeline(
        retraining_config: dict[str, Any],
        trigger_report_path: str | Path = 'data/results/retraining_trigger_report.json',
        champion_evaluation_path: str | Path = 'data/results/evaluation_metrics.json',
    ) -> dict[str, Any]:
    """
    Run retraining pipeline 

    Trains candidate model only if trigger report says retraining is needed
    """
    trigger_report = load_json_file(trigger_report_path)

    output_path = retraining_config['artifacts']['retraining_report_path']

    if should_skip_retraining(trigger_report):
        report = build_retraining_report(
            trigger_report= trigger_report,
            training_result= None,
            candidate_evaluation= None,
            champion_evaluation= None, 
            comparison= None, 
            registered_metadata= None, 
            skipped=  True,
        )

        save_json_file(report, output_path)

        return report

    training_result = train_model(
        config= retraining_config,
        use_mlflow=True,
    )

    candidate_evaluation = evaluate_model(retraining_config)
    champion_evaluation = load_json_file(champion_evaluation_path)

    promotion_config = retraining_config.get('promotion', {})

    comparison = compare_candidate_to_champion(
        candidate_metrics= candidate_evaluation, 
        champion_metrics= champion_evaluation,
        min_pr_auc_improvement= promotion_config.get(
            'min_pr_auc_improvement',
            0.01,
        ),
        max_expected_cost_increase= promotion_config.get(
            'max_expected_cost_increase',
            0.0,
        ),
        require_recall_at_least= promotion_config.get(
            'require_recall_at_least',
            0.70,
        ),
    )

    registered_metadata = None 

    if comparison['promote']:
        registered_metadata = register_model_version(retraining_config)

    report = build_retraining_report(
        trigger_report= trigger_report,
        training_result= training_result,
        candidate_evaluation= candidate_evaluation,
        champion_evaluation= champion_evaluation,
        comparison= comparison,
        registered_metadata= registered_metadata,
        skipped= False,
    )

    save_json_file(report,  output_path)

    return report