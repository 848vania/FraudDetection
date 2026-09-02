from app.models.retrain import (
    compare_candidate_to_champion,
    should_skip_retraining,
)


def test_should_skip_retraining_true():
    trigger_report = {
        'trigger_retraining': False,
    }

    assert should_skip_retraining(trigger_report=trigger_report) is True


def test_should_skip_retraining_false():
    trigger_report = {
        'trigger_retraining': True,
    }

    assert should_skip_retraining(trigger_report) is False


def test_compare_candidate_to_champio_promotes_better_model():
    candidate = {
        'test_pr_auc': 0.50,
        'expected_cost': 9000,
        'recall': 0.80,
    }

    champion = {
        'test_pr_auc': 0.45,
        'expected_cost': 10000,
        'recall': 0.75,
    }

    comparison = compare_candidate_to_champion(
        candidate_metrics= candidate,
        champion_metrics= champion,
        min_pr_auc_improvement= 0.01,
        max_expected_cost_increase= 0.0,
        require_recall_at_least= 0.70,
    )

    assert comparison['promote'] is True 


def test_compare_candidate_to_champion_rejects_worse_cost():
    candidate = {
        'test_pr_auc': 0.50,
        'expected_cost': 12000,
        'recall': 0.80,
    }

    champion = {
        'test_pr_auc': 0.45,
        'expected_cost': 10000,
        'recall': 0.75,
    }

    comparison = compare_candidate_to_champion(
        candidate_metrics= candidate,
        champion_metrics= champion,
        min_pr_auc_improvement= 0.01,
        max_expected_cost_increase= 0.0,
        require_recall_at_least= 0.70,
    )

    assert comparison['promote'] is False
    assert comparison['checks']['cost_passed'] is False


def test_compare_candidate_to_champion_rejects_low_recall():
    candidate = {
        'test_pr_auc': 0.50,
        'expected_cost': 9000,
        'recall': 0.50,
    }

    champion = {
        'test_pr_auc': 0.45,
        'expected_cost': 10000,
        'recall': 0.75,
    }

    comparison =  compare_candidate_to_champion(
        candidate_metrics= candidate,
        champion_metrics= champion,
        min_pr_auc_improvement= 0.01,
        max_expected_cost_increase= 0.0,
        require_recall_at_least= 0.70,
    )

    assert comparison['promote'] is False
    assert comparison['checks']['recall_passed'] is False