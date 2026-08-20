import pytest 

from app.models.thresholding import (
    calculate_confusion_counts,
    calculate_expected_cost,
    calculate_threshold_metrics,
    find_best_threshold,
    generate_threshold_grid,
    make_binary_predictions,
)


def test_make_binary_predictions():
    y_proba = [0.1, 0.4,  0.7,  0.9]

    y_pred = make_binary_predictions(y_proba, threshold=0.5)

    assert y_pred.tolist() == [0, 0, 1, 1]


def test_calculate_confusion_counts():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    counts = calculate_confusion_counts(y_true, y_pred)

    assert counts['true_negatives'] == 1
    assert counts['false_positives'] == 1
    assert counts['false_negatives'] == 1
    assert counts['true_positives'] == 1


def test_calculate_expected_cost():
    cost = calculate_expected_cost(
        false_positives=2,
        false_negatives=1,
        false_positive_cost= 25,
        false_negative_cost= 500,
    )

    assert cost == 550


def test_calculate_threshold_metrics():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1,  0.6, 0.4, 0.9]

    metrics = calculate_threshold_metrics(
        y_true= y_true,
        y_proba= y_proba,
        threshold= 0.5,
        false_positive_cost= 25,
        false_negative_cost= 500,
    )

    assert metrics['true_negatives'] == 1
    assert metrics['false_positives'] == 1
    assert metrics['false_negatives'] == 1
    assert metrics['true_positives'] == 1
    assert metrics['expected_cost'] == 525


def test_generate_threshold_grid():
    thresholds = generate_threshold_grid(grid_size=5)

    assert thresholds.tolist() == [0.0, 0.25, 0.50, 0.75, 1.0]


def test_generate_threshold_grid_fails_for_small_grid():
    with pytest.raises(ValueError, match= 'grid_size'):
        generate_threshold_grid(grid_size=1)


def test_find_best_threshold_returns_valid_threshold():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    report  = find_best_threshold(
        y_true= y_true,
        y_proba= y_proba,
        objective= 'min_expected_cost',
        grid_size= 11,
    )

    assert 0.0 <= report['best_threshold'] <= 1.0
    assert 'best_threshold_metrics' in report
    assert 'threshold_results' in report


def test_find_best_threshold_unsupported_objective_fails():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    with pytest.raises(ValueError, match='Unsupported threshold objective'):
        find_best_threshold(
            y_true= y_true,
            y_proba= y_proba,
            objective= 'unknown_objective',
        )