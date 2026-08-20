from typing import Any 

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def make_binary_predictions(
        y_proba: list[float] | np.ndarray,
        threshold: float,
    ) -> np.ndarray:
    """
    Convert probabilities to binary predictions using threshold
    """
    probabilities = np.asarray(y_proba)

    return (probabilities >= threshold).astype(int)


def calculate_confusion_counts(
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
    ) -> dict[str, int]:
    """
    Calculate confusion matrix counts
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels= [0,1],
    ).ravel()

    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }


def calculate_expected_cost(
        false_positives: int,
        false_negatives: int,
        false_positive_cost: float,
        false_negative_cost: float,
    ) -> float:
    """
    Calculate expected business cost 

    False negatives are missed fraud 
    False positives are legitimate transactions sent to  review 
    """
    return float(
        false_negatives * false_negative_cost
        + false_positives * false_positive_cost
    )


def calculate_threshold_metrics(
        y_true: list[int] | np.ndarray,
        y_proba: list[float] | np.ndarray,
        threshold: float,
        false_negative_cost: float = 500.0,
        false_positive_cost: float = 25.0,
    ) -> dict[str, Any]:
    """
    Calculate threshold-dependent fraud metrics
    """
    y_pred = make_binary_predictions(y_proba, threshold)
    counts = calculate_confusion_counts(y_true, y_pred)

    tn = counts['true_negatives']
    fp = counts['false_positives']
    fn = counts['false_negatives']
    tp = counts['true_positives']

    total = tn + fp + fn + tp 
    actual_positives = tp + fn 
    actual_negatives = tn + fp 

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    false_positive_rate = fp / actual_negatives if actual_negatives else 0.0
    false_negative_rate = fp / actual_positives if actual_positives else 0.0
    review_rate = (tp + fp) / total if total else 0.0 # Represents the total predicted positive rate, showing how often a model gives a positive decision
    fraud_capture_rate = recall

    expected_cost = calculate_expected_cost(
        false_positives=fp,
        false_negatives=fn,
        false_positive_cost= false_positive_cost,
        false_negative_cost= false_negative_cost,
    )

    return {
        'threshold': float(threshold),
        **counts,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate),
        'fraud_capture_rate': float(fraud_capture_rate),
        'review_rate': float(review_rate),
        'expected_cost': float(expected_cost),
    }


def generate_threshold_grid(grid_size: int = 101) -> np.ndarray:
    """
    Generate threshold candidates between 0.0 and 1.0
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    return  np.linspace(0.0, 1.0, grid_size)


def evaluate_thresholds(
        y_true: list[int] | np.ndarray,
        y_proba: list[float] | np.ndarray,
        thresholds: list[float] | np.ndarray,
        false_negative_cost: float = 500.0,
        false_positive_cost: float = 25.0,
    ) -> list[dict[str, Any]]:
    """
    Evaluate metrics across threshold candidates
    """
    return [
        calculate_threshold_metrics(
            y_true= y_true,
            y_proba= y_proba,
            threshold= float(threshold),
            false_negative_cost= false_negative_cost,
            false_positive_cost= false_positive_cost,
        )
        for threshold in thresholds
    ]


def select_best_threshold(
        threshold_results: list[dict[str, Any]],
        objective: str = 'min_expected_cost',
        min_precision: float | None = None,
        min_recall: float | None = None,
    ) -> dict[str, Any]:
    """
    Select best threshold based on objective
    """
    if not threshold_results:
        raise ValueError("threshold_results cannot be empty")

    candidates = threshold_results

    if min_precision is not None:
        candidates = [
            result for result in candidates
            if result['precision'] >= min_precision
        ]

    if min_recall is not None:
        candidates = [
            result for result in candidates
            if result['recall'] >= min_recall
        ]

    if not candidates:
        raise ValueError(
            "No threshold candidates satisfy the provided constraints"
        )

    if objective == 'min_expected_cost':
        return  min(
            candidates,
            key=lambda result: result['expected_cost']
        )

    if objective == 'max_f1':
        return max(
            candidates,
            key= lambda result: result['f1'],
        )

    if objective == 'max_recall':
        return max(
            candidates,
            key = lambda result: result['recall'],
        )

    raise ValueError(f'Unsupported threshold objective: {objective}')


def find_best_threshold(
        y_true: list[int] | np.ndarray,
        y_proba: list[float] | np.ndarray,
        objective: str = 'min_expected_cost',
        false_negative_cost: float = 500.0,
        false_positive_cost: float = 25.0,
        min_precision: float | None = None,
        min_recall: float | None = None,
        grid_size: int = 101,
    ) -> dict[str, Any]:
    """
    Find best threshold using validation labels and probabilities
    """
    thresholds = generate_threshold_grid(grid_size= grid_size)

    threshold_results = evaluate_thresholds(
        y_true = y_true,
        y_proba = y_proba,
        thresholds= thresholds,
        false_negative_cost= false_negative_cost,
        false_positive_cost= false_positive_cost,
    )

    best_threshold_result = select_best_threshold(
        threshold_results= threshold_results,
        objective= objective,
        min_precision= min_precision,
        min_recall= min_recall,
    )

    return {
        'objective': objective,
        'false_negative_cost': false_negative_cost,
        'false_positive_cost': false_positive_cost,
        'min_precision': min_precision,
        'min_recall': min_recall,
        'best_threshold': best_threshold_result['threshold'],
        'best_threshold_metrics': best_threshold_result,
        'threshold_results': threshold_results,
    }