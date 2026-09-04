import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)

from app.data.ingest import load_transactions
from app.features.build_features import split_features_and_target
from app.models.thresholding import (
    calculate_threshold_metrics,
    find_best_threshold,
)
from app.models.train import load_model_artifact


def predict_probabilities(
        model,
        X: pd.DataFrame,
    ) -> list[float]:
    """
    Predict positive-class probabilities
    """
    probabilities = model.predict_proba(X)[:, 1]

    return probabilities.tolist()


def calculate_probability_metrics(
        y_true,
        y_proba,
        prefix: str = 'test',
    ) -> dict[str, float]:
    """
    Calculate threshold-independent probability metrics
    """
    return {
        f'{prefix}_roc_auc': float(roc_auc_score(y_true, y_proba)),
        f'{prefix}_pr_auc': float(average_precision_score(y_true, y_proba)),
    }


def build_evaluation_report(
        model_name: str,
        model_path: str,
        selected_threshold: float,
        probability_metrics: dict[str, float],
        threshold_metrics: dict[str, Any],
    ) ->  dict[str, Any]:
    """
    Build final evaluation report
    """
    return {
        'model_name': model_name,
        'model_path': model_path,
        'selected_threshold': float(selected_threshold),
        **probability_metrics,
        **threshold_metrics,
    }


def save_json(
        data: dict[str, Any] | list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
    """
    Save JSON data
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok= True)

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_confusion_matrix_plot(
        y_true,
        y_pred,
        output_path: str | Path,
    ) -> None:
    """
    Save confusion matrix plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=['Non-fraud', 'Fraud'],
        values_format='d',
    )

    display.ax_.set_title('Confusion matrix')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_precision_recall_curve(
        y_true, 
        y_proba,
        output_path: str | Path,
    ) -> None:
    """
    Save precision-recall curve
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = PrecisionRecallDisplay.from_predictions(
        y_true,
        y_proba,
    )

    display.ax_.set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_roc_curve(
        y_true, 
        y_proba,
        output_path: str | Path,
    ) -> None:
    """
    Save ROC curve
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = RocCurveDisplay.from_predictions(
        y_true,
        y_proba,
    )

    display.ax_.set_title("ROC Curve")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_model(config: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate trained model and tune threshold

    Threshold is selected on validation data 
    Final metrics are calculated on test data
    """
    model_path = config['artifacts']['model_output_path']
    model_name = config['model']['name']

    valid_path = config["training"]["valid_path"]
    test_path = config["evaluation"]["test_path"]

    model = load_model_artifact(model_path)

    valid_df = load_transactions(valid_path)
    test_df = load_transactions(test_path)

    X_valid, y_valid = split_features_and_target(valid_df)
    X_test, y_test = split_features_and_target(test_df)

    valid_proba = predict_probabilities(model, X_valid)
    test_proba = predict_probabilities(model, X_test)

    evaluation_config = config.get('evaluation', {})

    threshold_report = find_best_threshold(
        y_true= y_valid,
        y_proba= valid_proba,
        objective=evaluation_config.get(
            'threshold_objective',
            'min_expected_cost',
        ),
        false_negative_cost= evaluation_config.get(
            'false_negative_cost',
            500.0,
        ),
        false_positive_cost= evaluation_config.get(
            'false_positive_cost',
            25.0,
        ),
        min_precision= evaluation_config.get('min_precision'),
        min_recall= evaluation_config.get('min_recall'),
        grid_size=evaluation_config.get('threshold_grid_size', 101),
    )

    selected_threshold = threshold_report['best_threshold']

    test_threshold_metrics = calculate_threshold_metrics(
        y_true= y_test,
        y_proba=test_proba,
        threshold= selected_threshold,
        false_negative_cost= evaluation_config.get(
            'false_negative_cost',
            500.0,
        ),
        false_positive_cost= evaluation_config.get(
            'false_positive_cost',
            25.0,
        ),
    )

    probability_metrics = calculate_probability_metrics(
        y_true= y_test,
        y_proba= test_proba,
        prefix= 'test',
    )

    evaluation_report = build_evaluation_report(
        model_name = model_name,
        model_path= model_path,
        selected_threshold= selected_threshold,
        probability_metrics= probability_metrics,
        threshold_metrics= test_threshold_metrics,
    )

    save_json(
        threshold_report,
        config['artifacts']['threshold_report_path'],
    )

    save_json(
        evaluation_report,
        config['artifacts']['evaluation_metrics_path'],
    )

    y_test_pred = [
        1 if probability >= selected_threshold else 0
        for probability in test_proba
    ]

    save_confusion_matrix_plot(
        y_true= y_test,
        y_pred = y_test_pred,
        output_path= config['artifacts']['confusion_matrix_path'],
    )

    save_precision_recall_curve(

        y_true = y_test,
        y_proba= test_proba,
        output_path= config['artifacts']['precision_recall_curve_path'],
    )

    save_roc_curve(
        y_true = y_test,
        y_proba= test_proba,
        output_path= config['artifacts']['roc_curve_path'],
    )

    return evaluation_report