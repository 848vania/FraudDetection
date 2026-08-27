import json 
from pathlib import Path
from typing import Any 

import numpy as np 
import pandas as pd 

from app.data.ingest import  load_transactions
from app.features.feature_schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def load_drift_data(
        reference_path: str | Path,
        current_path: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load reference and current datasets.
    """
    reference_df = load_transactions(reference_path)
    current_df = load_transactions(current_path)

    return reference_df, current_df


def calculate_numeric_drift(
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numeric_columns: list[str] = NUMERIC_COLUMNS,
        relative_mean_change_threshold: float = 0.25,
    ) -> list[dict[str, Any]]:
    """
    Detect numeric drift using relative mean average difference
    """
    results = []

    for column in numeric_columns:
        if column not in reference_df.columns or column not in current_df.columns:
            continue 

        reference_series = pd.to_numeric(
            reference_df[column],
            errors= 'coerce',
        ).dropna()

        current_series = pd.to_numeric(
            current_df[column],
            errors= 'coerce',
        ).dropna()

        if reference_series.empty or current_series.empty:
            continue

        reference_mean = float(reference_series.mean())
        current_mean = float(current_series.mean())

        denominator = abs(reference_mean) if abs(reference_mean) > 1e-9 else 1.0

        relative_change = abs(current_mean - reference_mean) / denominator

        drift_detected = relative_change >= relative_mean_change_threshold

        results.append(
            {
                'feature': column,
                'type': 'numeric',
                'reference_mean': reference_mean,
                'current_mean': current_mean,
                'relative_mean_change': float(relative_change),
                'threshold': relative_mean_change_threshold,
                'drift_detected': drift_detected,
            }
        )

    return results


def total_variation_distance(
        reference_distribution: dict[str, float],
        current_distribution: dict[str, float],
    ) -> float:
    """
    Calculate total variation distance between two categorical distributions.
    """
    keys = set(reference_distribution) | set(current_distribution)

    distance = 0.5 * sum(
        abs(
            reference_distribution.get(key, 0.0)
            - current_distribution.get(key, 0.0)
        )
        for key in keys
    )

    return float(distance)


def get_normalized_value_counts(series: pd.Series) -> dict[str, float]:
    """
    Return normalized value counts as dictionary
    """
    counts = series.astype(str).value_counts(normalize=True, dropna=False)

    return {
        str(key): float(value)
        for key,value in counts.to_dict().items()
    }


def calculate_categorical_drift(
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        categorical_columns: list[str] = CATEGORICAL_COLUMNS,
        tvd_threshold: float = 0.20,
    ) -> list[dict[str, Any]]:
    """
    Detect categorical drift using total variation distance
    """
    results = []

    for column in categorical_columns:
        if column not in reference_df.columns or column not in current_df.columns:
            continue

        reference_distribution = get_normalized_value_counts(
            reference_df[column]
        )

        current_distribution = get_normalized_value_counts(
            current_df[column]
        )

        distance = total_variation_distance(
            reference_distribution=reference_distribution,
            current_distribution= current_distribution,
        )

        drift_detected = distance >= tvd_threshold

        results.append(
            {
                'feature': column,
                'type': 'categorical',
                'total_variation_distance': distance,
                'threshold': tvd_threshold,
                'drift_detected': drift_detected,
                'reference_distribution': reference_distribution,
                'current_distribution': current_distribution,
            }
        )

    return results


def calculate_prediction_drift(
        reference_predictions: pd.DataFrame,
        current_predictions: pd.DataFrame,
        probability_column: str = 'fraud_probability',
        relative_mean_change_threshold: float = 0.25,
    ) -> dict[str, Any]:
    """
    Detect prediction probability  drift using mean probability shift
    """
    if (
        probability_column not in  reference_predictions.columns
        or probability_column not in current_predictions.columns
    ):
        return {
            'prediction_drift_checked': False,
            'reason': f'Missing probability column: {probability_column}',
            'predictiton_drift_detected': False,
        }

    reference_mean = float(reference_predictions[probability_column].mean())
    current_mean = float(current_predictions[probability_column].mean())

    denominator = abs(reference_mean) if abs(reference_mean) > 1e-9 else 1.0

    relative_change = abs(current_mean - reference_mean) / denominator

    drift_detected = relative_change >= relative_mean_change_threshold

    return {
        'prediction_drift_checked': True,
        'probabilit_column': probability_column,
        'reference_mean_probability': reference_mean,
        'current_mean_probability': current_mean,
        'relative_mean_change': float(relative_change),
        'threshold': relative_mean_change_threshold,
        'prediction_drift_detected': drift_detected,
    }


def calculate_risk_band_drift(
        reference_predictions: pd.DataFrame,
        current_predictions: pd.DataFrame,
        risk_band_column: str = 'risk_band',
        tvd_threshold: float = 0.20,
    ) -> dict[str, Any]:
    """
    Detect risk-band distribution drift
    """
    if (
        risk_band_column not in reference_predictions.columns
        or risk_band_column not in current_predictions.columns
    ):
        return {
            'risk_band_drift_checked': False,
            'reason': f'Missing risk band column: {risk_band_column}',
            'risk_band_drift_detected': False,
        }

    reference_distribution = get_normalized_value_counts(
        reference_predictions[risk_band_column]
    )

    current_distribution = get_normalized_value_counts(
        current_predictions[risk_band_column]
    )

    distance = total_variation_distance(
        reference_distribution= reference_distribution,
        current_distribution= current_distribution,
    )

    return {
        'risk_band_drift_checked': True,
        'total_variation_distance': distance,
        'threshold': tvd_threshold,
        'risk_band_drift_detected': distance >= tvd_threshold,
        'reference_distribution': reference_distribution,
        'current_distribution': current_distribution,
    }


def build_drift_summary(
        numeric_drift: list[dict[str, Any]],
        categorical_drift: list[dict[str, Any]],
        prediction_drift: dict[str, Any] | None = None,
        risk_band_drift: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
    """
    Build overall drift summary
    """
    all_feature_drift = numeric_drift + categorical_drift

    drifted_features = [
        item['feature']
        for item in all_feature_drift
        if item['drift_detected']
    ]

    prediction_drift_detected = (
        prediction_drift or {}
    ).get('prediction_drift_detected', False)

    risk_band_drift_detected = (
        risk_band_drift or {}
    ).get('risk_band_drift_detected', False)

    overall_drift_detected = (
        len(drifted_features) > 0
        or prediction_drift_detected
        or risk_band_drift_detected
    )

    if overall_drift_detected:
        recommendation = "investigate_or_retrain"
    else:
        recommendation = "no_action_needed"

    return {
        'overall_drift_detected': overall_drift_detected,
        'num_drifted_features': len(drifted_features),
        'drifted_features': drifted_features,
        'numeric_drift': numeric_drift,
        'categorical_drift': categorical_drift,
        'prediction_drift': prediction_drift,
        'recommendation': recommendation,
    }


def save_drift_summary(
        summary: dict[str, Any],
        output_path: str | Path
    ) -> None:
    """
    Save drift summary as JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def save_simple_html_report(
        summary: dict[str, Any],
        output_path: str | Path,
    ) -> None:
    """
    Save simple HTML drift report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drifted_features = summary.get('drifted_features', [])

    html = f"""
    <html>
    <head>
        <title>Fraud Model Drift Report</title>
    </head>
    <body>
        <h1>Fraud Model Drift Report</h1>

        <h2>Summary</h2>
        <p><strong>Overall drift detected:</strong> {summary.get("overall_drift_detected")}</p>
        <p><strong>Recommendation:</strong> {summary.get("recommendation")}</p>
        <p><strong>Number of drifted features:</strong> {summary.get("num_drifted_features")}</p>

        <h2>Drifted Features</h2>
        <ul>
            {''.join(f'<li>{feature}</li>' for feature in drifted_features)}
        </ul>

        <h2>Raw Summary</h2>
        <pre>{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>
    </body>
    </html>
    """

    with output_path.open('w', encoding= 'utf-8') as file:
        file.write(html)


def run_drift_check(
        reference_path: str | Path,
        current_path: str | Path,
        output_summary_path: str | Path = 'data/results/drift_summary.json',
        output_report_path: str | Path = 'data/results/drift_report.html',
        reference_predictions_path: str | Path | None = None,
        current_predictions_path: str | Path | None = None,
    ) -> dict[str, Any]:
    """
    Run full drift check
    """
    reference_df, current_df = load_drift_data(
        reference_path= reference_path,
        current_path= current_path,
    )

    numeric_drift = calculate_numeric_drift(
        reference_df= reference_df,
        current_df= current_df,
    )

    categorical_drift = calculate_categorical_drift(
        reference_df= reference_df,
        current_df= current_df,
    )

    prediction_drift = None
    risk_band_drift = None

    if reference_predictions_path and current_predictions_path:
        reference_predictions = load_transactions(reference_predictions_path)
        current_predictions =load_transactions(current_predictions_path)

        prediction_drift = calculate_prediction_drift(
            reference_predictions=  reference_predictions,
            current_predictions= current_predictions,
        )

        risk_band_drift = calculate_risk_band_drift(
            reference_predictions= reference_predictions,
            current_predictions= current_predictions,
        )

    summary  = build_drift_summary(
        numeric_drift= numeric_drift,
        categorical_drift= categorical_drift,
        prediction_drift= prediction_drift,
        risk_band_drift= risk_band_drift,
    )

    save_drift_summary(summary, output_summary_path)
    save_simple_html_report(summary, output_report_path)

    return summary