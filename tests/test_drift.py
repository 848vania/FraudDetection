import pandas as pd

from app.monitoring.drift import (
    build_drift_summary,
    calculate_categorical_drift,
    calculate_numeric_drift,
    calculate_prediction_drift,
    calculate_risk_band_drift,
    get_normalized_value_counts,
    total_variation_distance,
)


def make_reference_df():
    return pd.DataFrame(
        {
            'transaction_amount': [100, 110, 90, 105],
            'risk_score_external': [0.1, 0.2, 0.1, 0.2],
            'merchant_category': ['grocery', 'grocery', 'fashion', 'fashion'],
            'device_type': ['mobile', 'mobile', 'desktop', 'desktop'],
            'country': ['US', 'US', 'US', 'US'],
            'is_foreign_transaction': [False, False, False, False],
        }
    )


def make_current_df_shifted():
    return pd.DataFrame(
        {
            'transaction_amount': [300, 330, 290, 310],
            'risk_score_external': [0.8, 0.9, 0.7, 0.8],
            'merchant_category': ['travel', 'travel', 'travel', 'travel'],
            'device_type': ['mobile', 'mobile', 'mobile', 'mobile'],
            'country': ['GB', 'GB', 'GB', 'GB'],
            'is_foreign_transaction': [True, True, True, True]
        }
    )


def test_total_variation_distance():
    reference = {'A': 0.5, 'B': 0.5}
    current = {'A': 1.0, 'B': 0.0}

    distance = total_variation_distance(reference, current)

    assert distance == 0.5


def test_get_normalized_value_counts():
    series = pd.Series(['A', 'A', 'B'])

    counts = get_normalized_value_counts(series)

    assert counts['A'] == 2/3
    assert counts['B'] == 1/3


def test_calculate_numeric_drift_detects_shift():
    reference_df = make_reference_df()
    current_df = make_current_df_shifted()

    results = calculate_numeric_drift(
        reference_df= reference_df,
        current_df= current_df,
        numeric_columns= ['transaction_amount'],
        relative_mean_change_threshold= 0.25,
    )

    assert results[0]['drift_detected'] is True


def test_calculate_categorical_drift_detects_shift():
    reference_df = make_reference_df()
    current_df = make_current_df_shifted()

    results = calculate_categorical_drift(
        reference_df= reference_df,
        current_df= current_df,
        categorical_columns= ['merchant_category'],
        tvd_threshold= 0.20,
    )

    assert results[0]['drift_detected'] is True


def test_calculate_prediction_drift_detects_shift():
    reference_predictions = pd.DataFrame(
        {'fraud_probability': [0.1, 0.2, 0.1, 0.2]}
    )
    current_predictions = pd.DataFrame(
        {'fraud_probability': [0.7, 0.8, 0.9, 0.8]}
    )

    result = calculate_prediction_drift(
        reference_predictions= reference_predictions,
        current_predictions= current_predictions,
        relative_mean_change_threshold=0.25,
    )

    assert result['prediction_drift_detected'] is True


def test_calculate_risk_band_drift_detects_shift():
    reference_predictions = pd.DataFrame(
        {'risk_band': ['low', 'low', 'medium', 'medium']}
    )
    current_predictions = pd.DataFrame(
        {'risk_band': ['high', 'high', 'high', 'medium']}
    )

    result = calculate_risk_band_drift(
        reference_predictions= reference_predictions,
        current_predictions= current_predictions,
        tvd_threshold= 0.20,
    )

    assert result['risk_band_drift_detected'] is True


def test_build_drift_summary():
    numeric_drift = [
        {
            'feature': 'transaction_amount',
            'type': 'numeric',
            'drift_detected': True,
        }
    ]

    categorical_drift = [
        {
            'feature': 'merchant_category',
            'type': 'categorical',
            'drift_detected': False,
        }
    ]

    summary  = build_drift_summary(
        numeric_drift= numeric_drift,
        categorical_drift=  categorical_drift,
        prediction_drift= {
            'prediction_drift_detected': False
        },
        risk_band_drift= {
            'risk_band_drift_detected': False,
        },
    )

    assert summary['overall_drift_detected'] is True
    assert summary['num_drifted_features'] == 1
    assert summary['drifted_features'] == ['transaction_amount']