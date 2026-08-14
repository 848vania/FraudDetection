import pandas as pd 

from app.data.profiling import (
    generate_data_profile,
    get_class_imbalance_summary,
    get_categorical_summary,
    get_duplicate_transaction_count,
    get_missing_value_summary,
    get_numeric_correlations_with_target,
    get_numeric_summary,
    get_target_distribution,
    get_target_rate_by_category,
)


def make_profile_df():
    return pd.DataFrame(
        {
            'transaction_id': ['txn_1', 'txn_2', 'txn_3', 'txn_4'],
            'customer_id': ['cust_1', 'cust_2', 'cust_3', 'cust_4'],
            'transaction_amount': [100.0, 200.0, 50.0, 500.0],
            'transaction_hour': [10, 23, 2, 1],
            'merchant_category': [
                'grocery',
                'electronics',
                'grocery',
                'travel',
            ],
            'customer_tenuer_days': [300, 20, 500, 10],
            'num_transactions_24h': [1, 8, 2, 10],
            'num_failed_transactions_24h': [0, 3, 0, 4],
            'avg_transaction_amount_30d': [80.0, 100.0, 40.0, 120.0],
            'is_foreign_transaction': [False, True, False, True],
            'device_type': ['mobile', 'desktop', 'mobile', 'tablet'],
            'country': ['US', 'GB', 'US', 'US'],
            'previous_chargebacks': [0, 1, 0, 2],
            'account_age_days': [300, 20, 500, 10],
            'risk_score_external': [0.1, 0.8, 0.2, 0.9],
            'is_fraud': [0, 1, 0, 1],
        }
    )


def test_get_missing_value_summary():
    df = make_profile_df()
    df.loc[0, 'transaction_amount'] = None 

    summary = get_missing_value_summary(df)

    assert summary['transaction_amount'] == 1


def test_get_duplicate_transaction_count():
    df = make_profile_df()
    df.loc[1, 'transaction_id'] = 'txn_1'

    duplicate_count = get_duplicate_transaction_count(df)

    assert duplicate_count == 1 


def test_get_target_distribution():
    df = make_profile_df()

    summary = get_target_distribution(df)

    assert summary['positive_count'] == 2
    assert summary['negative_count'] == 2
    assert summary['fraud_rate'] == 0.5


def test_get_class_imbalance_summary():
    target_summary = {
        'positive_count': 1,
        'negative_count': 9,
    }

    summary = get_class_imbalance_summary(target_summary)

    assert summary['class_imbalance_ratio'] == 9
    assert summary['majority_class_accuracy'] == 0.9


def test_get_numeric_summary():
    df = make_profile_df()

    summary = get_numeric_summary(df)

    assert  'transaction_amount' in summary
    assert summary['transaction_amount']['min'] == 50.0
    assert summary['transaction_amount']['max'] == 500.0


def test_get_categorical_summary():
    df = make_profile_df()

    summary = get_categorical_summary(df)

    assert 'merchant_category' in summary
    assert summary['merchant_category']['grocery'] == 2


def test_get_target_rate_by_category():
    df = make_profile_df()

    result = get_target_rate_by_category(df)

    assert result['merchant_category']['electronics'] == 1.0
    assert result['merchant_category']['grocery'] == 0.0


def test_get_numeric_correlations_with_target():
    df = make_profile_df()

    correlations = get_numeric_correlations_with_target(df)

    assert isinstance(correlations, list)
    assert len(correlations) > 0
    assert 'feature' in correlations[0]
    assert 'correlation' in correlations[0]


def test_generate_data_profile():
    df = make_profile_df()

    profile = generate_data_profile(df)

    assert profile['num_rows'] == 4
    assert profile['num_columns'] == len(df.columns)
    assert profile['fraud_rate'] == 0.5
    assert 'numeric_summary' in profile
    assert 'categorical_summary' in profile
    assert 'profile_notes' in profile