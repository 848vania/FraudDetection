import pandas as pd 

from app.data.validation import (
    run_data_validation,
    validate_categorical_values,
    validate_missing_values,
    validate_numeric_ranges,
    validate_required_columns,
    validate_target_distribution,
    validate_target_values,
    validate_unique_transaction_id,
)


def make_valid_df():
    return pd.DataFrame(
        {
            "transaction_id": ["txn_1", "txn_2", "txn_3"],
            "customer_id": ["cust_1", "cust_2", "cust_3"],
            "transaction_amount": [100.0, 250.0, 30.0],
            "transaction_hour": [10, 23, 3],
            "merchant_category": ["grocery", "electronics", "travel"],
            "customer_tenure_days": [300, 20, 1000],
            "num_transactions_24h": [1, 7, 2],
            "num_failed_transactions_24h": [0, 2, 0],
            "avg_transaction_amount_30d": [80.0, 90.0, 40.0],
            "is_foreign_transaction": [False, True, False],
            "device_type": ["mobile", "desktop", "tablet"],
            "country": ["US", "GB", "IN"],
            "previous_chargebacks": [0, 1, 0],
            "account_age_days": [300, 20, 1000],
            "risk_score_external": [0.1, 0.8, 0.2],
            "is_fraud": [0, 1, 0],
        }
    )


def test_valid_dataframe_passes_validation():
    df = make_valid_df()

    report = run_data_validation(df)

    assert report['passed'] is True
    assert report['errors'] == []


def test_missing_required_column_fails():
    df = make_valid_df().drop(columns=['is_fraud'])

    errors = validate_required_columns(df)

    assert any('is_fraud' in error for error in errors)


def test_duplicate_transaction_id_fails():
    df = make_valid_df()
    df.loc[1, 'transaction_id'] = 'txn_1'

    errors = validate_unique_transaction_id(df)
    print(df)

    assert errors 
    assert 'duplicate' in errors[0]


def test_negative_transaction_amount_fails():
    df = make_valid_df()
    df.loc[0, 'transaction_amount'] = -10.0

    errors = validate_numeric_ranges(df)

    assert any('transaction_amount' in error for error in errors)


def test_invalid_transaction_hour_fails():
    df = make_valid_df()
    df.loc[0, 'transaction_hour'] = 25

    errors = validate_numeric_ranges(df)

    assert any('transaction_hour' in error for error in errors)


def test_invalid_risk_score_fails():
    df = make_valid_df()
    df.loc[0, 'risk_score_external'] = 1.5

    errors =  validate_numeric_ranges(df)

    assert any('risk_score_external' in error for error in errors)


def test_invalid_categorical_value_fails():
    df = make_valid_df()
    df.loc[0, 'device_type'] = 'smart_fridge'

    errors = validate_categorical_values(df)

    assert any('device_type' in error for error in errors)


def test_invalid_target_value_fails():
    df = make_valid_df()
    df.loc[0, 'is_fraud'] = 2

    errors = validate_target_values(df)

    assert any('is_fraud' in error for error in errors)


def test_no_positive_target_examples_fails():
    df = make_valid_df()
    df['is_fraud'] = 0

    summary, errors, warnings = validate_target_distribution(df)

    assert any('no positive fraud examples' in error for error in errors)


def test_missing_target_values_are_errors():
    df = make_valid_df()
    df.loc[0, 'is_fraud'] = None 

    errors, warnings = validate_missing_values(df)

    assert any('is_fraud' in error for error in errors)


def test_missing_feature_values_are_warnings():
    df = make_valid_df()
    df.loc[0, 'transaction_amount'] = None 

    errors, warnings = validate_missing_values(df)

    assert not errors
    assert any('transaction_amount' in warning for warning in warnings)