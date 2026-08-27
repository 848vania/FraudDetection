import pandas as pd 

from app.features.build_features import (
    add_amount_ratio_feature,
    add_derived_features,
    add_failed_transaction_rate_feature,
    add_high_velocity_feature,
    add_new_account_feature,
    add_night_transaction_feature,
    add_previous_chargeback_feature,
    build_feature_dataframe,
    safe_divide,
    split_features_and_target,
)
from app.features.feature_schema import  FEATURE_COLUMNS, TARGET_COLUMN
from app.features.preprocessing import  build_preprocessing_pipeline


def make_raw_feature_df():
    return  pd.DataFrame(
        {
            "transaction_id": ["txn_1", "txn_2", "txn_3"],
            "customer_id": ["cust_1", "cust_2", "cust_3"],
            "transaction_amount": [100.0, 250.0, 30.0],
            "transaction_hour": [10, 23, 3],
            "merchant_category": ["grocery", "electronics", "travel"],
            "customer_tenure_days": [300, 20, 1000],
            "num_transactions_24h": [1, 9, 2],
            "num_failed_transactions_24h": [0, 3, 0],
            "avg_transaction_amount_30d": [50.0, 100.0, 0.0],
            "is_foreign_transaction": [False, True, False],
            "device_type": ["mobile", "desktop", "tablet"],
            "country": ["US", "GB", "KR"],
            "previous_chargebacks": [0, 1, 0],
            "account_age_days": [300, 20, 1000],
            "risk_score_external": [0.1, 0.8, 0.2],
            "is_fraud": [0, 1, 0],
        }
    )


def test_safe_divide_handles_zero_denominator():
    numerator = pd.Series([10.0, 20.0])
    denominator = pd.Series([2.0, 0.0])

    result = safe_divide(numerator, denominator)

    assert result.iloc[0] == 5.0
    assert result.iloc[1] == 0.0


def test_add_amount_ratio_features():
    df = make_raw_feature_df()

    result = add_amount_ratio_feature(df)

    assert "amount_to_customer_avg_ratio" in result.columns
    assert result.loc[0, 'amount_to_customer_avg_ratio'] == 2.0
    assert result.loc[2, 'amount_to_customer_avg_ratio'] == 0.0


def test_add_night_transaction_feature():
    df = make_raw_feature_df()

    result = add_night_transaction_feature(df)

    assert "night_transaction" in result.columns
    assert result.loc[0, 'night_transaction'] == 0
    assert result.loc[1, 'night_transaction'] == 0
    assert result.loc[2, 'night_transaction'] == 1


def test_add_failed_transaction_rate_feature():
    df = make_raw_feature_df()

    result = add_failed_transaction_rate_feature(df)

    assert 'failed_transaction_rate_24h' in result.columns
    assert result.loc[0, 'failed_transaction_rate_24h'] == 0.0
    assert round(result.loc[1, 'failed_transaction_rate_24h'], 2) == 0.33


def test_add_high_velocity_feature():
    df = make_raw_feature_df()

    result = add_high_velocity_feature(df, velocity_threshold=8)

    assert 'high_velocity_flag' in result.columns
    assert result.loc[0, 'high_velocity_flag'] == 0
    assert result.loc[1, 'high_velocity_flag'] == 1
    assert result.loc[2, 'high_velocity_flag'] == 0


def test_add_new_account_feature():
    df = make_raw_feature_df()

    result = add_new_account_feature(df, account_age_threshold_days=30)

    assert 'new_account_flag' in result.columns
    assert result.loc[0, 'new_account_flag'] == 0
    assert result.loc[1, 'new_account_flag'] == 1


def test_add_previous_chargeback_feature():
    df = make_raw_feature_df()

    result = add_previous_chargeback_feature(df)

    assert 'has_previous_chargeback' in result.columns
    assert result.loc[0, 'has_previous_chargeback'] == 0
    assert result.loc[1, 'has_previous_chargeback'] == 1


def test_add_derived_features():
    df = make_raw_feature_df()

    result = add_derived_features(df)

    expected_columns = [
        'amount_to_customer_avg_ratio',
        'night_transaction',
        'failed_transaction_rate_24h',
        'high_velocity_flag',
        'new_account_flag',
        'has_previous_chargeback',
    ]

    for column in expected_columns:
        assert column in result.columns


def test_build_feature_dataframe_includes_expected_columns():
    df = make_raw_feature_df()

    result  = build_feature_dataframe(df, include_target=True)

    for column in FEATURE_COLUMNS:
        assert column in result.columns

    assert TARGET_COLUMN in result.columns
    assert 'transaction_id' in result.columns
    assert 'customer_id' in result.columns


def test_split_features_and_target():
    df = make_raw_feature_df()
    feature_df = build_feature_dataframe(df, include_target=True)

    x,y = split_features_and_target(feature_df)

    assert TARGET_COLUMN not in x.columns
    assert len(x) == len(y)
    assert list(y) == [0, 1, 0]


def test_preprocessing_pipeline_fits_and_transforms():
    df = make_raw_feature_df()
    feature_df = build_feature_dataframe(df, include_target= True)

    X,y = split_features_and_target(feature_df)

    preprocessor = build_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(X,y)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_preprocessing_pipeline_handles_unknows_category():
    train_df = make_raw_feature_df()
    train_feature_df = build_feature_dataframe(train_df, include_target=True)
    X_train, y_train = split_features_and_target(train_feature_df)

    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X_train, y_train)

    test_df = make_raw_feature_df()
    test_df.loc[0, 'merchant_category'] == 'new_unknown_category'
    test_feature_df = build_feature_dataframe(test_df, include_target= True)
    X_test, _ = split_features_and_target(test_feature_df)

    transformed = preprocessor.transform(X_test)

    assert transformed.shape[0] == len(X_test)