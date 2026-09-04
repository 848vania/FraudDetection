import pandas as pd
import pytest

from app.data.splitting import (
    build_split_summary,
    check_no_overlap,
    create_train_valid_test_split,
    get_fraud_rate,
    validate_split_inputs,
)


def make_split_df(num_rows: int = 1000) -> pd.DataFrame:
    """
    Create synthetic feature-like dataframe for split tests 

    Fraud rate is 5%
    """
    rows = []

    for i in range(num_rows):
        is_fraud = 1 if i < int(num_rows * 0.05) else 0

        rows.append(
            {
                'transaction_id': f'txn_{i:06d}',
                'customer_id': f'cust_{i % 100:04d}',
                'transaction_amount': float(i + 1),
                'transaction_hour': i % 24,
                'merchant_category': 'electronics' if i % 2 == 0 else 'grocery',
                'is_fraud': is_fraud,
            }
        )

    return pd.DataFrame(rows)


def test_validate_split_inputs_passes_for_valid_dataframe():
    df = make_split_df()

    validate_split_inputs(df)

    assert True


def test_validate_split_inputs_fails_if_target_missing():
    df = make_split_df().drop(columns=['is_fraud'])

    with pytest.raises(ValueError, match= "Target column not found"):
        validate_split_inputs(df)


def test_validate_split_inputs_fails_if_only_one_class():
    df = make_split_df()
    df['is_fraud'] = 0

    with pytest.raises(ValueError, match= "at least two classes"):
        validate_split_inputs(df)


def test_create_train_valid_test_split_sizes():
    df = make_split_df(num_rows=1000)

    splits = create_train_valid_test_split(
        df,
        test_size=0.2,
        valid_size=0.2,
        random_state=42,
    )

    assert len(splits['train']) ==  600
    assert len(splits['valid']) ==  200
    assert len(splits['test']) ==  200


def test_create_train_valid_test_split_preserves_fraud_rate():
    df = make_split_df(num_rows=1000)

    splits = create_train_valid_test_split(
        df,
        test_size=0.2,
        valid_size=0.2,
        random_state=42,
    )

    original_rate =  get_fraud_rate(df)
    train_rate = get_fraud_rate(splits['train'])
    valid_rate = get_fraud_rate(splits['valid'])
    test_rate = get_fraud_rate(splits['test'])

    assert abs(train_rate - original_rate) < 0.01
    assert abs(valid_rate - original_rate) < 0.01
    assert abs(test_rate - original_rate) < 0.01


def test_check_no_overlap_returns_true():
    df = make_split_df(num_rows=1000)

    splits = create_train_valid_test_split(df)

    assert check_no_overlap(splits) is True


def test_create_train_valid_test_split_is_reproducible():
    df = make_split_df(num_rows=1000)

    splits_1 = create_train_valid_test_split(df, random_state=42)
    splits_2 = create_train_valid_test_split(df, random_state=42)

    train_ids_1 = splits_1['train']['transaction_id'].tolist()
    train_ids_2 = splits_2['train']['transaction_id'].tolist()

    assert train_ids_1 == train_ids_2


def test_different_random_state_changes_split():
    df = make_split_df(num_rows=1000)

    splits_1 = create_train_valid_test_split(df, random_state=42)
    splits_2 = create_train_valid_test_split(df, random_state=123)

    train_ids_1 = splits_1['train']['transaction_id'].tolist()
    train_ids_2 = splits_2['train']['transaction_id'].tolist()
    
    assert train_ids_1 != train_ids_2


def test_invalid_split_sizes_fail():
    df = make_split_df(num_rows=1000)

    with pytest.raises(ValueError, match= 'less than 1.0'):
        create_train_valid_test_split(
            df,
            test_size=0.6,
            valid_size=0.5,
        )


def test_build_split_summary_contains_expected_fields():
    df = make_split_df(num_rows=1000)
    splits = create_train_valid_test_split(df)

    summary = build_split_summary(
        original_df= df,
        splits= splits,
        test_size= 0.2,
        valid_size= 0.2,
        random_state= 42,
    )

    assert summary['input_rows'] == 1000
    assert summary['train_rows'] == 600
    assert summary['valid_rows'] == 200
    assert summary['test_rows'] == 200
    assert 'train_fraud_rate' in summary
    assert 'valid_fraud_rate' in summary
    assert 'test_fraud_rate' in summary
    assert summary['no_overlap'] is True