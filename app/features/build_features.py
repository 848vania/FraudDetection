from pathlib import Path
from typing import Any 

import numpy as np 
import pandas as pd 

from app.features.feature_schema import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
)


def safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
        default_value: float = 0.0,
    ) -> pd.Series:
    """
    Safely divide two pandas Series.

    If denominator is zero or missing, return default_value.
    """
    denominator_safe = denominator.replace(0, np.nan)

    result  = numerator / denominator_safe

    return result.replace([np.inf, -np.inf], np.nan).fillna(default_value)


def add_amount_ratio_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio of transaction amount to customer's 30-day average amount
    """
    df = df.copy()

    df['amount_to_customer_avg_ratio'] = safe_divide(
        df['transaction_amount'],
        df['avg_transaction_amount_30d'],
        default_value=0.0,
    )

    return df 


def add_night_transaction_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flag for transaction occuring at night
    """
    df = df.copy()

    df['night_transaction'] = (
        df['transaction_hour'].between(0, 5)
    ).astype(int)

    return df 


def add_failed_transaction_rate_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio of failed transactions to total transactions in the last 24h 
    """
    df = df.copy()

    df['failed_transaction_rate_24h'] = safe_divide(
        df['num_failed_transactions_24h'],
        df['num_transactions_24h'],
        default_value=0.0,
    )

    return df 


def add_high_velocity_feature(
        df: pd.DataFrame,
        velocity_threshold: int = 8,
    ) -> pd.DataFrame:
    """
    Add binary flag for unusually high transaction volume in the last 24h
    """
    df = df.copy()

    df['high_velocity_flag'] = (
        df['num_transactions_24h'] >= velocity_threshold
    ).astype(int)

    return df 


def add_new_account_feature(
        df: pd.DataFrame,
        account_age_threshold_days: int = 30,
    ) -> pd.DataFrame:
    """
    Add binary flag for new accounts.
    """
    df = df.copy()
    df['new_account_flag'] = (
        df['account_age_days'] <= account_age_threshold_days
    ).astype(int)

    return df 


def add_previous_chargeback_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flag for customers with previous chargebacks
    """
    df = df.copy()

    df['has_previous_chargeback'] = (
        df['previous_chargebacks'] > 0
    ).astype(int)

    return df 


def normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize boolean-like columns
    """
    df = df.copy()

    if 'is_foreign_transaction' in df.columns:
        df['is_foreign_transaction'] = (
            df['is_foreign_transaction']
            .map(
                {
                    True: True,
                    False: False,
                    1: True,
                    0: False,
                    '1': True,
                    '0': False,
                    'true': True,
                    'false': False,
                    'True': True,
                    'False': False,
                }
            )
            .fillna(False)
        )

    return df 


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived features used by the model
    """
    df = df.copy()

    df = normalize_boolean_columns(df)
    df = add_amount_ratio_feature(df)
    df = add_night_transaction_feature(df)
    df = add_failed_transaction_rate_feature(df)
    df = add_high_velocity_feature(df)
    df = add_new_account_feature(df)
    df = add_previous_chargeback_feature(df)

    return df 


def build_feature_dataframe(
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
    """
    Build final feature dataframa used for training or prediction

    For training:
        include_target = True 

    For inference:
        include_target = False
    """

    df = add_derived_features(df)

    output_columns = ID_COLUMNS + FEATURE_COLUMNS

    if include_target and TARGET_COLUMN in df.columns:
        output_columns = output_columns + [TARGET_COLUMN]

    available_columns = [
        column for column in output_columns
        if column in df.columns
    ]

    return df[available_columns].copy()


def split_features_and_target(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a feature dataframe into X and y
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return x,y 


def save_feature_dataframe(
        df: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
    """
    Save feature dataframe as CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)