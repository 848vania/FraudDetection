import json 
from pathlib import Path
from typing import Any 

import pandas as pd 

from app.features.feature_schema import (
    RAW_CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


def get_dataset_shape(df: pd.DataFrame) -> dict[str, int]:
    """
    Return dataset row and column counts
    """
    return {
        'num_rows': int(len(df)),
        'num_columns': int(len(df.columns)),
    }


def get_missing_value_summary(df: pd.DataFrame) -> dict[str, int]:
    """
    Count missing values per each column 
    """

    return{
        column: int(df[column].isna().sum())
        for column in df.columns
    }


def get_duplicate_transaction_count(df: pd.DataFrame) -> int:
    """
    Count duplicate transaction IDs
    """
    if 'transaction_id' not in df.columns:
        return 0

    return int(df['transaction_id'].duplicated().sum())


def get_target_distribution(
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
    ) -> dict[str, Any]:
    """
    Calculate target distribution and fraud rate
    """
    if target_col not in df.columns:
        return {
            "target_column": target_col,
            "target_exists": False,
            "positive_count": 0,
            "negative_count": 0,
            "fraud_rate": None,
        }

    positive_count = int((df[target_col] == 1).sum())
    negative_count = int((df[target_col] == 0).sum())
    total_count = int(len(df))

    fraud_rate = positive_count / total_count if total_count else 0.0

    return {
        "target_column": target_col,
        "target_exists": True,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "fraud_rate": fraud_rate,
    }


def get_class_imbalance_summary(
        target_summary: dict[str, Any],
    ) -> dict[str, Any]:
    """
    Calculate class imbalance ration and majority-class baseline accuracy
    """
    positive_count = target_summary.get("positive_count", 0)
    negative_count = target_summary.get("negative_count", 0)

    total_count = positive_count + negative_count

    if total_count == 0:
        return {
            'class_imbalance_ratio': None,
            'majority_class_accuracy': None,
        }

    minority_count = min(positive_count, negative_count)
    majority_count = max(positive_count, negative_count)

    imbalance_ratio = (
        majority_count / minority_count
        if minority_count > 0
        else None
    )

    majority_class_accuracy = majority_count / total_count

    return {
        'class_imbalance_ratio': imbalance_ratio,
        'majority_class_accuracy': majority_class_accuracy,
    }


def get_numeric_summary(
        df: pd.DataFrame,
        numeric_colums: list[str] = RAW_NUMERIC_COLUMNS,
    ) -> dict[str, dict[str, float]]:
    """
    Calculate summary statistics for numeric columns
    """
    summary = {}

    for column in numeric_colums:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors='coerce').dropna()

        if series.empty:
            continue

        summary[column] = {
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'min': float(series.min()),
            'p25': float(series.quantile(0.25)),
            'p75': float(series.quantile(0.75)),
            'p95': float(series.quantile(0.95)),
            'p99': float(series.quantile(0.99)),
            'max': float(series.max()),
        }

    return summary


def get_categorical_summary(
        df: pd.DataFrame,
        categorical_columns: list[str] = RAW_CATEGORICAL_COLUMNS,
        top_n: int = 10,
    ) -> dict[str, dict[str, int]]:
    """
    Return top value counts for categorical columns
    """
    summary = {}

    for column in categorical_columns:
        if column not in df.columns:
            continue

        value_counts = (
            df[column]
            .astype(str)
            .value_counts(dropna=False)
            .head(top_n)
            .to_dict()
        )

        summary[column] = {
            str(key): int(value)
            for key, value in value_counts.items()
        }

    return summary


def get_target_rate_by_category(
        df: pd.DataFrame,
        categorical_columns: list[str] = RAW_CATEGORICAL_COLUMNS,
        target_col: str = TARGET_COLUMN,
    ) -> dict[str, dict[str, float]]:
    """
    Calculate target rate for each category value

    Example:
    merchant_category -> electronics -> fraud rate
    """
    if target_col not in df.columns:
        return {}

    result = {}

    for column in  categorical_columns:
        if column not in df.columns:
            continue

        rates = (
            df.groupby(column)[target_col]
            .mean()
            .sort_values(ascending=False)
            .to_dict()
        )

        result[column] = {
            str(key): float(value)
            for key, value in rates.items()
        }

    return result


def get_numeric_correlations_with_target(
        df: pd.DataFrame,
        numeric_columns: list[str] = RAW_NUMERIC_COLUMNS,
        target_col: str = TARGET_COLUMN,
        top_n: int = 10,
    ) -> list[dict[str, float]]:
    """
    Calculate absolute Pearson correlation between numeric features and target
    """
    if target_col not in df.columns:
        return []

    correlations = []

    for column in numeric_columns:
        if column not in df.columns:
            continue

        feature = pd.to_numeric(df[column], errors='coerce')
        target = pd.to_numeric(df[target_col], errors='coerce')

        valid = feature.notna() & target.notna()

        if valid.sum() < 2:
            continue

        correlation = feature[valid].corr(target[valid])

        if pd.isna(correlation):
            continue 

        correlations.append(
            {
                'feature': column,
                'correlation': float(correlation),
                'absolute_correlation': float(abs(correlation)),
            }
        )

    correlations = sorted(
        correlations,
        key  = lambda item: item['absolute_correlation'],
        reverse= True,
    )

    return correlations[:top_n]


def build_profile_notes(profile: dict[str, Any]) -> list[str]:
    """
    Build human-readable notes based on the profile
    """
    notes = []

    fraud_rate = profile.get('fraud_rate')
    majority_class_accuracy = profile.get('majority_class_accuracy')

    if fraud_rate is not None:
        notes.append(
            f"The fraud rate is {fraud_rate:.2%}, so the dataset is imbalanced."
        )

    if majority_class_accuracy is not None:
        notes.append(
            "Accuracy is misleading because a majority-class baseline would "
            f"achieve {majority_class_accuracy:.2%} accuracy."
        )

    notes.append(
        'Use PR-AUC, recall, precision, fraud capture rate, false positive rate, '
        "and expected cost instead of relying only on accuracy."
    )

    return notes 


def generate_data_profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a complete data profile for the fraud-risk dataset 
    """
    shape_summary = get_dataset_shape(df)
    missing_values = get_missing_value_summary(df)
    duplicate_transaction_ids = get_duplicate_transaction_count(df)
    target_summary = get_target_distribution(df)
    imbalance_summary = get_class_imbalance_summary(target_summary)
    numeric_summary = get_numeric_summary(df)
    categorical_summary = get_categorical_summary(df)
    target_rate_by_category = get_target_rate_by_category(df)
    correlations = get_numeric_correlations_with_target(df)

    profile = {
        **shape_summary,
        **target_summary,
        **imbalance_summary,
        'missing_values': missing_values,
        'duplicate_transaction_ids': duplicate_transaction_ids,
        'numeric_summary': numeric_summary,
        'categorical_summary': categorical_summary,
        'target_rate_by_category': target_rate_by_category,
        'top_numeric_correlations_with_target': correlations,
    }

    profile['profile_notes'] = build_profile_notes(profile)

    return profile 


def save_data_profile(
        profile: dict[str, Any],
        output_path: str | Path,
    ) -> None:
    """
    Save data profile as JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(profile, file, indent=2, ensure_ascii=False)