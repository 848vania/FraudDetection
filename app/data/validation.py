import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.features.feature_schema import (
    ALLOWED_VALUES,
    NUMERIC_RANGES,
    RAW_NUMERIC_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    TARGET_ALLOWED_VALUES,
    TARGET_COLUMN,
)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load raw transaction data  from a CSV file 

    Parameters
    -----------
    path:
        Path to the raw CSV file 

    Returns 
    --------
    pd.DataFrame 
        Loaded transaction dataset

    Raises 
    -------
    FileNotFoundError
        If the input path does not exist
    ValueError 
        If the file is empty 
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Raw data file is empty: {path}")

    return df

def validate_required_columns(df: pd.DataFrame) -> list[str]:
    """
    Check whether all required columns are present

    Return a list  of error messages
    Empty list  means this validation passed
    """
    errors = []

    missing_columns = [
        column for column in REQUIRED_RAW_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ', '.join(missing_columns)
        )

    return errors


def validate_no_unexpected_columns(df: pd.DataFrame) -> list[str]:
    """
    Warn if columns exist that are not part of the expected schema 

    Extra columns may be harmless, so this returns warning instead of errors 
    """
    warnings = []

    expected_columns = set(REQUIRED_RAW_COLUMNS)
    actual_columns = set(df.columns)

    unexpected_columns = sorted(actual_columns - expected_columns)

    if unexpected_columns:
        warnings.append(
            "Unexpected columns found: "
            + ", ".join(unexpected_columns)
        )

    return warnings


def validate_missing_values(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Check missing values

    Missing values in target or ID columns are errors 
    Missing values in feature columns are warning because preprocessing may handle them
    """
    errors = []
    warnings = []

    for column in REQUIRED_RAW_COLUMNS:
        if column not in df.columns:
            continue 

        missing_count = int(df[column].isna().sum())

        if missing_count == 0:
            continue 

        missing_rate = missing_count / len(df)

        message = (
            f"Column {column} has {missing_count} missing values "
            f"({missing_rate:.2%})."
        )

        if column in ['transaction_id', 'customer_id', TARGET_COLUMN]:
            errors.append(message)
        else:
            warnings.append(message)

    return errors, warnings


def validate_unique_transaction_id(df: pd.DataFrame) -> list[str]:
    """
    Check that  transaction_id is unique
    """
    errors = []

    if 'transaction_id' not in df.columns:
        return errors

    duplicate_count = int(df['transaction_id'].duplicated().sum())

    if duplicate_count > 0:
        errors.append(
            f"transaction_id contains {duplicate_count} duplicate values"
        )

    return errors


def validate_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Check whether expected numeric columns can be interpreted as numeric
    """
    errors = [] 

    for column in RAW_NUMERIC_COLUMNS:
        if column not in df.columns:
            continue 

        converted = pd.to_numeric(df[column], errors="coerce")

        invalid_count = int(converted.isna().sum() - df[column].isna().sum())

        if invalid_count > 0:
            errors.append(
                f"Column {column} contains {invalid_count} non-numeric values"
            )

    return errors 


def validate_numeric_ranges(df: pd.DataFrame) -> list[str]:
    """
    Check numeric columns agains configured min/max ranges
    """
    errors = []

    for column, range_config in NUMERIC_RANGES.items():
        if column not  in df.columns:
            continue

        series = pd.to_numeric(df[column], errors='coerce')

        min_value = range_config['min']
        max_value = range_config['max']

        below_min = int((series < min_value).sum())
        above_max = int((series > max_value).sum())

        if below_min > 0:
            errors.append(
                f"Column {column} has {below_min} values below {min_value}"
            )

        if above_max > 0:
            errors.append(
                f"Column {column} has {above_max} values above {max_value}"
            )

    return errors 


def validate_categorical_values(df: pd.DataFrame) -> list[str]:
    """
    Check whether categorical columns contain only alloed values
    """

    errors = []

    for column, allowed_values in ALLOWED_VALUES.items():
        if column not in df.columns:
            continue 

        observed_values = set(df[column].dropna().unique())
        allowed_set = set(allowed_values)

        invalid_values = sorted(observed_values - allowed_set)

        if invalid_values:
            errors.append(
                f"Column {column} contains invalid values: {invalid_values}"
            )

    return errors


def validate_target_values(df: pd.DataFrame) -> list[str]:
    """
    Check that  the target column contains only allowed binary values
    """
    errors = []

    if TARGET_COLUMN not in df.columns:
        return errors

    observed_values = set(df[TARGET_COLUMN].dropna().unique())
    allowed_values = set(TARGET_ALLOWED_VALUES)

    invalid_values = sorted(observed_values - allowed_values)

    if invalid_values:
        errors.append(
            f"Target column {TARGET_COLUMN} contains invalid values: {invalid_values}"
        )

    return errors


def validate_target_distribution(
        df: pd.DataFrame,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Validate and summarize target distribution

    Fraud datasets are usually imbalanced, but both classes must exist
    """
    errors = []
    warnings = [] 

    if TARGET_COLUMN not in df.columns:
        return {}, errors, warnings

    target_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()

    positive_count = int((df[TARGET_COLUMN] == 1).sum())
    negative_count = int((df[TARGET_COLUMN] == 0).sum())
    total_count = len(df)

    fraud_rate = positive_count / total_count if total_count else 0.0

    summary = {
        'target_counts': {
            str(key): int(value)
            for key, value in target_counts.items()
        },
        'positive_count': positive_count,
        'negative_count': negative_count,
        'fraud_rate': fraud_rate,
    }

    if positive_count == 0:
        errors.append("Target has no positive fraud examples.")

    if negative_count == 0:
        errors.append("Target has no negative non-fraud examples.")

    if fraud_rate < 0.001:
        warnings.append(
            f"Fraud rate is very low: {fraud_rate:.3%}. "
            "Model training may be unstable."
        )

    if fraud_rate > 0.30:
        warnings.append(
            f"Fraud rate is unusually high: {fraud_rate:.2%}. "
            "Check whether synthetic generation or labels are correct."
        )

    return summary, errors, warnings


def detect_outliers(df: pd.DataFrame) -> list[str]:
    """
    Detect extreme numeric values using quantiles 

    Outliers are warnings because fraud datasets  often naturally contain them
    """
    warnings = []

    outlier_columns = [
        'transaction_amount',
        'avg_transaction_amount_30d',
        'num_transactions_24h',
        'num_failed_transactions_24h',
    ]

    for column in outlier_columns:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors='coerce').dropna()

        if series.empty:
            continue

        q99 = series.quantile(0.99)
        q999 = series.quantile(0.999)

        if q999 >  q99 * 3 and q99 > 0:
            warnings.append(
                f"Column {column} contains extreme high outliers. "
                f"99th percentile={q99:.2f}, 99.9th percentile={q999:.2f}"
            )

    return warnings


def build_validation_report(
        df: pd.DataFrame,
        errors: list[str],
        warnings: list[str],
        target_summary: dict[str, Any],
    ) -> dict[str, Any]:
    """
    Build a structured validation report
    """
    report = {
        'passed': len(errors) == 0,
        'num_rows': int(len(df)),
        'num_columns': int(len(df.columns)),
        'columns': list(df.columns),
        'errors': errors,
        'warnings': warnings,
    }

    report.update(target_summary)

    return report 


def run_data_validation(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run all data validation checks and return a structured report
    """
    errors = []
    warnings = []

    errors.extend(validate_required_columns(df))
    warnings.extend(validate_no_unexpected_columns(df))

    missing_errors, missing_warnings = validate_missing_values(df)
    errors.extend(missing_errors)
    warnings.extend(missing_warnings)

    errors.extend(validate_unique_transaction_id(df))
    errors.extend(validate_numeric_columns(df))
    errors.extend(validate_numeric_ranges(df))
    errors.extend(validate_categorical_values(df))
    errors.extend(validate_target_values(df))

    target_summary, target_errors, target_warnings = validate_target_distribution(df)
    errors.extend(target_errors)
    warnings.extend(target_warnings)

    warnings.extend(detect_outliers(df))

    return build_validation_report(
        df = df,
        errors= errors,
        warnings= warnings,
        target_summary= target_summary,
    )


def save_validation_report(
        report: dict[str, Any],
        output_path: str | Path,
    ) -> None:
    """
    Save validation report as JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)