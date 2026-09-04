import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from app.features.feature_schema import TARGET_COLUMN


def validate_split_inputs(
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN
    ) -> None:
    """
    Valiadte that the datagrame can be split 

    Raises 
    -------
    ValueError 
        If target column is missing, dataframa is empty, or only  one class exists
    """
    if df.empty:
        raise ValueError("Cannot split an empty dataframe")

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    unique_classes = df[target_col].dropna().unique()

    if len(unique_classes) < 2:
        raise ValueError(
            f"Target column {target_col} must contain at least two classes"
        )


def create_train_valid_test_split(
        df: pd.DataFrame,
        target_col: str  = TARGET_COLUMN,
        test_size: float = 0.2,
        valid_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, pd.DataFrame]:
    """
    Create stratified train/validation/test splits

    test_size and valid_size are fractions of the full dataset
    """
    validate_split_inputs(df, target_col=target_col)

    if test_size <= 0 or valid_size <=0:
        raise ValueError('test_size and valid_size must be greater than 0.')

    if test_size + valid_size >= 1.0:
        raise ValueError('test_size + valid_size must be less than 1.0')

    train_valid_df, test_df = train_test_split(
        df,
        test_size= test_size,
        random_state= random_state,
        stratify= df[target_col],
    )

    adjusted_valid_size = valid_size / (1.0 - test_size)

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size= adjusted_valid_size,
        random_state= random_state,
        stratify= train_valid_df[target_col],
    )

    return {
        'train': train_df.reset_index(drop=True),
        'valid': valid_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True),
    }


def get_fraud_rate(
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
    ) -> float:
    """
    Calculate positive class rate
    """
    if df.empty:
        return 0.0

    return float(df[target_col].mean())


def check_no_overlap(
        splits: dict[str, pd.DataFrame],
        id_col: str = 'transaction_id',
    ) -> bool:
    """
    Check that split  datasets have no overlapping IDs
    """
    if any(id_col not in split_df.columns for split_df in splits.values()):
        return True

    train_ids = set(splits['train'][id_col])
    valid_ids = set(splits['valid'][id_col])
    test_ids = set(splits['test'][id_col])

    return (
        train_ids.isdisjoint(valid_ids)
        and train_ids.isdisjoint(test_ids)
        and valid_ids.isdisjoint(test_ids)
    )


def build_split_summary(
        original_df: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        target_col: str = TARGET_COLUMN,
        test_size: float = 0.2,
        valid_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
    """
    Build summary of train/validation/test splits
    """
    train_df = splits['train']
    valid_df = splits['valid']
    test_df = splits['test']

    return {
        'input_rows': int(len(original_df)),
        'train_rows': int(len(train_df)),
        'valid_rows': int(len(valid_df)),
        'test_rows': int(len(test_df)),
        'target_colum': target_col,
        'input_fraud_rate': get_fraud_rate(original_df, target_col),
        'train_fraud_rate': get_fraud_rate(train_df, target_col),
        'valid_fraud_rate': get_fraud_rate(valid_df, target_col),
        'test_fraud_rate': get_fraud_rate(test_df, target_col),
        'random_state': random_state,
        'test_size': test_size,
        'valid_size': valid_size,
        'no_overlap': check_no_overlap(splits),
    }


def save_splits(
        splits: dict[str, pd.DataFrame],
        output_dir: str | Path = 'data/processed',
    ) -> None:
    """
    Save train, validation, and test splits as CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits['train'].to_csv(output_dir / 'train.csv', index=False)
    splits['valid'].to_csv(output_dir / 'valid.csv', index=False)
    splits['test'].to_csv(output_dir / 'test.csv', index=False)


def save_split_summary(
        summary: dict[str, Any],
        output_path: str | Path,
    ) -> None:
    """
    Save split summary as JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def create_and_save_splits(
        df: pd.DataFrame,
        output_dir: str | Path = 'data/processed',
        target_col: str = TARGET_COLUMN, 
        test_size: float = 0.2,
        valid_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
    """
    Create train/valid/test splits, save them, and return split summary
    """
    splits = create_train_valid_test_split(
        df = df,
        target_col= target_col,
        test_size= test_size,
        valid_size= valid_size,
        random_state= random_state,
    )

    summary = build_split_summary(
        original_df= df,
        splits= splits,
        target_col= target_col,
        test_size= test_size,
        valid_size= valid_size,
        random_state= random_state,
    )

    save_splits(splits, output_dir= output_dir)
    save_split_summary(summary, Path(output_dir) / 'split_summary.json')

    return summary