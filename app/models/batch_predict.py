import json 
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd 

from app.data.ingest import load_transactions
from app.models.predict import (
    get_cached_model,
    get_cached_model_metadata,
    predict_transaction,   
)


def load_batch_input(input_path: str | Path) -> pd.DataFrame:
    """
    Load batch prediction input CSV
    """
    return load_transactions(input_path)


def dataframe_to_transactions(
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
    """
    Convert dataframe rows to transaction dictionaries 
    """
    return df.to_dict(orient= 'records')


def score_transactions(
        transactions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
    """
    Score a list of transaction dictionaries
    """
    model = get_cached_model()
    metadata = get_cached_model_metadata()

    predictions = []

    for transaction in transactions:
        prediction = predict_transaction(
            transaction= transaction,
            model= model,
            metadata= metadata,
        )

        predictions.append(prediction)

    return predictions


def add_batch_metadata(
        transactions: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        scored_at: str,
    ) -> list[dict[str, Any]]:
    """
    Add batch-level metadata to prediction records
    """
    enriched_predictions = []

    for transaction, prediction in zip(transactions, predictions):
        enriched_prediction = {
            'transaction_id': transaction.get('transaction_id'),
            'customer_id': transaction.get('customer_id'),
            **prediction,
            'scored_at': scored_at,
        }

        enriched_predictions.append(enriched_prediction)

    return enriched_predictions


def predictions_to_dataframe(
        predictions: list[dict[str, Any]],
    ) -> pd.DataFrame:
    """
    Convert predictions to dataframe
    """
    return pd.DataFrame(predictions)


def save_batch_predictions(
        prediction_df: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
    """
    Save batch predictions to CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(output_path, index= False)


def build_batch_summary(
        prediction_df: pd.DataFrame,
        input_path: str | Path, 
        output_path: str | Path,
        started_at: str,
        finished_at: str,
    ) -> dict[str, Any]:
    """
    Build summary of batch prediction job
    """
    total_rows = len(prediction_df)

    review_count = int((prediction_df['decision'] == 'review').sum()) if total_rows else 0
    high_risk_count = int((prediction_df['risk_band'] == 'high').sum()) if total_rows else 0 

    return {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'started_at': started_at,
        'finished_at': finished_at,
        'total_rows': int(total_rows),
        'review_count': review_count,
        'review_rate': review_count / total_rows if total_rows else 0.0,
        'high_risk_count': high_risk_count,
        'high_risk_rate': high_risk_count / total_rows if total_rows else 0.0,
        'average_fraud_probability': float(
            prediction_df['fraud_probability'].mean()
        ) if total_rows else 0.0,
        'risk_band_distribution': prediction_df['risk_band']
        .value_counts()
        .to_dict()
        if total_rows 
        else {},
        'decision_distributions': prediction_df['decision']
        .value_counts()
        .to_dict()
        if total_rows 
        else {},
    }


def save_batch_summary(
        summary: dict[str, Any],
        output_path: str | Path,
    ) -> None:
    """
    Save batch prediction summary as JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents= True, exist_ok= True)

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent= 2, ensure_ascii= False)


def make_timestamped_output_paths(
        output_dir: str | Path= 'data/predictions',
    ) -> tuple[Path, Path]:
    """
    Create timestamped prediction and summary output paths
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    predictions_path = output_dir / f'predictions_{timestamp}.csv'
    summary_path = output_dir / f'batch_prediction_summary_{timestamp}.json'

    return predictions_path, summary_path


def batch_predict(
        input_path: str | Path,
        output_path: str | Path | None = None, 
        summary_path: str | Path | None = None,
    ) -> dict[str, Any]:
    """
    Run batch prediction for a transaction CSV 
    """
    started_at = datetime.now(timezone.utc).isoformat()

    if output_path is None or summary_path is None:
        default_output_path, default_summary_path = make_timestamped_output_paths()

        if output_path is None:
            output_path = default_output_path

        if summary_path is None:
            summary_path = default_summary_path

    input_df = load_batch_input(input_path)
    transactions = dataframe_to_transactions(input_df)

    predictions = score_transactions(transactions)

    scored_at = datetime.now(timezone.utc).isoformat()

    enriched_predictions = add_batch_metadata(
        transactions= transactions,
        predictions= predictions,
        scored_at= scored_at,
    )

    prediction_df = predictions_to_dataframe(enriched_predictions)

    save_batch_predictions(
        prediction_df= prediction_df,
        output_path= output_path,
    )

    finished_at = datetime.now(timezone.utc).isoformat()

    summary = build_batch_summary(
        prediction_df= prediction_df,
        input_path= input_path,
        output_path= output_path,
        started_at= started_at,
        finished_at= finished_at,
    )

    save_batch_summary(
        summary= summary,
        output_path= summary_path,
    )

    summary['summary_path'] = str(summary_path)

    return summary