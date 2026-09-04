import os
from functools import lru_cache
from typing import Any

import pandas as pd

from app.features.build_features import build_feature_dataframe
from app.models.explain import explain_prediction
from app.models.registry import (
    load_model_for_inference,
    load_registered_model_metadata,
)
from app.models.train import load_training_config


def load_prediction_config(
        config_path: str | None = None,
    ) -> dict[str, Any]:
    """
    Load config for prediction
    """
    if config_path is None: 
        config_path = os.getenv(
            'DEFAULT_CONFIG_PATH',
            "configs/baseline_logistic.yaml",
        )
    return load_training_config(config_path)


@lru_cache(maxsize=1)
def get_default_config() -> dict[str, Any]:
    """
    Cached default prediction config
    """
    return load_prediction_config()


@lru_cache(maxsize=1)
def get_cached_model():
    """
    Load model once for inference
    """
    config = get_default_config()

    return load_model_for_inference(
        config= config,
        prefer_registry= False, #TODO: Change to 'True' when registry loading is stable 
    )


@lru_cache(maxsize=1)
def get_cached_model_metadata() -> dict[str, Any]:
    """
    Load registered model metadata once
    """
    config = get_default_config()

    return load_registered_model_metadata(config)


def transaction_to_dataframe(
        transaction: dict[str, Any],
    ) -> pd.DataFrame:
    """
    Convert transaction dictionary into single-row dataframe
    """
    return pd.DataFrame([transaction])


def prepare_features_for_prediction(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Build feature dataframe for inference 

    Target is not included during prediction
    """
    feature_df = build_feature_dataframe(
        df,
        include_target= False,
    )

    # The trained sklearn pipeline expects only FEATURE_COLUMNS, not ID_COLUMNS
    drop_columns = [
        column for column in ['transaction_id', 'customer_id']
        if column in feature_df.columns
    ]

    return feature_df.drop(columns=drop_columns, errors='ignore') 


# def predict_probability(
#         model,
#         features: pd.DataFrame,
#     ) -> float:
#     """
#     Predict fraud probability for a single transaction
#     """
#     probability = model.predict_proba(features)[0][1]

#     return float(probability)


def predict_probability(
        model,
        features: pd.DataFrame,
    ) -> float:
    """
    Predict fraud probability for a single transaction.

    Supports sklearn pipelines and MLflow pyfunc models 
    """
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features)[0][1]
        return float(probability)

    predictions = model.predict(features)

    if isinstance(predictions, pd.DataFrame):
        if 'prediction' in predictions.columns:
            return float(predictions['prediction'].iloc[0])
        return float(predictions.iloc[0, 0])

    if isinstance(predictions, list):
        return float(predictions[0])

    return float(predictions[0])


def assign_risk_band(probability: float) -> str:
    """
    Assign risk band from fraud probability
    """
    if probability < 0.30:
        return 'low'

    if probability < 0.70:
        return 'medium'

    return 'high'


def make_decision(
        probability: float,
        threshold: float,
    ) -> str:
    """
    Make business decision using selected threshold
    """
    if probability >= threshold:
        return 'review'

    return "approve"


def get_threshold_from_metadata(
        metadata: dict[str, Any],
        default_threshold: float = 0.5,
    ) -> float:
    """
    Return selected threshold from metadata
    """
    threshold = metadata.get('selected_threshold')

    if threshold is None:
        return default_threshold

    return float(threshold)


def build_prediction_response(
        transaction: dict[str, Any],
        probability: float,
        threshold: float,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
    """
    Build standard prediction response dictionary
    """
    return {
        'transaction_id': transaction.get('transaction_id'),
        'fraud_probability': probability,
        'risk_band': assign_risk_band(probability),
        'decision': make_decision(
            probability=probability, 
            threshold= threshold,
        ),
        'threshold_used': threshold,
        'model_name': metadata.get(
            'registered_model_name',
            metadata.get('model_name', 'unknown'),
        ),
        'model_version': metadata.get('model_version'),
        'model_alias': metadata.get('alias'),
    }


def predict_transaction(
        transaction: dict[str, Any],
        model = None,
        metadata: dict[str, Any] | None = None,
        include_explanation: bool = False,
    ) -> dict[str, Any]:
    """
    Predict fraud risk for one transaction
    """
    if model is None:
        model = get_cached_model()

    if metadata is None:
        metadata = get_cached_model_metadata()

    transaction_df = transaction_to_dataframe(transaction)

    features = prepare_features_for_prediction(transaction_df)

    probability = predict_probability(
        model= model,
        features= features,
    )

    threshold = get_threshold_from_metadata(metadata)

    response =  build_prediction_response(
        transaction= transaction,
        probability= probability,
        threshold= threshold,
        metadata= metadata,
    )

    if include_explanation:
        response['explanation'] = explain_prediction(
            model_pipeline= model,
            X = features,
            top_n= 5,
        )

    return response


def get_model_info(
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
    """
    Return model information for API/dashboard
    """
    if metadata is None:
        metadata = get_cached_model_metadata()

    return {
        'registered_model_name': metadata.get('registered_model_name'),
        'model_version': metadata.get('model_version'),
        'alias': metadata.get('alias'),
        'model_type': metadata.get('model_type'),
        'selected_threshold': metadata.get('selected_threshold'),
        'test_roc_auc': metadata.get('test_roc_auc'),
        'test_pr_auc': metadata.get('test_pr_auc'),
        'test_precision': metadata.get('test_precision'),
        'test_recall': metadata.get('test_recall'),
        'test_expected_cost': metadata.get('test_expected_cost'),
    }