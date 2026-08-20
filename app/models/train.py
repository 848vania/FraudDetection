import json
from pathlib import Path
from typing import Any 

import joblib
import pandas as pd 
import yaml 
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from app.data.ingest import load_transactions
from app.features.build_features import split_features_and_target
from app.features.preprocessing import build_preprocessing_pipeline
from app.models.tracking import (
    get_active_run_id,
    log_artifact_file,
    log_config_params,
    log_sklearn_model,
    log_training_metrics,
    setup_mlflow,
)


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load training configuration from a YAML file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise  FileNotFoundError(f"Training config not found: {config_path}")

    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(f"Training config is empty: {config_path}")

    return config


def load_training_data(
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation datasets from config paths
    """
    train_path = config['training']['train_path']
    valid_path = config['training']['valid_path']

    train_df = load_transactions(train_path)
    valid_df = load_transactions(valid_path)

    return train_df, valid_df


def build_model(config: dict[str, Any]):
    """
    Build model specified in config
    """
    model_config = config['model']
    model_type = model_config['type']

    if model_type == 'logistic_regression':
        return LogisticRegression(
            class_weight=model_config.get('class_weight', 'balanced'),
            max_iter  = model_config.get('max_iter', 1000),
            random_state= model_config.get('random_state', 42),
        )

    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators= model_config.get('n_estimators', 200),
            max_depth= model_config.get('max_depth'),
            class_weight= model_config.get('class_weight', 'balanced'),
            random_state= model_config.get('random_state', 42),
            n_jobs= model_config.get('n_jobs', -1),
        )

    raise ValueError(f'Unsupported model type: {model_type}')


def build_training_pipeline(config: dict[str, Any]) -> Pipeline:
    """
    Build full sklearn training pipeline 

    The pipeline includes preprocessing and classifier
    """
    preprocessor = build_preprocessing_pipeline()
    model = build_model(config)

    return Pipeline(
        steps  = [
            ('preprocessor', preprocessor),
            ('model', model)
        ],
    )


def fit_model(
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
    """
    Fit model pipeline
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_validation_probabilities(
        model_pipeline: Pipeline,
        X_valid: pd.DataFrame,
    ) -> list[float]:
    """
    Predict positive-class probabilities for validation data
    """
    if not hasattr(model_pipeline, 'predict_proba'):
        raise ValueError("Model pipeline does not support  predict_proba")

    probabilities = model_pipeline.predict_proba(X_valid)[:, 1]

    return probabilities.tolist()


def calculate_training_metrics(
        y_valid: pd.Series,
        y_valid_proba: list[float],
    ) -> dict[str, float]:
    """
    Calculate validation metrics for fraud-risk model.

    PR_AUC is specially important for imbalanced classification
    """
    return {
        'valid_roc_auc': float(
            roc_auc_score(y_valid, y_valid_proba)
        ),
        'valid_pr_auc': float(
            average_precision_score(y_valid, y_valid_proba)
        ),
        'valid_log_loss': float(
            log_loss(y_valid, y_valid_proba)
        ),
    }


def build_training_result(
        config: dict[str, Any],
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
    """
    Build training result summary
    """
    model_config = config['model']
    artifact_path = config['artifacts']['model_output_path']

    return {
        'model_name': model_config['name'],
        'model_type': model_config['type'],
        'train_rows': int(len(train_df)),
        'valid_rows': int(len(valid_df)),
        **metrics,
        'artifact_path': artifact_path,
    }


def save_model_artifact(
        model_pipeline: Pipeline,
        output_path: str | Path,
    ) -> None:
    """
    Save fitted model pipeline to disk 
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, output_path)


def load_model_artifact(artifact_path: str | Path):
    """
    Load saved model pipeline from disk
    """
    artifact_path = Path(artifact_path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    return joblib.load(artifact_path)


def save_training_metrics(
        result: dict[str, Any],
        output_path: str | Path,
    ) -> None:
    """
    Save training metrics to JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)


def train_model(
        config: dict[str, Any],
        use_mlflow: bool = True,
    ) -> dict[str, Any]:
    """
    Train model from config and save artifacts

    Returns 
    --------
    dict[str, Any]
        Training result summary
    """

    if use_mlflow:
        setup_mlflow(config)

    train_df, valid_df = load_training_data(config)

    X_train, y_train = split_features_and_target(train_df)
    X_valid, y_valid = split_features_and_target(valid_df)

    pipeline = build_training_pipeline(config)

    if use_mlflow:
        with mlflow.start_run(run_name=config['model']['name']):
            log_config_params(config)

            pipeline = fit_model(pipeline, X_train, y_train)

            y_valid_proba  = predict_validation_probabilities(pipeline, X_valid)

            metrics = calculate_training_metrics(
                y_valid= y_valid,
                y_valid_proba= y_valid_proba,
            )

            log_training_metrics(metrics)

            result = build_training_result(
                config= config,
                train_df= train_df,
                valid_df= valid_df,
                metrics= metrics,
            )

            run_id = get_active_run_id()
            result['mlflow_run_id'] = run_id

            save_model_artifact(
                model_pipeline= pipeline,
                output_path= config['artifacts']['model_output_path'],
            )

            save_training_metrics(
                result= result,
                output_path= config['artifacts']['metrics_output_path'],
            )

            log_artifact_file(config['artifacts']['model_output_path'])
            log_artifact_file(config['artifacts']['metrics_output_path'])

            if config.get('experiment', {}).get('log_model', True):
                log_sklearn_model(pipeline, artifact_path='model')

            return result

    pipeline = fit_model(pipeline, X_train, y_train)
    
    y_valid_proba  = predict_validation_probabilities(pipeline, X_valid)

    metrics = calculate_training_metrics(
        y_valid= y_valid,
        y_valid_proba= y_valid_proba,
    )

    result = build_training_result(
        config= config,
        train_df= train_df,
        valid_df= valid_df,
        metrics= metrics,
    )

    result['mlflow_run_id'] = None

    save_model_artifact(
        model_pipeline= pipeline,
        output_path= config['artifacts']['model_output_path'],
    )

    save_training_metrics(
        result= result,
        output_path= config['artifacts']['metrics_output_path'],
    )

    return result