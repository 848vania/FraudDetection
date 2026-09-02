from typing import Any

import numpy as np
import pandas as pd

from app.features.preprocessing import get_feature_names_after_preprocessing


def get_classifier_from_pipeline(model_pipeline):
    """
    Extract classifier from sklearn pipeline
    """
    if hasattr(model_pipeline, 'named_steps'):
        return model_pipeline.named_steps.get('classifier')

    return None


def get_preprocessor_from_pipeline(model_pipeline):
    """
    Extract preprocessor from sklearn pipeline
    """
    if hasattr(model_pipeline, 'named_steps'):
        return model_pipeline.named_steps.get('preprocecssor')

    return None 


def supports_linear_explanation(model_pipeline) -> bool:
    """
    Check whether model supports coefficient-based explanations
    """
    classifier = get_classifier_from_pipeline(model_pipeline)

    return classifier is not None and hasattr(classifier, 'coef_')


def calculate_linear_contributions(
        model_pipeline,
        X: pd.DataFrame,
    ) -> list[dict[str, Any]]:
    """
    Calculate feature contributions for a linear sklearn pipeline

    contribution = transformed dfeature value = model coefficient
    """
    if not  supports_linear_explanation(model_pipeline):
        raise ValueError("Model does not support linear coefficient explanation.")

    preprocessor = get_preprocessor_from_pipeline(model_pipeline)
    classifier = get_classifier_from_pipeline(model_pipeline)

    transformed = preprocessor.transform(X)

    coefficients =  classifier.coef_[0]

    feature_names = get_feature_names_after_preprocessing(preprocessor)

    row_values = transformed[0]

    contributions = []

    for feature_name, value, coefficient in zip(
        feature_names,
        row_values,
        coefficients,
    ):
        contribution = float(value * coefficient)

        contributions.append(
            {
                'feature': feature_name,
                'transformed_value':float(value),
                'coefficient': float(coefficient),
                'contribution': contribution,
                'absolute_contribution': abs(contribution),
            }
        )

    contributions = sorted(
        contributions, 
        key = lambda item: item['absolute_contribution'],
        reverse= True,
    )

    return contributions


def format_top_factors(
        contributions: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
    """
    Format top contribution factors
    """
    top_contributions = contributions[:top_n]

    factors = []

    for item in top_contributions:
        direction = 'increases_risk' if item['contribution'] > 0 else 'decreases_risk'

        factors.append(
            {
                'feature': item['feature'],
                'direction': direction,
                'contribution': item['contribution'],
            }
        )

    return factors


def explain_prediction(
        model_pipeline, 
        X: pd.DataFrame,
        top_n: int =  5,
    ) -> dict[str, Any]:
    """
    Explain one prediction
    """
    if supports_linear_explanation(model_pipeline):
        contributions = calculate_linear_contributions(
            model_pipeline= model_pipeline,
            X= X,
        )

        return {
            'explanation_type' : 'linear_coefficients',
            'top_factors': format_top_factors(
                contributions= contributions,
                top_n= top_n,
            ),
        }

    return {
        'explanation_type': 'not_available',
        'top_factors': [],
        'reason': 'Model type does not support the baseline explanation method'
    }


def get_global_feature_importance(
        model_pipeline,
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
    """
    Return global feature importance for supported models
    """
    classifier = get_classifier_from_pipeline(model_pipeline)
    preprocessor = get_preprocessor_from_pipeline(model_pipeline)

    if classifier is None or preprocessor is None:
        return []

    feature_names = get_feature_names_after_preprocessing(preprocessor)

    if hasattr(classifier, 'coef_'):
        values = np.abs(classifier.coef_[0])
        importance_type = 'absolute_coefficient'
    elif hasattr(classifier, 'feature_importances_'):
        values = classifier.feature_importances_
        importance_type = 'tree_feature_importance'
    else:
        return []

    rows = [] 

    for feature, value in  zip(feature_names,values):
        rows.append(
            {
                'feature': feature,
                'importance': float(value),
                'importance_type': importance_type,
            }
        )

    rows = sorted(
        rows,
        key = lambda item: item['importance'],
        reverse= True,
    )

    return rows[:top_n]