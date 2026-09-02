import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.models.explain import (
    format_top_factors,
    get_classifier_from_pipeline,
    supports_linear_explanation,
)


def make_simple_pipeline():
    X = pd.DataFrame(
        {
            'feature_a': [0.0, 1.0, 2.0, 3.0],
            'feature_b': [3.0, 2.0, 1.0, 0.0],
        }
    )
    y = [0, 0, 1, 1]

    pipeline = Pipeline(
        steps = [
            ('preprocessor', StandardScaler()),
            ('classifier', LogisticRegression()),
        ]
    )

    pipeline.fit(X, y)

    return pipeline, X


def test_get_classifier_from_pipeline():
    pipeline, _ = make_simple_pipeline()

    classifier = get_classifier_from_pipeline(pipeline)

    assert classifier is not None 
    assert classifier.__class__.__name__ == "LogisticRegression"


def test_supports_linear_explanation():
    pipeline, _ = make_simple_pipeline()

    assert supports_linear_explanation(pipeline) is True


def test_format_top_factors():
    contributions = [
        {
            'feature': 'feature_a',
            'contribution': 0.8,
            'absolute_contribution': 0.8,
        },
        {
            'feature': 'feature_b',
            'contribution': -0.2,
            'absolute_contribution': 0.2,
        },
    ]

    factors = format_top_factors(contributions, top_n=2)

    assert factors[0]['feature'] == 'feature_a'
    assert factors[0]['direction'] == 'increases_risk'
    assert factors[1]['direction'] == 'decreases_risk'