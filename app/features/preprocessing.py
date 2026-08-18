from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features.feature_schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_numeric_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for numeric features
    """
    return Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scaler', StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for categorical features
    """
    return Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy= 'most_frequent')),
            (
                'one_hot_encoder',
                OneHotEncoder(
                    handle_unknown= 'ignore',
                    sparse_output= False,
                ),
            ),
        ]
    )


def build_preprocessing_pipeline(
        numeric_columns: list[str] = NUMERIC_COLUMNS,
        categorical_columns: list[str] = CATEGORICAL_COLUMNS,
    ) -> ColumnTransformer:
    """
    Build full preprocessing pipeline for model training and inference
    """
    numeric_pipeline = build_numeric_pipeline()
    categorical_pipeline = build_categorical_pipeline()

    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, numeric_columns),
            ('categorical', categorical_pipeline, categorical_columns),
        ],
        remainder= 'drop'
    )


def get_feature_names_after_preprocessing(
        preprocessor: ColumnTransformer,
    ) -> list[str]:
    """
    Return feature names after proprocessing.

    Requires fitted preprocessor
    """
    feature_names = []

    numeric_features = preprocessor.transformers_[0][2]
    feature_names.extend(numeric_features)

    categorical_transformer = preprocessor.named_transformers_['categorical']
    one_hot_encoder = categorical_transformer.named_steps['one_hot_encoder']

    categorical_features = preprocessor.transformers_[1][2]
    encoded_names = one_hot_encoder.get_feature_names_out(categorical_features)

    feature_names.extend(encoded_names.tolist())

    return feature_names