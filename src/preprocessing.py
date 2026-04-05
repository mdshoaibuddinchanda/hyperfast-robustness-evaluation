"""Shared preprocessing pipeline for fair model comparison."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loading import get_feature_types


def build_shared_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build one train-fit preprocessing pipeline reused by all models."""
    feature_types = get_feature_types(features)
    numeric_columns = feature_types["numeric"]
    categorical_columns = feature_types["categorical"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")
