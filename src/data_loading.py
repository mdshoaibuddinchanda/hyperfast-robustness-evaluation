"""Dataset loading utilities for the HyperFast robustness study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

CREDIT_TARGET_COLUMN = "default payment next month"


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for a supported dataset."""

    name: str
    source_url: str
    folder: str
    canonical_file: str
    target_column: str
    notes: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "heart_disease": DatasetSpec(
        name="heart_disease",
        source_url="https://archive.ics.uci.edu/dataset/45/heart+disease",
        folder="heart_disease",
        canonical_file="processed.cleveland.data",
        target_column="num",
        notes="Binary target is derived as num > 0.",
    ),
    "adult_income": DatasetSpec(
        name="adult_income",
        source_url="https://archive.ics.uci.edu/ml/datasets/Adult",
        folder="adult_income",
        canonical_file="adult.data",
        target_column="income",
        notes="Labels are binarized as >50K vs <=50K.",
    ),
    "credit_default": DatasetSpec(
        name="credit_default",
        source_url=(
            "https://archive.ics.uci.edu/ml/datasets/"
            "default+of+credit+card+clients"
        ),
        folder="credit_default",
        canonical_file="default of credit card clients.xls",
        target_column=CREDIT_TARGET_COLUMN,
        notes="Default target is already binary in the original file.",
    ),
}


def _strip_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from object columns in-place and return frame."""
    object_columns = frame.select_dtypes(include=["object"]).columns.tolist()
    for column in object_columns:
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def _clean_income_labels(series: pd.Series) -> pd.Series:
    """Normalize Adult dataset labels by removing extra whitespace and dots."""
    return series.astype(str).str.strip().str.rstrip(".")


def load_heart_disease(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Heart Disease (Cleveland) with binary target conversion."""
    path = data_root / "heart_disease" / "processed.cleveland.data"
    frame = pd.read_csv(path, header=None, names=HEART_COLUMNS, na_values="?")
    frame["num"] = pd.to_numeric(frame["num"], errors="raise").astype(int)

    labels = (frame["num"] > 0).astype(int)
    features = frame.drop(columns=["num"])

    return features, labels


def load_adult_income(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Adult Income from adult.data with binary target conversion."""
    path = data_root / "adult_income" / "adult.data"
    frame = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        na_values=" ?",
        skipinitialspace=True,
    )
    frame = _strip_object_columns(frame)
    frame["income"] = _clean_income_labels(frame["income"])

    labels = (frame["income"] == ">50K").astype(int)
    features = frame.drop(columns=["income"])

    return features, labels


def load_credit_default(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Credit Default from the official .xls file."""
    path = data_root / "credit_default" / "default of credit card clients.xls"
    frame = pd.read_excel(path, header=1, engine="xlrd")

    if "ID" in frame.columns:
        frame = frame.drop(columns=["ID"])

    labels = pd.to_numeric(
        frame[CREDIT_TARGET_COLUMN], errors="raise"
    ).astype(int)
    features = frame.drop(columns=[CREDIT_TARGET_COLUMN])
    features = _strip_object_columns(features)

    return features, labels


def get_feature_types(features: pd.DataFrame) -> dict[str, list[str]]:
    """Infer numeric and categorical columns from dataframe dtypes."""
    numeric_columns = [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
    }


def load_dataset(
    dataset_name: str,
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series, DatasetSpec]:
    """Load one supported dataset by name."""
    loader_map = {
        "heart_disease": load_heart_disease,
        "adult_income": load_adult_income,
        "credit_default": load_credit_default,
    }

    if dataset_name not in loader_map:
        supported = ", ".join(sorted(loader_map))
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: {supported}."
        )

    features, labels = loader_map[dataset_name](data_root)
    spec = DATASET_SPECS[dataset_name]

    return features, labels, spec


def build_manifest_entry(
    spec: DatasetSpec,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, Any]:
    """Create a dataset manifest row from loaded data and static metadata."""
    feature_types = get_feature_types(features)
    class_counts = labels.value_counts(dropna=False).sort_index()

    return {
        **asdict(spec),
        "rows": int(len(features)),
        "num_features": int(features.shape[1]),
        "class_distribution": {
            str(label): int(count)
            for label, count in class_counts.items()
        },
        "feature_types": feature_types,
    }
