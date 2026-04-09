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
    "banknote_authentication": DatasetSpec(
        name="banknote_authentication",
        source_url="https://archive.ics.uci.edu/dataset/267/banknote+authentication",
        folder="banknote_authentication",
        canonical_file="data_banknote_authentication.txt",
        target_column="class",
        notes="Binary target is encoded as 0/1 in source data.",
    ),
    "breast_cancer_wisconsin_diagnostic": DatasetSpec(
        name="breast_cancer_wisconsin_diagnostic",
        source_url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        folder="breast_cancer_wisconsin_diagnostic",
        canonical_file="wdbc.data",
        target_column="diagnosis",
        notes="Binary target M/B converted to 1/0.",
    ),
    "haberman_survival": DatasetSpec(
        name="haberman_survival",
        source_url="https://archive.ics.uci.edu/dataset/43/haberman+s+survival",
        folder="haberman_survival",
        canonical_file="haberman.data",
        target_column="survival_status",
        notes="Target 2 (died within 5 years) is mapped to positive class.",
    ),
    "ionosphere": DatasetSpec(
        name="ionosphere",
        source_url="https://archive.ics.uci.edu/dataset/52/ionosphere",
        folder="ionosphere",
        canonical_file="ionosphere.data",
        target_column="class",
        notes="Labels g/b converted to 1/0.",
    ),
    "mushroom": DatasetSpec(
        name="mushroom",
        source_url="https://archive.ics.uci.edu/dataset/73/mushroom",
        folder="mushroom",
        canonical_file="agaricus-lepiota.data",
        target_column="class",
        notes="Poisonous/edible labels p/e converted to 1/0.",
    ),
    "pima_diabetes": DatasetSpec(
        name="pima_diabetes",
        source_url="https://archive.ics.uci.edu/dataset/34/diabetes",
        folder="pima_diabetes",
        canonical_file="pima-indians-diabetes.data",
        target_column="outcome",
        notes="Binary target is encoded as 0/1 in source data.",
    ),
    "sonar_mines_rocks": DatasetSpec(
        name="sonar_mines_rocks",
        source_url="https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks",
        folder="sonar_mines_rocks",
        canonical_file="sonar.all-data",
        target_column="class",
        notes="Labels M/R converted to 1/0.",
    ),
}


def _strip_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from object columns in-place and return frame."""
    object_columns = frame.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in object_columns:
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
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


def load_banknote_authentication(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Banknote Authentication (binary target in final column)."""
    path = data_root / "banknote_authentication" / "data_banknote_authentication.txt"
    columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    frame = pd.read_csv(path, header=None, names=columns)

    labels = pd.to_numeric(frame["class"], errors="raise").astype(int)
    features = frame.drop(columns=["class"])
    return features, labels


def load_breast_cancer_wisconsin_diagnostic(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Breast Cancer Wisconsin (Diagnostic)."""
    path = data_root / "breast_cancer_wisconsin_diagnostic" / "wdbc.data"
    frame = pd.read_csv(path, header=None)

    labels = (frame.iloc[:, 1].astype(str).str.strip() == "M").astype(int)
    features = frame.iloc[:, 2:].copy()
    features.columns = [f"feature_{idx + 1}" for idx in range(features.shape[1])]
    features = features.apply(pd.to_numeric, errors="coerce")
    return features, labels


def load_haberman_survival(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Haberman's Survival dataset."""
    path = data_root / "haberman_survival" / "haberman.data"
    columns = ["age", "operation_year", "positive_aux_nodes", "survival_status"]
    frame = pd.read_csv(path, header=None, names=columns)

    labels = (pd.to_numeric(frame["survival_status"], errors="raise") == 2).astype(int)
    features = frame.drop(columns=["survival_status"])
    return features, labels


def load_ionosphere(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Ionosphere with g/b target mapping."""
    path = data_root / "ionosphere" / "ionosphere.data"
    frame = pd.read_csv(path, header=None)

    labels = (frame.iloc[:, -1].astype(str).str.strip() == "g").astype(int)
    features = frame.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    features.columns = [f"feature_{idx + 1}" for idx in range(features.shape[1])]
    return features, labels


def load_mushroom(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Mushroom with poisonous/edible label mapping."""
    path = data_root / "mushroom" / "agaricus-lepiota.data"
    frame = pd.read_csv(path, header=None, na_values="?")
    frame = _strip_object_columns(frame)

    labels = (frame.iloc[:, 0].astype(str).str.strip() == "p").astype(int)
    features = frame.iloc[:, 1:].copy()
    features.columns = [f"feature_{idx + 1}" for idx in range(features.shape[1])]
    return features, labels


def load_pima_diabetes(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load Pima Indians Diabetes from UCI-style or OpenML-style CSV."""
    path = data_root / "pima_diabetes" / "pima-indians-diabetes.data"
    columns = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
        "outcome",
    ]
    frame_with_header = pd.read_csv(path)
    normalized_columns = [str(col).strip().lower() for col in frame_with_header.columns]

    if "class" in normalized_columns or "outcome" in normalized_columns:
        frame = frame_with_header.copy()
        frame.columns = normalized_columns
        frame = frame.rename(
            columns={
                "preg": "pregnancies",
                "plas": "glucose",
                "pres": "blood_pressure",
                "skin": "skin_thickness",
                "insu": "insulin",
                "mass": "bmi",
                "pedi": "diabetes_pedigree_function",
                "class": "outcome",
            }
        )

        if "outcome" not in frame.columns:
            raise ValueError("Pima dataset file is missing outcome/class target column.")

        labels_raw = frame["outcome"]
        labels_text = labels_raw.astype(str).str.strip().str.lower()
        if labels_text.isin(["tested_positive", "tested_negative"]).all():
            labels = (labels_text == "tested_positive").astype(int)
        else:
            labels = pd.to_numeric(labels_raw, errors="raise").astype(int)

        feature_columns = [name for name in columns[:-1] if name in frame.columns]
        if not feature_columns:
            feature_columns = [name for name in frame.columns if name != "outcome"]

        features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        return features, labels

    frame = pd.read_csv(path, header=None, names=columns)
    labels = pd.to_numeric(frame["outcome"], errors="raise").astype(int)
    features = frame.drop(columns=["outcome"])
    return features, labels


def load_sonar_mines_rocks(
    data_root: Path = RAW_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Sonar with mine/rock target mapping."""
    path = data_root / "sonar_mines_rocks" / "sonar.all-data"
    frame = pd.read_csv(path, header=None)

    labels = (frame.iloc[:, -1].astype(str).str.strip() == "M").astype(int)
    features = frame.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    features.columns = [f"feature_{idx + 1}" for idx in range(features.shape[1])]
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
        "banknote_authentication": load_banknote_authentication,
        "breast_cancer_wisconsin_diagnostic": load_breast_cancer_wisconsin_diagnostic,
        "haberman_survival": load_haberman_survival,
        "ionosphere": load_ionosphere,
        "mushroom": load_mushroom,
        "pima_diabetes": load_pima_diabetes,
        "sonar_mines_rocks": load_sonar_mines_rocks,
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
