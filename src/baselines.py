"""Classical baseline model definitions."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_classical_baselines(seed: int) -> dict[str, object]:
    """Return baseline estimators with stable defaults."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
