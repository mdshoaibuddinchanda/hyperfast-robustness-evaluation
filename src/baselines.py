"""Classical baseline model definitions."""

from __future__ import annotations

import os
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


_GPU_WARNING_EMITTED = False


def _cpu_baselines(seed: int) -> dict[str, Any]:
    """Return sklearn CPU baseline estimators."""
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


def _gpu_baselines(seed: int) -> dict[str, Any] | None:
    """Return cuML GPU baselines when RAPIDS stack is available."""
    try:
        from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
        from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
    except Exception:
        return None

    # cuML APIs do not fully match sklearn class_weight behavior.
    # Keep parameterization conservative and deterministic.
    return {
        "logistic_regression": CuMLLogisticRegression(
            max_iter=1000,
            fit_intercept=True,
            output_type="numpy",
        ),
        "random_forest": CuMLRandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_streams=1,
            output_type="numpy",
        ),
    }


def get_classical_baselines(seed: int, prefer_gpu: bool = False) -> dict[str, Any]:
    """Return baseline estimators with optional cuML GPU acceleration.

    GPU selection priority:
    1) Explicit prefer_gpu=True argument
    2) HF_USE_GPU_BASELINES=1 environment variable
    3) Fallback to CPU baselines when RAPIDS is unavailable
    """
    use_gpu = prefer_gpu or os.getenv("HF_USE_GPU_BASELINES", "0") == "1"
    if use_gpu:
        gpu_models = _gpu_baselines(seed)
        if gpu_models is not None:
            return gpu_models

        global _GPU_WARNING_EMITTED
        if not _GPU_WARNING_EMITTED:
            print(
                "[WARN] GPU baselines requested but RAPIDS/cuML is not available; "
                "falling back to sklearn CPU baselines."
            )
            _GPU_WARNING_EMITTED = True

    return _cpu_baselines(seed)
