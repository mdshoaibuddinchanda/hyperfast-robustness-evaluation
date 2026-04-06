"""HyperFast model builders and wrappers."""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hyperfast import HyperFastClassifier
from sklearn.metrics import balanced_accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHECKPOINT = PROJECT_ROOT / "hyperfast.ckpt"
CONFIG_ROOT = PROJECT_ROOT / "configs"
HYPERFAST_DEFAULT_CONFIG = CONFIG_ROOT / "hyperfast_default.json"
HYPERFAST_TUNED_CONFIG = CONFIG_ROOT / "hyperfast_tuned.json"


def _resolve_device() -> str:
    """Resolve target device for HyperFast execution."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_checkpoint_path() -> str | None:
    """Return local checkpoint path when available, else None."""
    if LOCAL_CHECKPOINT.exists():
        return str(LOCAL_CHECKPOINT)
    return None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Read JSON payload when file exists, otherwise return None."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_hyperfast_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize and type-cast HyperFast parameter dictionary."""
    normalized = {
        "n_ensemble": int(params.get("n_ensemble", 1)),
        "optimization": params.get("optimization", None),
        "stratify_sampling": bool(params.get("stratify_sampling", False)),
        "feature_bagging": bool(params.get("feature_bagging", False)),
    }

    if normalized["optimization"] in ("null", "None", ""):
        normalized["optimization"] = None

    return normalized


def build_hyperfast(
    seed: int,
    *,
    n_ensemble: int,
    optimization: str | None,
    stratify_sampling: bool,
    feature_bagging: bool,
) -> HyperFastClassifier:
    """Build one HyperFast instance from explicit parameters."""
    return HyperFastClassifier(
        device=_resolve_device(),
        n_ensemble=n_ensemble,
        optimization=optimization,
        stratify_sampling=stratify_sampling,
        feature_bagging=feature_bagging,
        seed=seed,
        custom_path=_resolve_checkpoint_path(),
    )


def load_hyperfast_default_params() -> dict[str, Any]:
    """Load default HyperFast parameters from config with safe fallback."""
    payload = _read_json_if_exists(HYPERFAST_DEFAULT_CONFIG)
    if payload is None:
        return _normalize_hyperfast_params({})

    params = payload.get("params", {})
    if not isinstance(params, dict):
        return _normalize_hyperfast_params({})

    return _normalize_hyperfast_params(params)


def load_hyperfast_tuned_grid() -> list[dict[str, Any]]:
    """Load and expand tuned HyperFast grid from config."""
    payload = _read_json_if_exists(HYPERFAST_TUNED_CONFIG)
    if payload is None:
        return [load_hyperfast_default_params()]

    grid = payload.get("params_grid", {})
    if not isinstance(grid, dict) or not grid:
        return [load_hyperfast_default_params()]

    keys = [
        "n_ensemble",
        "optimization",
        "stratify_sampling",
        "feature_bagging",
    ]
    values_per_key: list[list[Any]] = []
    for key in keys:
        values = grid.get(key, [load_hyperfast_default_params()[key]])
        if not isinstance(values, list) or not values:
            values = [load_hyperfast_default_params()[key]]
        values_per_key.append(values)

    expanded: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for combo in product(*values_per_key):
        candidate = _normalize_hyperfast_params(dict(zip(keys, combo, strict=True)))
        candidate_key = (
            candidate["n_ensemble"],
            candidate["optimization"],
            candidate["stratify_sampling"],
            candidate["feature_bagging"],
        )
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        expanded.append(candidate)

    if not expanded:
        return [load_hyperfast_default_params()]

    return expanded


def build_hyperfast_default(seed: int) -> HyperFastClassifier:
    """Build the fast/default-like HyperFast operating point."""
    params = load_hyperfast_default_params()
    return build_hyperfast(
        seed=seed,
        n_ensemble=int(params["n_ensemble"]),
        optimization=params["optimization"],
        stratify_sampling=bool(params["stratify_sampling"]),
        feature_bagging=bool(params["feature_bagging"]),
    )


def select_best_hyperfast_tuned(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[HyperFastClassifier, dict[str, Any], float, int, float]:
    """Select the best tuned HyperFast candidate using validation accuracy.

    Returns:
        Tuple of (best_model, best_params, best_val_bal_acc,
        n_candidates, total_fit_time_sec).
    """
    candidates = load_hyperfast_tuned_grid()
    best_model: HyperFastClassifier | None = None
    best_params: dict[str, Any] = {}
    best_score = float("-inf")
    last_error: str | None = None

    search_start = time.perf_counter()
    for params in candidates:
        try:
            model = build_hyperfast(
                seed=seed,
                n_ensemble=int(params["n_ensemble"]),
                optimization=params["optimization"],
                stratify_sampling=bool(params["stratify_sampling"]),
                feature_bagging=bool(params["feature_bagging"]),
            )
            model.fit(x_train, y_train)
            val_pred = model.predict(x_val)
            score = float(balanced_accuracy_score(y_val, val_pred))

            if score > best_score:
                best_score = score
                best_params = dict(params)
                best_model = model
        except Exception as exc:  # pragma: no cover - runtime guard
            last_error = str(exc)

    total_fit_time_sec = time.perf_counter() - search_start

    if best_model is None:
        raise RuntimeError(
            "HyperFast tuned selection failed for all candidates. "
            f"Last error: {last_error}"
        )

    return best_model, best_params, best_score, len(candidates), total_fit_time_sec
