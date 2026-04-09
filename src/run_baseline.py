"""Run a minimal clean-data baseline for HyperFast robustness study."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from baselines import get_classical_baselines
from data_loading import DATASET_SPECS, load_dataset
from hyperfast_runner import build_hyperfast_default, select_best_hyperfast_tuned
from metrics import compute_binary_classification_metrics
from preprocessing import build_shared_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CONFIG_PATH = PROJECT_ROOT / "configs" / "split_config.json"


def _load_default_datasets() -> list[str]:
    """Resolve allowed datasets from split config, with registry fallback."""
    if SPLIT_CONFIG_PATH.exists():
        split_cfg = json.loads(SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))
        configured = [str(name) for name in split_cfg.get("datasets", [])]
        configured = [name for name in configured if name.strip()]
        if configured:
            return configured

    return sorted(DATASET_SPECS.keys())


def _load_split(dataset: str, seed: int) -> dict[str, Any]:
    split_path = PROJECT_ROOT / "data" / "splits" / f"{dataset}_seed{seed}.json"
    return json.loads(split_path.read_text(encoding="utf-8"))


def _extract_positive_scores(model: Any, x_data: np.ndarray) -> np.ndarray | None:
    """Return positive class scores when available."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = _to_numpy_array(model.predict_proba(x_data))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None

    return probabilities[:, 1]


def _to_numpy_array(values: Any) -> np.ndarray:
    """Convert cupy/cudf/numpy/list-like outputs to a numpy array."""
    if isinstance(values, np.ndarray):
        return values

    if hasattr(values, "get"):
        try:
            return np.asarray(values.get())
        except Exception:
            pass

    if hasattr(values, "to_numpy"):
        try:
            as_numpy = values.to_numpy()
            if isinstance(as_numpy, np.ndarray):
                return as_numpy
            return np.asarray(as_numpy)
        except Exception:
            pass

    return np.asarray(values)


def _run_model(
    model_name: str,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Fit, evaluate, and time one model."""
    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_time_sec = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    val_pred = _to_numpy_array(model.predict(x_val))
    test_pred = _to_numpy_array(model.predict(x_test))
    predict_time_sec = time.perf_counter() - pred_start

    val_score = _extract_positive_scores(model, x_val)
    test_score = _extract_positive_scores(model, x_test)

    return {
        "model": model_name,
        "timing": {
            "fit_time_sec": fit_time_sec,
            "predict_time_sec": predict_time_sec,
            "total_time_sec": fit_time_sec + predict_time_sec,
        },
        "validation": compute_binary_classification_metrics(y_val, val_pred, val_score),
        "test": compute_binary_classification_metrics(y_test, test_pred, test_score),
        "predictions": {
            "val_pred": val_pred.tolist(),
            "test_pred": test_pred.tolist(),
            "val_score": None if val_score is None else val_score.tolist(),
            "test_score": None if test_score is None else test_score.tolist(),
        },
    }


def _run_hyperfast_tuned(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Tune HyperFast on validation split, then evaluate test split."""
    (
        model,
        best_params,
        best_val_bal_acc,
        candidate_count,
        fit_time_sec,
    ) = select_best_hyperfast_tuned(
        seed=seed,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    pred_start = time.perf_counter()
    val_pred = _to_numpy_array(model.predict(x_val))
    test_pred = _to_numpy_array(model.predict(x_test))
    predict_time_sec = time.perf_counter() - pred_start

    val_score = _extract_positive_scores(model, x_val)
    test_score = _extract_positive_scores(model, x_test)

    return {
        "model": "hyperfast_tuned",
        "selection": {
            "policy": "best_validation_balanced_accuracy",
            "candidate_count": candidate_count,
            "best_validation_balanced_accuracy": best_val_bal_acc,
            "best_params": best_params,
        },
        "timing": {
            "fit_time_sec": fit_time_sec,
            "predict_time_sec": predict_time_sec,
            "total_time_sec": fit_time_sec + predict_time_sec,
        },
        "validation": compute_binary_classification_metrics(y_val, val_pred, val_score),
        "test": compute_binary_classification_metrics(y_test, test_pred, test_score),
        "predictions": {
            "val_pred": val_pred.tolist(),
            "test_pred": test_pred.tolist(),
            "val_score": None if val_score is None else val_score.tolist(),
            "test_score": None if test_score is None else test_score.tolist(),
        },
    }


def run_baseline(
    dataset: str,
    seed: int,
    output_root: Path,
    use_gpu_baselines: bool = False,
) -> Path:
    """Run clean baseline models for one dataset/seed and save artifacts."""
    features, labels, _ = load_dataset(dataset)
    split_payload = _load_split(dataset, seed)

    train_idx = np.array(split_payload["indices"]["train"], dtype=int)
    val_idx = np.array(split_payload["indices"]["val"], dtype=int)
    test_idx = np.array(split_payload["indices"]["test"], dtype=int)

    x_train = features.iloc[train_idx]
    x_val = features.iloc[val_idx]
    x_test = features.iloc[test_idx]

    y_train = labels.iloc[train_idx].to_numpy(dtype=int)
    y_val = labels.iloc[val_idx].to_numpy(dtype=int)
    y_test = labels.iloc[test_idx].to_numpy(dtype=int)

    preprocessor = build_shared_preprocessor(x_train)
    x_train_t = preprocessor.fit_transform(x_train)
    x_val_t = preprocessor.transform(x_val)
    x_test_t = preprocessor.transform(x_test)

    model_builders: dict[str, Any] = {
        "hyperfast_default": lambda: build_hyperfast_default(seed),
        **{
            name: (lambda model=model: model)
            for name, model in get_classical_baselines(
                seed,
                prefer_gpu=use_gpu_baselines,
            ).items()
        },
    }

    model_results = []
    prediction_rows: list[dict[str, Any]] = []

    for model_name, model_builder in model_builders.items():
        try:
            model = model_builder()
            result = _run_model(
                model_name=model_name,
                model=model,
                x_train=np.asarray(x_train_t),
                y_train=y_train,
                x_val=np.asarray(x_val_t),
                y_val=y_val,
                x_test=np.asarray(x_test_t),
                y_test=y_test,
            )
            model_results.append(
                {k: v for k, v in result.items() if k != "predictions"}
            )

            for split_name, y_true_values, y_pred_values, y_score_values in [
                (
                    "val",
                    y_val,
                    result["predictions"]["val_pred"],
                    result["predictions"]["val_score"],
                ),
                (
                    "test",
                    y_test,
                    result["predictions"]["test_pred"],
                    result["predictions"]["test_score"],
                ),
            ]:
                for row_idx, (y_true_item, y_pred_item) in enumerate(
                    zip(y_true_values, y_pred_values, strict=False)
                ):
                    prediction_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model": model_name,
                            "split": split_name,
                            "row_number_within_split": row_idx,
                            "y_true": int(y_true_item),
                            "y_pred": int(y_pred_item),
                            "y_score": None
                            if y_score_values is None
                            else float(y_score_values[row_idx]),
                        }
                    )
        except Exception as exc:  # pragma: no cover - runtime robustness guard
            model_results.append(
                {
                    "model": model_name,
                    "error": str(exc),
                    "timing": None,
                    "validation": None,
                    "test": None,
                }
            )

    try:
        tuned_result = _run_hyperfast_tuned(
            seed=seed,
            x_train=np.asarray(x_train_t),
            y_train=y_train,
            x_val=np.asarray(x_val_t),
            y_val=y_val,
            x_test=np.asarray(x_test_t),
            y_test=y_test,
        )
        model_results.append({k: v for k, v in tuned_result.items() if k != "predictions"})

        for split_name, y_true_values, y_pred_values, y_score_values in [
            (
                "val",
                y_val,
                tuned_result["predictions"]["val_pred"],
                tuned_result["predictions"]["val_score"],
            ),
            (
                "test",
                y_test,
                tuned_result["predictions"]["test_pred"],
                tuned_result["predictions"]["test_score"],
            ),
        ]:
            for row_idx, (y_true_item, y_pred_item) in enumerate(
                zip(y_true_values, y_pred_values, strict=False)
            ):
                prediction_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": "hyperfast_tuned",
                        "split": split_name,
                        "row_number_within_split": row_idx,
                        "y_true": int(y_true_item),
                        "y_pred": int(y_pred_item),
                        "y_score": None
                        if y_score_values is None
                        else float(y_score_values[row_idx]),
                    }
                )
    except Exception as exc:  # pragma: no cover - runtime robustness guard
        model_results.append(
            {
                "model": "hyperfast_tuned",
                "error": str(exc),
                "timing": None,
                "validation": None,
                "test": None,
            }
        )

    output_dir = output_root / dataset / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "dataset": dataset,
        "seed": seed,
        "split_file": f"data/splits/{dataset}_seed{seed}.json",
        "results": model_results,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    predictions_path = output_dir / "predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    return metrics_path


def main() -> None:
    """CLI entry point."""
    default_datasets = _load_default_datasets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=default_datasets,
        default=default_datasets[0],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "baseline",
    )
    parser.add_argument(
        "--use-gpu-baselines",
        action="store_true",
        help=(
            "Use RAPIDS/cuML GPU Logistic Regression and Random Forest when "
            "available. Falls back to sklearn CPU baselines otherwise."
        ),
    )
    args = parser.parse_args()

    use_gpu_baselines = (
        args.use_gpu_baselines
        or os.getenv("HF_USE_GPU_BASELINES", "0") == "1"
    )
    if use_gpu_baselines:
        print("Using GPU baseline preference (RAPIDS/cuML if available).")

    metrics_path = run_baseline(
        dataset=args.dataset,
        seed=args.seed,
        output_root=args.output_root,
        use_gpu_baselines=use_gpu_baselines,
    )
    print(f"Saved baseline metrics: {metrics_path}")


if __name__ == "__main__":
    main()
