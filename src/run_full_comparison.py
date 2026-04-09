"""Run baseline and robustness experiments for all datasets and seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from baselines import get_classical_baselines
from data_loading import get_feature_types, load_dataset
from hyperfast_runner import (
    build_hyperfast,
    build_hyperfast_default,
    select_best_hyperfast_tuned,
)
from metrics import compute_binary_classification_metrics
from preprocessing import build_shared_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
RUNS_ROOT = PROJECT_ROOT / "runs"
RESULTS_ROOT = PROJECT_ROOT / "results"
SUMMARY_ROOT = RESULTS_ROOT / "summary_tables"
LOGS_ROOT = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for full comparison execution."""

    datasets: list[str]
    seeds: list[int]
    noise_sigmas: list[float]
    missing_rates: list[float]
    reduced_fractions: list[float]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON config file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config() -> ExperimentConfig:
    """Load all required experiment configuration files."""
    split_cfg = _read_json(CONFIG_ROOT / "split_config.json")
    noise_cfg = _read_json(CONFIG_ROOT / "noise_experiment.json")
    missing_cfg = _read_json(CONFIG_ROOT / "missingness_experiment.json")
    reduced_cfg = _read_json(CONFIG_ROOT / "reduced_data_experiment.json")

    return ExperimentConfig(
        datasets=list(split_cfg["datasets"]),
        seeds=[int(seed) for seed in split_cfg["seeds"]],
        noise_sigmas=[float(x) for x in noise_cfg["sigma_grid"]],
        missing_rates=[float(x) for x in missing_cfg["rates"]],
        reduced_fractions=[float(x) for x in reduced_cfg["train_fraction_grid"]],
    )


def _load_split(dataset: str, seed: int) -> dict[str, Any]:
    """Load precomputed split indices for one dataset and seed."""
    split_path = PROJECT_ROOT / "data" / "splits" / f"{dataset}_seed{seed}.json"
    return json.loads(split_path.read_text(encoding="utf-8"))


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


def _extract_positive_scores(model: Any, x_data: np.ndarray) -> np.ndarray | None:
    """Return positive-class scores when model provides predict_proba."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = _to_numpy_array(model.predict_proba(x_data))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None

    return probabilities[:, 1]


def _build_model_builders(seed: int, use_gpu_baselines: bool = False) -> dict[str, Any]:
    """Build estimator factories for the active seed."""
    builders: dict[str, Any] = {
        "hyperfast_default": lambda: build_hyperfast_default(seed),
    }

    for model_name, model in get_classical_baselines(
        seed,
        prefer_gpu=use_gpu_baselines,
    ).items():
        builders[model_name] = lambda model=model: model

    return builders


def _fit_and_evaluate(
    model_name: str,
    model_builder: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, list[Any]], Any | None]:
    """Fit one model and evaluate on validation and test splits."""
    try:
        model = model_builder()

        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_time = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        val_pred = _to_numpy_array(model.predict(x_val))
        test_pred = _to_numpy_array(model.predict(x_test))
        predict_time = time.perf_counter() - pred_start

        val_score = _extract_positive_scores(model, x_val)
        test_score = _extract_positive_scores(model, x_test)

        result = {
            "model": model_name,
            "timing": {
                "fit_time_sec": fit_time,
                "predict_time_sec": predict_time,
                "total_time_sec": fit_time + predict_time,
            },
            "validation": compute_binary_classification_metrics(
                y_val,
                val_pred,
                val_score,
            ),
            "test": compute_binary_classification_metrics(
                y_test,
                test_pred,
                test_score,
            ),
        }

        predictions = {
            "val_pred": val_pred.tolist(),
            "test_pred": test_pred.tolist(),
            "val_score": None if val_score is None else val_score.tolist(),
            "test_score": None if test_score is None else test_score.tolist(),
        }
        return result, predictions, model
    except Exception as exc:  # pragma: no cover - runtime guard
        return (
            {
                "model": model_name,
                "error": str(exc),
                "timing": None,
                "validation": None,
                "test": None,
            },
            {
                "val_pred": [],
                "test_pred": [],
                "val_score": None,
                "test_score": None,
            },
            None,
        )


def _fit_and_evaluate_hyperfast_tuned(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, list[Any]], Any | None]:
    """Select HyperFast tuned params on validation then evaluate test."""
    try:
        (
            model,
            best_params,
            best_val_bal_acc,
            candidate_count,
            total_fit_time,
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
        predict_time = time.perf_counter() - pred_start

        val_score = _extract_positive_scores(model, x_val)
        test_score = _extract_positive_scores(model, x_test)

        result = {
            "model": "hyperfast_tuned",
            "selection": {
                "policy": "best_validation_balanced_accuracy",
                "candidate_count": candidate_count,
                "best_validation_balanced_accuracy": best_val_bal_acc,
                "best_params": best_params,
            },
            "timing": {
                # Includes full candidate-search fitting cost.
                "fit_time_sec": total_fit_time,
                "predict_time_sec": predict_time,
                "total_time_sec": total_fit_time + predict_time,
            },
            "validation": compute_binary_classification_metrics(
                y_val,
                val_pred,
                val_score,
            ),
            "test": compute_binary_classification_metrics(
                y_test,
                test_pred,
                test_score,
            ),
        }

        predictions = {
            "val_pred": val_pred.tolist(),
            "test_pred": test_pred.tolist(),
            "val_score": None if val_score is None else val_score.tolist(),
            "test_score": None if test_score is None else test_score.tolist(),
        }
        return result, predictions, model
    except Exception as exc:  # pragma: no cover - runtime guard
        return (
            {
                "model": "hyperfast_tuned",
                "error": str(exc),
                "timing": None,
                "validation": None,
                "test": None,
            },
            {
                "val_pred": [],
                "test_pred": [],
                "val_score": None,
                "test_score": None,
            },
            None,
        )


def _evaluate_pretrained(
    model_name: str,
    model: Any,
    fit_time_sec: float,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Evaluate a pre-trained model under corrupted validation/test inputs."""
    pred_start = time.perf_counter()
    val_pred = _to_numpy_array(model.predict(x_val))
    test_pred = _to_numpy_array(model.predict(x_test))
    predict_time = time.perf_counter() - pred_start

    val_score = _extract_positive_scores(model, x_val)
    test_score = _extract_positive_scores(model, x_test)

    result = {
        "model": model_name,
        "timing": {
            "fit_time_sec": fit_time_sec,
            "predict_time_sec": predict_time,
            "total_time_sec": fit_time_sec + predict_time,
        },
        "validation": compute_binary_classification_metrics(y_val, val_pred, val_score),
        "test": compute_binary_classification_metrics(y_test, test_pred, test_score),
    }

    predictions = {
        "val_pred": val_pred.tolist(),
        "test_pred": test_pred.tolist(),
        "val_score": None if val_score is None else val_score.tolist(),
        "test_score": None if test_score is None else test_score.tolist(),
    }
    return result, predictions


def _apply_gaussian_noise(
    x_data: np.ndarray,
    n_numeric: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Apply additive Gaussian noise to transformed numeric columns only."""
    noisy = np.array(x_data, copy=True)
    if n_numeric <= 0:
        return noisy

    rng = np.random.default_rng(seed)
    noisy[:, :n_numeric] = noisy[:, :n_numeric] + rng.normal(
        loc=0.0,
        scale=sigma,
        size=(noisy.shape[0], n_numeric),
    )
    return noisy


def _apply_mcar_missing(
    frame: pd.DataFrame,
    rate: float,
    seed: int,
) -> pd.DataFrame:
    """Apply MCAR missingness independently across all cells."""
    rng = np.random.default_rng(seed)
    mask = rng.random(frame.shape) < rate
    return frame.mask(mask)


def _sample_train_subset_indices(
    y_train: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Sample a stratified training subset by fraction."""
    all_indices = np.arange(y_train.shape[0], dtype=int)
    if fraction >= 1.0:
        return all_indices

    sampled, _ = train_test_split(
        all_indices,
        train_size=fraction,
        random_state=seed,
        stratify=y_train,
    )
    return np.array(sampled, dtype=int)


def _build_prediction_rows(
    dataset: str,
    seed: int,
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: list[Any],
    y_score: list[Any] | None,
    experiment: str,
    condition_name: str,
    condition_value: str,
) -> list[dict[str, Any]]:
    """Convert per-model predictions to row records for CSV export."""
    rows: list[dict[str, Any]] = []
    for row_idx, (true_value, pred_value) in enumerate(
        zip(y_true, y_pred, strict=False)
    ):
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "experiment": experiment,
                "condition_name": condition_name,
                "condition_value": condition_value,
                "model": model_name,
                "split": split_name,
                "row_number_within_split": row_idx,
                "y_true": int(true_value),
                "y_pred": int(pred_value),
                "y_score": None if y_score is None else float(y_score[row_idx]),
            }
        )
    return rows


def _results_to_metric_rows(
    dataset: str,
    seed: int,
    experiment: str,
    condition_name: str,
    condition_value: str,
    split_file: str,
    artifact_path: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert condition result payload into long-form metric rows."""
    rows: list[dict[str, Any]] = []

    for result in results:
        model_name = str(result.get("model"))
        error = result.get("error")
        timing = result.get("timing")

        if error is not None:
            for split_name in ("validation", "test"):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "experiment": experiment,
                        "condition_name": condition_name,
                        "condition_value": condition_value,
                        "split_file": split_file,
                        "artifact_path": artifact_path,
                        "model": model_name,
                        "split": split_name,
                        "status": "error",
                        "error": str(error),
                        "balanced_accuracy": None,
                        "f1": None,
                        "precision": None,
                        "recall": None,
                        "auroc": None,
                        "fit_time_sec": None,
                        "predict_time_sec": None,
                        "total_time_sec": None,
                    }
                )
            continue

        for split_name, metric_key in (
            ("validation", "validation"),
            ("test", "test"),
        ):
            metrics = result.get(metric_key)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "experiment": experiment,
                    "condition_name": condition_name,
                    "condition_value": condition_value,
                    "split_file": split_file,
                    "artifact_path": artifact_path,
                    "model": model_name,
                    "split": split_name,
                    "status": "ok",
                    "error": None,
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "f1": metrics.get("f1"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "auroc": metrics.get("auroc"),
                    "fit_time_sec": timing.get("fit_time_sec"),
                    "predict_time_sec": timing.get("predict_time_sec"),
                    "total_time_sec": timing.get("total_time_sec"),
                }
            )

    return rows


def _save_condition_artifacts(
    dataset: str,
    seed: int,
    experiment: str,
    condition_name: str,
    condition_value: str,
    split_file: str,
    results: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> Path:
    """Persist one condition's metrics and predictions."""
    seed_dir = RUNS_ROOT / experiment / dataset / f"seed{seed}"
    if condition_name == "clean":
        output_dir = seed_dir
    else:
        output_dir = seed_dir / f"{condition_name}_{condition_value}"

    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": dataset,
        "seed": seed,
        "split_file": split_file,
        "experiment": experiment,
        "condition": {
            "name": condition_name,
            "value": condition_value,
        },
        "results": results,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    predictions_path = output_dir / "predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    return metrics_path


def _summarize_metrics(metrics_df: pd.DataFrame) -> None:
    """Write mean/std summary tables for test metrics."""
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    test_rows = metrics_df[metrics_df["split"] == "test"].copy()
    test_rows["attempted_runs"] = 1
    test_rows["ok_runs"] = (test_rows["status"] == "ok").astype(int)
    test_rows["error_runs"] = (test_rows["status"] != "ok").astype(int)

    status_coverage = (
        test_rows.groupby(
            ["experiment", "dataset", "model", "condition_name", "condition_value"],
            dropna=False,
        )
        .agg(
            attempted_runs=("attempted_runs", "sum"),
            ok_runs=("ok_runs", "sum"),
            error_runs=("error_runs", "sum"),
        )
        .reset_index()
    )
    status_coverage["ok_rate"] = np.where(
        status_coverage["attempted_runs"] > 0,
        status_coverage["ok_runs"] / status_coverage["attempted_runs"],
        np.nan,
    )
    status_coverage["error_rate"] = np.where(
        status_coverage["attempted_runs"] > 0,
        status_coverage["error_runs"] / status_coverage["attempted_runs"],
        np.nan,
    )
    status_coverage = status_coverage.sort_values(
        ["experiment", "dataset", "condition_name", "condition_value", "model"]
    )
    status_coverage.to_csv(
        SUMMARY_ROOT / "status_coverage_by_condition.csv",
        index=False,
    )

    status_by_model = (
        status_coverage.groupby("model", as_index=False)
        .agg(
            attempted_runs=("attempted_runs", "sum"),
            ok_runs=("ok_runs", "sum"),
            error_runs=("error_runs", "sum"),
        )
        .sort_values("attempted_runs", ascending=False)
    )
    status_by_model["ok_rate"] = np.where(
        status_by_model["attempted_runs"] > 0,
        status_by_model["ok_runs"] / status_by_model["attempted_runs"],
        np.nan,
    )
    status_by_model["error_rate"] = np.where(
        status_by_model["attempted_runs"] > 0,
        status_by_model["error_runs"] / status_by_model["attempted_runs"],
        np.nan,
    )
    status_by_model.to_csv(
        SUMMARY_ROOT / "status_coverage_by_model.csv",
        index=False,
    )

    test_ok = metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["status"] == "ok")
    ].copy()

    grouped = (
        test_ok.groupby(
            ["experiment", "dataset", "model", "condition_name", "condition_value"],
            dropna=False,
        )
        .agg(
            n_runs=("seed", "nunique"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            total_time_sec_mean=("total_time_sec", "mean"),
            total_time_sec_std=("total_time_sec", "std"),
        )
        .reset_index()
        .sort_values(
            by=["experiment", "dataset", "condition_name", "condition_value", "model"]
        )
    )
    grouped = grouped.merge(
        status_coverage,
        on=["experiment", "dataset", "model", "condition_name", "condition_value"],
        how="left",
    )

    grouped.to_csv(SUMMARY_ROOT / "test_mean_std_by_condition.csv", index=False)

    baseline_summary = grouped[
        (grouped["experiment"] == "baseline")
        & (grouped["condition_name"] == "clean")
    ].copy()
    baseline_summary.to_csv(
        SUMMARY_ROOT / "baseline_clean_mean_std.csv",
        index=False,
    )

    report_lines = [
        "# Full Comparison Summary",
        "",
        f"Total metric rows: {len(metrics_df)}",
        f"Total test rows attempted: {len(test_rows)}",
        f"Test rows used for summary: {len(test_ok)}",
        "",
        "## Baseline Clean (Mean Balanced Accuracy)",
        "",
    ]

    if baseline_summary.empty:
        report_lines.append("No baseline summary rows available.")
    else:
        for _, row in baseline_summary.iterrows():
            report_lines.append(
                "- "
                f"{row['dataset']} | {row['model']} | "
                f"bal_acc={row['balanced_accuracy_mean']:.6f} "
                f"+- {0.0 if pd.isna(row['balanced_accuracy_std']) else row['balanced_accuracy_std']:.6f}"
            )

    report_lines.extend(["", "## Test Reliability (Attempted vs Error)", ""])
    if status_by_model.empty:
        report_lines.append("No reliability rows available.")
    else:
        for _, row in status_by_model.iterrows():
            report_lines.append(
                "- "
                f"{row['model']}: attempted={int(row['attempted_runs'])}, "
                f"ok={int(row['ok_runs'])}, error={int(row['error_runs'])}, "
                f"ok_rate={row['ok_rate']:.3f}, error_rate={row['error_rate']:.3f}"
            )

    (LOGS_ROOT / "full_comparison_summary.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def run_full_comparison(
    config: ExperimentConfig,
    use_gpu_baselines: bool = False,
) -> None:
    """Run baseline plus robustness experiments for all datasets/seeds."""
    metric_rows: list[dict[str, Any]] = []

    for dataset in config.datasets:
        features, labels, _ = load_dataset(dataset)

        for seed in config.seeds:
            split_payload = _load_split(dataset, seed)
            split_file = f"data/splits/{dataset}_seed{seed}.json"

            train_idx = np.array(split_payload["indices"]["train"], dtype=int)
            val_idx = np.array(split_payload["indices"]["val"], dtype=int)
            test_idx = np.array(split_payload["indices"]["test"], dtype=int)

            x_train_raw = features.iloc[train_idx].copy()
            x_val_raw = features.iloc[val_idx].copy()
            x_test_raw = features.iloc[test_idx].copy()

            y_train = labels.iloc[train_idx].to_numpy(dtype=int)
            y_val = labels.iloc[val_idx].to_numpy(dtype=int)
            y_test = labels.iloc[test_idx].to_numpy(dtype=int)

            feature_types = get_feature_types(x_train_raw)
            n_numeric = len(feature_types["numeric"])

            preprocessor = build_shared_preprocessor(x_train_raw)
            x_train = np.asarray(preprocessor.fit_transform(x_train_raw))
            x_val = np.asarray(preprocessor.transform(x_val_raw))
            x_test = np.asarray(preprocessor.transform(x_test_raw))

            model_builders = _build_model_builders(
                seed,
                use_gpu_baselines=use_gpu_baselines,
            )
            baseline_results: list[dict[str, Any]] = []
            baseline_predictions: list[dict[str, Any]] = []
            fitted_models: dict[str, dict[str, Any]] = {}
            model_errors: dict[str, str] = {}
            tuned_best_params: dict[str, Any] | None = None

            for model_name, model_builder in model_builders.items():
                result, predictions, fitted_model = _fit_and_evaluate(
                    model_name=model_name,
                    model_builder=model_builder,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    y_test=y_test,
                )
                baseline_results.append(result)

                if fitted_model is not None:
                    fitted_models[model_name] = {
                        "model": fitted_model,
                        "fit_time_sec": result["timing"]["fit_time_sec"],
                    }
                    baseline_predictions.extend(
                        _build_prediction_rows(
                            dataset=dataset,
                            seed=seed,
                            model_name=model_name,
                            split_name="validation",
                            y_true=y_val,
                            y_pred=predictions["val_pred"],
                            y_score=predictions["val_score"],
                            experiment="baseline",
                            condition_name="clean",
                            condition_value="clean",
                        )
                    )
                    baseline_predictions.extend(
                        _build_prediction_rows(
                            dataset=dataset,
                            seed=seed,
                            model_name=model_name,
                            split_name="test",
                            y_true=y_test,
                            y_pred=predictions["test_pred"],
                            y_score=predictions["test_score"],
                            experiment="baseline",
                            condition_name="clean",
                            condition_value="clean",
                        )
                    )
                else:
                    model_errors[model_name] = str(result["error"])

            tuned_result, tuned_predictions, tuned_model = _fit_and_evaluate_hyperfast_tuned(
                seed=seed,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
            )
            baseline_results.append(tuned_result)
            if tuned_model is not None:
                fitted_models["hyperfast_tuned"] = {
                    "model": tuned_model,
                    "fit_time_sec": tuned_result["timing"]["fit_time_sec"],
                }
                selection = tuned_result.get("selection", {})
                if isinstance(selection, dict) and isinstance(
                    selection.get("best_params"),
                    dict,
                ):
                    tuned_best_params = dict(selection["best_params"])
                baseline_predictions.extend(
                    _build_prediction_rows(
                        dataset=dataset,
                        seed=seed,
                        model_name="hyperfast_tuned",
                        split_name="validation",
                        y_true=y_val,
                        y_pred=tuned_predictions["val_pred"],
                        y_score=tuned_predictions["val_score"],
                        experiment="baseline",
                        condition_name="clean",
                        condition_value="clean",
                    )
                )
                baseline_predictions.extend(
                    _build_prediction_rows(
                        dataset=dataset,
                        seed=seed,
                        model_name="hyperfast_tuned",
                        split_name="test",
                        y_true=y_test,
                        y_pred=tuned_predictions["test_pred"],
                        y_score=tuned_predictions["test_score"],
                        experiment="baseline",
                        condition_name="clean",
                        condition_value="clean",
                    )
                )
            else:
                model_errors["hyperfast_tuned"] = str(tuned_result["error"])

            all_model_names = list(model_builders.keys())
            if "hyperfast_tuned" not in all_model_names:
                all_model_names.append("hyperfast_tuned")

            baseline_metrics_path = _save_condition_artifacts(
                dataset=dataset,
                seed=seed,
                experiment="baseline",
                condition_name="clean",
                condition_value="clean",
                split_file=split_file,
                results=baseline_results,
                prediction_rows=baseline_predictions,
            )

            metric_rows.extend(
                _results_to_metric_rows(
                    dataset=dataset,
                    seed=seed,
                    experiment="baseline",
                    condition_name="clean",
                    condition_value="clean",
                    split_file=split_file,
                    artifact_path=str(baseline_metrics_path),
                    results=baseline_results,
                )
            )

            for sigma in config.noise_sigmas:
                sigma_text = f"{sigma:.2f}"
                x_val_noisy = _apply_gaussian_noise(
                    x_data=x_val,
                    n_numeric=n_numeric,
                    sigma=sigma,
                    seed=seed * 10_000 + int(sigma * 1_000) + 11,
                )
                x_test_noisy = _apply_gaussian_noise(
                    x_data=x_test,
                    n_numeric=n_numeric,
                    sigma=sigma,
                    seed=seed * 10_000 + int(sigma * 1_000) + 29,
                )

                noise_results: list[dict[str, Any]] = []
                noise_predictions: list[dict[str, Any]] = []

                for model_name in all_model_names:
                    if model_name in fitted_models:
                        result, predictions = _evaluate_pretrained(
                            model_name=model_name,
                            model=fitted_models[model_name]["model"],
                            fit_time_sec=float(fitted_models[model_name]["fit_time_sec"]),
                            x_val=x_val_noisy,
                            y_val=y_val,
                            x_test=x_test_noisy,
                            y_test=y_test,
                        )
                        noise_results.append(result)
                        noise_predictions.extend(
                            _build_prediction_rows(
                                dataset=dataset,
                                seed=seed,
                                model_name=model_name,
                                split_name="validation",
                                y_true=y_val,
                                y_pred=predictions["val_pred"],
                                y_score=predictions["val_score"],
                                experiment="noise",
                                condition_name="sigma",
                                condition_value=sigma_text,
                            )
                        )
                        noise_predictions.extend(
                            _build_prediction_rows(
                                dataset=dataset,
                                seed=seed,
                                model_name=model_name,
                                split_name="test",
                                y_true=y_test,
                                y_pred=predictions["test_pred"],
                                y_score=predictions["test_score"],
                                experiment="noise",
                                condition_name="sigma",
                                condition_value=sigma_text,
                            )
                        )
                    else:
                        noise_results.append(
                            {
                                "model": model_name,
                                "error": model_errors.get(model_name, "model unavailable"),
                                "timing": None,
                                "validation": None,
                                "test": None,
                            }
                        )

                noise_metrics_path = _save_condition_artifacts(
                    dataset=dataset,
                    seed=seed,
                    experiment="noise",
                    condition_name="sigma",
                    condition_value=sigma_text,
                    split_file=split_file,
                    results=noise_results,
                    prediction_rows=noise_predictions,
                )

                metric_rows.extend(
                    _results_to_metric_rows(
                        dataset=dataset,
                        seed=seed,
                        experiment="noise",
                        condition_name="sigma",
                        condition_value=sigma_text,
                        split_file=split_file,
                        artifact_path=str(noise_metrics_path),
                        results=noise_results,
                    )
                )

            for rate in config.missing_rates:
                rate_text = f"{rate:.2f}"
                x_val_missing_raw = _apply_mcar_missing(
                    frame=x_val_raw,
                    rate=rate,
                    seed=seed * 20_000 + int(rate * 1_000) + 7,
                )
                x_test_missing_raw = _apply_mcar_missing(
                    frame=x_test_raw,
                    rate=rate,
                    seed=seed * 20_000 + int(rate * 1_000) + 13,
                )

                x_val_missing = np.asarray(preprocessor.transform(x_val_missing_raw))
                x_test_missing = np.asarray(preprocessor.transform(x_test_missing_raw))

                missing_results: list[dict[str, Any]] = []
                missing_predictions: list[dict[str, Any]] = []

                for model_name in all_model_names:
                    if model_name in fitted_models:
                        result, predictions = _evaluate_pretrained(
                            model_name=model_name,
                            model=fitted_models[model_name]["model"],
                            fit_time_sec=float(fitted_models[model_name]["fit_time_sec"]),
                            x_val=x_val_missing,
                            y_val=y_val,
                            x_test=x_test_missing,
                            y_test=y_test,
                        )
                        missing_results.append(result)
                        missing_predictions.extend(
                            _build_prediction_rows(
                                dataset=dataset,
                                seed=seed,
                                model_name=model_name,
                                split_name="validation",
                                y_true=y_val,
                                y_pred=predictions["val_pred"],
                                y_score=predictions["val_score"],
                                experiment="missingness",
                                condition_name="rate",
                                condition_value=rate_text,
                            )
                        )
                        missing_predictions.extend(
                            _build_prediction_rows(
                                dataset=dataset,
                                seed=seed,
                                model_name=model_name,
                                split_name="test",
                                y_true=y_test,
                                y_pred=predictions["test_pred"],
                                y_score=predictions["test_score"],
                                experiment="missingness",
                                condition_name="rate",
                                condition_value=rate_text,
                            )
                        )
                    else:
                        missing_results.append(
                            {
                                "model": model_name,
                                "error": model_errors.get(model_name, "model unavailable"),
                                "timing": None,
                                "validation": None,
                                "test": None,
                            }
                        )

                missing_metrics_path = _save_condition_artifacts(
                    dataset=dataset,
                    seed=seed,
                    experiment="missingness",
                    condition_name="rate",
                    condition_value=rate_text,
                    split_file=split_file,
                    results=missing_results,
                    prediction_rows=missing_predictions,
                )

                metric_rows.extend(
                    _results_to_metric_rows(
                        dataset=dataset,
                        seed=seed,
                        experiment="missingness",
                        condition_name="rate",
                        condition_value=rate_text,
                        split_file=split_file,
                        artifact_path=str(missing_metrics_path),
                        results=missing_results,
                    )
                )

            for fraction in config.reduced_fractions:
                fraction_text = f"{fraction:.2f}"
                subset_indices = _sample_train_subset_indices(
                    y_train=y_train,
                    fraction=fraction,
                    seed=seed * 30_000 + int(fraction * 1_000) + 17,
                )

                x_train_sub_raw = x_train_raw.iloc[subset_indices].copy()
                y_train_sub = y_train[subset_indices]

                preprocessor_sub = build_shared_preprocessor(x_train_sub_raw)
                x_train_sub = np.asarray(preprocessor_sub.fit_transform(x_train_sub_raw))
                x_val_sub = np.asarray(preprocessor_sub.transform(x_val_raw))
                x_test_sub = np.asarray(preprocessor_sub.transform(x_test_raw))

                reduced_results: list[dict[str, Any]] = []
                reduced_predictions: list[dict[str, Any]] = []

                for model_name, model_builder in model_builders.items():
                    result, predictions, _ = _fit_and_evaluate(
                        model_name=model_name,
                        model_builder=model_builder,
                        x_train=x_train_sub,
                        y_train=y_train_sub,
                        x_val=x_val_sub,
                        y_val=y_val,
                        x_test=x_test_sub,
                        y_test=y_test,
                    )
                    reduced_results.append(result)

                    if "error" not in result:
                        reduced_predictions.extend(
                            _build_prediction_rows(
                                dataset=dataset,
                                seed=seed,
                                model_name=model_name,
                                split_name="validation",
                                y_true=y_val,
                                y_pred=predictions["val_pred"],
                                y_score=predictions["val_score"],
                                experiment="reduced_data",
                                condition_name="fraction",
                                condition_value=fraction_text,
                            )
                        )
                        reduced_predictions.extend(
                            _build_prediction_rows(
                                dataset=dataset,
                                seed=seed,
                                model_name=model_name,
                                split_name="test",
                                y_true=y_test,
                                y_pred=predictions["test_pred"],
                                y_score=predictions["test_score"],
                                experiment="reduced_data",
                                condition_name="fraction",
                                condition_value=fraction_text,
                            )
                        )

                if tuned_best_params is not None:
                    tuned_result, tuned_predictions, _ = _fit_and_evaluate(
                        model_name="hyperfast_tuned",
                        model_builder=lambda params=tuned_best_params: build_hyperfast(
                            seed=seed,
                            n_ensemble=int(params.get("n_ensemble", 1)),
                            optimization=params.get("optimization", None),
                            stratify_sampling=bool(
                                params.get("stratify_sampling", False)
                            ),
                            feature_bagging=bool(params.get("feature_bagging", False)),
                        ),
                        x_train=x_train_sub,
                        y_train=y_train_sub,
                        x_val=x_val_sub,
                        y_val=y_val,
                        x_test=x_test_sub,
                        y_test=y_test,
                    )
                else:
                    tuned_result, tuned_predictions, _ = _fit_and_evaluate_hyperfast_tuned(
                        seed=seed,
                        x_train=x_train_sub,
                        y_train=y_train_sub,
                        x_val=x_val_sub,
                        y_val=y_val,
                        x_test=x_test_sub,
                        y_test=y_test,
                    )
                reduced_results.append(tuned_result)
                if "error" not in tuned_result:
                    reduced_predictions.extend(
                        _build_prediction_rows(
                            dataset=dataset,
                            seed=seed,
                            model_name="hyperfast_tuned",
                            split_name="validation",
                            y_true=y_val,
                            y_pred=tuned_predictions["val_pred"],
                            y_score=tuned_predictions["val_score"],
                            experiment="reduced_data",
                            condition_name="fraction",
                            condition_value=fraction_text,
                        )
                    )
                    reduced_predictions.extend(
                        _build_prediction_rows(
                            dataset=dataset,
                            seed=seed,
                            model_name="hyperfast_tuned",
                            split_name="test",
                            y_true=y_test,
                            y_pred=tuned_predictions["test_pred"],
                            y_score=tuned_predictions["test_score"],
                            experiment="reduced_data",
                            condition_name="fraction",
                            condition_value=fraction_text,
                        )
                    )

                reduced_metrics_path = _save_condition_artifacts(
                    dataset=dataset,
                    seed=seed,
                    experiment="reduced_data",
                    condition_name="fraction",
                    condition_value=fraction_text,
                    split_file=split_file,
                    results=reduced_results,
                    prediction_rows=reduced_predictions,
                )

                metric_rows.extend(
                    _results_to_metric_rows(
                        dataset=dataset,
                        seed=seed,
                        experiment="reduced_data",
                        condition_name="fraction",
                        condition_value=fraction_text,
                        split_file=split_file,
                        artifact_path=str(reduced_metrics_path),
                        results=reduced_results,
                    )
                )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)
    _summarize_metrics(metrics_df)


def _parse_list_argument(raw: str | None, cast_type: Any) -> list[Any] | None:
    """Parse comma-separated CLI argument into list or return None."""
    if raw is None or raw.strip() == "":
        return None
    return [cast_type(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated integer seeds.",
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

    config = _load_config()

    override_datasets = _parse_list_argument(args.datasets, str)
    override_seeds = _parse_list_argument(args.seeds, int)

    if override_datasets is not None:
        config = ExperimentConfig(
            datasets=override_datasets,
            seeds=config.seeds,
            noise_sigmas=config.noise_sigmas,
            missing_rates=config.missing_rates,
            reduced_fractions=config.reduced_fractions,
        )

    if override_seeds is not None:
        config = ExperimentConfig(
            datasets=config.datasets,
            seeds=override_seeds,
            noise_sigmas=config.noise_sigmas,
            missing_rates=config.missing_rates,
            reduced_fractions=config.reduced_fractions,
        )

    use_gpu_baselines = (
        args.use_gpu_baselines
        or os.getenv("HF_USE_GPU_BASELINES", "0") == "1"
    )
    if use_gpu_baselines:
        print("Using GPU baseline preference (RAPIDS/cuML if available).")

    run_full_comparison(config, use_gpu_baselines=use_gpu_baselines)
    print(f"Saved long-form metrics: {RESULTS_ROOT / 'metrics.csv'}")
    print(
        "Saved summary tables: "
        f"{SUMMARY_ROOT / 'test_mean_std_by_condition.csv'}"
    )


if __name__ == "__main__":
    main()
