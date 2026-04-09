"""Generate analysis-layer artifacts for comparison and paper writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_CONFIG_PATH = PROJECT_ROOT / "configs" / "analysis_artifacts.json"
SUMMARY_PATH = PROJECT_ROOT / "results" / "summary_tables" / "test_mean_std_by_condition.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.csv"
SUMMARY_OUT = PROJECT_ROOT / "results" / "summary_tables"
PLOTS_OUT = PROJECT_ROOT / "plots"
ERROR_OUT = PROJECT_ROOT / "error_analysis"
REPORT_OUT = PROJECT_ROOT / "report"


def _load_analysis_config(path: Path) -> dict[str, Any]:
    """Load and validate analysis artifact configuration."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "figure_dpi",
        "export_formats",
        "dataset_order",
        "dataset_labels",
        "model_order",
        "model_labels",
        "model_short_labels",
        "model_markers",
        "model_colors",
        "worst_case_rules",
        "line_plots",
        "runtime_plot_stem",
        "drop_plot_stem",
        "robust_claim_experiments",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        missing_keys = ", ".join(missing)
        raise KeyError(f"Missing keys in analysis config: {missing_keys}")
    return payload


ANALYSIS_CONFIG = _load_analysis_config(ANALYSIS_CONFIG_PATH)

FIGURE_DPI = int(ANALYSIS_CONFIG["figure_dpi"])
EXPORT_FORMATS = tuple(str(fmt) for fmt in ANALYSIS_CONFIG["export_formats"])

DATASET_ORDER = list(ANALYSIS_CONFIG["dataset_order"])
DATASET_LABELS = dict(ANALYSIS_CONFIG["dataset_labels"])

MODEL_ORDER = list(ANALYSIS_CONFIG["model_order"])
MODEL_LABELS = dict(ANALYSIS_CONFIG["model_labels"])
MODEL_SHORT_LABELS = dict(ANALYSIS_CONFIG["model_short_labels"])
MODEL_MARKERS = dict(ANALYSIS_CONFIG["model_markers"])
MODEL_COLORS = dict(ANALYSIS_CONFIG["model_colors"])

WORST_CASE_RULES: list[dict[str, Any]] = list(ANALYSIS_CONFIG["worst_case_rules"])
LINE_PLOT_SPECS: list[dict[str, Any]] = list(ANALYSIS_CONFIG["line_plots"])
RUNTIME_PLOT_STEM = str(ANALYSIS_CONFIG["runtime_plot_stem"])
DROP_PLOT_STEM = str(ANALYSIS_CONFIG["drop_plot_stem"])
ROBUST_CLAIM_EXPERIMENTS = {
    str(name) for name in ANALYSIS_CONFIG["robust_claim_experiments"]
}

CONDITION_GROUP_COLS = [
    "dataset",
    "experiment",
    "condition_name",
    "condition_value",
]


def _build_worst_case_conditions(drop_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Infer worst-case conditions from available experiment values."""
    selected: list[dict[str, Any]] = []

    for order, rule in enumerate(WORST_CASE_RULES):
        scoped = drop_df[
            (drop_df["experiment"] == rule["experiment"])
            & (drop_df["condition_name"] == rule["condition_name"])
        ].copy()
        if scoped.empty:
            continue

        condition_values = pd.to_numeric(scoped["condition_value"], errors="coerce")
        condition_values = condition_values.dropna()
        if condition_values.empty:
            continue

        if rule["selector"] == "max":
            selected_value = float(condition_values.max())
        else:
            selected_value = float(condition_values.min())

        label = rule["label_template"].format(
            value=selected_value,
            percent=100.0 * selected_value,
        )

        selected.append(
            {
                "experiment": rule["experiment"],
                "condition_name": rule["condition_name"],
                "condition_value": format(selected_value, ".12g"),
                "label": label,
                "condition_order": order,
            }
        )

    return selected


def _apply_research_plot_style() -> None:
    """Apply a clean, publication-ready visual style."""
    plt.style.use("default")
    sns.set_palette([MODEL_COLORS[m] for m in MODEL_ORDER])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "grid.color": "#CFCFCF",
            "lines.linewidth": 2.2,
            "lines.markersize": 6.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": FIGURE_DPI,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 6,
        }
    )


def _save_figure(fig: Figure, output_stem: Path) -> None:
    """Save each figure as both high-resolution PNG and PDF."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in EXPORT_FORMATS:
        fig.savefig(
            output_stem.with_suffix(f".{fmt}"),
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
        )


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load summary and detailed metric tables."""
    summary_df = pd.read_csv(SUMMARY_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)

    summary_df["condition_num"] = pd.to_numeric(
        summary_df["condition_value"],
        errors="coerce",
    )
    metrics_df["condition_num"] = pd.to_numeric(
        metrics_df["condition_value"],
        errors="coerce",
    )
    return summary_df, metrics_df


def _clean_lookup(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract clean baseline score lookup for drop analysis."""
    clean_df = summary_df[
        (summary_df["experiment"] == "baseline")
        & (summary_df["condition_name"] == "clean")
        & (summary_df["condition_value"].astype(str) == "clean")
    ][["dataset", "model", "balanced_accuracy_mean"]].copy()
    clean_df = clean_df.rename(columns={"balanced_accuracy_mean": "clean_score"})
    return clean_df


def generate_drop_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create clean-vs-corrupted drop table for all conditions."""
    clean_df = _clean_lookup(summary_df)
    non_clean = summary_df[summary_df["condition_name"] != "clean"].copy()

    merged = non_clean.merge(clean_df, on=["dataset", "model"], how="left")
    merged["drop_abs"] = merged["clean_score"] - merged["balanced_accuracy_mean"]
    merged["drop_pct"] = np.where(
        merged["clean_score"] > 0,
        100.0 * merged["drop_abs"] / merged["clean_score"],
        np.nan,
    )

    merged = merged[
        [
            "dataset",
            "experiment",
            "condition_name",
            "condition_value",
            "model",
            "clean_score",
            "balanced_accuracy_mean",
            "drop_abs",
            "drop_pct",
            "n_runs",
            "total_time_sec_mean",
        ]
    ].sort_values(
        ["dataset", "experiment", "condition_name", "condition_value", "model"]
    )

    out_path = SUMMARY_OUT / "performance_drop_vs_clean.csv"
    merged.to_csv(out_path, index=False)
    return merged


def generate_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create condition-wise model rankings."""
    ranked = summary_df.copy()
    ranked["rank"] = ranked.groupby(
        ["dataset", "experiment", "condition_name", "condition_value"]
    )["balanced_accuracy_mean"].rank(method="min", ascending=False)

    best = ranked.groupby(
        ["dataset", "experiment", "condition_name", "condition_value"]
    )["balanced_accuracy_mean"].transform("max")
    ranked["gap_to_best"] = best - ranked["balanced_accuracy_mean"]

    ranked = ranked[
        [
            "dataset",
            "experiment",
            "condition_name",
            "condition_value",
            "model",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "rank",
            "gap_to_best",
            "n_runs",
        ]
    ].sort_values(
        [
            "dataset",
            "experiment",
            "condition_name",
            "condition_value",
            "rank",
            "model",
        ]
    )

    out_path = SUMMARY_OUT / "condition_wise_rankings.csv"
    ranked.to_csv(out_path, index=False)
    return ranked


def generate_status_coverage(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create reliability tables with attempted/ok/error counts and rates."""
    test_rows = metrics_df[metrics_df["split"] == "test"].copy()
    if test_rows.empty:
        empty_cols = [
            "dataset",
            "experiment",
            "condition_name",
            "condition_value",
            "model",
            "attempted_runs",
            "ok_runs",
            "error_runs",
            "ok_rate",
            "error_rate",
        ]
        empty_df = pd.DataFrame(columns=empty_cols)
        empty_df.to_csv(SUMMARY_OUT / "status_coverage_by_condition.csv", index=False)
        empty_df.to_csv(SUMMARY_OUT / "status_coverage_by_model.csv", index=False)
        return empty_df

    test_rows["attempted_runs"] = 1
    test_rows["ok_runs"] = (test_rows["status"] == "ok").astype(int)
    test_rows["error_runs"] = (test_rows["status"] != "ok").astype(int)

    coverage = (
        test_rows.groupby(
            ["dataset", "experiment", "condition_name", "condition_value", "model"],
            as_index=False,
        )
        .agg(
            attempted_runs=("attempted_runs", "sum"),
            ok_runs=("ok_runs", "sum"),
            error_runs=("error_runs", "sum"),
        )
        .sort_values(["dataset", "experiment", "condition_name", "condition_value", "model"])
    )
    coverage["ok_rate"] = np.where(
        coverage["attempted_runs"] > 0,
        coverage["ok_runs"] / coverage["attempted_runs"],
        np.nan,
    )
    coverage["error_rate"] = np.where(
        coverage["attempted_runs"] > 0,
        coverage["error_runs"] / coverage["attempted_runs"],
        np.nan,
    )

    coverage.to_csv(SUMMARY_OUT / "status_coverage_by_condition.csv", index=False)

    by_model = (
        coverage.groupby("model", as_index=False)
        .agg(
            attempted_runs=("attempted_runs", "sum"),
            ok_runs=("ok_runs", "sum"),
            error_runs=("error_runs", "sum"),
        )
        .sort_values("attempted_runs", ascending=False)
    )
    by_model["ok_rate"] = np.where(
        by_model["attempted_runs"] > 0,
        by_model["ok_runs"] / by_model["attempted_runs"],
        np.nan,
    )
    by_model["error_rate"] = np.where(
        by_model["attempted_runs"] > 0,
        by_model["error_runs"] / by_model["attempted_runs"],
        np.nan,
    )
    by_model.to_csv(SUMMARY_OUT / "status_coverage_by_model.csv", index=False)

    return coverage


def generate_runtime_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create runtime comparison table from detailed test metrics."""
    test_ok = metrics_df[(metrics_df["split"] == "test") & (metrics_df["status"] == "ok")]

    runtime = (
        test_ok.groupby(
            ["dataset", "model", "experiment", "condition_name", "condition_value"],
            as_index=False,
        )
        .agg(
            fit_time_sec_mean=("fit_time_sec", "mean"),
            fit_time_sec_std=("fit_time_sec", "std"),
            predict_time_sec_mean=("predict_time_sec", "mean"),
            predict_time_sec_std=("predict_time_sec", "std"),
            total_time_sec_mean=("total_time_sec", "mean"),
            total_time_sec_std=("total_time_sec", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            n_runs=("seed", "nunique"),
        )
        .sort_values(["dataset", "experiment", "condition_name", "condition_value", "model"])
    )

    runtime.to_csv(SUMMARY_OUT / "runtime_by_condition.csv", index=False)

    clean_runtime = runtime[
        (runtime["experiment"] == "baseline")
        & (runtime["condition_name"] == "clean")
        & (runtime["condition_value"].astype(str) == "clean")
    ].copy()
    clean_runtime.to_csv(SUMMARY_OUT / "runtime_clean_comparison.csv", index=False)

    return runtime


def generate_efficiency_tradeoff(runtime_df: pd.DataFrame) -> pd.DataFrame:
    """Create an accuracy-vs-inference-cost tradeoff score table."""
    tradeoff = runtime_df.copy()
    group_cols = ["dataset", "experiment", "condition_name", "condition_value"]

    tradeoff["best_accuracy_in_group"] = tradeoff.groupby(group_cols)[
        "balanced_accuracy_mean"
    ].transform("max")
    tradeoff["fastest_predict_time_sec_in_group"] = tradeoff.groupby(group_cols)[
        "predict_time_sec_mean"
    ].transform("min")

    tradeoff["accuracy_ratio"] = np.where(
        tradeoff["best_accuracy_in_group"] > 0,
        tradeoff["balanced_accuracy_mean"] / tradeoff["best_accuracy_in_group"],
        np.nan,
    )
    tradeoff["inference_efficiency_ratio"] = np.where(
        tradeoff["predict_time_sec_mean"] > 0,
        tradeoff["fastest_predict_time_sec_in_group"]
        / tradeoff["predict_time_sec_mean"],
        np.nan,
    )

    raw_score = tradeoff["accuracy_ratio"] * tradeoff["inference_efficiency_ratio"]
    raw_score = raw_score.clip(lower=0.0)
    tradeoff["accuracy_inference_tradeoff_score"] = np.sqrt(raw_score)

    tradeoff = tradeoff.sort_values(
        [
            "dataset",
            "experiment",
            "condition_name",
            "condition_value",
            "accuracy_inference_tradeoff_score",
        ],
        ascending=[True, True, True, True, False],
    )
    tradeoff.to_csv(SUMMARY_OUT / "accuracy_inference_tradeoff.csv", index=False)

    by_model = (
        tradeoff.groupby("model", as_index=False)
        .agg(
            mean_tradeoff_score=("accuracy_inference_tradeoff_score", "mean"),
            median_tradeoff_score=("accuracy_inference_tradeoff_score", "median"),
            q25_tradeoff_score=(
                "accuracy_inference_tradeoff_score",
                lambda s: float(np.nanpercentile(s, 25)),
            ),
            q75_tradeoff_score=(
                "accuracy_inference_tradeoff_score",
                lambda s: float(np.nanpercentile(s, 75)),
            ),
        )
        .sort_values("mean_tradeoff_score", ascending=False)
    )
    by_model.to_csv(SUMMARY_OUT / "accuracy_inference_tradeoff_by_model.csv", index=False)

    return tradeoff


def _interpret_paired_result(
    ci_low: float,
    ci_high: float,
    p_value: float,
) -> str:
    """Map paired comparison statistics to a plain-language interpretation."""
    if np.isnan(p_value):
        return "insufficient_data"
    if p_value < 0.05 and ci_high < 0:
        return "significantly_worse_than_best"
    if p_value < 0.05 and ci_low > 0:
        return "significantly_better_than_best"
    return "not_significantly_different"


def _holm_adjusted_pvalues(raw_p_values: list[float]) -> list[float]:
    """Apply Holm correction while preserving original row order."""
    if not raw_p_values:
        return []

    indexed = list(enumerate(raw_p_values))
    indexed.sort(key=lambda item: item[1])

    adjusted = [np.nan] * len(raw_p_values)
    running_max = 0.0
    m = len(raw_p_values)

    for rank, (original_idx, p_val) in enumerate(indexed):
        scaled = min(1.0, (m - rank) * float(p_val))
        running_max = max(running_max, scaled)
        adjusted[original_idx] = min(1.0, running_max)

    return adjusted


def _cohen_d_paired(diffs: np.ndarray) -> float:
    """Compute paired-sample Cohen's d from seed-wise differences."""
    std_diff = float(np.std(diffs, ddof=1))
    mean_diff = float(np.mean(diffs))
    if np.isclose(std_diff, 0.0):
        if np.isclose(mean_diff, 0.0):
            return 0.0
        return float(np.sign(mean_diff) * np.inf)
    return mean_diff / std_diff


def _effect_size_label(cohen_d: float) -> str:
    """Map |d| to conventional effect-size buckets."""
    if not np.isfinite(cohen_d):
        return "large"
    abs_d = abs(float(cohen_d))
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def _resolve_p_value_with_fallback(
    diffs: np.ndarray,
    p_value_wilcoxon: float,
    p_value_ttest: float,
) -> tuple[float, str]:
    """Resolve paired-test p-value with robust fallback for degenerate cases."""
    if np.isfinite(p_value_wilcoxon):
        return float(p_value_wilcoxon), "wilcoxon"
    if np.isfinite(p_value_ttest):
        return float(p_value_ttest), "paired_ttest"

    finite_diffs = diffs[np.isfinite(diffs)]
    if finite_diffs.size == 0:
        return np.nan, "insufficient_data"

    # If all paired differences are numerically zero, treat as no detectable gap.
    non_zero_diffs = finite_diffs[~np.isclose(finite_diffs, 0.0)]
    if non_zero_diffs.size == 0:
        return 1.0, "sign_test_fallback"

    positive_count = int((non_zero_diffs > 0).sum())
    trial_count = int(non_zero_diffs.size)
    try:
        sign_test = stats.binomtest(positive_count, n=trial_count, p=0.5)
        return float(sign_test.pvalue), "sign_test_fallback"
    except Exception:  # pragma: no cover - defensive numeric guard
        return np.nan, "insufficient_data"


def _safe_finite_float(value: Any) -> float | None:
    """Convert a value to a finite float, or return None when not possible."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def generate_statistical_confidence(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute paired significance, CI, and effect size vs condition winner."""
    test_ok = metrics_df[(metrics_df["split"] == "test") & (metrics_df["status"] == "ok")]

    rows: list[dict[str, Any]] = []
    for group_key, group in test_ok.groupby(CONDITION_GROUP_COLS, sort=False):
        model_means = (
            group.groupby("model", as_index=False)
            .agg(balanced_accuracy=("balanced_accuracy", "mean"))
            .sort_values("balanced_accuracy", ascending=False)
            .reset_index(drop=True)
        )
        if model_means.empty:
            continue

        best_model = str(model_means.iloc[0]["model"])
        best_mean = float(model_means.iloc[0]["balanced_accuracy"])
        pivot = group.pivot_table(
            index="seed",
            columns="model",
            values="balanced_accuracy",
            aggfunc="mean",
        )

        holm_target_indices: list[int] = []
        holm_raw_p_values: list[float] = []

        for _, model_row in model_means.iterrows():
            model = str(model_row["model"])
            model_mean = float(model_row["balanced_accuracy"])

            if model == best_model:
                rows.append(
                    {
                        "dataset": group_key[0],
                        "experiment": group_key[1],
                        "condition_name": group_key[2],
                        "condition_value": group_key[3],
                        "best_model": best_model,
                        "best_mean": best_mean,
                        "model": model,
                        "model_mean": model_mean,
                        "mean_diff_vs_best": 0.0,
                        "cohen_d_vs_best": 0.0,
                        "effect_size": "reference_best",
                        "ci95_low": 0.0,
                        "ci95_high": 0.0,
                        "p_value_ttest": np.nan,
                        "p_value_wilcoxon": np.nan,
                        "p_value_raw": np.nan,
                        "p_value_holm": np.nan,
                        "p_value": np.nan,
                        "test_basis": "reference_best",
                        "n_common_seeds": int(pivot[best_model].dropna().shape[0]),
                        "interpretation": "reference_best",
                    }
                )
                continue

            if best_model not in pivot.columns or model not in pivot.columns:
                rows.append(
                    {
                        "dataset": group_key[0],
                        "experiment": group_key[1],
                        "condition_name": group_key[2],
                        "condition_value": group_key[3],
                        "best_model": best_model,
                        "best_mean": best_mean,
                        "model": model,
                        "model_mean": model_mean,
                        "mean_diff_vs_best": np.nan,
                        "cohen_d_vs_best": np.nan,
                        "effect_size": "insufficient_data",
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "p_value_ttest": np.nan,
                        "p_value_wilcoxon": np.nan,
                        "p_value_raw": np.nan,
                        "p_value_holm": np.nan,
                        "p_value": np.nan,
                        "test_basis": "insufficient_data",
                        "n_common_seeds": 0,
                        "interpretation": "insufficient_data",
                    }
                )
                continue

            paired = pivot[[best_model, model]].dropna()
            n_common = int(len(paired))

            if n_common < 2:
                rows.append(
                    {
                        "dataset": group_key[0],
                        "experiment": group_key[1],
                        "condition_name": group_key[2],
                        "condition_value": group_key[3],
                        "best_model": best_model,
                        "best_mean": best_mean,
                        "model": model,
                        "model_mean": model_mean,
                        "mean_diff_vs_best": np.nan,
                        "cohen_d_vs_best": np.nan,
                        "effect_size": "insufficient_data",
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "p_value_ttest": np.nan,
                        "p_value_wilcoxon": np.nan,
                        "p_value_raw": np.nan,
                        "p_value_holm": np.nan,
                        "p_value": np.nan,
                        "test_basis": "insufficient_data",
                        "n_common_seeds": n_common,
                        "interpretation": "insufficient_data",
                    }
                )
                continue

            diffs = paired[model].to_numpy(dtype=float) - paired[best_model].to_numpy(dtype=float)
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1))
            se_diff = std_diff / np.sqrt(n_common)
            t_crit = float(stats.t.ppf(0.975, n_common - 1))
            ci_low = mean_diff - t_crit * se_diff
            ci_high = mean_diff + t_crit * se_diff
            cohen_d = _cohen_d_paired(diffs)
            effect_size = _effect_size_label(cohen_d)
            _, p_val = stats.ttest_rel(
                paired[model].to_numpy(dtype=float),
                paired[best_model].to_numpy(dtype=float),
            )
            p_value_ttest = float(p_val) if np.isfinite(p_val) else np.nan

            p_value_wilcoxon = np.nan
            try:
                wilcoxon_result = stats.wilcoxon(diffs)
                p_wilcoxon = getattr(wilcoxon_result, "pvalue", None)
                if p_wilcoxon is None and isinstance(wilcoxon_result, tuple):
                    if len(wilcoxon_result) >= 2:
                        p_wilcoxon = wilcoxon_result[1]
                p_wilcoxon_float = _safe_finite_float(p_wilcoxon)
                if p_wilcoxon_float is not None:
                    p_value_wilcoxon = p_wilcoxon_float
            except Exception:  # pragma: no cover - defensive numeric guard
                p_value_wilcoxon = np.nan

            p_value_raw = (
                p_value_wilcoxon
                if np.isfinite(p_value_wilcoxon)
                else p_value_ttest
            )
            test_basis = (
                "wilcoxon"
                if np.isfinite(p_value_wilcoxon)
                else "paired_ttest"
            )

            p_value_raw, test_basis = _resolve_p_value_with_fallback(
                diffs=diffs,
                p_value_wilcoxon=p_value_wilcoxon,
                p_value_ttest=p_value_ttest,
            )

            row_idx = len(rows)
            if np.isfinite(p_value_raw):
                holm_target_indices.append(row_idx)
                holm_raw_p_values.append(float(p_value_raw))

            rows.append(
                {
                    "dataset": group_key[0],
                    "experiment": group_key[1],
                    "condition_name": group_key[2],
                    "condition_value": group_key[3],
                    "best_model": best_model,
                    "best_mean": best_mean,
                    "model": model,
                    "model_mean": model_mean,
                    "mean_diff_vs_best": mean_diff,
                    "cohen_d_vs_best": cohen_d,
                    "effect_size": effect_size,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_value_ttest": p_value_ttest,
                    "p_value_wilcoxon": p_value_wilcoxon,
                    "p_value_raw": p_value_raw,
                    "p_value_holm": np.nan,
                    "p_value": np.nan,
                    "test_basis": test_basis,
                    "n_common_seeds": n_common,
                    "interpretation": (
                        "pending" if np.isfinite(p_value_raw) else "insufficient_data"
                    ),
                }
            )

        holm_adjusted = _holm_adjusted_pvalues(holm_raw_p_values)
        for idx, adjusted_p in zip(
            holm_target_indices,
            holm_adjusted,
            strict=True,
        ):
            rows[idx]["p_value_holm"] = float(adjusted_p)

        for idx in holm_target_indices:
            row = rows[idx]
            p_for_interpretation = (
                float(row["p_value_holm"])
                if np.isfinite(row["p_value_holm"])
                else float(row["p_value_raw"])
            )
            rows[idx]["p_value"] = p_for_interpretation
            rows[idx]["test_basis"] = f"{row['test_basis']}_holm"
            rows[idx]["interpretation"] = _interpret_paired_result(
                float(row["ci95_low"]),
                float(row["ci95_high"]),
                p_for_interpretation,
            )

        # Safety net: unresolved pending rows are treated as insufficient_data.
        for row in rows:
            if row.get("interpretation") == "pending":
                if np.isfinite(float(row.get("p_value_raw", np.nan))):
                    continue
                row["interpretation"] = "insufficient_data"
                row["test_basis"] = "insufficient_data"

    if not rows:
        significance = pd.DataFrame(
            columns=[
                "dataset",
                "experiment",
                "condition_name",
                "condition_value",
                "best_model",
                "best_mean",
                "model",
                "model_mean",
                "mean_diff_vs_best",
                "cohen_d_vs_best",
                "effect_size",
                "ci95_low",
                "ci95_high",
                "p_value_ttest",
                "p_value_wilcoxon",
                "p_value_raw",
                "p_value_holm",
                "p_value",
                "test_basis",
                "n_common_seeds",
                "interpretation",
            ]
        )
    else:
        significance = pd.DataFrame(rows).sort_values(
            [
                "dataset",
                "experiment",
                "condition_name",
                "condition_value",
                "model_mean",
            ],
            ascending=[True, True, True, True, False],
        )
    significance.to_csv(SUMMARY_OUT / "statistical_confidence_vs_best.csv", index=False)

    compare_only = significance[
        (significance["interpretation"] != "reference_best")
        & (significance["interpretation"] != "insufficient_data")
    ].copy()
    if compare_only.empty:
        summary_by_model = pd.DataFrame(
            columns=[
                "model",
                "n_conditions",
                "significantly_worse_count",
                "not_significantly_different_count",
                "significantly_better_count",
            ]
        )
    else:
        compare_only["sig_worse"] = (
            compare_only["interpretation"] == "significantly_worse_than_best"
        ).astype(int)
        compare_only["not_sig_diff"] = (
            compare_only["interpretation"] == "not_significantly_different"
        ).astype(int)
        compare_only["sig_better"] = (
            compare_only["interpretation"] == "significantly_better_than_best"
        ).astype(int)

        summary_by_model = (
            compare_only.groupby("model", as_index=False)
            .agg(
                n_conditions=("model", "count"),
                significantly_worse_count=("sig_worse", "sum"),
                not_significantly_different_count=("not_sig_diff", "sum"),
                significantly_better_count=("sig_better", "sum"),
                mean_abs_cohen_d=("cohen_d_vs_best", lambda s: float(np.nanmean(np.abs(s)))),
                large_effect_count=("effect_size", lambda s: int((s == "large").sum())),
            )
            .sort_values("n_conditions", ascending=False)
        )
    summary_by_model.to_csv(SUMMARY_OUT / "significance_summary_by_model.csv", index=False)

    report_lines = [
        "# Statistical Confidence Interpretation",
        "",
        (
            "Wilcoxon signed-rank p-values (fallback paired t-test) compare each "
            "model against the best-mean model per condition."
        ),
        "Holm correction is applied within each condition across pairwise comparisons.",
        "95% CI and paired-sample Cohen's d are computed on seed-wise differences.",
        "",
    ]

    if summary_by_model.empty:
        report_lines.append("No pairwise comparisons with sufficient data are available.")
    else:
        for _, row in summary_by_model.iterrows():
            report_lines.append(
                "- "
                f"{row['model']}: "
                f"n={int(row['n_conditions'])}, "
                f"significantly_worse={int(row['significantly_worse_count'])}, "
                f"not_significant={int(row['not_significantly_different_count'])}, "
                f"significantly_better={int(row['significantly_better_count'])}, "
                f"mean_abs_cohen_d={float(row['mean_abs_cohen_d']):.3f}, "
                f"large_effects={int(row['large_effect_count'])}."
            )

    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "statistical_interpretation.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    return significance


def generate_condition_conclusions(
    ranking_df: pd.DataFrame,
    significance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create winner/runner-up/worst conclusions for each condition."""
    rows: list[dict[str, Any]] = []

    for group_key, group in ranking_df.groupby(CONDITION_GROUP_COLS, sort=False):
        ordered = group.sort_values(["rank", "model"]).reset_index(drop=True)
        if ordered.empty:
            continue

        winner = ordered.iloc[0]
        worst = ordered.iloc[-1]
        runner_up = ordered.iloc[1] if len(ordered) > 1 else None

        runner_up_model = None
        runner_up_score = np.nan
        runner_up_gap = np.nan
        runner_up_vs_winner = "n/a"
        runner_up_p_value = np.nan

        if runner_up is not None:
            runner_up_model = runner_up["model"]
            runner_up_score = float(runner_up["balanced_accuracy_mean"])
            runner_up_gap = float(winner["balanced_accuracy_mean"] - runner_up_score)

            sig_row = significance_df[
                (significance_df["dataset"] == group_key[0])
                & (significance_df["experiment"] == group_key[1])
                & (significance_df["condition_name"] == group_key[2])
                & (significance_df["condition_value"].astype(str) == str(group_key[3]))
                & (significance_df["model"] == runner_up_model)
            ]
            if not sig_row.empty:
                runner_up_vs_winner = str(sig_row.iloc[0]["interpretation"])
                runner_up_p_value = float(sig_row.iloc[0]["p_value"])

        rows.append(
            {
                "dataset": group_key[0],
                "experiment": group_key[1],
                "condition_name": group_key[2],
                "condition_value": group_key[3],
                "winner_model": winner["model"],
                "winner_score": float(winner["balanced_accuracy_mean"]),
                "runner_up_model": runner_up_model,
                "runner_up_score": runner_up_score,
                "runner_up_gap_to_winner": runner_up_gap,
                "runner_up_vs_winner": runner_up_vs_winner,
                "runner_up_p_value": runner_up_p_value,
                "worst_model": worst["model"],
                "worst_score": float(worst["balanced_accuracy_mean"]),
            }
        )

    conclusions = pd.DataFrame(rows)
    conclusions["condition_num"] = pd.to_numeric(
        conclusions["condition_value"],
        errors="coerce",
    )
    conclusions = conclusions.sort_values(
        [
            "dataset",
            "experiment",
            "condition_name",
            "condition_num",
            "condition_value",
        ]
    )
    conclusions = conclusions.drop(columns=["condition_num"])
    conclusions.to_csv(SUMMARY_OUT / "condition_conclusions.csv", index=False)

    lines = [
        "# Condition-wise Conclusions",
        "",
        "Winner, runner-up, and worst model per condition with runner-up significance.",
        "",
    ]

    for dataset in sorted(conclusions["dataset"].unique()):
        lines.append(f"## {dataset}")
        lines.append("")

        subset = conclusions[conclusions["dataset"] == dataset]
        for _, row in subset.iterrows():
            p_text = "p=n/a" if np.isnan(row["runner_up_p_value"]) else f"p={row['runner_up_p_value']:.4f}"
            lines.append(
                "- "
                f"{row['experiment']} "
                f"({row['condition_name']}={row['condition_value']}): "
                f"winner={row['winner_model']} ({row['winner_score']:.4f}), "
                f"runner_up={row['runner_up_model']} ({row['runner_up_score']:.4f}, "
                f"gap={row['runner_up_gap_to_winner']:.4f}, "
                f"{row['runner_up_vs_winner']}, {p_text}), "
                f"worst={row['worst_model']} ({row['worst_score']:.4f})."
            )
        lines.append("")

    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "condition_conclusions.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return conclusions


def _save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    output_stem: Path,
) -> None:
    """Save one multi-panel line chart split by dataset."""
    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.9), sharey=True)
    fig.patch.set_facecolor("white")
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset].copy()
        subset = subset.sort_values(x_col)
        ax = axes[idx]
        ax.set_facecolor("white")
        for model in MODEL_ORDER:
            model_subset = subset[subset["model"] == model].copy()
            if model_subset.empty:
                continue
            ax.plot(
                model_subset[x_col],
                model_subset["balanced_accuracy_mean"],
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, "#4C72B0"),
                marker=MODEL_MARKERS.get(model, "o"),
                markerfacecolor="white",
                markeredgewidth=1.1,
            )

        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        unique_x = sorted(pd.unique(subset[x_col].dropna()))
        if unique_x:
            ax.set_xticks(unique_x)
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel("Balanced Accuracy" if idx == 0 else "")
        axes[idx].set_ylim(0.0, 1.0)
        axes[idx].grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
        axes[idx].grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=False,
            title="Model",
        )
    fig.suptitle(title, fontsize=14, y=1.03, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    _save_figure(fig, output_stem)
    plt.close(fig)


def _select_worst_case_drop_rows(
    drop_df: pd.DataFrame,
    selected_conditions: list[dict[str, Any]],
) -> pd.DataFrame:
    """Select rows used for the worst-case drop comparison bar chart."""
    selected_rows: list[pd.DataFrame] = []

    for cond in selected_conditions:
        scoped = drop_df[
            (drop_df["experiment"] == cond["experiment"])
            & (drop_df["condition_name"] == cond["condition_name"])
        ].copy()
        scoped["condition_num"] = pd.to_numeric(scoped["condition_value"], errors="coerce")

        target_value = float(cond["condition_value"])
        scoped = scoped[(scoped["condition_num"] - target_value).abs() < 1e-12]

        if scoped.empty:
            continue

        scoped["condition_label"] = cond["label"]
        scoped["condition_order"] = int(cond["condition_order"])
        selected_rows.append(scoped)

    if not selected_rows:
        return pd.DataFrame()

    worst_case = pd.concat(selected_rows, ignore_index=True)
    worst_case = worst_case.sort_values(
        ["dataset", "condition_order", "model"],
    )
    return worst_case


def generate_drop_comparison_plot(drop_df: pd.DataFrame) -> None:
    """Plot percentage drop vs clean baseline under worst-case conditions."""
    selected_conditions = _build_worst_case_conditions(drop_df)
    worst_case = _select_worst_case_drop_rows(drop_df, selected_conditions)
    if worst_case.empty:
        return

    worst_case = worst_case[[
        "dataset",
        "experiment",
        "condition_name",
        "condition_value",
        "condition_label",
        "model",
        "clean_score",
        "balanced_accuracy_mean",
        "drop_abs",
        "drop_pct",
    ]].copy()

    worst_case["dataset"] = pd.Categorical(
        worst_case["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )
    worst_case["model"] = pd.Categorical(
        worst_case["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    condition_order = [c["label"] for c in selected_conditions]

    (SUMMARY_OUT / "drop_comparison_worst_case.csv").parent.mkdir(parents=True, exist_ok=True)
    worst_case.sort_values(["dataset", "condition_label", "model"]).to_csv(
        SUMMARY_OUT / "drop_comparison_worst_case.csv",
        index=False,
    )

    datasets = [d for d in DATASET_ORDER if d in set(worst_case["dataset"].astype(str))]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.4 * len(datasets), 5.2),
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    if len(datasets) == 1:
        axes = [axes]

    x = np.arange(len(condition_order))
    plot_models = [
        model for model in MODEL_ORDER
        if model in set(worst_case["model"].astype(str))
    ]
    if not plot_models:
        return

    width = min(0.8 / len(plot_models), 0.28)
    offsets = (np.arange(len(plot_models)) - (len(plot_models) - 1) / 2.0) * width

    all_drop_values = pd.to_numeric(worst_case["drop_pct"], errors="coerce")
    y_min = min(-1.0, float(np.floor(all_drop_values.min() - 0.6)))
    y_max = max(1.0, float(np.ceil(all_drop_values.max() + 0.6)))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ax.set_facecolor("white")
        subset = worst_case[worst_case["dataset"] == dataset].copy()

        for model_idx, model in enumerate(plot_models):
            model_vals: list[float] = []
            for cond_label in condition_order:
                row = subset[
                    (subset["condition_label"] == cond_label)
                    & (subset["model"] == model)
                ]
                value = np.nan if row.empty else float(row.iloc[0]["drop_pct"])
                model_vals.append(value)

            ax.bar(
                x + offsets[model_idx],
                model_vals,
                width=width,
                label=MODEL_LABELS.get(model, model) if idx == 0 else None,
                color=MODEL_COLORS.get(model, "#4C72B0"),
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )

        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(condition_order)
        ax.set_ylim(y_min, y_max)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("Worst-case condition")
        ax.set_ylabel("Drop vs clean baseline (%)" if idx == 0 else "")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.2, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=False,
            title="Model",
        )

    fig.suptitle("Performance Drop Under Worst-Case Conditions", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.95))
    _save_figure(fig, PLOTS_OUT / DROP_PLOT_STEM)
    plt.close(fig)


def generate_plots(summary_df: pd.DataFrame, runtime_df: pd.DataFrame, drop_df: pd.DataFrame) -> None:
    """Generate required robustness and runtime plots."""
    _apply_research_plot_style()
    generate_drop_comparison_plot(drop_df)

    for spec in LINE_PLOT_SPECS:
        scoped = summary_df[summary_df["experiment"] == str(spec["experiment"])]
        if scoped.empty:
            continue

        plot_df = scoped.copy()
        plot_df["x"] = plot_df["condition_num"]
        _save_line_plot(
            df=plot_df,
            x_col="x",
            title=str(spec["title"]),
            xlabel=str(spec["xlabel"]),
            output_stem=PLOTS_OUT / str(spec["output_stem"]),
        )

    clean_runtime = runtime_df[
        (runtime_df["experiment"] == "baseline")
        & (runtime_df["condition_name"] == "clean")
        & (runtime_df["condition_value"].astype(str) == "clean")
    ].copy()

    datasets = [d for d in DATASET_ORDER if d in set(clean_runtime["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.9), sharey=True)
    fig.patch.set_facecolor("white")
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ax.set_facecolor("white")
        subset = clean_runtime[clean_runtime["dataset"] == dataset].copy()

        for model in MODEL_ORDER:
            model_row = subset[subset["model"] == model]
            if model_row.empty:
                continue

            x_val = float(model_row.iloc[0]["total_time_sec_mean"])
            y_val = float(model_row.iloc[0]["balanced_accuracy_mean"])
            label = MODEL_LABELS.get(model, model) if idx == 0 else None

            ax.scatter(
                x_val,
                y_val,
                label=label,
                s=140,
                color=MODEL_COLORS.get(model, "#4C72B0"),
                marker=MODEL_MARKERS.get(model, "o"),
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )
            ax.annotate(
                MODEL_SHORT_LABELS.get(model, model),
                (x_val, y_val),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xscale("log")
        ax.set_xlabel("Total Time (s, log scale)")
        ax.set_ylabel("Balanced Accuracy" if idx == 0 else "")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=False,
            title="Model",
        )

    fig.suptitle("Runtime vs Performance Trade-off (Clean Baseline)", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    _save_figure(fig, PLOTS_OUT / RUNTIME_PLOT_STEM)
    plt.close(fig)


def _match_condition(
    frame: pd.DataFrame,
    condition_name: str,
    condition_value: Any,
) -> pd.DataFrame:
    """Filter rows by condition with numeric tolerance support."""
    scoped = frame[frame["condition_name"] == condition_name]
    if condition_name == "clean":
        return scoped[scoped["condition_value"].astype(str) == str(condition_value)]

    value_num = float(condition_value)
    return scoped[(scoped["condition_num"] - value_num).abs() < 1e-12]


def generate_failure_analysis(drop_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Generate failure-pattern summary and representative misclassified rows."""
    ERROR_OUT.mkdir(parents=True, exist_ok=True)

    worst = (
        drop_df.sort_values(["dataset", "model", "drop_abs"], ascending=[True, True, False])
        .groupby(["dataset", "model"], as_index=False)
        .head(1)
        .copy()
    )

    failure_rows: list[dict[str, Any]] = []
    sample_rows: list[pd.DataFrame] = []

    for _, row in worst.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        experiment = row["experiment"]
        condition_name = row["condition_name"]
        condition_value = row["condition_value"]

        test_rows = metrics_df[
            (metrics_df["dataset"] == dataset)
            & (metrics_df["model"] == model)
            & (metrics_df["experiment"] == experiment)
            & (metrics_df["split"] == "test")
            & (metrics_df["status"] == "ok")
        ].copy()
        test_rows = _match_condition(test_rows, condition_name, condition_value)

        if test_rows.empty:
            continue

        worst_seed_row = test_rows.sort_values("balanced_accuracy", ascending=True).iloc[0]
        predictions_path = Path(str(worst_seed_row["artifact_path"])).with_name("predictions.csv")

        pred_df = pd.read_csv(predictions_path)
        pred_df = pred_df[
            (pred_df["dataset"] == dataset)
            & (pred_df["model"] == model)
            & (pred_df["experiment"] == experiment)
            & (pred_df["split"] == "test")
            & (pred_df["condition_name"] == condition_name)
        ].copy()

        if condition_name == "clean":
            pred_df = pred_df[pred_df["condition_value"].astype(str) == str(condition_value)]
        else:
            cond_num = pd.to_numeric(pred_df["condition_value"], errors="coerce")
            pred_df = pred_df[(cond_num - float(condition_value)).abs() < 1e-12]

        misclassified = pred_df[pred_df["y_true"] != pred_df["y_pred"]].copy()
        misclassified_count = int(len(misclassified))
        total_count = int(len(pred_df))

        failure_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "worst_experiment": experiment,
                "condition_name": condition_name,
                "condition_value": condition_value,
                "clean_score": row["clean_score"],
                "worst_score": row["balanced_accuracy_mean"],
                "drop_abs": row["drop_abs"],
                "drop_pct": row["drop_pct"],
                "seed_example": int(worst_seed_row["seed"]),
                "misclassified_count": misclassified_count,
                "test_rows": total_count,
                "error_rate_example": (
                    misclassified_count / total_count if total_count > 0 else np.nan
                ),
            }
        )

        if not misclassified.empty:
            sample = misclassified.head(20).copy()
            sample_rows.append(sample)

    failure_df = pd.DataFrame(failure_rows).sort_values(
        ["dataset", "drop_abs"],
        ascending=[True, False],
    )
    failure_df.to_csv(ERROR_OUT / "failure_patterns.csv", index=False)

    if sample_rows:
        sample_df = pd.concat(sample_rows, ignore_index=True)
    else:
        sample_df = pd.DataFrame()
    sample_df.to_csv(ERROR_OUT / "misclassified_samples.csv", index=False)

    lines = [
        "# Failure Case Analysis",
        "",
        "Representative worst-condition patterns per dataset/model.",
        "",
    ]

    if failure_df.empty:
        lines.append("No failure patterns found.")
    else:
        for _, fr in failure_df.iterrows():
            lines.append(
                "- "
                f"{fr['dataset']} | {fr['model']} | "
                f"worst: {fr['worst_experiment']} "
                f"({fr['condition_name']}={fr['condition_value']}) | "
                f"drop={fr['drop_abs']:.4f} ({fr['drop_pct']:.2f}%) | "
                f"example error rate={fr['error_rate_example']:.4f}"
            )

    (ERROR_OUT / "failure_cases.md").write_text("\n".join(lines), encoding="utf-8")


def generate_claims(
    summary_df: pd.DataFrame,
    drop_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    status_coverage_df: pd.DataFrame,
    tradeoff_df: pd.DataFrame,
) -> None:
    """Write evidence-based claims for manuscript draft usage."""
    clean = summary_df[
        (summary_df["experiment"] == "baseline")
        & (summary_df["condition_name"] == "clean")
        & (summary_df["condition_value"].astype(str) == "clean")
    ]
    clean_winners = (
        clean.sort_values(["dataset", "balanced_accuracy_mean"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )

    selected_conditions = _build_worst_case_conditions(drop_df)
    robust_frames: list[pd.DataFrame] = []

    for condition in selected_conditions:
        if condition["experiment"] not in ROBUST_CLAIM_EXPERIMENTS:
            continue

        scoped = drop_df[
            (drop_df["experiment"] == condition["experiment"])
            & (drop_df["condition_name"] == condition["condition_name"])
        ].copy()
        condition_num = pd.to_numeric(scoped["condition_value"], errors="coerce")
        target_value = float(condition["condition_value"])
        scoped = scoped[(condition_num - target_value).abs() < 1e-12]
        robust_frames.append(scoped)

    if robust_frames:
        robust_focus = pd.concat(robust_frames, ignore_index=True)
    else:
        robust_focus = pd.DataFrame(columns=drop_df.columns)

    robust_drop = (
        robust_focus.groupby("model", as_index=False)
        .agg(drop_abs=("drop_abs", "mean"))
        .sort_values("drop_abs", ascending=False)
    )

    sig_compare = significance_df[
        significance_df["interpretation"] != "reference_best"
    ].copy()
    sig_worse = int((sig_compare["interpretation"] == "significantly_worse_than_best").sum())
    sig_not_diff = int((sig_compare["interpretation"] == "not_significantly_different").sum())
    sig_total = int(len(sig_compare))

    clean_runtime = runtime_df[
        (runtime_df["experiment"] == "baseline")
        & (runtime_df["condition_name"] == "clean")
        & (runtime_df["condition_value"].astype(str) == "clean")
    ].copy()

    fastest = (
        clean_runtime.sort_values(["dataset", "total_time_sec_mean"], ascending=[True, True])
        .groupby("dataset", as_index=False)
        .head(1)
        .rename(
            columns={
                "model": "fastest_model",
                "total_time_sec_mean": "fastest_total_time_sec",
            }
        )
    )
    hyperfast_runtime = clean_runtime[
        clean_runtime["model"].isin(["hyperfast_default", "hyperfast_tuned"])
    ].copy()
    best_hyperfast = (
        hyperfast_runtime.sort_values(
            ["dataset", "total_time_sec_mean"],
            ascending=[True, True],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .rename(
            columns={
                "model": "best_hyperfast_model",
                "total_time_sec_mean": "best_hyperfast_total_time_sec",
            }
        )
    )
    speed_compare = fastest.merge(best_hyperfast, on="dataset", how="left")
    speed_compare["best_hyperfast_vs_fastest_factor"] = (
        speed_compare["best_hyperfast_total_time_sec"]
        / speed_compare["fastest_total_time_sec"]
    )

    lines = ["# Evidence-Based Claims", ""]

    winner_lines = [
        f"{row['dataset']}: {row['model']} ({row['balanced_accuracy_mean']:.4f})"
        for _, row in clean_winners.iterrows()
    ]
    lines.append("1. Clean-data winners by dataset: " + "; ".join(winner_lines) + ".")

    if not robust_drop.empty:
        top_drop = robust_drop.iloc[0]
        robust_condition_labels = [
            str(condition["label"])
            for condition in selected_conditions
            if condition["experiment"] in ROBUST_CLAIM_EXPERIMENTS
        ]
        robust_desc = (
            ", ".join(robust_condition_labels)
            if robust_condition_labels
            else "the strongest available corruption settings"
        )

        if sig_total > 0:
            sig_text = (
                f"paired tests vs condition winners show {sig_worse}/{sig_total} "
                f"significantly worse comparisons and {sig_not_diff}/{sig_total} "
                "not-significant gaps"
            )
        else:
            sig_text = "insufficient paired data prevented significance counting"

        lines.append(
            f"2. Under {robust_desc}, the largest mean performance drop is for "
            f"{top_drop['model']} ({top_drop['drop_abs']:.4f}); {sig_text}."
        )

    if not speed_compare.empty:
        fastest_desc = "; ".join(
            [
                f"{row['dataset']}: {row['fastest_model']} ({row['fastest_total_time_sec']:.4f}s)"
                for _, row in speed_compare.iterrows()
            ]
        )
        factor_min = float(speed_compare["best_hyperfast_vs_fastest_factor"].min())
        factor_max = float(speed_compare["best_hyperfast_vs_fastest_factor"].max())
        lines.append(
            "3. Clean runtime ranking favors: "
            f"{fastest_desc}; in this setup, the fastest HyperFast variant is "
            f"{factor_min:.1f}x-{factor_max:.1f}x slower than the fastest "
            "model per dataset."
        )

    clean_tradeoff = tradeoff_df[
        (tradeoff_df["experiment"] == "baseline")
        & (tradeoff_df["condition_name"] == "clean")
        & (tradeoff_df["condition_value"].astype(str) == "clean")
    ].copy()
    if not clean_tradeoff.empty:
        classical = clean_tradeoff[
            clean_tradeoff["model"].isin(["logistic_regression", "random_forest"])
        ].copy()
        hyperfast = clean_tradeoff[
            clean_tradeoff["model"].isin(["hyperfast_default", "hyperfast_tuned"])
        ].copy()

        best_classical = (
            classical.sort_values(
                ["dataset", "accuracy_inference_tradeoff_score"],
                ascending=[True, False],
            )
            .groupby("dataset", as_index=False)
            .head(1)
            .rename(columns={"accuracy_inference_tradeoff_score": "classical_score"})
        )
        best_hyperfast = (
            hyperfast.sort_values(
                ["dataset", "accuracy_inference_tradeoff_score"],
                ascending=[True, False],
            )
            .groupby("dataset", as_index=False)
            .head(1)
            .rename(columns={"accuracy_inference_tradeoff_score": "hyperfast_score"})
        )
        merged_tradeoff = best_hyperfast.merge(
            best_classical[["dataset", "classical_score"]],
            on="dataset",
            how="inner",
        )
        if not merged_tradeoff.empty:
            merged_tradeoff["score_ratio"] = (
                merged_tradeoff["hyperfast_score"]
                / merged_tradeoff["classical_score"]
            )
            ratio_min = float(merged_tradeoff["score_ratio"].min())
            ratio_max = float(merged_tradeoff["score_ratio"].max())
            lines.append(
                "4. Training-free does not mean deployment-free: under the "
                "accuracy-vs-inference-cost score, the best HyperFast variant "
                f"achieves only {ratio_min:.2f}x-{ratio_max:.2f}x of the best "
                "classical model score across datasets."
            )

    robust_status = status_coverage_df[
        status_coverage_df["experiment"].isin(ROBUST_CLAIM_EXPERIMENTS)
    ].copy()
    if not robust_status.empty:
        reliability = (
            robust_status.groupby("model", as_index=False)
            .agg(
                attempted_runs=("attempted_runs", "sum"),
                ok_runs=("ok_runs", "sum"),
                error_runs=("error_runs", "sum"),
                mean_error_rate=("error_rate", "mean"),
                max_error_rate=("error_rate", "max"),
            )
            .sort_values("mean_error_rate", ascending=False)
        )

        highest_error = reliability.iloc[0]
        lowest_error = reliability.iloc[-1]
        lines.append(
            "5. Reliability under robustness conditions: highest mean error rate is "
            f"{highest_error['model']} ({highest_error['mean_error_rate']:.3f}, "
            f"max={highest_error['max_error_rate']:.3f}); lowest is "
            f"{lowest_error['model']} ({lowest_error['mean_error_rate']:.3f})."
        )

    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "claims.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run all analysis artifact generation steps."""
    summary_df, metrics_df = _load_data()
    drop_df = generate_drop_table(summary_df)
    ranking_df = generate_rankings(summary_df)
    status_coverage_df = generate_status_coverage(metrics_df)
    runtime_df = generate_runtime_table(metrics_df)
    tradeoff_df = generate_efficiency_tradeoff(runtime_df)
    significance_df = generate_statistical_confidence(metrics_df)
    _ = generate_condition_conclusions(ranking_df, significance_df)
    generate_plots(summary_df, runtime_df, drop_df)
    generate_failure_analysis(drop_df, metrics_df)
    generate_claims(
        summary_df,
        drop_df,
        runtime_df,
        significance_df,
        status_coverage_df,
        tradeoff_df,
    )

    print(f"drop table: {(SUMMARY_OUT / 'performance_drop_vs_clean.csv').as_posix()}")
    print(f"rankings: {(SUMMARY_OUT / 'condition_wise_rankings.csv').as_posix()}")
    print(f"status coverage: {(SUMMARY_OUT / 'status_coverage_by_condition.csv').as_posix()}")
    print(f"runtime table: {(SUMMARY_OUT / 'runtime_by_condition.csv').as_posix()}")
    print(
        "tradeoff table: "
        f"{(SUMMARY_OUT / 'accuracy_inference_tradeoff.csv').as_posix()}"
    )
    print(
        "significance table: "
        f"{(SUMMARY_OUT / 'statistical_confidence_vs_best.csv').as_posix()}"
    )
    print(
        "condition conclusions: "
        f"{(SUMMARY_OUT / 'condition_conclusions.csv').as_posix()}"
    )
    print(f"plots: {PLOTS_OUT.as_posix()}")
    print(f"failure analysis: {ERROR_OUT.as_posix()}")
    print(f"claims: {(REPORT_OUT / 'claims.md').as_posix()}")


if __name__ == "__main__":
    main()
