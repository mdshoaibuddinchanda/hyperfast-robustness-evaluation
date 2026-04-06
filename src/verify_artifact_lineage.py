"""Verify lineage from raw run outputs to summary tables and plot sources."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


NUM_TOL = 1e-10


def _add_check(checks: list[dict[str, str]], name: str, ok: bool, detail: str) -> None:
    """Append one verification check result."""
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _canon_condition_value(value: object) -> str:
    """Convert condition values to a stable string representation."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text

    if np.isfinite(number):
        return format(number, ".12g")
    return text


def _normalize_condition_value(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical condition_value strings."""
    out = df.copy()
    if "condition_value" in out.columns:
        out["condition_value"] = out["condition_value"].map(_canon_condition_value)
    return out


def _load_raw_runs(runs_dir: Path) -> pd.DataFrame:
    """Load every runs/**/metrics.json file into a normalized table."""
    rows: list[dict[str, object]] = []

    for metrics_path in runs_dir.glob("**/metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        dataset = payload["dataset"]
        seed = int(payload["seed"])
        experiment = payload["experiment"]
        condition_name = payload["condition"]["name"]
        condition_value = str(payload["condition"]["value"])

        for result in payload["results"]:
            model = result["model"]
            timing = result.get("timing")
            error = result.get("error")

            for split in ["validation", "test"]:
                metrics = result.get(split)
                if error is not None or metrics is None or timing is None:
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "experiment": experiment,
                            "condition_name": condition_name,
                            "condition_value": condition_value,
                            "model": model,
                            "split": split,
                            "status": "error",
                            "balanced_accuracy": np.nan,
                            "f1": np.nan,
                            "precision": np.nan,
                            "recall": np.nan,
                            "auroc": np.nan,
                            "fit_time_sec": np.nan,
                            "predict_time_sec": np.nan,
                            "total_time_sec": np.nan,
                        }
                    )
                else:
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "experiment": experiment,
                            "condition_name": condition_name,
                            "condition_value": condition_value,
                            "model": model,
                            "split": split,
                            "status": "ok",
                            "balanced_accuracy": float(metrics["balanced_accuracy"]),
                            "f1": float(metrics["f1"]),
                            "precision": float(metrics["precision"]),
                            "recall": float(metrics["recall"]),
                            "auroc": float(metrics["auroc"]),
                            "fit_time_sec": float(timing["fit_time_sec"]),
                            "predict_time_sec": float(timing["predict_time_sec"]),
                            "total_time_sec": float(timing["total_time_sec"]),
                        }
                    )

    columns = [
        "dataset",
        "seed",
        "experiment",
        "condition_name",
        "condition_value",
        "model",
        "split",
        "status",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "auroc",
        "fit_time_sec",
        "predict_time_sec",
        "total_time_sec",
    ]
    return pd.DataFrame(rows, columns=columns)


def _max_abs_diff(joined: pd.DataFrame, columns: list[str]) -> float:
    """Compute max absolute difference across paired _file/_calc columns."""
    max_diff = 0.0
    both = joined[joined["_merge"] == "both"]

    for column in columns:
        diff = (both[f"{column}_file"] - both[f"{column}_calc"]).abs().max()
        if pd.notna(diff):
            max_diff = max(max_diff, float(diff))

    return max_diff


def verify(project_root: Path) -> tuple[bool, list[dict[str, str]]]:
    """Run all lineage checks and return overall status with details."""
    checks: list[dict[str, str]] = []

    runs_dir = project_root / "runs"
    results_dir = project_root / "results"
    summary_dir = results_dir / "summary_tables"
    plots_dir = project_root / "plots"
    analysis_config_path = project_root / "configs" / "analysis_artifacts.json"
    analysis_cfg = json.loads(analysis_config_path.read_text(encoding="utf-8"))
    line_plot_specs = list(analysis_cfg.get("line_plots", []))
    runtime_plot_stem = str(analysis_cfg.get("runtime_plot_stem", "runtime_tradeoff"))
    drop_plot_stem = str(analysis_cfg.get("drop_plot_stem", "drop_comparison"))
    export_formats = [str(fmt) for fmt in analysis_cfg.get("export_formats", ["png", "pdf"])]

    raw_df = _load_raw_runs(runs_dir)
    raw_df = _normalize_condition_value(raw_df)
    raw_ok = raw_df[raw_df["status"] == "ok"].copy()

    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    metrics_ok = metrics_df[metrics_df["status"] == "ok"].copy()
    metrics_ok = _normalize_condition_value(metrics_ok)

    key_cols = [
        "dataset",
        "seed",
        "experiment",
        "condition_name",
        "condition_value",
        "model",
        "split",
    ]
    value_cols = [
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "auroc",
        "fit_time_sec",
        "predict_time_sec",
        "total_time_sec",
    ]

    merged_raw_metrics = metrics_ok[key_cols + value_cols].merge(
        raw_ok[key_cols + value_cols],
        on=key_cols,
        how="outer",
        suffixes=("_metrics", "_raw"),
        indicator=True,
    )
    missing_left = int((merged_raw_metrics["_merge"] == "left_only").sum())
    missing_right = int((merged_raw_metrics["_merge"] == "right_only").sum())

    max_diff_raw_metrics = 0.0
    both = merged_raw_metrics[merged_raw_metrics["_merge"] == "both"]
    for column in value_cols:
        diff = (both[f"{column}_metrics"] - both[f"{column}_raw"]).abs().max()
        if pd.notna(diff):
            max_diff_raw_metrics = max(max_diff_raw_metrics, float(diff))

    _add_check(
        checks,
        "raw_runs_to_metrics_csv",
        missing_left == 0 and missing_right == 0 and max_diff_raw_metrics <= NUM_TOL,
        (
            f"rows_raw_ok={len(raw_ok)}, rows_metrics_ok={len(metrics_ok)}, "
            f"left_only={missing_left}, right_only={missing_right}, "
            f"max_abs_diff={max_diff_raw_metrics:.3e}"
        ),
    )

    calc_status = (
        metrics_df[metrics_df["split"] == "test"]
        .assign(
            attempted_runs=1,
            ok_runs=lambda d: (d["status"] == "ok").astype(int),
            error_runs=lambda d: (d["status"] != "ok").astype(int),
        )
        .groupby(
            ["dataset", "experiment", "condition_name", "condition_value", "model"],
            as_index=False,
        )
        .agg(
            attempted_runs=("attempted_runs", "sum"),
            ok_runs=("ok_runs", "sum"),
            error_runs=("error_runs", "sum"),
        )
    )
    calc_status["ok_rate"] = np.where(
        calc_status["attempted_runs"] > 0,
        calc_status["ok_runs"] / calc_status["attempted_runs"],
        np.nan,
    )
    calc_status["error_rate"] = np.where(
        calc_status["attempted_runs"] > 0,
        calc_status["error_runs"] / calc_status["attempted_runs"],
        np.nan,
    )

    file_status = _normalize_condition_value(
        pd.read_csv(summary_dir / "status_coverage_by_condition.csv")
    )
    joined_status = file_status.merge(
        calc_status,
        on=["dataset", "experiment", "condition_name", "condition_value", "model"],
        how="outer",
        suffixes=("_file", "_calc"),
        indicator=True,
    )
    status_left = int((joined_status["_merge"] == "left_only").sum())
    status_right = int((joined_status["_merge"] == "right_only").sum())
    status_max_diff = _max_abs_diff(
        joined_status,
        ["attempted_runs", "ok_runs", "error_runs", "ok_rate", "error_rate"],
    )

    _add_check(
        checks,
        "metrics_to_status_coverage",
        status_left == 0 and status_right == 0 and status_max_diff <= NUM_TOL,
        (
            f"left_only={status_left}, right_only={status_right}, "
            f"max_abs_diff={status_max_diff:.3e}"
        ),
    )

    calc_status_model = (
        calc_status.groupby("model", as_index=False)
        .agg(
            attempted_runs=("attempted_runs", "sum"),
            ok_runs=("ok_runs", "sum"),
            error_runs=("error_runs", "sum"),
        )
    )
    calc_status_model["ok_rate"] = np.where(
        calc_status_model["attempted_runs"] > 0,
        calc_status_model["ok_runs"] / calc_status_model["attempted_runs"],
        np.nan,
    )
    calc_status_model["error_rate"] = np.where(
        calc_status_model["attempted_runs"] > 0,
        calc_status_model["error_runs"] / calc_status_model["attempted_runs"],
        np.nan,
    )

    file_status_model = pd.read_csv(summary_dir / "status_coverage_by_model.csv")
    joined_status_model = file_status_model.merge(
        calc_status_model,
        on=["model"],
        how="outer",
        suffixes=("_file", "_calc"),
        indicator=True,
    )
    status_model_left = int((joined_status_model["_merge"] == "left_only").sum())
    status_model_right = int((joined_status_model["_merge"] == "right_only").sum())
    status_model_max_diff = _max_abs_diff(
        joined_status_model,
        ["attempted_runs", "ok_runs", "error_runs", "ok_rate", "error_rate"],
    )

    _add_check(
        checks,
        "metrics_to_status_coverage_by_model",
        (
            status_model_left == 0
            and status_model_right == 0
            and status_model_max_diff <= NUM_TOL
        ),
        (
            f"left_only={status_model_left}, right_only={status_model_right}, "
            f"max_abs_diff={status_model_max_diff:.3e}"
        ),
    )

    summary = _normalize_condition_value(
        pd.read_csv(summary_dir / "test_mean_std_by_condition.csv")
    )
    calc_summary = (
        metrics_ok[metrics_ok["split"] == "test"]
        .groupby(
            ["experiment", "dataset", "model", "condition_name", "condition_value"],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            total_time_sec_mean=("total_time_sec", "mean"),
            total_time_sec_std=("total_time_sec", "std"),
        )
    )

    joined_summary = summary.merge(
        calc_summary,
        on=["experiment", "dataset", "model", "condition_name", "condition_value"],
        how="outer",
        suffixes=("_file", "_calc"),
        indicator=True,
    )
    summary_left = int((joined_summary["_merge"] == "left_only").sum())
    summary_right = int((joined_summary["_merge"] == "right_only").sum())
    summary_max_diff = _max_abs_diff(
        joined_summary,
        [
            "n_runs",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "f1_mean",
            "f1_std",
            "auroc_mean",
            "auroc_std",
            "total_time_sec_mean",
            "total_time_sec_std",
        ],
    )

    _add_check(
        checks,
        "metrics_to_test_mean_std",
        summary_left == 0 and summary_right == 0 and summary_max_diff <= NUM_TOL,
        (
            f"left_only={summary_left}, right_only={summary_right}, "
            f"max_abs_diff={summary_max_diff:.3e}"
        ),
    )

    clean_lookup = summary[
        (summary["experiment"] == "baseline")
        & (summary["condition_name"] == "clean")
        & (summary["condition_value"].astype(str) == "clean")
    ][["dataset", "model", "balanced_accuracy_mean"]].rename(
        columns={"balanced_accuracy_mean": "clean_score"}
    )
    non_clean = summary[summary["condition_name"] != "clean"].copy()
    calc_drop = non_clean.merge(clean_lookup, on=["dataset", "model"], how="left")
    calc_drop["drop_abs"] = calc_drop["clean_score"] - calc_drop["balanced_accuracy_mean"]
    calc_drop["drop_pct"] = np.where(
        calc_drop["clean_score"] > 0,
        100.0 * calc_drop["drop_abs"] / calc_drop["clean_score"],
        np.nan,
    )
    calc_drop = calc_drop[
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
    ]

    file_drop = _normalize_condition_value(
        pd.read_csv(summary_dir / "performance_drop_vs_clean.csv")
    )
    joined_drop = file_drop.merge(
        calc_drop,
        on=["dataset", "experiment", "condition_name", "condition_value", "model"],
        how="outer",
        suffixes=("_file", "_calc"),
        indicator=True,
    )
    drop_left = int((joined_drop["_merge"] == "left_only").sum())
    drop_right = int((joined_drop["_merge"] == "right_only").sum())
    drop_max_diff = _max_abs_diff(
        joined_drop,
        [
            "clean_score",
            "balanced_accuracy_mean",
            "drop_abs",
            "drop_pct",
            "n_runs",
            "total_time_sec_mean",
        ],
    )

    _add_check(
        checks,
        "summary_to_performance_drop",
        drop_left == 0 and drop_right == 0 and drop_max_diff <= NUM_TOL,
        (
            f"left_only={drop_left}, right_only={drop_right}, "
            f"max_abs_diff={drop_max_diff:.3e}"
        ),
    )

    calc_runtime = (
        metrics_ok[metrics_ok["split"] == "test"]
        .groupby(
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
            n_runs=("seed", "count"),
        )
    )
    file_runtime = _normalize_condition_value(
        pd.read_csv(summary_dir / "runtime_by_condition.csv")
    )
    joined_runtime = file_runtime.merge(
        calc_runtime,
        on=["dataset", "model", "experiment", "condition_name", "condition_value"],
        how="outer",
        suffixes=("_file", "_calc"),
        indicator=True,
    )
    runtime_left = int((joined_runtime["_merge"] == "left_only").sum())
    runtime_right = int((joined_runtime["_merge"] == "right_only").sum())
    runtime_max_diff = _max_abs_diff(
        joined_runtime,
        [
            "fit_time_sec_mean",
            "fit_time_sec_std",
            "predict_time_sec_mean",
            "predict_time_sec_std",
            "total_time_sec_mean",
            "total_time_sec_std",
            "balanced_accuracy_mean",
            "n_runs",
        ],
    )

    _add_check(
        checks,
        "metrics_to_runtime_by_condition",
        runtime_left == 0 and runtime_right == 0 and runtime_max_diff <= NUM_TOL,
        (
            f"left_only={runtime_left}, right_only={runtime_right}, "
            f"max_abs_diff={runtime_max_diff:.3e}"
        ),
    )

    calc_rankings = summary.copy()
    calc_rankings["rank"] = calc_rankings.groupby(
        ["dataset", "experiment", "condition_name", "condition_value"]
    )["balanced_accuracy_mean"].rank(method="min", ascending=False)
    best = calc_rankings.groupby(
        ["dataset", "experiment", "condition_name", "condition_value"]
    )["balanced_accuracy_mean"].transform("max")
    calc_rankings["gap_to_best"] = best - calc_rankings["balanced_accuracy_mean"]
    calc_rankings = calc_rankings[
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
    ]

    file_rankings = _normalize_condition_value(
        pd.read_csv(summary_dir / "condition_wise_rankings.csv")
    )
    joined_rankings = file_rankings.merge(
        calc_rankings,
        on=["dataset", "experiment", "condition_name", "condition_value", "model"],
        how="outer",
        suffixes=("_file", "_calc"),
        indicator=True,
    )
    rank_left = int((joined_rankings["_merge"] == "left_only").sum())
    rank_right = int((joined_rankings["_merge"] == "right_only").sum())
    rank_max_diff = _max_abs_diff(
        joined_rankings,
        ["balanced_accuracy_mean", "balanced_accuracy_std", "rank", "gap_to_best", "n_runs"],
    )

    _add_check(
        checks,
        "summary_to_condition_rankings",
        rank_left == 0 and rank_right == 0 and rank_max_diff <= NUM_TOL,
        (
            f"left_only={rank_left}, right_only={rank_right}, "
            f"max_abs_diff={rank_max_diff:.3e}"
        ),
    )

    file_drop_plot = _normalize_condition_value(
        pd.read_csv(summary_dir / "drop_comparison_worst_case.csv")
    )

    if file_drop_plot.empty:
        calc_drop_plot = pd.DataFrame(
            columns=[
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
            ]
        )
    else:
        selection_keys = file_drop_plot[
            ["experiment", "condition_name", "condition_value", "condition_label"]
        ].drop_duplicates()

        calc_drop_plot = file_drop.merge(
            selection_keys,
            on=["experiment", "condition_name", "condition_value"],
            how="inner",
        )
        calc_drop_plot = calc_drop_plot[
            [
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
            ]
        ]

    joined_drop_plot = file_drop_plot.merge(
        calc_drop_plot,
        on=[
            "dataset",
            "experiment",
            "condition_name",
            "condition_value",
            "condition_label",
            "model",
        ],
        how="outer",
        suffixes=("_file", "_calc"),
        indicator=True,
    )
    drop_plot_left = int((joined_drop_plot["_merge"] == "left_only").sum())
    drop_plot_right = int((joined_drop_plot["_merge"] == "right_only").sum())
    drop_plot_max_diff = _max_abs_diff(
        joined_drop_plot,
        ["clean_score", "balanced_accuracy_mean", "drop_abs", "drop_pct"],
    )

    _add_check(
        checks,
        "drop_table_to_drop_comparison_worst_case",
        (
            drop_plot_left == 0
            and drop_plot_right == 0
            and drop_plot_max_diff <= NUM_TOL
        ),
        (
            f"left_only={drop_plot_left}, right_only={drop_plot_right}, "
            f"max_abs_diff={drop_plot_max_diff:.3e}"
        ),
    )

    trained_models = sorted(raw_ok["model"].unique().tolist())
    metrics_models = sorted(metrics_ok["model"].unique().tolist())
    summary_models = sorted(summary["model"].unique().tolist())
    drop_plot_models = sorted(file_drop_plot["model"].unique().tolist())
    models_match = trained_models == metrics_models == summary_models == drop_plot_models

    _add_check(
        checks,
        "model_set_consistency",
        models_match,
        (
            f"trained={trained_models}; metrics={metrics_models}; "
            f"summary={summary_models}; drop_comparison={drop_plot_models}"
        ),
    )

    plot_sources: dict[str, int] = {}
    for spec in line_plot_specs:
        stem = str(spec["output_stem"])
        experiment_name = str(spec["experiment"])
        plot_sources[stem] = int(len(summary[summary["experiment"] == experiment_name]))

    plot_sources[runtime_plot_stem] = int(
        len(pd.read_csv(summary_dir / "runtime_clean_comparison.csv"))
    )
    plot_sources[drop_plot_stem] = int(len(file_drop_plot))

    plot_files = [
        f"{stem}.{fmt}"
        for stem in plot_sources
        for fmt in export_formats
    ]
    missing_plots = [name for name in plot_files if not (plots_dir / name).exists()]
    source_ok = all(rows > 0 for rows in plot_sources.values())

    _add_check(
        checks,
        "plot_files_and_sources",
        len(missing_plots) == 0 and source_ok,
        f"missing_plot_files={missing_plots}; source_rows={plot_sources}",
    )

    overall_ok = all(check["ok"] for check in checks)
    return overall_ok, checks


def write_report(project_root: Path, overall_ok: bool, checks: list[dict[str, str]]) -> Path:
    """Write markdown report summarizing the lineage verification."""
    report_path = project_root / "report" / "artifact_lineage_verification.md"
    lines = [
        "# Artifact Lineage Verification",
        "",
        f"Overall status: {'PASS' if overall_ok else 'FAIL'}",
        "",
        "## Checks",
    ]

    for check in checks:
        status = "PASS" if check["ok"] else "FAIL"
        lines.append(f"- {check['name']}: {status} | {check['detail']}")

    lines.extend(["", "## Conclusion"])
    if overall_ok:
        lines.append(
            "All tables and all plot source datasets are consistent with trained "
            "run outputs under runs/**/metrics.json."
        )
    else:
        lines.append("At least one lineage check failed. Review failed items above.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    """Execute verification and print concise terminal summary."""
    project_root = Path(__file__).resolve().parents[1]
    overall_ok, checks = verify(project_root)
    report_path = write_report(project_root, overall_ok, checks)

    print(f"verification_report: {report_path.as_posix()}")
    print(f"overall_status: {'PASS' if overall_ok else 'FAIL'}")
    for check in checks:
        print(f"{check['name']} => {'PASS' if check['ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()