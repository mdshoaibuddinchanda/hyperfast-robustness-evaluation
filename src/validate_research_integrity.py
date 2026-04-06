"""Validate that research-hardening safeguards are present in generated artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
SUMMARY_ROOT = RESULTS_ROOT / "summary_tables"
CONFIG_ROOT = PROJECT_ROOT / "configs"

REQUIRED_MODELS = {
    "hyperfast_default",
    "hyperfast_tuned",
    "logistic_regression",
    "random_forest",
}


def _check_file_exists(path: Path, findings: list[str]) -> bool:
    """Append finding when file is missing and return existence status."""
    if not path.exists():
        findings.append(f"Missing required file: {path.as_posix()}")
        return False
    return True


def _check_columns(
    frame: pd.DataFrame,
    required_cols: set[str],
    frame_name: str,
    findings: list[str],
) -> bool:
    """Validate required columns exist in a dataframe."""
    missing = sorted(required_cols - set(frame.columns))
    if missing:
        findings.append(f"{frame_name} missing columns: {missing}")
        return False
    return True


def _validate_metrics(metrics_df: pd.DataFrame, findings: list[str]) -> None:
    """Run structural checks for long-form metrics output."""
    required_cols = {
        "dataset",
        "seed",
        "experiment",
        "condition_name",
        "condition_value",
        "model",
        "split",
        "status",
    }
    if not _check_columns(metrics_df, required_cols, "results/metrics.csv", findings):
        return

    test_rows = metrics_df[metrics_df["split"] == "test"].copy()
    present_models = set(test_rows["model"].dropna().astype(str).unique().tolist())
    missing_models = sorted(REQUIRED_MODELS - present_models)
    if missing_models:
        findings.append(
            "results/metrics.csv missing expected models in test split: "
            f"{missing_models}"
        )

    bad_status = sorted(
        set(test_rows["status"].dropna().astype(str).unique().tolist())
        - {"ok", "error"}
    )
    if bad_status:
        findings.append(f"Unexpected status values in metrics: {bad_status}")


def _validate_status_coverage(
    metrics_df: pd.DataFrame,
    status_df: pd.DataFrame,
    findings: list[str],
) -> None:
    """Check reliability table consistency against metrics.csv."""
    required_cols = {
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
    }
    if not _check_columns(
        status_df,
        required_cols,
        "results/summary_tables/status_coverage_by_condition.csv",
        findings,
    ):
        return

    test_rows = metrics_df[metrics_df["split"] == "test"].copy()
    calc = (
        test_rows.assign(
            attempted_runs=1,
            ok_runs=(test_rows["status"] == "ok").astype(int),
            error_runs=(test_rows["status"] != "ok").astype(int),
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

    merged = status_df.merge(
        calc,
        on=["dataset", "experiment", "condition_name", "condition_value", "model"],
        how="outer",
        suffixes=("_file", "_calc"),
        indicator=True,
    )

    if (merged["_merge"] != "both").any():
        findings.append(
            "status_coverage_by_condition row mismatch against metrics.csv "
            f"(non-overlap rows={int((merged['_merge'] != 'both').sum())})"
        )

    both = merged[merged["_merge"] == "both"]
    for col in ["attempted_runs", "ok_runs", "error_runs"]:
        diff = (both[f"{col}_file"] - both[f"{col}_calc"]).abs()
        if diff.notna().any() and float(diff.max()) > 0.0:
            findings.append(f"status_coverage mismatch in {col}")
            break


def _validate_significance(significance_df: pd.DataFrame, findings: list[str]) -> None:
    """Check statistical table contains hardened significance columns."""
    required_cols = {
        "p_value_ttest",
        "p_value_wilcoxon",
        "p_value_holm",
        "p_value",
        "test_basis",
        "interpretation",
    }
    if not _check_columns(
        significance_df,
        required_cols,
        "results/summary_tables/statistical_confidence_vs_best.csv",
        findings,
    ):
        return

    pending_count = int((significance_df["interpretation"] == "pending").sum())
    if pending_count > 0:
        findings.append(
            "statistical_confidence_vs_best contains pending interpretations "
            f"(count={pending_count})"
        )


def validate() -> tuple[bool, list[str]]:
    """Run full integrity validation suite."""
    findings: list[str] = []

    metrics_path = RESULTS_ROOT / "metrics.csv"
    status_path = SUMMARY_ROOT / "status_coverage_by_condition.csv"
    status_model_path = SUMMARY_ROOT / "status_coverage_by_model.csv"
    significance_path = SUMMARY_ROOT / "statistical_confidence_vs_best.csv"
    summary_path = SUMMARY_ROOT / "test_mean_std_by_condition.csv"
    analysis_config_path = CONFIG_ROOT / "analysis_artifacts.json"

    needed = [
        metrics_path,
        status_path,
        status_model_path,
        significance_path,
        summary_path,
        analysis_config_path,
    ]
    if not all(_check_file_exists(path, findings) for path in needed):
        return False, findings

    metrics_df = pd.read_csv(metrics_path)
    status_df = pd.read_csv(status_path)
    significance_df = pd.read_csv(significance_path)
    summary_df = pd.read_csv(summary_path)

    _validate_metrics(metrics_df, findings)
    _validate_status_coverage(metrics_df, status_df, findings)
    _validate_significance(significance_df, findings)

    if "ok_rate" not in summary_df.columns or "error_rate" not in summary_df.columns:
        findings.append(
            "test_mean_std_by_condition.csv is missing ok_rate/error_rate coverage columns"
        )

    analysis_cfg = json.loads(analysis_config_path.read_text(encoding="utf-8"))
    robust_claim_experiments = set(
        str(item) for item in analysis_cfg.get("robust_claim_experiments", [])
    )
    if "reduced_data" not in robust_claim_experiments:
        findings.append(
            "analysis_artifacts.json robust_claim_experiments does not include reduced_data"
        )

    return len(findings) == 0, findings


def main() -> None:
    """CLI entry point."""
    ok, findings = validate()
    print(f"integrity_status: {'PASS' if ok else 'FAIL'}")
    if findings:
        print("findings:")
        for item in findings:
            print(f"- {item}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
