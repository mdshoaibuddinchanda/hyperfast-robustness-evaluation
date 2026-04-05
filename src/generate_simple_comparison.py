"""Generate a compact all-results comparison report from summary tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "summary_tables" / "test_mean_std_by_condition.csv"
OUTPUT_PATH = PROJECT_ROOT / "logs" / "all_results_simple_comparison.md"


def _fmt(value: float | int | str) -> str:
    """Format numeric values for readable report cells."""
    if isinstance(value, str):
        return value
    return f"{value:.4f}"


def _build_condition_specs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Infer report condition columns dynamically from available summary data."""
    specs: list[dict[str, Any]] = [
        {
            "column": "clean",
            "experiment": "baseline",
            "condition_name": "clean",
            "condition_value": "clean",
        }
    ]

    prefix_map = {
        "noise": "noise",
        "missingness": "missing",
        "reduced_data": "frac",
    }

    experiments = sorted(
        [
            str(name)
            for name in pd.unique(df["experiment"].dropna())
            if str(name) != "baseline"
        ]
    )

    for experiment in experiments:
        experiment_df = df[df["experiment"] == experiment].copy()
        if experiment_df.empty:
            continue

        prefix = prefix_map.get(experiment, experiment)
        condition_names = sorted(experiment_df["condition_name"].dropna().unique().tolist())
        include_condition_name = len(condition_names) > 1

        for condition_name in condition_names:
            scoped = experiment_df[experiment_df["condition_name"] == condition_name].copy()
            condition_values = pd.to_numeric(scoped["condition_value"], errors="coerce")
            unique_values = sorted(pd.unique(condition_values.dropna()))

            for value in unique_values:
                if include_condition_name:
                    column = f"{prefix}_{condition_name}_{value:.2f}"
                else:
                    column = f"{prefix}_{value:.2f}"

                specs.append(
                    {
                        "column": column,
                        "experiment": experiment,
                        "condition_name": condition_name,
                        "condition_value": float(value),
                    }
                )

    return specs


def _metric_matrix(
    df: pd.DataFrame,
    dataset: str,
    condition_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build one matrix row per model with dynamic condition columns."""
    rows = []
    dataset_df = df[df["dataset"] == dataset]

    for model in sorted(dataset_df["model"].unique()):
        model_df = dataset_df[dataset_df["model"] == model]

        def pick(spec: dict[str, Any]) -> float:
            scoped = model_df[
                (model_df["experiment"] == spec["experiment"])
                & (model_df["condition_name"] == spec["condition_name"])
            ]

            if spec["condition_name"] == "clean":
                match = scoped[
                    scoped["condition_value"].astype(str)
                    == str(spec["condition_value"])
                ]
            else:
                target_value = float(spec["condition_value"])
                numeric_values = pd.to_numeric(scoped["condition_value"], errors="coerce")
                match = scoped[(numeric_values - target_value).abs() < 1e-12]

            if match.empty:
                return float("nan")
            return float(match.iloc[0]["balanced_accuracy_mean"])

        row: dict[str, Any] = {"model": model}
        for spec in condition_specs:
            row[spec["column"]] = pick(spec)
        rows.append(row)

    matrix = pd.DataFrame(rows)
    metric_columns = [column for column in matrix.columns if column != "model"]
    for column in metric_columns:
        matrix[column] = matrix[column].map(_fmt)

    return matrix


def _clean_runtime_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build clean-data runtime and AUROC comparison table."""
    baseline = df[
        (df["experiment"] == "baseline")
        & (df["condition_name"] == "clean")
        & (df["condition_value"].astype(str) == "clean")
    ].copy()
    baseline = baseline[
        [
            "dataset",
            "model",
            "balanced_accuracy_mean",
            "auroc_mean",
            "total_time_sec_mean",
            "n_runs",
        ]
    ]
    baseline = baseline.sort_values(["dataset", "model"])
    for column in ["balanced_accuracy_mean", "auroc_mean", "total_time_sec_mean"]:
        baseline[column] = baseline[column].map(_fmt)
    return baseline


def _to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a markdown table without external dependencies."""
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in frame.iterrows():
        row_values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join([header, separator, *rows])


def main() -> None:
    """Create markdown report with full simple comparison tables."""
    df = pd.read_csv(INPUT_PATH)
    condition_specs = _build_condition_specs(df)
    datasets = sorted(df["dataset"].dropna().astype(str).unique().tolist())

    lines: list[str] = []
    lines.append("# All Results Simple Comparison")
    lines.append("")
    lines.append("Primary metric shown below: balanced accuracy mean across 5 seeds.")
    lines.append("")

    lines.append("## Clean Baseline + Runtime")
    lines.append("")
    lines.append(_to_markdown_table(_clean_runtime_table(df)))
    lines.append("")

    for dataset in datasets:
        lines.append(f"## Dataset: {dataset}")
        lines.append("")
        lines.append(
            _to_markdown_table(_metric_matrix(df, dataset, condition_specs))
        )
        lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_PATH.as_posix())


if __name__ == "__main__":
    main()
