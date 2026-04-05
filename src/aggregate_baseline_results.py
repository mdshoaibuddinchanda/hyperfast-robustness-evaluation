"""Aggregate per-dataset baseline metrics into one CSV log."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "baseline"
CONFIG_PATH = PROJECT_ROOT / "configs" / "split_config.json"


def _build_row(dataset: str, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Create one flattened CSV row from model result payload."""
    validation = result.get("validation")
    test = result.get("test")
    timing = result.get("timing")

    return {
        "dataset": dataset,
        "seed": payload.get("seed"),
        "split_file": payload.get("split_file"),
        "model": result.get("model"),
        "status": "error" if result.get("error") else "ok",
        "error": result.get("error"),
        "val_balanced_accuracy": None
        if validation is None
        else validation.get("balanced_accuracy"),
        "test_balanced_accuracy": None
        if test is None
        else test.get("balanced_accuracy"),
        "val_auroc": None if validation is None else validation.get("auroc"),
        "test_auroc": None if test is None else test.get("auroc"),
        "fit_time_sec": None if timing is None else timing.get("fit_time_sec"),
        "predict_time_sec": None if timing is None else timing.get("predict_time_sec"),
        "total_time_sec": None if timing is None else timing.get("total_time_sec"),
    }


def _parse_list_argument(raw: str | None) -> list[str] | None:
    """Parse comma-separated string into list or return None."""
    if raw is None or raw.strip() == "":
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values if values else None


def _default_datasets() -> list[str]:
    """Resolve default dataset list from split config or baseline run folders."""
    if CONFIG_PATH.exists():
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        configured = payload.get("datasets", [])
        if configured:
            return [str(name) for name in configured]

    if RUN_ROOT.exists():
        return sorted([entry.name for entry in RUN_ROOT.iterdir() if entry.is_dir()])

    return []


def aggregate(seed: int, datasets: list[str], log_path: Path) -> Path:
    """Read baseline metrics and write the consolidated CSV log."""
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        metrics_path = RUN_ROOT / dataset / f"seed{seed}" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing metrics file for dataset='{dataset}', seed={seed}: "
                f"{metrics_path.as_posix()}"
            )

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            rows.append(_build_row(dataset, payload, result))

    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "seed",
        "split_file",
        "model",
        "status",
        "error",
        "val_balanced_accuracy",
        "test_balanced_accuracy",
        "val_auroc",
        "test_auroc",
        "fit_time_sec",
        "predict_time_sec",
        "total_time_sec",
    ]

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return log_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1, help="Seed to aggregate.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names. Defaults to split_config datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    args = parser.parse_args()

    datasets = _parse_list_argument(args.datasets)
    if datasets is None:
        datasets = _default_datasets()

    if not datasets:
        raise ValueError("No datasets available to aggregate.")

    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "logs" / f"baseline_all_datasets_seed{args.seed}.csv"

    written_path = aggregate(seed=args.seed, datasets=datasets, log_path=output_path)
    print(written_path.as_posix())


if __name__ == "__main__":
    main()
