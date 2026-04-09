"""All-in-one pipeline runner for the HyperFast robustness benchmark.

This script performs the full workflow in one command:
1) Preflight checks (optional requirement installation)
2) Dataset availability checks (optional auto-download)
3) HyperFast checkpoint availability (optional auto-download)
4) Split availability checks (optional generation)
5) Full comparison run
6) Analysis artifact generation
7) Lineage verification
8) Research-integrity validation

Safety default:
- Refuses to start if another `run_full_comparison.py` process is active,
  unless `--allow-concurrent-run` is provided.

Auto behavior default:
- Missing requirements/data/checkpoint/splits are handled automatically.
- Use `--no-auto-*` flags to disable any automatic step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
CHECKPOINT_PATH = PROJECT_ROOT / "hyperfast.ckpt"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

DATA_SOURCES: dict[Path, str] = {
    RAW_ROOT / "heart_disease" / "processed.cleveland.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    ),
    RAW_ROOT / "adult_income" / "adult.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "adult/adult.data"
    ),
    RAW_ROOT / "adult_income" / "adult.test": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "adult/adult.test"
    ),
    RAW_ROOT / "credit_default" / "default of credit card clients.xls": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00350/default%20of%20credit%20card%20clients.xls"
    ),
    RAW_ROOT / "banknote_authentication" / "data_banknote_authentication.txt": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00267/data_banknote_authentication.txt"
    ),
    RAW_ROOT
    / "breast_cancer_wisconsin_diagnostic"
    / "wdbc.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "breast-cancer-wisconsin/wdbc.data"
    ),
    RAW_ROOT / "haberman_survival" / "haberman.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "haberman/haberman.data"
    ),
    RAW_ROOT / "ionosphere" / "ionosphere.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "ionosphere/ionosphere.data"
    ),
    RAW_ROOT / "mushroom" / "agaricus-lepiota.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "mushroom/agaricus-lepiota.data"
    ),
    RAW_ROOT / "pima_diabetes" / "pima-indians-diabetes.data": (
        "https://www.openml.org/data/get_csv/37/"
        "dataset_37_diabetes.arff"
    ),
    RAW_ROOT / "sonar_mines_rocks" / "sonar.all-data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "undocumented/connectionist-bench/sonar/sonar.all-data"
    ),
}


@dataclass
class StepResult:
    """One pipeline step result for final summary logging."""

    name: str
    ok: bool
    elapsed_sec: float


def _run_command(label: str, args: list[str]) -> None:
    """Run one subprocess step from project root and fail fast on errors."""
    print(f"\n[STEP] {label}")
    print("[CMD] " + " ".join(args))
    started = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Step failed: {label} (exit={completed.returncode}, elapsed={elapsed:.1f}s)"
        )
    print(f"[OK] {label} ({elapsed:.1f}s)")


def _parse_requirement_line(line: str) -> tuple[str, str | None] | None:
    """Parse one requirements line into package name and exact pinned version."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("--"):
        return None

    exact_match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s]+)$", stripped)
    if exact_match:
        return exact_match.group(1), exact_match.group(2)

    name_match = re.match(r"^([A-Za-z0-9_.\-]+)", stripped)
    if name_match:
        return name_match.group(1), None
    return None


def _missing_requirements(requirements_path: Path) -> list[str]:
    """Return list of missing or version-mismatched requirements."""
    missing: list[str] = []
    lines = requirements_path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        parsed = _parse_requirement_line(line)
        if parsed is None:
            continue
        package_name, pinned = parsed

        try:
            installed_version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            missing.append(f"{package_name} (not installed)")
            continue

        if pinned is not None and installed_version != pinned:
            missing.append(
                f"{package_name} (installed={installed_version}, required={pinned})"
            )

    return missing


def _ensure_requirements(auto_install: bool) -> None:
    """Validate requirements and optionally install/upgrade missing entries."""
    if not REQUIREMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS_PATH}")

    missing = _missing_requirements(REQUIREMENTS_PATH)
    if not missing:
        print("[OK] Requirements check passed.")
        return

    print("[WARN] Requirement issues detected:")
    for item in missing:
        print(f"  - {item}")

    if not auto_install:
        raise RuntimeError(
            "Missing requirements found and auto-install is disabled. "
            "Re-run with --auto-install-requirements."
        )

    if shutil.which("uv") is not None:
        _run_command(
            "Install/upgrade requirements with uv",
            ["uv", "pip", "install", "-r", str(REQUIREMENTS_PATH)],
        )
    else:
        _run_command(
            "Install/upgrade requirements with pip",
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)],
        )

    missing_after = _missing_requirements(REQUIREMENTS_PATH)
    if missing_after:
        raise RuntimeError(
            "Requirements still missing after installation: "
            + ", ".join(missing_after)
        )
    print("[OK] Requirements are now satisfied.")


def _download_file(url: str, destination: Path) -> None:
    """Download one file with streaming writes and basic size validation."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded file is empty: {destination}")


def _ensure_datasets(auto_download: bool) -> None:
    """Ensure required raw dataset files are present, optionally downloading."""
    missing = [path for path in DATA_SOURCES if not path.exists()]
    if not missing:
        print("[OK] All required dataset files are present.")
        return

    if not auto_download:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            "Missing dataset files and auto-download is disabled:\n" + missing_text
        )

    print("[STEP] Download missing dataset files")
    for path in missing:
        url = DATA_SOURCES[path]
        print(f"[DOWNLOAD] {url} -> {path}")
        _download_file(url, path)
    print("[OK] Dataset download completed.")


def _ensure_checkpoint(auto_download: bool) -> None:
    """Ensure local HyperFast checkpoint exists, optionally downloading."""
    if CHECKPOINT_PATH.exists() and CHECKPOINT_PATH.stat().st_size > 0:
        print("[OK] HyperFast checkpoint is present.")
        return

    if not auto_download:
        raise RuntimeError(
            "HyperFast checkpoint is missing and auto-download is disabled."
        )

    print("[STEP] Download HyperFast checkpoint")
    from download_hyperfast_checkpoint import download_checkpoint

    download_checkpoint(CHECKPOINT_PATH)
    print("[OK] HyperFast checkpoint downloaded.")


def _expected_split_paths() -> list[Path]:
    """Compute expected split JSON paths from split_config."""
    split_config_path = CONFIG_ROOT / "split_config.json"
    payload = json.loads(split_config_path.read_text(encoding="utf-8"))
    datasets = [str(name) for name in payload["datasets"]]
    seeds = [int(seed) for seed in payload["seeds"]]
    return [
        DATA_ROOT / "splits" / f"{dataset}_seed{seed}.json"
        for dataset in datasets
        for seed in seeds
    ]


def _ensure_splits(auto_generate: bool) -> None:
    """Ensure configured split files exist, optionally generating all splits."""
    expected_paths = _expected_split_paths()
    missing_paths = [path for path in expected_paths if not path.exists()]

    if not missing_paths:
        print("[OK] Split files are present.")
        return

    if not auto_generate:
        raise RuntimeError(
            "Split files are missing and auto-generation is disabled."
        )

    _run_command(
        "Generate deterministic splits",
        [sys.executable, str(SRC_ROOT / "generate_splits.py")],
    )

    missing_after = [path for path in expected_paths if not path.exists()]
    if missing_after:
        raise RuntimeError("Split generation incomplete; some files are still missing.")


def _detect_active_full_run_windows() -> list[dict[str, Any]]:
    """Detect active run_full_comparison.py processes on Windows."""
    if os.name != "nt":
        return []

    ps_query = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python|pythonw' -and "
        "$_.CommandLine -match 'run_full_comparison.py' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_query],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    payload = result.stdout.strip()
    if not payload:
        return []

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _guard_concurrent_full_run(allow_concurrent_run: bool) -> None:
    """Prevent concurrent writes to runs/results by default."""
    active = _detect_active_full_run_windows()
    if not active:
        return

    if allow_concurrent_run:
        print("[WARN] Active full comparison process detected, continuing by request.")
        return

    detail_lines = [
        f"  - PID={item.get('ProcessId')} | {item.get('CommandLine', '')}"
        for item in active
    ]
    raise RuntimeError(
        "Detected active run_full_comparison.py process(es). "
        "To avoid artifact corruption, this script will not run concurrently.\n"
        + "\n".join(detail_lines)
        + "\nRe-run with --allow-concurrent-run only if you accept concurrent writes."
    )


def run_pipeline(args: argparse.Namespace) -> list[StepResult]:
    """Execute end-to-end pipeline with preflight and post-run validation."""
    summary: list[StepResult] = []

    def run_step(name: str, fn: Any) -> None:
        start = time.perf_counter()
        try:
            fn()
            summary.append(StepResult(name=name, ok=True, elapsed_sec=time.perf_counter() - start))
        except Exception:
            summary.append(StepResult(name=name, ok=False, elapsed_sec=time.perf_counter() - start))
            raise

    run_step(
        "concurrency_guard",
        lambda: _guard_concurrent_full_run(args.allow_concurrent_run),
    )
    run_step(
        "requirements_check",
        lambda: _ensure_requirements(args.auto_install_requirements),
    )
    run_step(
        "dataset_check_or_download",
        lambda: _ensure_datasets(args.auto_download_data),
    )
    run_step(
        "checkpoint_check_or_download",
        lambda: _ensure_checkpoint(args.auto_download_checkpoint),
    )
    run_step(
        "split_check_or_generate",
        lambda: _ensure_splits(args.auto_generate_splits),
    )
    run_step(
        "run_full_comparison",
        lambda: _run_command(
            "Run full comparison",
            [sys.executable, str(SRC_ROOT / "run_full_comparison.py")],
        ),
    )
    run_step(
        "generate_analysis_artifacts",
        lambda: _run_command(
            "Generate analysis artifacts",
            [sys.executable, str(SRC_ROOT / "generate_analysis_artifacts.py")],
        ),
    )
    run_step(
        "verify_artifact_lineage",
        lambda: _run_command(
            "Verify artifact lineage",
            [sys.executable, str(SRC_ROOT / "verify_artifact_lineage.py")],
        ),
    )
    run_step(
        "validate_research_integrity",
        lambda: _run_command(
            "Validate research integrity",
            [sys.executable, str(SRC_ROOT / "validate_research_integrity.py")],
        ),
    )

    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for all-in-one pipeline options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-concurrent-run",
        action="store_true",
        help="Allow running even when another run_full_comparison.py is active.",
    )

    # Backward-compatible enable flags (kept hidden), plus explicit disable flags.
    parser.add_argument(
        "--auto-install-requirements",
        dest="auto_install_requirements",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-auto-install-requirements",
        dest="auto_install_requirements",
        action="store_false",
        help="Do not auto-install missing packages from requirements.txt.",
    )
    parser.add_argument(
        "--auto-download-data",
        dest="auto_download_data",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-auto-download-data",
        dest="auto_download_data",
        action="store_false",
        help="Do not auto-download missing dataset files.",
    )
    parser.add_argument(
        "--auto-download-checkpoint",
        dest="auto_download_checkpoint",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-auto-download-checkpoint",
        dest="auto_download_checkpoint",
        action="store_false",
        help="Do not auto-download missing hyperfast.ckpt.",
    )
    parser.add_argument(
        "--auto-generate-splits",
        dest="auto_generate_splits",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-auto-generate-splits",
        dest="auto_generate_splits",
        action="store_false",
        help="Do not auto-generate missing split files.",
    )

    parser.set_defaults(
        auto_install_requirements=True,
        auto_download_data=True,
        auto_download_checkpoint=True,
        auto_generate_splits=True,
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    print("[INFO] All-in-one pipeline runner starting...")
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(
        "[INFO] Auto mode: "
        f"requirements={args.auto_install_requirements}, "
        f"data={args.auto_download_data}, "
        f"checkpoint={args.auto_download_checkpoint}, "
        f"splits={args.auto_generate_splits}"
    )

    started = time.perf_counter()
    summary: list[StepResult] = []
    try:
        summary = run_pipeline(args)
    finally:
        print("\n[SUMMARY]")
        for item in summary:
            status = "OK" if item.ok else "FAIL"
            print(f"- {item.name}: {status} ({item.elapsed_sec:.1f}s)")
        print(f"[INFO] Total elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
