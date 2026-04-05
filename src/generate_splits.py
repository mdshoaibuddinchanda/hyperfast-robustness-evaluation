"""Generate fixed stratified train/val/test splits for robustness experiments."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loading import build_manifest_entry, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "split_config.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


@dataclass(frozen=True)
class SplitConfig:
    """Runtime configuration for deterministic split generation."""

    datasets: list[str]
    test_size: float
    val_size: float
    test_seed: int
    seeds: list[int]


def _read_config(path: Path) -> SplitConfig:
    """Load split configuration from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitConfig(
        datasets=list(payload["datasets"]),
        test_size=float(payload["test_size"]),
        val_size=float(payload["val_size"]),
        test_seed=int(payload["test_seed"]),
        seeds=[int(seed) for seed in payload["seeds"]],
    )


def _class_distribution(labels: np.ndarray) -> dict[str, int]:
    """Return class counts as a JSON-safe dictionary."""
    unique_values, counts = np.unique(labels, return_counts=True)
    return {
        str(int(label)): int(count)
        for label, count in zip(unique_values, counts, strict=False)
    }


def _validate_split(
    all_indices: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Ensure split partitions are disjoint and complete."""
    if np.intersect1d(train_indices, val_indices).size != 0:
        raise ValueError("Train/validation overlap detected.")
    if np.intersect1d(train_indices, test_indices).size != 0:
        raise ValueError("Train/test overlap detected.")
    if np.intersect1d(val_indices, test_indices).size != 0:
        raise ValueError("Validation/test overlap detected.")

    combined = np.concatenate([train_indices, val_indices, test_indices])
    unique_combined = np.unique(combined)
    if unique_combined.size != all_indices.size:
        raise ValueError("Split does not cover all dataset rows exactly once.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON artifact with deterministic key order and indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def generate_splits(config: SplitConfig) -> None:
    """Generate split files and dataset metadata for all configured datasets."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest_payload: dict[str, Any] = {
        "split_policy": {
            "test_size": config.test_size,
            "val_size": config.val_size,
            "test_seed": config.test_seed,
            "seeds": config.seeds,
        },
        "datasets": {},
    }

    for dataset_name in config.datasets:
        features, labels, spec = load_dataset(dataset_name)
        labels_array = labels.to_numpy(dtype=int)
        all_indices = np.arange(labels_array.shape[0], dtype=int)

        train_val_indices, test_indices = train_test_split(
            all_indices,
            test_size=config.test_size,
            random_state=config.test_seed,
            stratify=labels_array,
        )

        val_ratio_within_train = config.val_size / (1.0 - config.test_size)

        manifest_payload["datasets"][dataset_name] = build_manifest_entry(
            spec=spec,
            features=features,
            labels=labels,
        )

        for seed in config.seeds:
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_ratio_within_train,
                random_state=seed,
                stratify=labels_array[train_val_indices],
            )

            _validate_split(
                all_indices=all_indices,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )

            payload = {
                "dataset": dataset_name,
                "seed": seed,
                "split_policy": {
                    "test_size": config.test_size,
                    "val_size": config.val_size,
                    "test_seed": config.test_seed,
                },
                "indices": {
                    "train": train_indices.tolist(),
                    "val": val_indices.tolist(),
                    "test": test_indices.tolist(),
                },
                "class_distribution": {
                    "full": _class_distribution(labels_array),
                    "train": _class_distribution(labels_array[train_indices]),
                    "val": _class_distribution(labels_array[val_indices]),
                    "test": _class_distribution(labels_array[test_indices]),
                },
            }

            split_path = SPLITS_DIR / f"{dataset_name}_seed{seed}.json"
            _write_json(split_path, payload)

    _write_json(METADATA_DIR / "dataset_manifest.json", manifest_payload)


def main() -> None:
    """CLI entry point for split generation."""
    config = _read_config(CONFIG_PATH)
    generate_splits(config)
    print(f"Wrote split files to: {SPLITS_DIR}")
    print(f"Wrote dataset manifest to: {METADATA_DIR / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
