"""HyperFast model builders and wrappers."""

from __future__ import annotations

from pathlib import Path

import torch
from hyperfast import HyperFastClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHECKPOINT = PROJECT_ROOT / "hyperfast.ckpt"


def build_hyperfast_default(seed: int) -> HyperFastClassifier:
    """Build the fast/default-like HyperFast operating point."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    custom_path = str(LOCAL_CHECKPOINT) if LOCAL_CHECKPOINT.exists() else None
    return HyperFastClassifier(
        device=device,
        n_ensemble=1,
        optimization=None,
        stratify_sampling=False,
        feature_bagging=False,
        seed=seed,
        custom_path=custom_path,
    )
