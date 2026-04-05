"""Download HyperFast checkpoint to a local file for offline reuse."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "hyperfast.ckpt"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/43484094"
CHECKPOINT_SHA256_ENV = "HYPERFAST_CHECKPOINT_SHA256"


def download_checkpoint(target_path: Path = CHECKPOINT_PATH) -> Path:
    """Download checkpoint with streaming and basic validation."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()

    with requests.get(DOWNLOAD_URL, stream=True, timeout=None) as response:
        response.raise_for_status()
        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    hasher.update(chunk)

    size = target_path.stat().st_size
    if size <= 0:
        raise RuntimeError("Downloaded checkpoint is empty.")

    sha256_hex = hasher.hexdigest()
    expected_sha256 = os.environ.get(CHECKPOINT_SHA256_ENV)
    if expected_sha256:
        normalized_expected = expected_sha256.strip().lower()
        if sha256_hex.lower() != normalized_expected:
            raise RuntimeError(
                "Checkpoint SHA256 mismatch: "
                f"expected={normalized_expected}, actual={sha256_hex}"
            )

    print(f"Saved checkpoint: {target_path}")
    print(f"Size bytes: {size}")
    print(f"SHA256: {sha256_hex}")
    return target_path


def main() -> None:
    """CLI entry point."""
    download_checkpoint()


if __name__ == "__main__":
    main()
