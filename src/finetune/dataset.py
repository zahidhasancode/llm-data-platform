"""Load a dataset version built by the Spark job, and refuse it if the files have changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_split(version_dir: str, split: str) -> list[dict]:
    """Return one split after checking its bytes against the recorded SHA-256."""
    root = Path(version_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    body = (root / f"{split}.jsonl").read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    if actual != metadata["split_hashes"][split]:
        raise ValueError(f"{split}.jsonl does not match the hash in metadata.json")
    return [json.loads(line) for line in body.decode("utf-8").splitlines()]


def dataset_hash(version_dir: str) -> str:
    metadata = json.loads((Path(version_dir) / "metadata.json").read_text(encoding="utf-8"))
    return metadata["dataset_hash"]
