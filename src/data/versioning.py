"""
Deterministic dataset versioning for the LLM Data Platform.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .ingestion import Sample


def _sample_to_dict(s: Sample) -> dict:
    return {"id": s.id, "input": s.input, "output": s.output, "source": s.source}


def compute_dataset_hash(samples: list[Sample]) -> str:
    """
    Compute a stable SHA256 hash of the dataset content.
    Order of samples is preserved; same list produces same hash.
    """
    hasher = hashlib.sha256()
    for s in samples:
        line = json.dumps(_sample_to_dict(s), sort_keys=True, ensure_ascii=False) + "\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def create_dataset_version(
    samples: list[Sample],
    version_name: str,
    config: dict,
    output_dir: str = "artifacts/datasets",
) -> str:
    """
    Write a versioned dataset to disk: data.jsonl and metadata.json.
    Creates parent directories if missing. Returns the path to the version directory.
    """
    version_path = Path(output_dir) / version_name
    version_path.mkdir(parents=True, exist_ok=True)

    data_path = version_path / "data.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            line = json.dumps(_sample_to_dict(s), sort_keys=True, ensure_ascii=False) + "\n"
            f.write(line)

    dataset_hash = compute_dataset_hash(samples)
    metadata = {
        "dataset_version": version_name,
        "num_samples": len(samples),
        "config": config,
        "dataset_hash": dataset_hash,
    }
    metadata_path = version_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return str(version_path)
