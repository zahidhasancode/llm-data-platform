"""
Trainer module for the LLM Data Platform.
Platform abstraction for training runs; uses config and dataset artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import TrainingConfig, load_training_config

DATASETS_BASE = "artifacts/datasets"


def count_dataset_samples(dataset_path: str) -> int:
    """
    Count the number of samples in a versioned dataset by reading data.jsonl.
    dataset_path is the path to the dataset version folder (containing data.jsonl).
    """
    path = Path(dataset_path) / "data.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset data file not found: {path}")
    count = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def train_model(
    training_config_path: str,
    model_version: str,
    artifacts_dir: str = "artifacts/models",
) -> str:
    """
    Run a training job from a config: load config, locate dataset, count samples,
    simulate training, and write model artifact metadata.
    Returns the path to the model version directory.
    """
    config = load_training_config(training_config_path)

    dataset_path = Path(DATASETS_BASE) / config.dataset_version
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset version not found: {dataset_path}")

    num_training_samples = count_dataset_samples(str(dataset_path))

    # Simulate training run (no real training)
    _simulate_training(config, num_training_samples)

    output_path = Path(artifacts_dir) / model_version
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_version": model_version,
        "base_model": config.base_model,
        "dataset_version": config.dataset_version,
        "training_config": asdict(config),
        "num_training_samples": num_training_samples,
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return str(output_path)


def _simulate_training(config: TrainingConfig, num_samples: int) -> None:
    """Simulate a training run; no real training. Deterministic no-op."""
    _ = config.epochs * (num_samples // max(config.batch_size, 1))
    return None
