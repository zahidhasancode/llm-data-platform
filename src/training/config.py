"""
Training configuration for the LLM Data Platform.
Loads and validates YAML training configs into a typed dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REQUIRED_FIELDS = ("base_model", "dataset_version", "learning_rate", "epochs", "batch_size")


@dataclass
class TrainingConfig:
    base_model: str
    dataset_version: str
    learning_rate: float
    epochs: int
    batch_size: int

    @classmethod
    def from_dict(cls, data: dict) -> TrainingConfig:
        """Build TrainingConfig from a dict; validates required fields and types."""
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"Training config missing required fields: {missing}")

        base_model = data["base_model"]
        dataset_version = data["dataset_version"]
        learning_rate = data["learning_rate"]
        epochs = data["epochs"]
        batch_size = data["batch_size"]

        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError("base_model must be a non-empty string")
        if not isinstance(dataset_version, str) or not dataset_version.strip():
            raise ValueError("dataset_version must be a non-empty string")
        if not isinstance(learning_rate, (int, float)):
            raise ValueError("learning_rate must be a number")
        learning_rate = float(learning_rate)
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        return cls(
            base_model=base_model.strip(),
            dataset_version=dataset_version.strip(),
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )


def load_training_config(path: str) -> TrainingConfig:
    """
    Load a YAML training config from path, validate required fields, and return a TrainingConfig.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Training config must be a YAML object (dict)")

    return TrainingConfig.from_dict(data)
