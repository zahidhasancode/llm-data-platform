"""
End-to-end training pipeline for the LLM Data Platform.
Runs training, registration, and evaluation in sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..evaluation.evaluator import evaluate_model
from . import registry
from . import trainer

MODELS_DIR = "artifacts/models"
REGISTRY_PATH = "artifacts/models/registry.json"
EVALUATIONS_DIR = "artifacts/evaluations"


def load_model_metadata(model_version: str, models_dir: str = MODELS_DIR) -> dict:
    """
    Read metadata.json from the model folder and return it as a dictionary.
    """
    path = Path(models_dir) / model_version / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_training_pipeline(
    training_config_path: str,
    model_version: str,
) -> dict:
    """
    Run the full pipeline: train, register, evaluate; return evaluation results.
    """
    trainer.train_model(
        training_config_path,
        model_version,
        artifacts_dir=MODELS_DIR,
    )

    metadata = load_model_metadata(model_version, MODELS_DIR)
    registry.register_model(metadata, REGISTRY_PATH)

    evaluate_model(
        model_version,
        models_dir=MODELS_DIR,
        output_dir=EVALUATIONS_DIR,
    )

    eval_path = Path(EVALUATIONS_DIR) / model_version / "evaluation.json"
    with open(eval_path, encoding="utf-8") as f:
        evaluation_results = json.load(f)

    return evaluation_results
