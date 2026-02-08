"""
Model evaluation module for the LLM Data Platform.
Loads model metadata and produces deterministic evaluation artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_model_metadata(model_version: str, models_dir: str) -> dict:
    """
    Load metadata.json for the given model version and return it as a dict.
    models_dir is the base path (e.g. artifacts/models); metadata is at
    <models_dir>/<model_version>/metadata.json
    """
    path = Path(models_dir) / model_version / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _simulate_metrics(model_version: str, dataset_version: str) -> dict:
    """
    Produce deterministic fake metrics from model_version and dataset_version.
    Same inputs always yield the same metrics.
    """
    seed = f"{model_version}:{dataset_version}".encode("utf-8")
    h = hashlib.sha256(seed).hexdigest()
    # Use hash bytes to derive fixed floats/ints in [0, 1) or similar ranges
    quality_score = 0.5 + (int(h[:8], 16) % 5000) / 10000.0  # 0.5 - 1.0
    latency_ms = 20 + (int(h[8:16], 16) % 80)  # 20 - 100 ms
    cost_per_1k_tokens = 0.01 + (int(h[16:24], 16) % 90) / 10000.0  # 0.01 - 0.01+0.009
    return {
        "quality_score": round(quality_score, 4),
        "latency_ms": latency_ms,
        "cost_per_1k_tokens": round(cost_per_1k_tokens, 4),
    }


def evaluate_model(
    model_version: str,
    models_dir: str = "artifacts/models",
    output_dir: str = "artifacts/evaluations",
) -> str:
    """
    Load model metadata, simulate evaluation metrics, and write evaluation.json.
    Returns the path to the evaluation directory.
    """
    metadata = load_model_metadata(model_version, models_dir)
    dataset_version = metadata.get("dataset_version", "")

    metrics = _simulate_metrics(model_version, dataset_version)

    eval_path = Path(output_dir) / model_version
    eval_path.mkdir(parents=True, exist_ok=True)

    result = {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "metrics": metrics,
    }
    out_file = eval_path / "evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return str(eval_path)
