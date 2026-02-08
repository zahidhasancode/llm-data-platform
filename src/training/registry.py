"""
Simple model registry for the LLM Data Platform.
Stores and retrieves model metadata in a JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_registry(registry_path: str) -> dict:
    """Load registry from disk; return empty structure if file does not exist."""
    path = Path(registry_path)
    if not path.exists():
        return {"models": []}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "models" not in data:
        return {"models": []}
    if not isinstance(data["models"], list):
        return {"models": []}
    return data


def _save_registry(registry_path: str, data: dict) -> None:
    """Write registry to disk; create parent directories if missing."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def register_model(model_metadata: dict, registry_path: str) -> None:
    """
    Append a model entry to the registry and save.
    Creates the registry file and parent directories if they do not exist.
    """
    data = _load_registry(registry_path)
    data["models"].append(model_metadata)
    _save_registry(registry_path, data)


def list_models(registry_path: str) -> list:
    """
    Return the list of all registered models.
    Returns an empty list if the registry file does not exist.
    """
    data = _load_registry(registry_path)
    return list(data["models"])


def get_model(model_version: str, registry_path: str) -> dict:
    """
    Return the metadata dict for the given model_version.
    Raises KeyError if no model with that version is registered.
    """
    models = list_models(registry_path)
    for m in models:
        if isinstance(m, dict) and m.get("model_version") == model_version:
            return m
    raise KeyError(f"Model not found: {model_version}")
