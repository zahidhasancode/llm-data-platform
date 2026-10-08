"""Tests for the JSON model registry in src/training/registry.py."""

from __future__ import annotations

import json

import pytest

from src.training.registry import get_model, list_models, register_model


def _meta(version: str, samples: int = 10) -> dict:
    return {"model_version": version, "dataset_version": "d1", "num_training_samples": samples}


def test_missing_registry_lists_nothing(tmp_path):
    assert list_models(str(tmp_path / "registry.json")) == []


def test_register_creates_file_and_parent_dirs(tmp_path):
    path = tmp_path / "models" / "nested" / "registry.json"
    register_model(_meta("m1"), str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == {"models": [_meta("m1")]}


def test_models_are_listed_in_registration_order(tmp_path):
    path = str(tmp_path / "registry.json")
    for v in ("m1", "m2", "m3"):
        register_model(_meta(v), path)
    assert [m["model_version"] for m in list_models(path)] == ["m1", "m2", "m3"]


def test_get_model_returns_metadata_or_raises(tmp_path):
    path = str(tmp_path / "registry.json")
    register_model(_meta("m1", samples=5), path)
    assert get_model("m1", path)["num_training_samples"] == 5
    with pytest.raises(KeyError):
        get_model("missing", path)


def test_reregistering_a_version_replaces_its_entry(tmp_path):
    path = str(tmp_path / "registry.json")
    register_model(_meta("m1", samples=5), path)
    register_model(_meta("m2"), path)
    register_model(_meta("m1", samples=99), path)
    models = list_models(path)
    assert [m["model_version"] for m in models] == ["m1", "m2"]
    assert get_model("m1", path)["num_training_samples"] == 99


@pytest.mark.parametrize("content", ['[]', '{"other": 1}', '{"models": "x"}'])
def test_malformed_registry_is_treated_as_empty(tmp_path, content):
    path = tmp_path / "registry.json"
    path.write_text(content, encoding="utf-8")
    assert list_models(str(path)) == []
