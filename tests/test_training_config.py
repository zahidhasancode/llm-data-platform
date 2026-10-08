"""Tests for src/training/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.config import TrainingConfig, load_training_config

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID = {
    "base_model": "mistral-7b",
    "dataset_version": "support_v1",
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 4,
}


def test_shipped_training_config_loads():
    """The example config in configs/training must load (it uses `2e-5`)."""
    config = load_training_config(str(REPO_ROOT / "configs" / "training" / "mistral_finetune.yaml"))
    assert config.base_model == "mistral-7b"
    assert config.learning_rate == pytest.approx(2e-5)
    assert config.epochs == 3
    assert config.batch_size == 4


def test_yaml_exponent_without_dot_is_parsed_as_float(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        "base_model: m\ndataset_version: d\nlearning_rate: 1e-4\nepochs: 1\nbatch_size: 2\n",
        encoding="utf-8",
    )
    assert load_training_config(str(path)).learning_rate == pytest.approx(1e-4)


def test_from_dict_strips_and_converts():
    config = TrainingConfig.from_dict({**VALID, "base_model": "  m  ", "learning_rate": 1})
    assert config.base_model == "m"
    assert isinstance(config.learning_rate, float)


def test_missing_fields_are_reported():
    with pytest.raises(ValueError, match="missing required fields"):
        TrainingConfig.from_dict({"base_model": "m"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_model", "   "),
        ("dataset_version", ""),
        ("learning_rate", "fast"),
        ("learning_rate", True),
        ("epochs", 0),
        ("epochs", True),
        ("epochs", 2.5),
        ("batch_size", -1),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({**VALID, field: value})


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_config(str(tmp_path / "nope.yaml"))


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML object"):
        load_training_config(str(path))
