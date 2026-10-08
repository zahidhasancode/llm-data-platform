"""
End-to-end tests for the dataset pipeline (src/data/pipeline.py) and the
training pipeline (src/training/pipeline.py) on the toy example in examples/.

Training and evaluation are simulated in this repo (no model download, no GPU),
so these tests run the real code with no mocking.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.data.pipeline import build_dataset_from_config
from src.evaluation.evaluator import evaluate_model
from src.training.pipeline import run_training_pipeline
from src.training.registry import get_model
from src.training.trainer import train_model

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A copy of examples/ and configs/ in a temp dir, used as the working directory.

    Both pipelines resolve artifacts/ relative to the current directory.
    """
    shutil.copytree(REPO_ROOT / "examples", tmp_path / "examples")
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_dataset_pipeline_on_example(workdir):
    version_path = Path(build_dataset_from_config("configs/datasets/example_support.yaml"))
    assert version_path == Path("artifacts/datasets/support_example_v1")

    rows = _read_jsonl(version_path / "data.jsonl")
    # 9 rows in: one duplicate, one empty answer, one too short ("Hi"), one noisy row are dropped.
    assert [r["id"] for r in rows] == [
        "support_example_0",
        "support_example_2",
        "support_example_3",
        "support_example_6",
        "support_example_8",
    ]
    meta = json.loads((version_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["num_samples"] == 5
    assert meta["config"]["input_path"] == "examples/support_sample.csv"


def test_dataset_pipeline_is_reproducible(workdir):
    first = Path(build_dataset_from_config("configs/datasets/example_support.yaml"))
    first_bytes = {n: (first / n).read_bytes() for n in ("data.jsonl", "metadata.json")}
    second = Path(build_dataset_from_config("configs/datasets/example_support.yaml"))
    assert {n: (second / n).read_bytes() for n in first_bytes} == first_bytes


def test_dataset_pipeline_honours_output_dir(workdir):
    cfg = workdir / "custom.yaml"
    cfg.write_text(
        "source: s\ninput_path: examples/support_sample.csv\nversion_name: raw\noutput_dir: out/ds\n",
        encoding="utf-8",
    )
    path = Path(build_dataset_from_config(str(cfg)))
    assert path == Path("out/ds/raw")
    # No min_length / duplicate / noise rules: only the empty-answer row goes.
    assert json.loads((path / "metadata.json").read_text(encoding="utf-8"))["num_samples"] == 8


def test_dataset_pipeline_missing_config(workdir):
    with pytest.raises(FileNotFoundError):
        build_dataset_from_config("configs/datasets/does_not_exist.yaml")


def test_trainer_requires_the_dataset_version(workdir):
    with pytest.raises(FileNotFoundError, match="Dataset version not found"):
        train_model("configs/training/example_finetune.yaml", "model_v1")


def test_training_pipeline_on_example(workdir):
    build_dataset_from_config("configs/datasets/example_support.yaml")
    result = run_training_pipeline("configs/training/example_finetune.yaml", "model_v1")

    assert result["model_version"] == "model_v1"
    assert result["dataset_version"] == "support_example_v1"
    metrics = result["metrics"]
    assert set(metrics) == {"quality_score", "latency_ms", "cost_per_1k_tokens"}
    assert 0.5 <= metrics["quality_score"] < 1.0
    assert 20 <= metrics["latency_ms"] < 100
    assert 0.01 <= metrics["cost_per_1k_tokens"] < 0.02

    model_meta = json.loads(Path("artifacts/models/model_v1/metadata.json").read_text(encoding="utf-8"))
    assert model_meta["num_training_samples"] == 5
    assert model_meta["training_config"]["learning_rate"] == pytest.approx(2e-5)
    assert get_model("model_v1", "artifacts/models/registry.json") == model_meta
    assert Path("artifacts/evaluations/model_v1/evaluation.json").exists()


def test_simulated_metrics_are_deterministic(workdir):
    build_dataset_from_config("configs/datasets/example_support.yaml")
    first = run_training_pipeline("configs/training/example_finetune.yaml", "model_v1")
    second = run_training_pipeline("configs/training/example_finetune.yaml", "model_v1")
    assert first == second
    other = run_training_pipeline("configs/training/example_finetune.yaml", "model_v2")
    assert other["metrics"] != first["metrics"]


def test_evaluator_requires_model_metadata(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model metadata not found"):
        evaluate_model("ghost", models_dir=str(tmp_path / "models"), output_dir=str(tmp_path / "evals"))
