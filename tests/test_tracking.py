"""MLflow tracking: runs compare correctly and each version traces to its data."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient  # noqa: E402

from src.finetune.tracking import MODEL_NAME, champion_accuracy, compare_runs, log_run, promote  # noqa: E402


def results(root: Path, dataset: str, base: float, qlora: float) -> tuple[str, str]:
    root.mkdir()
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    result = root / "results"
    result.mkdir()
    train = {
        "base_model": "m", "dataset_hash": dataset, "train_examples": 10, "val_examples": 2,
        "epochs": 1, "learning_rate": 0.1, "lora_rank": 16, "effective_batch": 16, "max_len": 64,
        "precision": "fp16", "gpu": "T4", "train_minutes": 3.5,
        "log_history": [{"step": 10, "loss": 0.5}, {"step": 10, "eval_loss": 0.4}],
    }
    (result / "train_log.json").write_text(json.dumps(train))
    metrics = {"execution_accuracy": 0, "exact_match": 1, "valid_sql_rate": 2}
    report = {
        "dataset_hash": dataset,
        "conditions": {
            "base": {**metrics, "execution_accuracy": base},
            "qlora": {**metrics, "execution_accuracy": qlora},
        },
    }
    (result / "metrics.json").write_text(json.dumps(report))
    return str(result), str(adapter)


@pytest.fixture
def uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path}/mlflow.db"


def test_two_runs_compare_best_first(tmp_path: Path, uri: str) -> None:
    log_run(*results(tmp_path / "a", "hash-a", 60.0, 70.0), uri, "first")
    log_run(*results(tmp_path / "b", "hash-b", 60.0, 75.0), uri, "second")
    assert [r["run_name"] for r in compare_runs(uri)] == ["second", "first"]


def test_a_version_traces_to_its_dataset_and_run(tmp_path: Path, uri: str) -> None:
    logged = log_run(*results(tmp_path / "a", "hash-a", 60.0, 70.0), uri, "first")
    version = MlflowClient(uri).get_model_version(MODEL_NAME, logged["model_version"])
    assert version.tags["dataset_hash"] == "hash-a"
    assert version.run_id == logged["run_id"]


def test_each_run_gets_a_new_version(tmp_path: Path, uri: str) -> None:
    first = log_run(*results(tmp_path / "a", "h", 60.0, 70.0), uri, "first")
    second = log_run(*results(tmp_path / "b", "h", 60.0, 71.0), uri, "second")
    assert (first["model_version"], second["model_version"]) == (1, 2)


def test_results_from_different_datasets_are_refused(tmp_path: Path, uri: str) -> None:
    result, adapter = results(tmp_path / "a", "hash-a", 60.0, 70.0)
    report = json.loads(Path(result, "metrics.json").read_text())
    Path(result, "metrics.json").write_text(json.dumps({**report, "dataset_hash": "other"}))
    with pytest.raises(ValueError, match="different datasets"):
        log_run(result, adapter, uri, "bad")


def test_results_without_a_dataset_hash_are_refused(tmp_path: Path, uri: str) -> None:
    result, adapter = results(tmp_path / "a", "", 60.0, 70.0)
    with pytest.raises(ValueError, match="no dataset hash"):
        log_run(result, adapter, uri, "bad")


def test_no_champion_at_first(tmp_path: Path, uri: str) -> None:
    assert champion_accuracy(uri) is None
    log_run(*results(tmp_path / "a", "h", 60.0, 70.0), uri, "first")
    assert champion_accuracy(uri) is None


def test_promote_sets_the_champion(tmp_path: Path, uri: str) -> None:
    first = log_run(*results(tmp_path / "a", "h", 60.0, 70.0), uri, "first")
    second = log_run(*results(tmp_path / "b", "h", 60.0, 75.0), uri, "second")
    promote(uri, first["model_version"])
    assert champion_accuracy(uri) == 70.0
    promote(uri, second["model_version"])
    assert champion_accuracy(uri) == 75.0
