"""Record finished runs in MLflow, and tie each model version to its exact data.

A run needs `train_log.json` and `metrics.json`, both written by real training and evaluation.
Nothing here computes a metric, so MLflow can only show what was measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

EXPERIMENT = "text2sql-qlora"
MODEL_NAME = "text2sql-qlora"
CHAMPION = "champion"
PARAMS = (
    "base_model",
    "dataset_hash",
    "train_examples",
    "val_examples",
    "epochs",
    "learning_rate",
    "lora_rank",
    "effective_batch",
    "max_len",
    "precision",
    "gpu",
)


def _load(results_dir: Path) -> tuple[dict, dict]:
    train = json.loads((results_dir / "train_log.json").read_text(encoding="utf-8"))
    report = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    if train.get("dataset_hash") != report.get("dataset_hash"):
        raise ValueError("Training and evaluation used different datasets")
    if not train.get("dataset_hash"):
        raise ValueError("Results have no dataset hash, so the model cannot be traced to its data")
    return train, report


def log_run(results_dir: str, adapter_dir: str, tracking_uri: str, run_name: str) -> dict:
    """Log one finished run, register its adapter, and return its ids."""
    train, report = _load(Path(results_dir))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: train[k] for k in PARAMS})
        mlflow.log_metric("train_minutes", train["train_minutes"])
        for entry in train.get("log_history", []):
            step = entry.get("step", 0)
            if "loss" in entry:
                mlflow.log_metric("train_loss", entry["loss"], step=step)
            if "eval_loss" in entry:
                mlflow.log_metric("eval_loss", entry["eval_loss"], step=step)
        for condition, metrics in report["conditions"].items():
            for key in ("execution_accuracy", "exact_match", "valid_sql_rate"):
                mlflow.log_metric(f"{condition}_{key}", metrics[key])
        mlflow.log_artifacts(results_dir, "results")
        mlflow.log_artifacts(adapter_dir, "adapter")
        run_id = run.info.run_id
        source = f"{run.info.artifact_uri}/adapter"

    if not client.search_registered_models(filter_string=f"name='{MODEL_NAME}'"):
        client.create_registered_model(MODEL_NAME)
    version = client.create_model_version(
        MODEL_NAME, source, run_id=run_id, tags={"dataset_hash": train["dataset_hash"]}
    )
    return {"run_id": run_id, "model_version": version.version, "dataset_hash": train["dataset_hash"]}


def champion_accuracy(tracking_uri: str) -> float | None:
    """Fine-tuned accuracy of the current champion, or None if there is no champion yet."""
    client = MlflowClient(tracking_uri)
    if not client.search_registered_models(filter_string=f"name='{MODEL_NAME}'"):
        return None
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics["qlora_execution_accuracy"]


def promote(tracking_uri: str, version: int) -> None:
    """Point the champion alias at a version. The server loads whatever this alias names."""
    MlflowClient(tracking_uri).set_registered_model_alias(MODEL_NAME, CHAMPION, version)


def compare_runs(tracking_uri: str) -> list[dict]:
    """Every run in the experiment, best fine-tuned accuracy first."""
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["metrics.qlora_execution_accuracy DESC"]
    )
    return [
        {
            "run_name": r.info.run_name,
            "dataset_hash": r.data.params["dataset_hash"],
            "base": r.data.metrics["base_execution_accuracy"],
            "qlora": r.data.metrics["qlora_execution_accuracy"],
            "train_minutes": r.data.metrics["train_minutes"],
        }
        for r in runs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("adapter")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    print(json.dumps(log_run(args.results, args.adapter, args.tracking_uri, args.run_name), indent=2))


if __name__ == "__main__":
    main()
