"""Pipeline: take a finished training run, record it, gate it, and promote it if it passes.

Training runs on Kaggle, so this DAG starts from the results that run wrote to S3. Each stage
retries, and a failure raises an alert in the task log.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

log = logging.getLogger(__name__)
BUCKET = os.environ["ARTIFACT_BUCKET"]
TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]


def alert(context: dict) -> None:
    log.error("ALERT: task %s failed: %s", context["task_instance"].task_id, context.get("exception"))


@dag(
    dag_id="text2sql_pipeline",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    params={"run": "smoke"},
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
        "on_failure_callback": alert,
    },
)
def text2sql_pipeline():
    @task
    def fetch(params: dict | None = None) -> str:
        """Copy one run's results and adapter from S3."""
        run = params["run"]
        target = Path("/tmp/runs") / run
        bucket = boto3.resource("s3").Bucket(BUCKET)
        for item in bucket.objects.filter(Prefix=f"runs/{run}/"):
            path = target / item.key.removeprefix(f"runs/{run}/")
            path.parent.mkdir(parents=True, exist_ok=True)
            bucket.download_file(item.key, str(path))
        if not (target / "results" / "metrics.json").exists():
            raise FileNotFoundError(f"No results in s3://{BUCKET}/runs/{run}")
        return str(target)

    @task
    def record(folder: str, params: dict | None = None) -> dict:
        from src.finetune.tracking import log_run

        return log_run(f"{folder}/results", f"{folder}/adapter", TRACKING_URI, params["run"])

    @task
    def gate(folder: str, logged: dict) -> dict:
        from src.finetune.tracking import champion_accuracy
        from src.pipeline.gate import decide

        report = json.loads(Path(folder, "results", "metrics.json").read_text())
        decision = decide(report, champion_accuracy(TRACKING_URI))
        log.info("Gate for version %s: %s", logged["model_version"], decision)
        if not decision.promote:
            raise AirflowSkipException(f"Blocked: {'; '.join(decision.reasons)}")
        return logged

    @task
    def promote_model(logged: dict) -> None:
        from src.finetune.tracking import promote

        promote(TRACKING_URI, logged["model_version"])
        log.info("Version %s is now the champion", logged["model_version"])

    folder = fetch()
    logged = record(folder)
    promote_model(gate(folder, logged))


text2sql_pipeline()
