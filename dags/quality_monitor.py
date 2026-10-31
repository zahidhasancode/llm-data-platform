"""Every few minutes, check live quality; if it has dropped, start the pipeline."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import psycopg2
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

log = logging.getLogger(__name__)
WINDOW = 200
BASELINE = float(os.environ.get("VALID_SQL_BASELINE", "95"))


@dag(
    dag_id="quality_monitor",
    start_date=datetime(2026, 10, 1),
    schedule=timedelta(minutes=5),
    catchup=False,
    max_active_runs=1,
)
def quality_monitor():
    @task.short_circuit
    def quality_dropped() -> bool:
        from src.pipeline.monitor import check

        with psycopg2.connect(os.environ["EVAL_DATABASE_URL"]) as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT (metadata->>'valid_sql')::boolean
                FROM llm_responses
                WHERE metadata ? 'valid_sql'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (WINDOW,),
            )
            flags = [row[0] for row in cursor.fetchall()]
        reading = check(flags, BASELINE)
        log.info("Quality reading: %s", reading)
        return reading.dropped

    retrain = TriggerDagRunOperator(
        task_id="start_pipeline",
        trigger_dag_id="text2sql_pipeline",
        conf={"run": "{{ var.value.get('next_run', 'smoke') }}"},
    )

    quality_dropped() >> retrain


quality_monitor()
