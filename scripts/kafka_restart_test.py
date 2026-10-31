"""Real broker test: kill the consumer mid-stream, restart it, and count what was stored."""

import json
import os
import subprocess
import sys
import time
import uuid

import psycopg
from confluent_kafka import Producer

COUNT = 300
BROKER = "kafka:9092"
DATABASE = os.environ["TEST_DATABASE_URL"]
CONSUMER = [sys.executable, "-m", "events.consumer", "--bootstrap-servers", BROKER, "--group-id", "restart-test"]


def event(i: int) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "prompt_name": "sql",
        "prompt_version": "1",
        "model_name": "qwen",
        "model_version": "1",
        "application_id": "test",
        "input_text": f"question {i}",
        "response": {"status": "success", "output_text": "SELECT 1;"},
    }


def stored() -> int:
    with psycopg.connect(DATABASE) as conn:
        return conn.execute("select count(*) from llm_requests").fetchone()[0]


def main() -> None:
    producer = Producer({"bootstrap.servers": BROKER, "acks": "all"})
    events = [event(i) for i in range(COUNT)]
    for e in events:
        producer.produce("llm.requests", key=e["event_id"], value=json.dumps(e))
    producer.produce("llm.requests", key="bad", value=b'{"event_id": "not-a-uuid"}')
    assert producer.flush(30) == 0

    first = subprocess.Popen(CONSUMER)
    while stored() < 50:
        time.sleep(0.01)
    first.kill()
    first.wait()
    at_kill = stored()

    second = subprocess.Popen(CONSUMER)
    deadline = time.time() + 120
    while stored() < COUNT and time.time() < deadline:
        time.sleep(0.2)
    time.sleep(5)
    second.terminate()
    second.wait()

    print(json.dumps({"sent": COUNT, "stored_when_killed": at_kill, "stored_at_end": stored()}))
    if stored() != COUNT:
        raise SystemExit("rows lost or duplicated")


if __name__ == "__main__":
    main()
