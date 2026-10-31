"""Publish real events to Kafka with a chosen valid-SQL flag. Used to plant a quality drop."""

import json
import sys
import uuid

from confluent_kafka import Producer

count, valid = int(sys.argv[1]), sys.argv[2] == "valid"
producer = Producer({"bootstrap.servers": "kafka:9092", "acks": "all"})
for i in range(count):
    event = {
        "event_id": str(uuid.uuid4()),
        "prompt_name": "text2sql",
        "prompt_version": "1",
        "model_name": "text2sql-qlora",
        "model_version": "planted",
        "application_id": "quality-test",
        "input_text": f"planted question {i}",
        "response": {"status": "success", "output_text": "SELECT 1;", "metadata": {"valid_sql": valid}},
    }
    producer.produce("llm.requests", key=event["event_id"], value=json.dumps(event))
assert producer.flush(30) == 0
print(f"published {count} valid={valid}")
