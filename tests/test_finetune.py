"""The parts of fine-tuning that need no GPU: data checks and scoring."""

import hashlib
import json
from pathlib import Path

import pytest

from src.finetune.dataset import dataset_hash, load_split
from src.finetune.scoring import score

CONTEXT = "CREATE TABLE t (a INT); INSERT INTO t VALUES (1), (2);"
ROW = {"question": "How many rows?", "context": CONTEXT, "sql": "SELECT COUNT(*) FROM t;"}


def version(tmp_path: Path) -> Path:
    body = (json.dumps(ROW) + "\n").encode()
    (tmp_path / "test.jsonl").write_bytes(body)
    meta = {"dataset_hash": "abc", "split_hashes": {"test": hashlib.sha256(body).hexdigest()}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return tmp_path


def test_load_split_returns_the_rows(tmp_path: Path) -> None:
    root = version(tmp_path)
    assert load_split(str(root), "test") == [ROW]
    assert dataset_hash(str(root)) == "abc"


def test_a_changed_file_is_refused(tmp_path: Path) -> None:
    root = version(tmp_path)
    (root / "test.jsonl").write_text(json.dumps({**ROW, "sql": "SELECT 1;"}) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        load_split(str(root), "test")


def test_score_counts_a_different_query_with_the_same_rows_as_correct() -> None:
    metrics, predictions = score([ROW], ["SELECT count(a) FROM t;"])
    assert metrics == {"n": 1, "execution_accuracy": 100.0, "exact_match": 0.0, "valid_sql_rate": 100.0}
    assert predictions[0]["exec_match"]


def test_score_marks_broken_sql() -> None:
    metrics, _ = score([ROW], ["SELEC nothing"])
    assert (metrics["execution_accuracy"], metrics["valid_sql_rate"]) == (0.0, 0.0)


def test_score_refuses_a_missing_prediction() -> None:
    with pytest.raises(ValueError):
        score([ROW], [])
