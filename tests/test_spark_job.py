"""Spark job: rejected rows, split safety and repeatable hashes."""

import json
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

from src.data.spark_job import run, start_spark  # noqa: E402


def row(q: object, sql: object = "SELECT 1", context: object = "CREATE TABLE t (a INT)") -> dict:
    return {"question": q, "context": context, "sql": sql}


@pytest.fixture(scope="module")
def spark():
    session = start_spark()
    yield session
    session.stop()


def write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


ROWS = [row(f"Question number {i}?") for i in range(200)]


def test_bad_rows_are_rejected_with_a_reason(spark, tmp_path: Path) -> None:
    source = write(
        tmp_path / "in.jsonl",
        ROWS + [row(""), row("No SQL here?", sql="  "), row("Big?", context="x" * 100)],
    )
    meta = run(spark, source, str(tmp_path / "out"), max_chars=60)
    assert meta["rejected"]["missing_question"] == 1
    assert meta["rejected"]["missing_sql"] == 1
    assert meta["rejected"]["too_long"] == 1
    reasons = {r["reject_reason"] for r in read(tmp_path / "out" / "rejected.jsonl")}
    assert {"missing_question", "missing_sql", "too_long"} <= reasons


def test_duplicates_ignore_case_and_spacing(spark, tmp_path: Path) -> None:
    source = write(tmp_path / "in.jsonl", [row("What is it?"), row("what   IS it?"), row("Other?")])
    meta = run(spark, source, str(tmp_path / "out"))
    assert sum(meta["counts"].values()) == 2
    assert meta["rejected"] == {"duplicate_question": 1}


def test_same_input_gives_the_same_hash_in_any_order(spark, tmp_path: Path) -> None:
    first = run(spark, write(tmp_path / "a.jsonl", ROWS), str(tmp_path / "a"))
    second = run(spark, write(tmp_path / "b.jsonl", ROWS[::-1]), str(tmp_path / "b"))
    assert first["dataset_hash"] == second["dataset_hash"]
    assert (tmp_path / "a" / "train.jsonl").read_bytes() == (tmp_path / "b" / "train.jsonl").read_bytes()


def test_changed_data_changes_the_hash(spark, tmp_path: Path) -> None:
    first = run(spark, write(tmp_path / "a.jsonl", ROWS), str(tmp_path / "a"))
    second = run(spark, write(tmp_path / "b.jsonl", ROWS + [row("New?")]), str(tmp_path / "b"))
    assert first["dataset_hash"] != second["dataset_hash"]


def test_a_question_keeps_its_split_and_never_crosses_splits(spark, tmp_path: Path) -> None:
    run(spark, write(tmp_path / "a.jsonl", ROWS), str(tmp_path / "a"))
    run(spark, write(tmp_path / "b.jsonl", ROWS + [row(f"Extra {i}?") for i in range(50)]), str(tmp_path / "b"))
    splits = ("train", "validation", "test")
    sets = {s: {r["question"] for r in read(tmp_path / "a" / f"{s}.jsonl")} for s in splits}
    assert not (sets["train"] & sets["test"] or sets["train"] & sets["validation"])
    assert not sets["test"] & sets["validation"]
    for s in splits:
        after = {r["question"] for r in read(tmp_path / "b" / f"{s}.jsonl")}
        assert sets[s] <= after


def test_a_source_without_a_required_column_is_refused(spark, tmp_path: Path) -> None:
    source = write(tmp_path / "in.jsonl", [{"question": "Q?", "context": "C"}])
    with pytest.raises(ValueError, match="sql"):
        run(spark, source, str(tmp_path / "out"))
