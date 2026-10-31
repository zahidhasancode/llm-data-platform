"""Spark job: clean, deduplicate, split and version a JSONL dataset.

Rows that fail a check go to `rejected.jsonl` with a reason, so nothing disappears silently.
The split comes from a hash of the question, so a row keeps its split when other rows change,
and the same question can never appear in two splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

REQUIRED = ("question", "context", "sql")
SPLITS = ("train", "validation", "test")


def start_spark() -> SparkSession:
    return (
        SparkSession.builder.master("local[2]")
        .appName("llm-data-pipeline")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def _clean(df: DataFrame) -> DataFrame:
    for column in REQUIRED:
        df = df.withColumn(column, F.trim(F.col(column)))
    return df


def _with_reject_reason(df: DataFrame, max_chars: int) -> DataFrame:
    missing = F.lit(None).cast("string")
    for column in reversed(REQUIRED):
        missing = F.when(
            F.col(column).isNull() | (F.col(column) == ""), F.lit(f"missing_{column}")
        ).otherwise(missing)
    too_long = F.length("question") + F.length("context") + F.length("sql") > max_chars
    return df.withColumn(
        "reject_reason",
        F.coalesce(missing, F.when(too_long, F.lit("too_long"))),
    )


def _mark_duplicates(df: DataFrame) -> DataFrame:
    """The first row of each normalised question keeps its place; later copies are rejected."""
    key = F.lower(F.regexp_replace("question", r"\s+", " "))
    order = Window.partitionBy(key).orderBy("sql", "context")
    return df.withColumn("_rank", F.row_number().over(order)).withColumn(
        "reject_reason",
        F.coalesce(
            F.col("reject_reason"),
            F.when(F.col("_rank") > 1, F.lit("duplicate_question")),
        ),
    ).drop("_rank")


def _with_split(df: DataFrame, validation_pct: int, test_pct: int) -> DataFrame:
    key = F.lower(F.regexp_replace("question", r"\s+", " "))
    bucket = F.conv(F.substring(F.sha2(key, 256), 1, 8), 16, 10).cast("long") % 100
    return df.withColumn(
        "split",
        F.when(bucket < test_pct, "test")
        .when(bucket < test_pct + validation_pct, "validation")
        .otherwise("train"),
    )


def _write_jsonl(rows: list[dict], path: Path) -> str:
    """Write sorted rows as canonical JSON and return their SHA-256."""
    lines = [
        json.dumps(r, sort_keys=True, ensure_ascii=False)
        for r in sorted(rows, key=lambda r: (r["question"] or "", r["sql"] or "", r["context"] or ""))
    ]
    body = "".join(line + "\n" for line in lines).encode("utf-8")
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def run(
    spark: SparkSession,
    source: str,
    output: str,
    *,
    validation_pct: int = 5,
    test_pct: int = 10,
    max_chars: int = 20_000,
) -> dict:
    """Build a dataset version in `output` and return its metadata."""
    raw = spark.read.json(source)
    absent = [c for c in REQUIRED if c not in raw.columns]
    if absent:
        raise ValueError(f"Source has no column: {', '.join(absent)}")

    checked = _mark_duplicates(_with_reject_reason(_clean(raw.select(*REQUIRED)), max_chars))
    good = _with_split(checked.filter(F.col("reject_reason").isNull()), validation_pct, test_pct)
    bad = checked.filter(F.col("reject_reason").isNotNull())

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r.asDict() for r in good.select(*REQUIRED, "split").toLocalIterator()]
    rejected = [r.asDict() for r in bad.toLocalIterator()]

    hashes, counts = {}, {}
    for split in SPLITS:
        chosen = [{c: r[c] for c in REQUIRED} for r in rows if r["split"] == split]
        hashes[split] = _write_jsonl(chosen, out / f"{split}.jsonl")
        counts[split] = len(chosen)
    _write_jsonl(rejected, out / "rejected.jsonl")

    reasons: dict[str, int] = {}
    for r in rejected:
        reasons[r["reject_reason"]] = reasons.get(r["reject_reason"], 0) + 1

    combined = hashlib.sha256("".join(hashes[s] for s in SPLITS).encode()).hexdigest()
    metadata = {
        "source": Path(source).name,
        "dataset_hash": combined,
        "split_hashes": hashes,
        "counts": counts,
        "rejected": dict(sorted(reasons.items())),
        "settings": {
            "validation_pct": validation_pct,
            "test_pct": test_pct,
            "max_chars": max_chars,
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    spark = start_spark()
    try:
        print(json.dumps(run(spark, args.source, args.output), indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
