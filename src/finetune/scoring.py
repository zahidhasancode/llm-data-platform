"""Score predicted SQL by running it. Plain Python, so it is tested without a GPU."""

from __future__ import annotations

from src.finetune.sqlcheck import execution_match, extract_sql, normalize, run


def is_valid(context: str, sql: str) -> bool:
    try:
        run(context, sql)
        return True
    except Exception:
        return False


def score(rows: list[dict], raw: list[str]) -> tuple[dict, list[dict]]:
    """Return summary metrics and one record per question."""
    if not rows or len(rows) != len(raw):
        raise ValueError("Need one prediction for each row, and at least one row")
    predictions = []
    for row, text in zip(rows, raw, strict=True):
        sql = extract_sql(text)
        predictions.append(
            {
                "question": row["question"],
                "gold": row["sql"],
                "pred": sql,
                "exec_match": execution_match(row["context"], row["sql"], sql),
                "exact": normalize(sql) == normalize(row["sql"]),
                "valid": is_valid(row["context"], sql),
            }
        )
    n = len(rows)
    metrics = {
        "n": n,
        "execution_accuracy": round(100 * sum(p["exec_match"] for p in predictions) / n, 1),
        "exact_match": round(100 * sum(p["exact"] for p in predictions) / n, 1),
        "valid_sql_rate": round(100 * sum(p["valid"] for p in predictions) / n, 1),
    }
    return metrics, predictions
