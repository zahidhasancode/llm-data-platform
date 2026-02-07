"""
Deterministic data ingestion for the LLM Data Platform.
Loads JSON, CSV, and plain text files into a normalized Sample structure.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    id: str
    input: str
    output: str
    source: str


def _load_json(path: str, source: str) -> list[Sample]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of objects with 'input' and 'output' keys")
    samples = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"JSON item at index {i} must be an object")
        inp = item.get("input", "")
        out = item.get("output", "")
        if not isinstance(inp, str):
            inp = str(inp)
        if not isinstance(out, str):
            out = str(out)
        samples.append(
            Sample(
                id=f"{source}_{i}",
                input=inp,
                output=out,
                source=source,
            )
        )
    return samples


def _load_csv(path: str, source: str) -> list[Sample]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return []
    key_lower = {f.lower(): f for f in fieldnames}
    input_col = key_lower.get("input", fieldnames[0] if fieldnames else "")
    output_col = key_lower.get("output", fieldnames[1] if len(fieldnames) > 1 else fieldnames[0])
    if not input_col or not output_col:
        raise ValueError("CSV must have 'input' and 'output' columns or at least two columns")
    samples = []
    for i, row in enumerate(rows):
        inp = row.get(input_col, "")
        out = row.get(output_col, "")
        if inp is None:
            inp = ""
        if out is None:
            out = ""
        if not isinstance(inp, str):
            inp = str(inp)
        if not isinstance(out, str):
            out = str(out)
        samples.append(
            Sample(
                id=f"{source}_{i}",
                input=inp,
                output=out,
                source=source,
            )
        )
    return samples


def _load_text(path: str, source: str) -> list[Sample]:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    samples = []
    for i, line in enumerate(lines):
        line = line.rstrip("\n\r")
        if "\t" in line:
            parts = line.split("\t", 1)
            inp = parts[0]
            out = parts[1] if len(parts) > 1 else ""
        else:
            inp = line
            out = ""
        samples.append(
            Sample(
                id=f"{source}_{i}",
                input=inp,
                output=out,
                source=source,
            )
        )
    return samples


def load_dataset(path: str, source: str) -> list[Sample]:
    """
    Load a dataset from a file. Detects format by extension and returns
    a list of normalized Sample objects.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return _load_json(path, source)
    if suffix == ".csv":
        return _load_csv(path, source)
    if suffix in (".txt", ".text") or suffix == "":
        return _load_text(path, source)
    raise ValueError(f"Unsupported file type: {suffix}. Use .json, .csv, or .txt")
