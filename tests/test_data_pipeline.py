"""
Unit tests for the LLM Data Platform data pipeline.
Run with: pytest tests/test_data_pipeline.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.data.ingestion import Sample, load_dataset
from src.data.cleaning import remove_empty_samples, remove_duplicate_samples
from src.data.versioning import create_dataset_version


# --- Ingestion tests ---


def test_load_dataset_returns_samples_json(tmp_path):
    """Load a small JSON dataset; ensure we get Sample objects."""
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps([
            {"input": "Hello", "output": "Hi there"},
            {"input": "Bye", "output": "Goodbye"},
        ]),
        encoding="utf-8",
    )
    samples = load_dataset(str(path), source="test")
    assert len(samples) == 2
    for s in samples:
        assert isinstance(s, Sample)
        assert s.source == "test"
        assert hasattr(s, "id") and hasattr(s, "input") and hasattr(s, "output")
    assert samples[0].input == "Hello" and samples[0].output == "Hi there"
    assert samples[1].input == "Bye" and samples[1].output == "Goodbye"


def test_load_dataset_returns_samples_csv(tmp_path):
    """Load a small CSV dataset; ensure we get Sample objects."""
    path = tmp_path / "data.csv"
    path.write_text("input,output\nHello,Hi there\nBye,Goodbye\n", encoding="utf-8")
    samples = load_dataset(str(path), source="test")
    assert len(samples) == 2
    for s in samples:
        assert isinstance(s, Sample)
        assert s.source == "test"
    assert samples[0].input == "Hello" and samples[0].output == "Hi there"
    assert samples[1].input == "Bye" and samples[1].output == "Goodbye"


def test_load_dataset_returns_samples_txt(tmp_path):
    """Load a small text dataset (tab-separated); ensure we get Sample objects."""
    path = tmp_path / "data.txt"
    path.write_text("Hello\tHi there\nBye\tGoodbye\n", encoding="utf-8")
    samples = load_dataset(str(path), source="test")
    assert len(samples) == 2
    for s in samples:
        assert isinstance(s, Sample)
    assert samples[0].input == "Hello" and samples[0].output == "Hi there"
    assert samples[1].input == "Bye" and samples[1].output == "Goodbye"


# --- Cleaning tests ---


def _sample(id_: str, input_: str, output: str, source: str = "test") -> Sample:
    return Sample(id=id_, input=input_, output=output, source=source)


def test_remove_empty_samples_removes_blank():
    """Cleaning removes samples with empty or blank input/output."""
    samples = [
        _sample("0", "a", "b"),
        _sample("1", "  ", "x"),
        _sample("2", "y", ""),
        _sample("3", "", "z"),
        _sample("4", "p", "q"),
    ]
    result = remove_empty_samples(samples)
    assert len(result) == 2
    assert result[0].input == "a" and result[0].output == "b"
    assert result[1].input == "p" and result[1].output == "q"


def test_remove_duplicate_samples_keeps_first():
    """Cleaning removes duplicate (input, output) pairs; first occurrence is kept."""
    samples = [
        _sample("0", "same", "out"),
        _sample("1", "other", "val"),
        _sample("2", "same", "out"),
        _sample("3", "other", "val"),
    ]
    result = remove_duplicate_samples(samples)
    assert len(result) == 2
    assert result[0].id == "0" and result[0].input == "same"
    assert result[1].id == "1" and result[1].input == "other"


# --- Versioning tests ---


def test_create_dataset_version_writes_files_and_metadata(tmp_path):
    """create_dataset_version creates data.jsonl and metadata.json with correct sample count."""
    samples = [
        _sample("0", "in1", "out1"),
        _sample("1", "in2", "out2"),
    ]
    output_dir = str(tmp_path / "artifacts" / "datasets")
    version_path = create_dataset_version(
        samples,
        version_name="test_v1",
        config={"min_length": 1},
        output_dir=output_dir,
    )
    base = Path(version_path)
    assert base.is_dir()
    data_file = base / "data.jsonl"
    meta_file = base / "metadata.json"
    assert data_file.exists()
    assert meta_file.exists()

    lines = data_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "id" in obj and "input" in obj and "output" in obj and "source" in obj

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["dataset_version"] == "test_v1"
    assert meta["num_samples"] == 2
    assert meta["config"] == {"min_length": 1}
    assert "dataset_hash" in meta and len(meta["dataset_hash"]) == 64
