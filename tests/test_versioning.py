"""Tests for content hashing and version folders in src/data/versioning.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.data.ingestion import Sample
from src.data.versioning import compute_dataset_hash, create_dataset_version


def _samples() -> list[Sample]:
    return [
        Sample(id="s_0", input="¿Dónde está?", output="Aquí", source="s"),
        Sample(id="s_1", input="q2", output="a2", source="s"),
    ]


def test_hash_is_stable_across_calls_and_equal_lists():
    assert compute_dataset_hash(_samples()) == compute_dataset_hash(_samples())


def test_hash_changes_with_content_and_order():
    base = compute_dataset_hash(_samples())
    edited = _samples()
    edited[1].output = "a2!"
    assert compute_dataset_hash(edited) != base
    assert compute_dataset_hash(list(reversed(_samples()))) != base


def test_hash_of_empty_dataset_is_sha256_of_nothing():
    assert compute_dataset_hash([]) == hashlib.sha256(b"").hexdigest()


def test_hash_matches_bytes_written_to_data_jsonl(tmp_path):
    """The docs promise the hash can be re-checked from data.jsonl alone."""
    version = Path(create_dataset_version(_samples(), "v1", {}, output_dir=str(tmp_path)))
    meta = json.loads((version / "metadata.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((version / "data.jsonl").read_bytes()).hexdigest() == meta["dataset_hash"]


def test_non_ascii_text_is_written_unescaped(tmp_path):
    version = Path(create_dataset_version(_samples(), "v1", {}, output_dir=str(tmp_path)))
    assert "¿Dónde está?" in (version / "data.jsonl").read_text(encoding="utf-8")


def test_same_input_produces_byte_identical_versions(tmp_path):
    a = Path(create_dataset_version(_samples(), "v1", {"k": 1}, output_dir=str(tmp_path / "a")))
    b = Path(create_dataset_version(_samples(), "v1", {"k": 1}, output_dir=str(tmp_path / "b")))
    for name in ("data.jsonl", "metadata.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_rewriting_a_version_replaces_its_content(tmp_path):
    create_dataset_version(_samples(), "v1", {}, output_dir=str(tmp_path))
    version = Path(create_dataset_version(_samples()[:1], "v1", {}, output_dir=str(tmp_path)))
    meta = json.loads((version / "metadata.json").read_text(encoding="utf-8"))
    assert meta["num_samples"] == 1
    assert len((version / "data.jsonl").read_text(encoding="utf-8").splitlines()) == 1
