"""Edge cases for src/data/ingestion.py (the happy paths are in test_data_pipeline.py)."""

from __future__ import annotations

import json

import pytest

from src.data.cleaning import remove_empty_samples
from src.data.ingestion import load_dataset


def _write_json(tmp_path, items):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return str(path)


def test_json_ids_are_source_and_index(tmp_path):
    samples = load_dataset(_write_json(tmp_path, [{"input": "a", "output": "b"}] * 3), source="crm")
    assert [s.id for s in samples] == ["crm_0", "crm_1", "crm_2"]


def test_json_null_and_missing_fields_become_empty_strings(tmp_path):
    path = _write_json(
        tmp_path,
        [
            {"input": None, "output": "answer"},
            {"input": "question", "output": None},
            {"input": "only input"},
        ],
    )
    samples = load_dataset(path, source="t")
    assert [(s.input, s.output) for s in samples] == [("", "answer"), ("question", ""), ("only input", "")]
    # ...so the cleaning step drops them instead of keeping the text "None".
    assert remove_empty_samples(samples) == []


def test_json_non_string_values_are_stringified(tmp_path):
    samples = load_dataset(_write_json(tmp_path, [{"input": 42, "output": 3.5}]), source="t")
    assert (samples[0].input, samples[0].output) == ("42", "3.5")


def test_json_root_must_be_a_list(tmp_path):
    with pytest.raises(ValueError, match="must be a list"):
        load_dataset(_write_json(tmp_path, {"input": "a"}), source="t")


def test_json_items_must_be_objects(tmp_path):
    with pytest.raises(ValueError, match="index 1"):
        load_dataset(_write_json(tmp_path, [{"input": "a", "output": "b"}, "oops"]), source="t")


def test_text_line_without_tab_has_empty_output(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("just a line\nq\ta\tb\n", encoding="utf-8")
    samples = load_dataset(str(path), source="t")
    assert (samples[0].input, samples[0].output) == ("just a line", "")
    # Only the first tab splits; the rest stays in the output.
    assert (samples[1].input, samples[1].output) == ("q", "a\tb")


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataset(str(path), source="t")


def _write_csv(tmp_path, text):
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_csv_named_columns_are_case_insensitive_and_win_over_position(tmp_path):
    samples = load_dataset(_write_csv(tmp_path, "id,Output,INPUT\n7,answer,question\n"), source="t")
    assert (samples[0].input, samples[0].output) == ("question", "answer")


def test_csv_without_named_columns_uses_first_two(tmp_path):
    samples = load_dataset(_write_csv(tmp_path, "question,answer,extra\nq,a,x\n"), source="t")
    assert (samples[0].input, samples[0].output) == ("q", "a")


def test_csv_with_only_input_named_takes_output_from_another_column(tmp_path):
    # Before the fix, output fell back to the 2nd column, which here *is* "input".
    samples = load_dataset(_write_csv(tmp_path, "answer,input\na,q\n"), source="t")
    assert (samples[0].input, samples[0].output) == ("q", "a")


def test_csv_with_a_single_column_is_rejected(tmp_path):
    # Before the fix, the one column was copied into both input and output.
    with pytest.raises(ValueError, match="at least two columns"):
        load_dataset(_write_csv(tmp_path, "text\nhello\n"), source="t")


def test_csv_short_rows_give_empty_fields(tmp_path):
    samples = load_dataset(_write_csv(tmp_path, "input,output\nonly-input\n"), source="t")
    assert (samples[0].input, samples[0].output) == ("only-input", "")


def test_csv_with_header_only_is_empty(tmp_path):
    assert load_dataset(_write_csv(tmp_path, "input,output\n"), source="t") == []
