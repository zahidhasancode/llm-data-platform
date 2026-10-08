"""Tests for the length/noise filters and clean_and_filter in src/data/filtering.py."""

from __future__ import annotations

import pytest

from src.data.filtering import clean_and_filter, filter_by_min_length, filter_noise
from src.data.ingestion import Sample


def _s(id_: str, inp: str, out: str) -> Sample:
    return Sample(id=id_, input=inp, output=out, source="t")


def test_min_length_applies_to_both_fields_and_is_inclusive():
    samples = [_s("0", "abc", "abcd"), _s("1", "abcd", "ab"), _s("2", "abcd", "abcd")]
    assert [s.id for s in filter_by_min_length(samples, 3)] == ["0", "2"]


@pytest.mark.parametrize("min_length", [0, -5])
def test_non_positive_min_length_keeps_everything(min_length):
    samples = [_s("0", "a", "b")]
    assert filter_by_min_length(samples, min_length) == samples


def test_noise_threshold_means_more_than_max_repeat():
    exactly_ten = "x" * 10
    eleven = "x" * 11
    samples = [_s("ok", f"a{exactly_ten}b", "fine"), _s("noisy_in", eleven, "fine"), _s("noisy_out", "fine", eleven)]
    assert [s.id for s in filter_noise(samples, max_repeat=10)] == ["ok"]


def test_noise_threshold_is_configurable():
    samples = [_s("0", "aaab", "out")]
    assert filter_noise(samples, max_repeat=2) == []
    assert filter_noise(samples, max_repeat=3) == samples


def test_clean_and_filter_with_empty_config_only_drops_empty_samples():
    samples = [_s("0", "q", "a"), _s("1", "q", "a"), _s("2", " ", "a"), _s("3", "z" * 30, "a")]
    assert [s.id for s in clean_and_filter(samples, {})] == ["0", "1", "3"]


def test_clean_and_filter_applies_every_rule_in_order():
    samples = [
        _s("0", "long enough", "long enough"),
        _s("1", "long enough", "long enough"),  # duplicate of 0
        _s("2", "short", "long enough"),  # input below min_length
        _s("3", "long enough", ""),  # empty output
        _s("4", "long enough", "nooooooooooooo"),  # 13 repeated 'o'
        _s("5", "also long enough", "also long enough"),
    ]
    config = {"remove_duplicates": True, "min_length": 8, "filter_noise": True, "noise_max_repeat": 10}
    assert [s.id for s in clean_and_filter(samples, config)] == ["0", "5"]


def test_clean_and_filter_is_deterministic_and_does_not_mutate_input():
    samples = [_s(str(i), f"input {i % 3}", f"output {i % 3}") for i in range(9)]
    before = list(samples)
    config = {"remove_duplicates": True, "min_length": 3}
    first = clean_and_filter(samples, config)
    second = clean_and_filter(samples, config)
    assert first == second
    assert [s.id for s in first] == ["0", "1", "2"]
    assert samples == before
