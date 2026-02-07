"""
Deterministic data filtering for the LLM Data Platform.
"""

from __future__ import annotations

import re

from .ingestion import Sample
from . import cleaning


def filter_by_min_length(samples: list[Sample], min_length: int) -> list[Sample]:
    """
    Remove samples where input or output is shorter than min_length (in characters).
    """
    if min_length <= 0:
        return samples
    return [s for s in samples if len(s.input) >= min_length and len(s.output) >= min_length]


def _has_excessive_repeat(text: str, max_repeat: int = 10) -> bool:
    """True if text has the same character repeated more than max_repeat times."""
    if not text or max_repeat < 1:
        return False
    pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}")
    return pattern.search(text) is not None


def filter_noise(samples: list[Sample], max_repeat: int = 10) -> list[Sample]:
    """
    Remove samples where input or output looks like noise (e.g. excessive repeated characters).
    """
    return [
        s
        for s in samples
        if not _has_excessive_repeat(s.input, max_repeat) and not _has_excessive_repeat(s.output, max_repeat)
    ]


def clean_and_filter(
    samples: list[Sample],
    config: dict,
) -> list[Sample]:
    """
    Pipeline: apply cleaning then filtering based on config.

    config may contain:
      - min_length (int): drop samples with input or output shorter than this
      - remove_duplicates (bool): drop duplicate (input, output) pairs
      - filter_noise (bool): drop samples with excessive repeated characters
      - noise_max_repeat (int): used when filter_noise True; default 10
    """
    out = cleaning.remove_empty_samples(samples)
    if config.get("remove_duplicates", False):
        out = cleaning.remove_duplicate_samples(out)
    if "min_length" in config:
        out = filter_by_min_length(out, config["min_length"])
    if config.get("filter_noise", False):
        out = filter_noise(out, config.get("noise_max_repeat", 10))
    return out
