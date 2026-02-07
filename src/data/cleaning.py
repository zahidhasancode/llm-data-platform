"""
Deterministic data cleaning for the LLM Data Platform.
"""

from __future__ import annotations

from .ingestion import Sample


def remove_empty_samples(samples: list[Sample]) -> list[Sample]:
    """
    Remove samples where input or output is missing or blank.
    """
    result = []
    for s in samples:
        inp = s.input if s.input is not None else ""
        out = s.output if s.output is not None else ""
        if inp.strip() and out.strip():
            result.append(s)
    return result


def remove_duplicate_samples(samples: list[Sample]) -> list[Sample]:
    """
    Remove duplicates by (input, output) pair. First occurrence is kept.
    """
    seen: set[tuple[str, str]] = set()
    result = []
    for s in samples:
        key = (s.input, s.output)
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result
