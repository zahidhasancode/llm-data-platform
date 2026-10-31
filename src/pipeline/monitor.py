"""Decide from live results whether the served model's quality has dropped."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reading:
    samples: int
    valid_rate: float | None
    dropped: bool


def check(flags: list[bool], baseline: float, *, max_drop: float = 10.0, min_samples: int = 50) -> Reading:
    """Compare the valid-SQL rate with its baseline.

    Too few samples is "no reading", never "quality is fine" and never a drop.
    """
    if len(flags) < min_samples:
        return Reading(len(flags), None, False)
    rate = 100 * sum(flags) / len(flags)
    return Reading(len(flags), round(rate, 1), baseline - rate > max_drop)
