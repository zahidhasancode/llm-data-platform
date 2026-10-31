"""Quality monitor: boundaries, and no decision from too little data."""

from src.pipeline.monitor import check


def flags(good: int, bad: int) -> list[bool]:
    return [True] * good + [False] * bad


def test_too_few_samples_is_no_reading() -> None:
    reading = check(flags(0, 49), 95.0)
    assert (reading.valid_rate, reading.dropped) == (None, False)


def test_a_steady_rate_is_not_a_drop() -> None:
    assert not check(flags(95, 5), 95.0).dropped


def test_exactly_the_limit_is_not_a_drop() -> None:
    assert not check(flags(85, 15), 95.0).dropped


def test_past_the_limit_is_a_drop() -> None:
    reading = check(flags(84, 16), 95.0)
    assert reading.dropped
    assert reading.valid_rate == 84.0
