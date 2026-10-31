"""Quality gate: boundaries, and that a block always says why."""

from src.pipeline.gate import decide


def report(base: float, qlora: float, valid: float = 95.0, n: int = 100) -> dict:
    return {
        "conditions": {
            "base": {"n": n, "execution_accuracy": base, "valid_sql_rate": 90.0},
            "qlora": {"n": n, "execution_accuracy": qlora, "valid_sql_rate": valid},
        }
    }


def test_a_good_model_is_promoted() -> None:
    assert decide(report(60, 70), None).promote


def test_the_gain_boundary() -> None:
    assert decide(report(60, 62), None).promote
    assert not decide(report(60, 61.9), None).promote


def test_the_valid_sql_boundary() -> None:
    assert decide(report(60, 70, valid=90.0), None).promote
    assert not decide(report(60, 70, valid=89.9), None).promote


def test_a_model_worse_than_the_champion_is_blocked() -> None:
    decision = decide(report(60, 70), 71.0)
    assert not decision.promote
    assert "champion" in decision.reasons[0]


def test_equal_to_the_champion_passes() -> None:
    assert decide(report(60, 70), 70.0).promote


def test_every_failed_check_is_reported() -> None:
    assert len(decide(report(60, 60, valid=50), 80.0).reasons) == 3


def test_missing_results_are_blocked() -> None:
    assert decide({}, None).reasons == ("missing_results",)


def test_different_test_sizes_are_blocked() -> None:
    bad = report(60, 70)
    bad["conditions"]["qlora"]["n"] = 50
    assert decide(bad, None).reasons == ("different_test_sizes",)
