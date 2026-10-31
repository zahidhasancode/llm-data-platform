"""Quality gate: decide from measured results whether a candidate model may be promoted."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    promote: bool
    reasons: tuple[str, ...]


def decide(
    report: dict,
    champion_accuracy: float | None,
    *,
    min_gain_over_base: float = 2.0,
    min_valid_sql: float = 90.0,
    max_drop_from_champion: float = 0.0,
) -> Decision:
    """Promote only if every check passes. All reasons are returned, so a block can be explained."""
    conditions = report.get("conditions", {})
    if "base" not in conditions or "qlora" not in conditions:
        return Decision(False, ("missing_results",))
    base, candidate = conditions["base"], conditions["qlora"]
    if base["n"] != candidate["n"]:
        return Decision(False, ("different_test_sizes",))

    reasons = []
    gain = candidate["execution_accuracy"] - base["execution_accuracy"]
    if gain < min_gain_over_base:
        reasons.append(f"gain_over_base {gain:.1f} is below {min_gain_over_base}")
    if candidate["valid_sql_rate"] < min_valid_sql:
        reasons.append(f"valid_sql {candidate['valid_sql_rate']} is below {min_valid_sql}")
    if champion_accuracy is not None:
        drop = champion_accuracy - candidate["execution_accuracy"]
        if drop > max_drop_from_champion:
            reasons.append(f"accuracy {drop:.1f} below the current champion")
    return Decision(not reasons, tuple(reasons) or ("passed",))
