"""Regression tests for app/measurement.py -- the deterministic, read-only measurement
layer around OOW-agent decisions (Checks A/B/C, see that module's docstring and
OOW_Mission_Sim_Analysis_V1.md Sec.2 #1/#2/#4). Edge cases matter more than the happy path
here: this layer's entire value is in NOT producing false positives/negatives, since a
false "error" would corrupt any future DPO-mining built on top of it.
"""
import copy
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.measurement import measure_decision_quality  # noqa: E402
from app.simulation import VesselConstraints  # noqa: E402

CONSTRAINTS = VesselConstraints()  # min_cpa_m=500, turn_rate_deg_s=3.0, time_step_s=10.0


def contact(cpa_m: float, tcpa_s: float = 100.0) -> dict:
    """Minimal stand-in for narrate.contact_line()'s return dict -- measure_decision_quality
    reads "cpa_m" and "tcpa_s" off each entry (quality-review STAP 1: Check A's real_risk()
    needs both). Default tcpa_s=100.0 sits well within RISK_HORIZON_S, so every existing
    test that only varies cpa_m keeps its original CPA-only behaviour."""
    return {"cpa_m": cpa_m, "tcpa_s": tcpa_s}


# ── Check A: fabricated risk ──────────────────────────────────────────────
def test_a_fires_when_rule_cited_with_no_real_risk() -> None:
    decision = {"action": "turn_right", "degrees": 10.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(3000.0)], CONSTRAINTS)
    assert "A_fabricated_risk" in result["checks_fired"]
    assert result["details"]["A"]["cited_encounter_rule"] == "Rule 15"
    assert result["details"]["A"]["cited_conduct_rule"] == "Rule 16"
    assert result["details"]["A"]["min_cpa_m"] == 3000.0
    assert result["details"]["A"]["safe_distance_m"] == 500.0


def test_a_fires_when_only_one_field_is_non_none() -> None:
    # A malformed/inconsistent pair (would fail validate_action_json, but measurement
    # must still not silently ignore it) -- either field alone citing a rule counts.
    decision = {"action": "turn_right", "degrees": 10.0, "encounter_rule": "none",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(3000.0)], CONSTRAINTS)
    assert "A_fabricated_risk" in result["checks_fired"]


# ── Fase C0: backward-compat with the pre-B3 "rule_applied" schema ────────
def test_old_rule_applied_schema_is_read_as_both_fields() -> None:
    """The 147 archived units_v1 checkpoints predate encounter_rule/conduct_rule and only
    ever recorded "rule_applied" -- measure_decision_quality() must still be able to fire
    Check A against them (falls back to rule_applied for BOTH fields)."""
    decision = {"action": "hold_course", "degrees": 0, "rule_applied": "Rule 15",
               "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(3000.0)], CONSTRAINTS)
    assert "A_fabricated_risk" in result["checks_fired"]
    assert result["details"]["A"]["cited_encounter_rule"] == "Rule 15"
    assert result["details"]["A"]["cited_conduct_rule"] == "Rule 15"


def test_new_schema_decision_never_reads_a_stray_rule_applied_key() -> None:
    """A decision that already has real encounter_rule/conduct_rule keys must use THOSE,
    never fall back to a (hypothetical, stray) rule_applied key -- the fallback is
    strictly for old-schema decisions missing the new keys entirely."""
    decision = {"action": "hold_course", "degrees": 0, "encounter_rule": "none",
               "conduct_rule": "none", "rule_applied": "Rule 15", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(3000.0)], CONSTRAINTS)
    assert "A_fabricated_risk" not in result["checks_fired"]


def test_a_does_not_fire_on_tcpa_zero_already_passed_case() -> None:
    # The classic misread: TCPA=0 because the closest point already happened, but CPA is
    # large -- situation carries only cpa_m (per contact_line()'s schema), so this is
    # naturally CPA-only already; a rule cited here is still a real fabrication check on
    # CPA alone, matching the system prompt's own definition of real risk.
    decision = {"action": "hold_course", "degrees": None, "encounter_rule": "none",
               "conduct_rule": "none", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(11818.0, tcpa_s=0.0)], CONSTRAINTS)
    assert result["checks_fired"] == []


# ── Quality-review STAP 1 (2026-09-23): Check A now needs BOTH cpa_m and tcpa_s ───────
def test_a_fires_on_low_cpa_and_tcpa_within_horizon() -> None:
    decision = {"action": "turn_right", "degrees": 10.0, "encounter_rule": "none",
               "conduct_rule": "none", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(300.0, tcpa_s=100.0)], CONSTRAINTS)
    assert result["checks_fired"] == []  # no rule cited, so Check A correctly doesn't fire


def test_a_fires_when_rule_cited_but_tcpa_beyond_horizon_even_with_low_cpa() -> None:
    """Before this fix, Check A was CPA-only: a low CPA far off in time (TCPA beyond the
    real-risk horizon) would NOT have fired Check A even though it should have -- a rule
    cited against a contact that isn't yet a real risk IS a fabrication."""
    decision = {"action": "turn_right", "degrees": 10.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(300.0, tcpa_s=301.0)], CONSTRAINTS)
    assert "A_fabricated_risk" in result["checks_fired"]


def test_a_fires_when_rule_cited_but_tcpa_already_negative() -> None:
    decision = {"action": "turn_right", "degrees": 10.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(300.0, tcpa_s=-5.0)], CONSTRAINTS)
    assert "A_fabricated_risk" in result["checks_fired"]


def test_a_does_not_fire_when_no_rule_cited() -> None:
    decision = {"action": "hold_course", "degrees": None, "encounter_rule": "none",
               "conduct_rule": "none", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(9999.0)], CONSTRAINTS)
    assert result["checks_fired"] == []


def test_a_does_not_fire_when_real_risk_exists() -> None:
    decision = {"action": "turn_right", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "A_fabricated_risk" not in result["checks_fired"]


# ── Check B: wrong turn direction ─────────────────────────────────────────
def test_b_fires_on_rule15_turn_left_with_real_risk() -> None:
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "B_wrong_direction" in result["checks_fired"]
    assert result["details"]["B"] == {"cited_encounter_rule": "Rule 15",
                                      "cited_conduct_rule": "Rule 16", "action": "turn_left"}


def test_b_fires_on_conduct_rule14_head_on_too() -> None:
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 14",
               "conduct_rule": "Rule 14", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "B_wrong_direction" in result["checks_fired"]


def test_b_fires_on_conduct_rule16_too() -> None:
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "B_wrong_direction" in result["checks_fired"]


def test_b_does_not_fire_on_rule13_turn_left() -> None:
    # Overtaking may legitimately pass on either side.
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 13",
               "conduct_rule": "Rule 13", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "B_wrong_direction" not in result["checks_fired"]
    assert "B" not in result["details"]


def test_b_does_not_fire_on_rule17_turn_left_but_logs_suspect() -> None:
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 17", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "B_wrong_direction" not in result["checks_fired"]
    assert "B" not in result["details"]
    assert result["details"]["B_suspect_rule17"] == {"cited_encounter_rule": "Rule 15",
                                                      "cited_conduct_rule": "Rule 17",
                                                      "action": "turn_left"}


def test_b_rule17_is_exclusive_of_the_give_way_fallback() -> None:
    """A stand-on vessel's own 17(b) turn during a crossing/head-on encounter must NOT
    also match the encounter_rule Rule 14/15 fallback and false-positive as "wrong
    direction" -- conduct_rule=="Rule 17" takes priority, exactly like the original
    single-rule_applied version's if/elif exclusivity."""
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 14",
               "conduct_rule": "Rule 17", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert result["checks_fired"] == []
    assert "B" not in result["details"]
    assert "B_suspect_rule17" in result["details"]


def test_b_does_not_fire_on_rule19() -> None:
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 19", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert result["checks_fired"] == []
    assert "B_suspect_rule17" not in result["details"]


def test_b_does_not_fire_on_give_way_stop_rule8() -> None:
    # Rule 8 (give-way emergency stop) is never a turn -- action isn't turn_left here,
    # but also conduct_rule=="Rule 8" must never be treated as a give-way TURN rule.
    decision = {"action": "stop", "degrees": None, "encounter_rule": "Rule 14",
               "conduct_rule": "Rule 8", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert result["checks_fired"] == []


def test_b_does_not_fire_when_a_already_fired() -> None:
    # Order-dependency: no real risk (Check A fires on the fabricated Rule 15 citation),
    # so Check B must not ALSO fire on the same turn_left, even though the rule/action
    # combination would otherwise qualify.
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(3000.0)], CONSTRAINTS)
    assert result["checks_fired"] == ["A_fabricated_risk"]
    assert "B" not in result["details"]


def test_b_does_not_fire_on_turn_right() -> None:
    decision = {"action": "turn_right", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "B_wrong_direction" not in result["checks_fired"]


# ── Check C: physically impossible turn request ───────────────────────────
def test_c_fires_over_limit_and_logs_both_values() -> None:
    # limit_degrees = turn_rate_deg_s(3.0) * time_step_s(10.0) = 30.0
    decision = {"action": "turn_right", "degrees": 90.0, "encounter_rule": "Rule 14",
               "conduct_rule": "Rule 14", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "C_degrees_over_limit" in result["checks_fired"]
    assert result["details"]["C"] == {"requested_degrees": 90.0, "limit_degrees": 30.0}


def test_c_does_not_fire_at_or_below_limit() -> None:
    decision = {"action": "turn_right", "degrees": 30.0, "encounter_rule": "Rule 14",
               "conduct_rule": "Rule 14", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert "C_degrees_over_limit" not in result["checks_fired"]


def test_c_ignores_non_turn_actions() -> None:
    decision = {"action": "slow_down", "degrees": None, "encounter_rule": "none",
               "conduct_rule": "none", "reasoning": "..."}
    result = measure_decision_quality(decision, [], CONSTRAINTS)
    assert result["checks_fired"] == []


def test_c_uses_constraints_not_a_hardcoded_limit() -> None:
    tight = VesselConstraints(turn_rate_deg_s=1.0, time_step_s=10.0)  # limit=10 deg
    decision = {"action": "turn_left", "degrees": 15.0, "encounter_rule": "none",
               "conduct_rule": "none", "reasoning": "..."}
    result = measure_decision_quality(decision, [], tight)
    assert "C_degrees_over_limit" in result["checks_fired"]
    assert result["details"]["C"]["limit_degrees"] == 10.0


# ── Fully correct decision -> nothing fires ───────────────────────────────
def test_nothing_fires_on_a_fully_correct_decision() -> None:
    decision = {"action": "turn_right", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(100.0)], CONSTRAINTS)
    assert result["checks_fired"] == []
    assert result["details"] == {}


def test_nothing_fires_on_quiet_hold_course() -> None:
    decision = {"action": "hold_course", "degrees": None, "encounter_rule": "none",
               "conduct_rule": "none", "reasoning": "..."}
    result = measure_decision_quality(decision, [contact(9999.0)], CONSTRAINTS)
    assert result["checks_fired"] == []
    assert result["details"] == {}


# ── Purity: never mutates the input decision dict ─────────────────────────
def test_never_mutates_decision() -> None:
    decision = {"action": "turn_left", "degrees": 90.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "some reasoning text"}
    before = copy.deepcopy(decision)
    measure_decision_quality(decision, [contact(3000.0)], CONSTRAINTS)
    assert decision == before
