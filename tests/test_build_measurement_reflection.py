"""Fase C3 tests for pipeline/track2/build_measurement_reflection.py -- the 3-check-
question reflection critique, replacing the old "too vague, add parameters" pattern."""
from pipeline.track2.build_measurement_reflection import build_critique, build_refined


def _measurement(checks_fired, **details):
    return {"checks_fired": checks_fired, "details": details}


def test_critique_names_check_1_when_fabricated_risk_fires() -> None:
    m = _measurement(["A_fabricated_risk"],
                     A={"min_cpa_m": 3000.0, "safe_distance_m": 500.0,
                        "cited_encounter_rule": "Rule 15", "cited_conduct_rule": "Rule 16"})
    critique = build_critique(m)
    assert "Check 1" in critique and "NO --" in critique
    assert "3000" in critique and "500" in critique
    assert "Check 2" in critique and "consistent" in critique
    assert "Check 3" in critique and "consistent" in critique


def test_critique_names_check_2_when_wrong_direction_fires() -> None:
    m = _measurement(["B_wrong_direction"],
                     B={"cited_encounter_rule": "Rule 15", "cited_conduct_rule": "Rule 16"})
    critique = build_critique(m)
    assert "Check 2" in critique and "NO --" in critique
    assert "starboard" in critique


def test_critique_names_check_3_when_degrees_over_limit_fires() -> None:
    m = _measurement(["C_degrees_over_limit"], C={"requested_degrees": 45.0, "limit_degrees": 30.0})
    critique = build_critique(m)
    assert "Check 3" in critique and "45" in critique and "30" in critique


def test_refined_forces_hold_course_when_fabricated_risk_fires() -> None:
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    m = _measurement(["A_fabricated_risk"], A={"min_cpa_m": 3000.0, "safe_distance_m": 500.0})
    refined = build_refined(decision, m)
    assert refined["action"] == "hold_course"
    assert refined["encounter_rule"] == "none" and refined["conduct_rule"] == "none"


def test_refined_flips_direction_when_wrong_direction_fires() -> None:
    decision = {"action": "turn_left", "degrees": 20.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    m = _measurement(["B_wrong_direction"], B={})
    refined = build_refined(decision, m)
    assert refined["action"] == "turn_right"
    assert refined["degrees"] == 20.0


def test_refined_caps_degrees_when_over_limit_fires() -> None:
    decision = {"action": "turn_right", "degrees": 45.0, "encounter_rule": "Rule 15",
               "conduct_rule": "Rule 16", "reasoning": "..."}
    m = _measurement(["C_degrees_over_limit"], C={"requested_degrees": 45.0, "limit_degrees": 30.0})
    refined = build_refined(decision, m)
    assert refined["degrees"] == 30.0
