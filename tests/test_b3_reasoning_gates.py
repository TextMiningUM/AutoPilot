"""Unit tests for pipeline/track2/b3_reasoning_gates.py -- pure logic, no API calls."""
from pipeline.track2.b3_reasoning_gates import (
    gate_schema, gate_decision_match, gate_rule_match, gate_contact_consistency,
    gate_number_consistency, gate_threshold_wording, gate_risk_consistency, run_gates,
    extract_first_json_object, extract_numbers_with_units,
)

VALID_OBJ = {"action": "turn_right", "degrees": 30.0, "encounter_rule": "Rule 14",
            "conduct_rule": "Rule 14", "reasoning": "x"}


def test_gate_schema_accepts_valid_and_rejects_invalid() -> None:
    assert gate_schema(VALID_OBJ) is None
    assert gate_schema({"action": "reverse_thrust"}) is not None
    assert gate_schema(None) is not None


def test_gate_decision_match() -> None:
    assert gate_decision_match(VALID_OBJ, "turn_right", 30.0) is None
    assert gate_decision_match(VALID_OBJ, "turn_left", 30.0) is not None
    assert gate_decision_match(VALID_OBJ, "turn_right", 25.0) is not None
    assert gate_decision_match({"action": "hold_course", "degrees": None}, "hold_course", None) is None
    assert gate_decision_match({"action": "hold_course", "degrees": 5.0}, "hold_course", None) is not None


def test_gate_rule_match_exact() -> None:
    reasoning = "This is a head-on encounter under Rule 14, so I alter to starboard."
    assert gate_rule_match(VALID_OBJ, "Rule 14", "Rule 14", reasoning) is None


def test_gate_rule_match_field_mismatch() -> None:
    obj = {**VALID_OBJ, "encounter_rule": "Rule 15"}
    assert gate_rule_match(obj, "Rule 14", "Rule 14", "Rule 14 applies") is not None


def test_gate_rule_match_extra_rule_cited() -> None:
    reasoning = "Under Rule 14 and also Rule 15, I alter to starboard."
    assert gate_rule_match(VALID_OBJ, "Rule 14", "Rule 14", reasoning) is not None


def test_gate_rule_match_standing_rules_are_never_extra() -> None:
    reasoning = "Rule 7 requires risk assessment; this is Rule 14, so I alter to starboard."
    assert gate_rule_match(VALID_OBJ, "Rule 14", "Rule 14", reasoning) is None


def test_gate_rule_match_missing_expected_citation() -> None:
    reasoning = "I am altering course to starboard for safety."
    assert gate_rule_match(VALID_OBJ, "Rule 14", "Rule 14", reasoning) is not None


def test_gate_rule_match_no_risk_case_needs_no_citation() -> None:
    obj = {"action": "hold_course", "degrees": None, "encounter_rule": "none", "conduct_rule": "none"}
    assert gate_rule_match(obj, "none", "none", "No risk, holding course.") is None


def test_gate_contact_consistency() -> None:
    assert gate_contact_consistency("RANDOM_TS1 is ahead", "RANDOM_TS1") is None
    assert gate_contact_consistency("RANDOM_TS2 is ahead", "RANDOM_TS1") is not None
    assert gate_contact_consistency("no contacts", None) is None


def test_gate_number_consistency_accepts_numbers_from_situation() -> None:
    situation = "range 1408 m, rel.bearing -3.0 deg, CPA 1 m, TCPA 2.3 min"
    reasoning = "At 1408 metres, CPA of 1 m in 2.3 minutes."
    assert gate_number_consistency(reasoning, situation) is None


def test_gate_number_consistency_rejects_invented_numbers() -> None:
    situation = "range 1408 m, CPA 1 m, TCPA 2.3 min"
    reasoning = "At 1408 metres, but the CPA is actually 9999 m."
    assert gate_number_consistency(reasoning, situation) is not None


def test_gate_number_consistency_allows_the_given_degrees() -> None:
    situation = "range 1408 m, CPA 1 m"
    reasoning = "Turning 30 degrees to starboard."
    assert gate_number_consistency(reasoning, situation, frozenset({30.0})) is None
    assert gate_number_consistency(reasoning, situation, frozenset()) is not None


def test_gate_number_consistency_tolerates_dropping_the_sign_on_a_bearing() -> None:
    """rel.bearing is signed (negative=port) in the situation text, but naturally-phrased
    reasoning almost always drops the sign in favour of "X degrees to port/starboard"
    wording -- that is correct prose, not an invented number."""
    situation = "rel.bearing -57.6 deg, range 1019 m"
    reasoning = "The contact bears 57.6 degrees to port at 1019 metres."
    assert gate_number_consistency(reasoning, situation) is None


def test_gate_number_consistency_allows_the_224_5_overtaking_sector_constant() -> None:
    situation = "range 1074 m, CPA 0 m"
    reasoning = "She is approaching from more than 22.5 degrees abaft our beam, overtaking us."
    assert gate_number_consistency(reasoning, situation) is None


def test_gate_threshold_wording_catches_the_leo00022_failure() -> None:
    reasoning = "CPA of 272 m, well above the 500 m safe passing distance."
    assert gate_threshold_wording(reasoning, cpa_m=272.0, safe_distance_m=500.0) is not None


def test_gate_threshold_wording_accepts_correct_direction() -> None:
    reasoning = "CPA of 272 m, well below the 500 m safe passing distance -- a real risk."
    assert gate_threshold_wording(reasoning, cpa_m=272.0, safe_distance_m=500.0) is None


def test_gate_threshold_wording_no_op_when_not_discussed() -> None:
    assert gate_threshold_wording("Holding course, no contacts nearby.", cpa_m=272.0, safe_distance_m=500.0) is None


def test_gate_risk_consistency() -> None:
    assert gate_risk_consistency("There is no real risk of collision here.", real_risk=True) is not None
    assert gate_risk_consistency("There is no real risk of collision here.", real_risk=False) is None
    assert gate_risk_consistency("Rule 14 applies, I am altering course.", real_risk=True) is None


def test_extract_first_json_object_from_prose_wrapped_response() -> None:
    text = 'Sure, here it is:\n{"action": "hold_course", "degrees": null, "encounter_rule": "none", "conduct_rule": "none", "reasoning": "x"}\nDone.'
    obj = extract_first_json_object(text)
    assert obj is not None and obj["action"] == "hold_course"


def test_extract_numbers_with_units_ignores_bare_integers() -> None:
    nums = extract_numbers_with_units("Rule 14 applies at 1408 m and 9.5 knots.")
    assert 1408.0 in nums and 9.5 in nums
    assert 14.0 not in nums  # bare rule number, no unit


def test_run_gates_all_pass() -> None:
    situation = "range 1408 m, rel.bearing -3.0 deg, CPA 1 m, TCPA 2.3 min, safe passing distance is 500m"
    obj = {"action": "turn_right", "degrees": 30.0, "encounter_rule": "Rule 14",
          "conduct_rule": "Rule 14", "reasoning": "x"}
    reasoning = "RANDOM_TS1 CPA of 1 m in 2.3 minutes, well below the 500m threshold -- head-on under Rule 14."
    failures = run_gates(
        obj, reasoning, situation=situation, expected_action="turn_right", expected_degrees=30.0,
        expected_encounter_rule="Rule 14", expected_conduct_rule="Rule 14", real_risk=True,
        cpa_m=1.0, safe_distance_m=500.0, decisive_contact_name="RANDOM_TS1",
    )
    assert failures == {}


def test_run_gates_none_response_fails_schema() -> None:
    failures = run_gates(
        None, None, situation="x", expected_action="hold_course", expected_degrees=None,
        expected_encounter_rule="none", expected_conduct_rule="none", real_risk=False,
        cpa_m=None, safe_distance_m=500.0, decisive_contact_name=None,
    )
    assert "schema" in failures
