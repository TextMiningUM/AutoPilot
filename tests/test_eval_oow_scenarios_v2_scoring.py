"""RAG-rebuild-v2 plan point 4: unit tests for eval_oow_scenarios.py's --schema v2
deterministic scoring functions (no LLM/model needed -- pure logic on synthetic answers).
"""
from pipeline.eval.eval_oow_scenarios import (
    parse_action_json, v2_action_correct_score, v2_direction_correct_score, v2_rule_correct_score,
)

GOLD_TURN = {"action": "turn_right", "degrees": 30.0, "encounter_rule": "Rule 14", "conduct_rule": "Rule 14"}
GOLD_HOLD = {"action": "hold_course", "degrees": None, "encounter_rule": "none", "conduct_rule": "none"}


def test_parse_action_json_extracts_a_valid_object() -> None:
    answer = ('I will do this: {"action": "turn_right", "degrees": 25, "encounter_rule": "Rule 14", '
             '"conduct_rule": "Rule 14", "reasoning": "x"}')
    parsed = parse_action_json(answer)
    assert parsed is not None
    assert parsed["action"] == "turn_right"


def test_parse_action_json_returns_none_for_no_json() -> None:
    assert parse_action_json("I will turn right 30 degrees under Rule 14.") is None


def test_parse_action_json_returns_none_for_invalid_schema() -> None:
    # Missing required "reasoning" / invalid action name -> validate_action_json fails
    answer = '{"action": "reverse_thrust", "degrees": 10, "encounter_rule": "Rule 14", "conduct_rule": "Rule 14"}'
    assert parse_action_json(answer) is None


def test_action_correct_requires_exact_match() -> None:
    parsed = {"action": "turn_right", "degrees": 30.0, "encounter_rule": "Rule 14",
             "conduct_rule": "Rule 14", "reasoning": "x"}
    assert v2_action_correct_score(parsed, GOLD_TURN) == 1.0
    wrong = {**parsed, "action": "turn_left"}
    assert v2_action_correct_score(wrong, GOLD_TURN) == 0.0
    assert v2_action_correct_score(None, GOLD_TURN) == 0.0


def test_direction_correct_is_vacuous_none_for_non_turn_gold() -> None:
    parsed = {"action": "hold_course", "degrees": None, "encounter_rule": "none",
             "conduct_rule": "none", "reasoning": "x"}
    assert v2_direction_correct_score(parsed, GOLD_HOLD, tolerance=10.0) is None


def test_direction_correct_within_tolerance() -> None:
    parsed = {"action": "turn_right", "degrees": 28.0, "encounter_rule": "Rule 14",
             "conduct_rule": "Rule 14", "reasoning": "x"}
    assert v2_direction_correct_score(parsed, GOLD_TURN, tolerance=10.0) == 1.0
    far = {**parsed, "degrees": 5.0}
    assert v2_direction_correct_score(far, GOLD_TURN, tolerance=10.0) == 0.0


def test_direction_correct_requires_matching_action_too() -> None:
    """A degree figure attached to the WRONG action name is not direction-correct, even
    if the number happens to be close -- direction only makes sense conditioned on the
    right manoeuvre being chosen."""
    wrong_action_right_degrees = {"action": "turn_left", "degrees": 30.0, "encounter_rule": "Rule 14",
                                  "conduct_rule": "Rule 14", "reasoning": "x"}
    assert v2_direction_correct_score(wrong_action_right_degrees, GOLD_TURN, tolerance=10.0) == 0.0


def test_rule_correct_exact_or_none() -> None:
    parsed = {"action": "turn_right", "degrees": 30.0, "encounter_rule": "Rule 14",
             "conduct_rule": "Rule 14", "reasoning": "x"}
    assert v2_rule_correct_score(parsed, GOLD_TURN, "encounter_rule") == 1.0
    assert v2_rule_correct_score(parsed, GOLD_TURN, "conduct_rule") == 1.0
    wrong = {**parsed, "encounter_rule": "Rule 15"}
    assert v2_rule_correct_score(wrong, GOLD_TURN, "encounter_rule") == 0.0
    assert v2_rule_correct_score(wrong, GOLD_TURN, "conduct_rule") == 1.0
    assert v2_rule_correct_score(None, GOLD_TURN, "encounter_rule") == 0.0


def test_rule_correct_scores_encounter_and_conduct_independently() -> None:
    """A response can get the encounter right but the conduct wrong (or vice versa) --
    the two fields must be scored completely independently."""
    parsed = {"action": "hold_course", "degrees": None, "encounter_rule": "Rule 15",
             "conduct_rule": "Rule 15", "reasoning": "x"}  # conduct should be Rule 17
    gold = {"action": "hold_course", "degrees": None, "encounter_rule": "Rule 15", "conduct_rule": "Rule 17"}
    assert v2_rule_correct_score(parsed, gold, "encounter_rule") == 1.0
    assert v2_rule_correct_score(parsed, gold, "conduct_rule") == 0.0
