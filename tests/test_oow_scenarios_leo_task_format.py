"""Fase B2 (RAG-rebuild-v2 plan, 2026-09-22): task-format unification tests for
pipeline/track2/build_oow_scenarios_leo.py -- the generator must emit the SAME task
format Basic Simulator/app/agents.py evaluates on (pipeline/oow_agent_spec.py), and must
never leak the COLREG classification the model is meant to derive itself into the user
prompt."""
import json

from pipeline.oow_agent_spec import SYSTEM_OOW_AGENT, ACTIONS, validate_action_json
from pipeline.track2.build_oow_scenarios_leo import (
    build_assistant_json, build_user_message, wrong_action_variant, render_leo_narrative,
    leo_choose_action, stratified_sample, LEO_FILE,
)

FORBIDDEN_USER_PHRASES = ("Applicable COLREG rules", "We are meeting head-on",
                          "Both vessels are give-way")


def test_build_assistant_json_validates_against_the_shared_schema() -> None:
    decision = {"action": "turn_right", "degrees": 20.0, "rule_applied": "Rule 15"}
    obj = build_assistant_json(decision, "reasoning text")
    assert not validate_action_json(obj)


def test_wrong_action_variant_still_validates_and_differs_from_the_original() -> None:
    for decision in (
        {"action": "turn_right", "degrees": 20.0, "rule_applied": "Rule 15"},
        {"action": "turn_left", "degrees": 15.0, "rule_applied": "Rule 13"},
        {"action": "hold_course", "degrees": None, "rule_applied": "none"},
        {"action": "stop", "degrees": None, "rule_applied": "Rule 17"},
    ):
        wrong = wrong_action_variant(decision)
        obj = build_assistant_json(wrong, "reasoning text")
        assert not validate_action_json(obj)
        assert wrong["action"] != decision["action"] or wrong["degrees"] != decision["degrees"]


def test_user_message_never_contains_forbidden_answer_leaking_phrases() -> None:
    """A synthetic row deliberately constructed WITH a forbidden phrase must be
    detectable -- proves the check below has teeth, not just a vacuously-true scan."""
    poisoned = build_user_message("some situation\nApplicable COLREG rules: Rule 15")
    assert any(p in poisoned for p in FORBIDDEN_USER_PHRASES)


def test_real_leo_narratives_never_contain_forbidden_phrases_or_rule_numbers() -> None:
    """Scans real rendered narratives (not synthetic) -- the actual regression test that
    matters. Rule numbers ("Rule 14", ...) must not appear anywhere in the user-facing
    situation report; that classification is exactly what the model must derive itself."""
    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    sample = stratified_sample(all_recs, 40, seed=1)
    for r in sample:
        narrative = render_leo_narrative(r["state"])
        for phrase in FORBIDDEN_USER_PHRASES:
            assert phrase not in narrative, f"leo_id={r['id']!r} narrative contains {phrase!r}"
        assert "Rule " not in narrative, f"leo_id={r['id']!r} narrative leaks a rule number"


def test_narrative_includes_goal_course_check_and_safe_distance() -> None:
    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    sample = stratified_sample(all_recs, 10, seed=2)
    for r in sample:
        narrative = render_leo_narrative(r["state"])
        assert "GOAL COURSE CHECK" in narrative
        assert "safe passing distance" in narrative


def test_sft_row_only_uses_actions_from_the_shared_vocabulary() -> None:
    """B6: any assistant JSON this generator could emit must use an action name from
    ACTIONS -- built directly from leo_choose_action()'s own output space."""
    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    sample = stratified_sample(all_recs, 60, seed=3)
    for r in sample:
        decision = leo_choose_action(r["state"])
        if decision["action"] is None:
            continue  # paused frame -- never becomes a training row
        assert decision["action"] in ACTIONS
        obj = build_assistant_json(decision, "placeholder reasoning")
        assert not validate_action_json(obj)


def test_system_prompt_is_the_shared_object() -> None:
    from pipeline.track2 import build_oow_scenarios_leo
    assert build_oow_scenarios_leo.SYSTEM_OOW_AGENT is SYSTEM_OOW_AGENT
