"""Fase B2 (RAG-rebuild-v2 plan, 2026-09-22): task-format unification tests for
pipeline/track2/build_oow_scenarios.py -- the generator's training rows and its v2 eval
file must share the exact same renderer/action-mapping, and must never leak the COLREG
classification the model is meant to derive itself into the user prompt.
"""
import json

from core import AgentPaths
from pipeline.oow_agent_spec import ACTIONS, SYSTEM_OOW_AGENT, validate_action_json
from pipeline.track2.build_oow_scenarios import (
    FIXED_QUESTION_UNIFIED, N_EVAL_PER_CATEGORY_DEFAULT, N_TRAIN_PER_CATEGORY_DEFAULT,
    generate_population, render_scenario_situation, split_eval_train, to_unified_action,
    wrong_action_variant,
)

paths = AgentPaths.oow()
V2_FILE = paths.eval_dir / "oow_colreg_scenarios_v2.json"

FORBIDDEN_USER_PHRASES = ("Applicable COLREG rules", "We are meeting head-on",
                          "Both vessels are give-way", "give-way vessel", "stand-on vessel")


def _fresh_pop(seed: int = 7):
    pop = generate_population(N_EVAL_PER_CATEGORY_DEFAULT + N_TRAIN_PER_CATEGORY_DEFAULT, seed=seed)
    return split_eval_train(pop, N_EVAL_PER_CATEGORY_DEFAULT)


def test_to_unified_action_always_validates_against_the_shared_schema() -> None:
    eval_recs, train_recs = _fresh_pop()
    for rec in eval_recs + train_recs:
        unified = to_unified_action(rec)
        obj = {**unified, "reasoning": "placeholder"}
        assert not validate_action_json(obj), (rec["category"], unified)


def test_wrong_action_variant_still_validates_and_differs() -> None:
    for decision in (
        {"action": "turn_right", "degrees": 20.0, "rule_applied": "Rule 15"},
        {"action": "turn_left", "degrees": 15.0, "rule_applied": "Rule 13"},
        {"action": "hold_course", "degrees": None, "rule_applied": "none"},
        {"action": "stop", "degrees": None, "rule_applied": "Rule 17"},
    ):
        wrong = wrong_action_variant(decision)
        obj = {**wrong, "reasoning": "placeholder"}
        assert not validate_action_json(obj)
        assert wrong["action"] != decision["action"] or wrong["degrees"] != decision["degrees"]


def test_no_risk_records_never_get_a_turn_or_stop_action() -> None:
    """No-risk consistency test (parallel to test_leo_goal_course_consistency.py): this
    generator's own-ship is ALWAYS on the goal bearing by construction (mission waypoint
    is always straight ahead, see render_scenario_situation()'s docstring), so this is
    expected to be trivially satisfied -- but is still asserted so a future change to the
    geometry (off-bearing own-ship) would be caught here rather than silently mislabeling
    a goal-course turn as if it were collision-avoidance."""
    eval_recs, train_recs = _fresh_pop()
    for rec in eval_recs + train_recs:
        unified = to_unified_action(rec)
        if unified["rule_applied"] == "none":
            assert unified["action"] in ("hold_course", "speed_up"), (rec["category"], unified)


def test_situation_text_never_contains_forbidden_leaking_phrases() -> None:
    eval_recs, train_recs = _fresh_pop()
    for rec in eval_recs + train_recs:
        situation = render_scenario_situation(rec)
        for phrase in FORBIDDEN_USER_PHRASES:
            assert phrase not in situation, (rec["category"], phrase)
        assert "Rule " not in situation, f"{rec['category']} situation leaks a rule number"


def test_situation_text_includes_goal_course_check_and_safe_distance() -> None:
    eval_recs, _ = _fresh_pop()
    for rec in eval_recs[:5]:
        situation = render_scenario_situation(rec)
        assert "GOAL COURSE CHECK" in situation
        assert "safe passing distance" in situation


def test_a_synthetic_row_with_a_forbidden_phrase_is_detected() -> None:
    """Proves the phrase scan above has teeth, not just a vacuously-true check."""
    poisoned = "some situation\nApplicable COLREG rules: Rule 15"
    assert any(p in poisoned for p in FORBIDDEN_USER_PHRASES)


def test_v2_eval_record_situation_is_byte_identical_to_what_a_training_row_would_embed() -> None:
    """The shared-renderer identity requirement: a v2 eval record's situation text and a
    training row's situation text must be byte-identical for the same geometry, because
    both call the exact same render_scenario_situation() function -- verified here against
    the REAL, already-written v2 file (not just an in-memory re-derivation), by
    regenerating the same deterministic population and matching by position."""
    if not V2_FILE.exists():
        return
    v2 = json.loads(V2_FILE.read_text(encoding="utf-8"))
    eval_recs, _ = _fresh_pop(seed=0)
    assert len(eval_recs) == len(v2)
    for rec, v2_rec in zip(eval_recs[:20], v2[:20]):
        # What write_scenario_sft_files() would embed as the user turn for this geometry:
        training_user_text = f"Situation:\n{render_scenario_situation(rec)}\n\n{FIXED_QUESTION_UNIFIED}"
        assert training_user_text == f"Situation:\n{v2_rec['situation']}\n\n{v2_rec['question']}", (
            f"v2 record {v2_rec['id']} situation text diverges from what a training row "
            "would embed for the same geometry -- the two paths must call the same renderer"
        )


def test_system_prompt_is_the_shared_object() -> None:
    from pipeline.track2 import build_oow_scenarios
    assert build_oow_scenarios.SYSTEM_OOW_AGENT is SYSTEM_OOW_AGENT
