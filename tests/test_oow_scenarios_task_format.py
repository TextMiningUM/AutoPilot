"""Fase B2 (RAG-rebuild-v2 plan, 2026-09-22): task-format unification tests for
pipeline/track2/build_oow_scenarios.py -- the generator's training rows and its v2 eval
file must share the exact same renderer/action-mapping, and must never leak the COLREG
classification the model is meant to derive itself into the user prompt.
"""
import json

import pytest

from core import AgentPaths
from pipeline.oow_agent_spec import ACTIONS, SYSTEM_OOW_AGENT, validate_action_json, fixed_limits
from pipeline.track2.build_oow_scenarios import (
    FIXED_QUESTION_UNIFIED, N_EVAL_PER_CATEGORY_DEFAULT, N_TRAIN_PER_CATEGORY_DEFAULT,
    generate_population, render_scenario_situation, split_eval_train, to_unified_action,
    wrong_action_variant, build_teacher_payload, contact_name_for_index,
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
        {"action": "turn_right", "degrees": 20.0, "encounter_rule": "Rule 15", "conduct_rule": "Rule 16"},
        {"action": "turn_left", "degrees": 15.0, "encounter_rule": "Rule 13", "conduct_rule": "Rule 13"},
        {"action": "hold_course", "degrees": None, "encounter_rule": "none", "conduct_rule": "none"},
        {"action": "stop", "degrees": None, "encounter_rule": "Rule 15", "conduct_rule": "Rule 8"},
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
        if unified["encounter_rule"] == "none":
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
    regenerating the same deterministic population and matching by position.

    Quality-review STAP 2 (2026-09-23) added max_turn_deg/risk_horizon_s to the shared
    constraint_line() -- the CURRENTLY-frozen oow_colreg_scenarios_v2.json predates that
    change (single safe-distance sentence only) and is explicitly scheduled for
    regeneration in STAP 4 (BLOK II), not yet approved/run -- skip rather than fail until
    that regeneration lands, so this stays a genuine identity check (not a stale-fixture
    false negative) once v2 is rebuilt against the new renderer.

    Screening-set-B audit follow-up (2026-09-23): constraint_line() was reworded again
    (encounter-identification vs action-is-mandatory split into two explicit sentences,
    replacing the single "BOTH...AND..." conjunction sentence) -- same situation, same
    skip-not-fail policy, gated on a phrase unique to the NEW wording."""
    if not V2_FILE.exists():
        return
    v2 = json.loads(V2_FILE.read_text(encoding="utf-8"))
    if ("may request at most" not in v2[0]["situation"]
            or "you must identify the encounter and the applicable steering rule" not in v2[0]["situation"]):
        pytest.skip("oow_colreg_scenarios_v2.json predates the current constraint_line() "
                   "wording -- pending its own regeneration")
    eval_recs, _ = _fresh_pop(seed=0)
    assert len(eval_recs) == len(v2)
    for rec, v2_rec in zip(eval_recs[:20], v2[:20]):
        limits = fixed_limits(v2_rec["safe_distance_m"], v2_rec["max_turn_deg"], rec["own_speed"])
        # What write_scenario_sft_files() would embed as the user turn for this geometry:
        training_user_text = f"Situation:\n{render_scenario_situation(rec, limits)}\n\n{FIXED_QUESTION_UNIFIED}"
        assert training_user_text == f"Situation:\n{v2_rec['situation']}\n\n{v2_rec['question']}", (
            f"v2 record {v2_rec['id']} situation text diverges from what a training row "
            "would embed for the same geometry -- the two paths must call the same renderer"
        )


def test_system_prompt_is_the_shared_object() -> None:
    from pipeline.track2 import build_oow_scenarios
    assert build_oow_scenarios.SYSTEM_OOW_AGENT is SYSTEM_OOW_AGENT


# ── Fase B3 point 2: the teacher-only decisive-contact hint must NEVER reach Qwen ──
def test_qwen_user_content_never_contains_the_decisive_contact_hint() -> None:
    """build_teacher_payload()'s `decisive_contact` field (name + CPA) is answer-side
    context for the reasoning-generation teacher only -- the actual Qwen user turn
    (render_scenario_situation() + FIXED_QUESTION_UNIFIED) must never mention it, and
    must be byte-identical to render_scenario_situation() alone (i.e. building the
    teacher payload must not have mutated/augmented the situation text itself)."""
    eval_recs, train_recs = _fresh_pop()
    multi_contact_recs = [r for r in train_recs if len(r["targets"]) > 1][:10]
    assert multi_contact_recs, "expected at least one multi-contact training record to test"
    for i, rec in enumerate(multi_contact_recs):
        situation = render_scenario_situation(rec)
        qwen_user_content = f"Situation:\n{situation}\n\n{FIXED_QUESTION_UNIFIED}"
        payload, _expected = build_teacher_payload(rec, f"t{i:05d}")
        assert payload["situation"] == situation
        if "decisive_contact" in payload:
            assert "decisive_contact" not in qwen_user_content
            assert "decisive" not in qwen_user_content.lower()


def test_contact_name_for_index_matches_the_pool_used_in_the_situation_text() -> None:
    eval_recs, train_recs = _fresh_pop()
    multi = next(r for r in train_recs if len(r["targets"]) > 1)
    situation = render_scenario_situation(multi)
    for i in range(len(multi["targets"])):
        assert contact_name_for_index(i) in situation


def test_written_sft_file_rows_all_pass_the_shared_schema_and_action_vocabulary() -> None:
    """B6 (real integration check, not just a unit-level property test): once Fase B3's
    full population has actually been written, every row in the real
    oow_scenario_sft_direct.jsonl must have an assistant JSON that both parses and passes
    validate_action_json() -- catching any real generation-time bug the in-memory tests
    above can't see. No-ops (returns) if the file hasn't been generated yet."""
    path = paths.cache_dir / "oow_scenario_sft_direct.jsonl"
    if not path.exists():
        return
    n_checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert row["messages"][-1]["role"] == "assistant"
        obj = json.loads(row["messages"][-1]["content"])
        errors = validate_action_json(obj)
        assert not errors, (row.get("category"), errors)
        assert obj["action"] in ACTIONS
        n_checked += 1
    assert n_checked > 0, f"{path} exists but is empty"
