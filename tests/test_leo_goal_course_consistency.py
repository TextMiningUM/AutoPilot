"""Fase B2 consistency test (RAG-rebuild-v2 plan, 2026-09-22): for every generated
training example WITHOUT real collision risk (Fase B1's Rule-7 gate says "no risk"), the
label must equal exactly what GOAL COURSE CHECK recommends for that state's OWN input --
hold_course if already on the goal bearing, else the named turn+degrees (and speed_up if
under cruise speed). A mismatch would mean the model is being trained to IGNORE its own
situation report -- see the master plan's point 2 for the full rationale.

This test currently runs over a moderate sample of Leo's raw states; the user's plan asks
for it to be run over the FULL 7928-state set once Fase B3 (real reasoning generation)
lands -- it is deterministic and needs no LLM output, so it already exercises the real
logic today.
"""
import json

from pipeline.oow_agent_spec import goal_course_action
from pipeline.track2.build_oow_scenarios_leo import (
    leo_choose_action, stratified_sample, LEO_FILE, RESUME_SPEED_MARGIN,
)

NO_RISK_BUCKETS = {"resume", "clear"}


def test_no_risk_labels_match_goal_course_check_on_a_sample() -> None:
    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    sample = stratified_sample(all_recs, 200, seed=7)
    checked = 0
    for r in sample:
        state = r["state"]
        decision = leo_choose_action(state)
        if decision["bucket"] not in NO_RISK_BUCKETS:
            continue
        checked += 1
        own, mission = state["own_ship"], state["mission"]
        goal_action, goal_degrees = goal_course_action(own["x"], own["y"], own["heading"],
                                                       mission["x"], mission["y"])
        if goal_action != "hold_course":
            assert decision["action"] == goal_action, r["id"]
            assert decision["degrees"] == goal_degrees, r["id"]
        elif own["speed"] < own["target_speed"] - RESUME_SPEED_MARGIN:
            assert decision["action"] == "speed_up", r["id"]
        else:
            assert decision["action"] == "hold_course", r["id"]
        assert decision["encounter_rule"] == "none" and decision["conduct_rule"] == "none", r["id"]
    assert checked > 0, "sample contained no no-real-risk records -- test didn't check anything"


def test_full_leo_dataset_no_risk_labels_match_goal_course_check() -> None:
    """The FULL-dataset version the master plan asks to run after Fase B3 -- already
    runnable today since it only needs the deterministic action label, not the [LLM]
    reasoning text. Kept as its own (slower) test rather than folding into the sampled
    one above so CI can skip/mark it separately if the full 7928-state pass is ever too
    slow to run on every commit."""
    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    mismatches = []
    for r in all_recs:
        state = r["state"]
        decision = leo_choose_action(state)
        if decision["bucket"] not in NO_RISK_BUCKETS:
            continue
        own, mission = state["own_ship"], state["mission"]
        goal_action, goal_degrees = goal_course_action(own["x"], own["y"], own["heading"],
                                                       mission["x"], mission["y"])
        expected = goal_action
        if goal_action == "hold_course" and own["speed"] < own["target_speed"] - RESUME_SPEED_MARGIN:
            expected = "speed_up"
        if decision["action"] != expected or (goal_action != "hold_course" and decision["degrees"] != goal_degrees):
            mismatches.append(r["id"])
    assert not mismatches, f"{len(mismatches)} no-real-risk record(s) disagree with GOAL COURSE CHECK: {mismatches[:10]}"
