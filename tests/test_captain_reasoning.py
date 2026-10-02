"""Tests for pipeline/captain_reasoning.py (design_captain_missions.md Sec 16.2/16.4
skeleton) -- the deterministic plan/reasoning renderer. No GPU/API key needed."""
import pytest

from pipeline.captain_agent_spec import CaptainAction, CostDimensions, RolloutResult
from pipeline.captain_reasoning import assess_goal_achievability, render_captain_response

GOALS = [{"goal": "cargo delivered intact", "status": "open"},
        {"goal": "arrive within the stated ETA deadline", "status": "open"}]


def test_assess_goal_achievability_open_when_slack_is_positive():
    status, note = assess_goal_achievability(5.0)
    assert status == "open"
    assert "still achievable" in note


def test_assess_goal_achievability_at_risk_when_slack_is_negative():
    status, note = assess_goal_achievability(-3.0)
    assert status == "at_risk"
    assert "miss the deadline" in note.lower()
    assert "3.0" in note


def test_assess_goal_achievability_handles_no_stated_deadline():
    status, note = assess_goal_achievability(None)
    assert status == "open"
    assert "no stated" in note.lower()


def test_render_captain_response_shape_and_content():
    rollout = RolloutResult(dims=CostDimensions(duty_breaches=0.0, goal_shortfall=0.1, commercial=5.0),
                            notes={"time_h": 10.0})
    applied = CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": 8.0})
    response = render_captain_response(
        event_type="engine_failure", applied=applied, chosen_rollout=rollout, regret_value=0.0,
        shield_substituted=False, goals=GOALS, deadline_slack_h=4.0,
    )
    assert response.action is applied
    assert "continue_at_capped_speed" in response.reasoning
    assert "zero regret" in response.reasoning
    assert "commercial=5.0" in response.reasoning
    assert response.plan["resource_note"] and "still achievable" in response.plan["resource_note"]
    assert all(g["status"] == "open" for g in response.plan["updated_goals"])


def test_render_captain_response_marks_the_deadline_goal_at_risk():
    rollout = RolloutResult(dims=CostDimensions())
    applied = CaptainAction(tool="hold", params={})
    response = render_captain_response(
        event_type="commercial_instruction", applied=applied, chosen_rollout=rollout, regret_value=2.5,
        shield_substituted=True, goals=GOALS, deadline_slack_h=-1.5,
    )
    updated = {g["goal"]: g["status"] for g in response.plan["updated_goals"]}
    assert updated["arrive within the stated ETA deadline"] == "at_risk"
    assert updated["cargo delivered intact"] == "open"  # only the deadline goal changes
    assert "regret=2.5" in response.reasoning
    assert "shield" in response.reasoning.lower()


def test_render_captain_response_never_mutates_the_input_goals_list():
    original = [dict(g) for g in GOALS]
    rollout = RolloutResult(dims=CostDimensions())
    render_captain_response(event_type="fog", applied=CaptainAction(tool="hold", params={}),
                            chosen_rollout=rollout, regret_value=0.0, shield_substituted=False,
                            goals=GOALS, deadline_slack_h=-10.0)
    assert GOALS == original
