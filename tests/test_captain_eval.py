"""Captain walking-skeleton Phase 5 tests: the minimal composite evaluation -- only
Safety/Mission-outcome/Resource-efficiency (design_captain_missions.md Sec 10.1/10.2/
13.C.11). Pure pipeline/ module -- no app/ dependency, no GPU/API key needed."""
import pytest

from pipeline.captain_eval import (
    MissionOutcomeFacts, ResourceEfficiencyFacts, evaluate_captain_mission,
    mission_outcome_axis, resource_efficiency_axis, safety_axis_captain,
)
from pipeline.captain_types import BrownEnvelopeEvent

_EVENT = BrownEnvelopeEvent(event_id="ev1", type="engine_failure", severity="moderate",
                            params={"capped_speed_kn": 8.0})
_EVENTS_BY_ID = {"ev1": _EVENT}

_CLEAN_DECISION_LOG = [
    {"t": 0.0, "field_path": "active_hazards", "old": None, "new": "ev1", "cause": "ev1"},
    {"t": 0.0, "field_path": "captain_decision",
     "new": {"tool": "continue_at_capped_speed", "params": {"speed_kn": 8.0}, "shield_substituted": False},
     "old": None, "cause": "ev1"},
]


# --- safety_axis_captain -------------------------------------------------------------------

def test_safety_axis_passes_for_a_clean_decision_log():
    passed, score, violations = safety_axis_captain(_CLEAN_DECISION_LOG, _EVENTS_BY_ID)
    assert passed
    assert score == 1.0
    assert violations == []


def test_safety_axis_fails_when_an_applied_action_actually_exceeds_the_cap():
    bad_log = [
        {"t": 0.0, "field_path": "captain_decision",
         "new": {"tool": "continue_at_capped_speed", "params": {"speed_kn": 10.0}, "shield_substituted": False},
         "old": None, "cause": "ev1"},
    ]
    passed, score, violations = safety_axis_captain(bad_log, _EVENTS_BY_ID)
    assert not passed
    assert score == 0.0
    assert len(violations) == 1


def test_safety_axis_ignores_non_decision_log_entries():
    log = [{"t": 0.0, "field_path": "active_hazards", "old": None, "new": "ev1", "cause": "ev1"}]
    passed, score, _ = safety_axis_captain(log, _EVENTS_BY_ID)
    assert passed and score == 1.0


def test_safety_axis_skips_a_decision_whose_event_has_no_declared_cap():
    log = [{"t": 0.0, "field_path": "captain_decision",
           "new": {"tool": "hold", "params": {}, "shield_substituted": False}, "old": None, "cause": "unknown_event"}]
    passed, score, _ = safety_axis_captain(log, _EVENTS_BY_ID)
    assert passed and score == 1.0


# --- mission_outcome_axis ------------------------------------------------------------------

def test_mission_outcome_all_criteria_met():
    facts = MissionOutcomeFacts(reached_destination=True, fuel_tonnes_remaining=100.0,
                                elapsed_h=28.0, eta_deadline_h=32.0)
    score, unmet = mission_outcome_axis(facts)
    assert score == 1.0
    assert unmet == []


def test_mission_outcome_scores_zero_when_never_arrived():
    facts = MissionOutcomeFacts(reached_destination=False, fuel_tonnes_remaining=100.0,
                                elapsed_h=10.0, eta_deadline_h=32.0)
    score, unmet = mission_outcome_axis(facts)
    assert score == 0.0
    assert len(unmet) == 2


def test_mission_outcome_partial_when_deadline_breached_but_cargo_delivered():
    facts = MissionOutcomeFacts(reached_destination=True, fuel_tonnes_remaining=100.0,
                                elapsed_h=40.0, eta_deadline_h=32.0)
    score, unmet = mission_outcome_axis(facts)
    assert score == 0.5
    assert unmet == ["arrive within the stated ETA deadline"]


def test_mission_outcome_unmet_when_fuel_ran_out_despite_arriving():
    facts = MissionOutcomeFacts(reached_destination=True, fuel_tonnes_remaining=0.0,
                                elapsed_h=20.0, eta_deadline_h=32.0)
    score, unmet = mission_outcome_axis(facts)
    assert "cargo delivered intact" in unmet


def test_mission_outcome_no_deadline_means_that_criterion_is_always_met():
    facts = MissionOutcomeFacts(reached_destination=True, fuel_tonnes_remaining=1.0,
                                elapsed_h=999.0, eta_deadline_h=None)
    score, unmet = mission_outcome_axis(facts)
    assert score == 1.0


# --- resource_efficiency_axis --------------------------------------------------------------

def test_resource_efficiency_perfect_when_actual_matches_reference():
    facts = ResourceEfficiencyFacts(reference_fuel_t=100.0, reference_time_h=10.0,
                                    actual_fuel_t=100.0, actual_time_h=10.0)
    result = resource_efficiency_axis(facts)
    assert result["fuel_score"] == 1.0
    assert result["time_score"] == 1.0
    assert result["score"] == 1.0


def test_resource_efficiency_clamps_at_one_when_actual_beats_the_reference():
    facts = ResourceEfficiencyFacts(reference_fuel_t=100.0, reference_time_h=10.0,
                                    actual_fuel_t=50.0, actual_time_h=10.0)
    result = resource_efficiency_axis(facts)
    assert result["fuel_score"] == 1.0  # no extra credit for beating the unconstrained ideal


def test_resource_efficiency_degrades_when_actual_costs_more_than_reference():
    facts = ResourceEfficiencyFacts(reference_fuel_t=100.0, reference_time_h=10.0,
                                    actual_fuel_t=100.0, actual_time_h=20.0)
    result = resource_efficiency_axis(facts)
    assert result["time_score"] == pytest.approx(0.5)


# --- evaluate_captain_mission (composite) --------------------------------------------------

def test_composite_pass_for_a_clean_on_time_mission():
    outcome = MissionOutcomeFacts(reached_destination=True, fuel_tonnes_remaining=400.0,
                                  elapsed_h=28.35, eta_deadline_h=32.0)
    resource = ResourceEfficiencyFacts(reference_fuel_t=722.5, reference_time_h=20.01,
                                       actual_fuel_t=445.9, actual_time_h=28.35)
    result = evaluate_captain_mission(_CLEAN_DECISION_LOG, _EVENTS_BY_ID, outcome, resource)
    assert result.verdict == "PASS"
    assert result.safety_passed
    assert result.mission_outcome_score == 1.0
    assert 0.0 < result.composite_score <= (0.25 + 0.20 + 0.07)


def test_composite_hard_gate_1_zeroes_out_on_a_safety_violation():
    bad_log = [
        {"t": 0.0, "field_path": "captain_decision",
         "new": {"tool": "continue_at_capped_speed", "params": {"speed_kn": 10.0}, "shield_substituted": False},
         "old": None, "cause": "ev1"},
    ]
    outcome = MissionOutcomeFacts(reached_destination=True, fuel_tonnes_remaining=400.0,
                                  elapsed_h=28.35, eta_deadline_h=32.0)
    resource = ResourceEfficiencyFacts(reference_fuel_t=722.5, reference_time_h=20.01,
                                       actual_fuel_t=445.9, actual_time_h=28.35)
    result = evaluate_captain_mission(bad_log, _EVENTS_BY_ID, outcome, resource)
    assert result.composite_score == 0.0
    assert result.verdict.startswith("FAIL -- safety")


def test_composite_hard_gate_2_caps_the_score_when_destination_never_reached():
    outcome = MissionOutcomeFacts(reached_destination=False, fuel_tonnes_remaining=0.0,
                                  elapsed_h=50.0, eta_deadline_h=32.0)
    resource = ResourceEfficiencyFacts(reference_fuel_t=722.5, reference_time_h=20.01,
                                       actual_fuel_t=850.0, actual_time_h=50.0)
    result = evaluate_captain_mission(_CLEAN_DECISION_LOG, _EVENTS_BY_ID, outcome, resource)
    assert result.composite_score <= 0.2
    assert result.verdict.startswith("FAIL -- did not reach")
