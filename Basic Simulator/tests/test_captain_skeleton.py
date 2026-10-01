"""Captain walking-skeleton Phase 4 tests: end-to-end wiring of MissionSim + the route
planner + the procedure library/shield/decision layer, driven via the Sec 15.3 debug
control set, over the one concrete scenario file
(Data/Captain/Scenarios/skeleton_engine_failure_v1.json). No GPU/API key needed."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_skeleton import (  # noqa: E402
    CaptainSkeleton, build_mission_order_from_scenario, build_region_data, load_scenario,
)

REPO_ROOT = APP_ROOT.parent
SCENARIO_PATH = REPO_ROOT / "Data" / "Captain" / "Scenarios" / "skeleton_engine_failure_v1.json"


def _fresh_skeleton() -> CaptainSkeleton:
    return CaptainSkeleton.from_scenario_file(SCENARIO_PATH)


# --- Scenario loading ----------------------------------------------------------------

def test_load_scenario_has_the_expected_top_level_shape():
    scenario = load_scenario(SCENARIO_PATH)
    assert scenario["mission_id"] == "MSN-SKELETON-0001"
    assert len(scenario["brown_envelopes"]) == 1
    assert scenario["brown_envelopes"][0]["type"] == "engine_failure"


def test_build_mission_order_injects_waypoints_and_events():
    scenario = load_scenario(SCENARIO_PATH)
    order = build_mission_order_from_scenario(scenario)
    assert order.mission_id == "MSN-SKELETON-0001"
    assert len(order.waypoints) == 3
    assert len(order.events) == 1
    assert order.events[0].type == "engine_failure"
    assert order.speed_of_advance_kn == 12.0


def test_build_region_data_resolves_the_stub_corridor_ports():
    scenario = load_scenario(SCENARIO_PATH)
    zones, ports = build_region_data(scenario)
    assert zones == []  # the scenario's own stub region has no exclusion zones
    port_ids = {p.id for p in ports}
    assert "port_skeleton_refuge" in port_ids


# --- CaptainSkeleton construction ------------------------------------------------------

def test_skeleton_builds_with_a_fresh_mission_state():
    skeleton = _fresh_skeleton()
    assert skeleton.sim.state.elapsed_s == 0.0
    assert skeleton.sim.state.distance_travelled_nm == 0.0
    assert skeleton.sim.state.fuel_tonnes == 850.0
    assert skeleton.sim.state.current_speed_kn == 12.0
    assert skeleton.fired_event_ids == set()
    assert len(skeleton.pending_events) == 1


# --- Debug control set (Sec 15.3) ------------------------------------------------------

def test_run_to_next_event_fires_the_engine_failure_at_the_scripted_trigger():
    skeleton = _fresh_skeleton()
    event = skeleton.run_to_next_event()
    assert event is not None
    assert event.event_id == "ev1"
    assert "ev1" in skeleton.fired_event_ids
    assert skeleton.sim.state.distance_travelled_nm == pytest.approx(40.0, abs=1e-6)


def test_engine_failure_decision_caps_the_speed_and_logs_a_delta():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()
    # Deadline pressure (Sec 14's own worked example): diverting's ~30h detour+repair
    # breaches the Mission Order's ETA deadline, continuing at the capped speed does not --
    # the deterministic oracle_best baseline Captain should therefore choose to continue.
    assert skeleton.sim.state.current_speed_kn == 8.0
    decisions = [d for d in skeleton.state.event_log if d["field_path"] == "captain_decision"]
    assert len(decisions) == 1
    assert decisions[0]["new"]["tool"] == "continue_at_capped_speed"
    assert decisions[0]["new"]["shield_substituted"] is False


def test_run_to_next_event_returns_none_once_the_mission_is_exhausted():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()  # fires ev1
    assert skeleton.run_to_next_event() is None  # no more pending events before destination
    assert skeleton.sim.reached_destination()


def test_force_event_fires_regardless_of_its_own_trigger_condition():
    skeleton = _fresh_skeleton()
    event = skeleton.force_event("ev1")
    assert event.event_id == "ev1"
    assert skeleton.sim.state.distance_travelled_nm == 0.0  # forced before the trigger distance
    assert "ev1" in skeleton.fired_event_ids


def test_force_event_raises_for_an_unknown_event_id():
    skeleton = _fresh_skeleton()
    with pytest.raises(ValueError):
        skeleton.force_event("does_not_exist")


def test_force_event_raises_if_already_fired():
    skeleton = _fresh_skeleton()
    skeleton.force_event("ev1")
    with pytest.raises(ValueError):
        skeleton.force_event("ev1")


def test_step_mission_advances_the_clock_and_stops_at_the_destination():
    skeleton = _fresh_skeleton()
    # step_mission() fires the scripted engine failure mid-route, which caps the speed at
    # 8kn for the remainder -- budget steps generously at the SLOWEST post-event speed, not
    # at the original (faster) nominal_duration_s, so the budget is never too tight.
    worst_case_duration_s = (skeleton.sim.total_route_distance_nm / 8.0) * 3600.0
    total_steps = int(worst_case_duration_s // skeleton.sim.dt_mission_s) + 5
    skeleton.step_mission(total_steps)
    assert skeleton.sim.reached_destination()


def test_show_mission_state_mentions_the_mission_id():
    skeleton = _fresh_skeleton()
    assert skeleton.state.mission_id in skeleton.show_mission_state()


def test_show_procedure_lookup_reports_the_engine_failure_shield():
    skeleton = _fresh_skeleton()
    event = skeleton.order.events[0]
    text = skeleton.show_procedure_lookup(event)
    assert "safe speed" in text
    assert "fault log entry" in text


def test_show_mpr_reports_fuel_and_eta_before_any_event_fires():
    skeleton = _fresh_skeleton()
    report = skeleton.show_mpr()
    assert "MISSION PROGRESS REPORT" in report
    assert "850.0" in report
    assert "none" in report  # no active hazards yet


def test_show_mpr_reflects_the_capped_speed_after_the_engine_failure():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()
    report = skeleton.show_mpr()
    assert "engine_failure" in report
    assert "8.0 kn" in report


# --- Minimal evaluation (Sec 10, Phase 5) -----------------------------------------------

def _run_to_completion(skeleton: CaptainSkeleton) -> None:
    worst_case_duration_s = (skeleton.sim.total_route_distance_nm / 8.0) * 3600.0
    total_steps = int(worst_case_duration_s // skeleton.sim.dt_mission_s) + 5
    skeleton.step_mission(total_steps)


def test_evaluate_a_full_run_passes_with_no_safety_violation():
    skeleton = _fresh_skeleton()
    _run_to_completion(skeleton)
    result = skeleton.evaluate()
    assert result.verdict == "PASS"
    assert result.safety_passed
    assert result.safety_violations == []
    assert result.mission_outcome_score == 1.0  # arrived, within deadline, fuel remaining


def test_evaluate_resource_efficiency_reflects_the_capped_speed_detour():
    skeleton = _fresh_skeleton()
    _run_to_completion(skeleton)
    result = skeleton.evaluate()
    # The capped-speed leg costs real extra TIME vs the unconstrained 12kn oracle baseline
    # -- time_score must be meaningfully below 1.0 even though the mission still passed.
    assert result.resource_efficiency["time_score"] < 1.0
    assert 0.0 < result.composite_score <= (0.25 + 0.20 + 0.07)


def test_evaluate_before_the_mission_completes_hits_hard_gate_2():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()  # engine failure fires, but destination not yet reached
    result = skeleton.evaluate()
    assert not skeleton.sim.reached_destination()
    assert result.composite_score <= 0.2
    assert result.verdict.startswith("FAIL -- did not reach")
