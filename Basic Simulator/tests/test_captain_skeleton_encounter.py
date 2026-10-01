"""Captain walking-skeleton Phase 9c/9d tests: the encounter-sim splice wired end-to-end
(design_captain_missions.md Sec 13.A.1's 5 concretising points) -- Poisson-scheduled
ambient traffic spliced into the EXISTING live encounter-sim (app.oracle_planner.plan()
as the deterministic OOW, Sec 13.C.15), while the mission-sim's own clock/fuel keep
advancing, and the mission's own exclusion zones (point 5) are projected into the live
encounter frame and passed into the oracle's hard gate. No GPU/API key needed."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_skeleton import ENCOUNTER_DT_S, MAX_ENCOUNTER_STEPS, CaptainSkeleton  # noqa: E402
from pipeline.captain_types import ExclusionZone  # noqa: E402

REPO_ROOT = APP_ROOT.parent
AMBIENT_SCENARIO = REPO_ROOT / "Data" / "Captain" / "Scenarios" / "skeleton_ambient_encounter_v1.json"
ENGINE_FAILURE_SCENARIO = REPO_ROOT / "Data" / "Captain" / "Scenarios" / "skeleton_engine_failure_v1.json"


def _fresh_skeleton() -> CaptainSkeleton:
    return CaptainSkeleton.from_scenario_file(AMBIENT_SCENARIO)


def _encounters(skeleton: CaptainSkeleton) -> list[dict]:
    return [d["new"] for d in skeleton.state.event_log if d["field_path"] == "encounter"]


def _run_to_completion(skeleton: CaptainSkeleton) -> None:
    worst_case_s = (skeleton.sim.total_route_distance_nm / skeleton.order.speed_of_advance_kn) * 3600.0
    total_steps = int(worst_case_s // skeleton.sim.dt_mission_s) + 20
    for _ in range(total_steps):
        if skeleton.sim.reached_destination():
            return
        skeleton.step_mission(1)


# --- Backward compatibility -----------------------------------------------------------

def test_a_scenario_with_no_ambient_key_has_no_arrival_schedule():
    skeleton = CaptainSkeleton.from_scenario_file(ENGINE_FAILURE_SCENARIO)
    assert skeleton._ambient_arrival_times_s == []


def test_existing_engine_failure_scenario_is_completely_unaffected():
    # No "ambient" key at all -- _check_ambient() must be a pure no-op, zero behaviour
    # change for every scenario built before Phase 9c.
    skeleton = CaptainSkeleton.from_scenario_file(ENGINE_FAILURE_SCENARIO)
    skeleton.run_to_next_event()
    assert _encounters(skeleton) == []


# --- The ambient-traffic scenario --------------------------------------------------------

def test_dense_traffic_produces_a_non_empty_arrival_schedule():
    skeleton = _fresh_skeleton()
    assert len(skeleton._ambient_arrival_times_s) > 0


def test_full_mission_runs_to_completion_through_multiple_spliced_encounters():
    skeleton = _fresh_skeleton()
    _run_to_completion(skeleton)
    assert skeleton.sim.reached_destination()
    encounters = _encounters(skeleton)
    assert len(encounters) == len(skeleton._ambient_arrival_times_s)
    for e in encounters:
        assert e["duration_s"] < MAX_ENCOUNTER_STEPS * ENCOUNTER_DT_S  # actually resolved, didn't hit the cap
        assert e["ended_at_s"] >= e["started_at_s"]


def test_mission_clock_and_fuel_advance_consistently_across_encounters():
    skeleton = _fresh_skeleton()
    fuel_before = skeleton.sim.state.fuel_tonnes
    elapsed_before = skeleton.sim.state.elapsed_s
    _run_to_completion(skeleton)
    assert skeleton.sim.state.fuel_tonnes < fuel_before  # real fuel was consumed
    assert skeleton.sim.state.elapsed_s > elapsed_before
    assert skeleton.sim.state.distance_travelled_nm == pytest.approx(skeleton.sim.total_route_distance_nm, abs=1e-6)


def test_speed_regime_is_restored_after_an_encounter_ends():
    skeleton = _fresh_skeleton()
    skeleton.step_mission(1)  # advance past t=0 so _check_ambient() has a chance to fire
    while not _encounters(skeleton) and not skeleton.sim.reached_destination():
        skeleton.step_mission(1)
    assert _encounters(skeleton)  # at least one encounter happened
    # Back to the Captain's own planned SOA (no active caps in this scenario) once clear.
    assert skeleton.sim.state.current_speed_kn == pytest.approx(skeleton.order.speed_of_advance_kn)


def test_full_mission_evaluates_cleanly_despite_the_encounters():
    skeleton = _fresh_skeleton()
    _run_to_completion(skeleton)
    result = skeleton.evaluate()
    assert result.verdict == "PASS"
    assert result.safety_passed
    assert result.mission_outcome_score == 1.0


# --- avoid_zone projection into the encounter frame (Sec 13.A.1 point 5, Phase 9d) ---------

def test_an_active_exclusion_zone_does_not_break_the_mission():
    # A real zone positioned directly on the corridor's own route, near the start --
    # exercises the project-zones-into-the-encounter-frame + oracle hard-gate plumbing on
    # every one of this scenario's 23 encounters without special-casing any of them.
    skeleton = _fresh_skeleton()
    skeleton.zones = [ExclusionZone(id="test_zone", type="dynamic_hazard",
                                    polygon=[(52.05, 2.999), (52.05, 3.001),
                                            (52.06, 3.001), (52.06, 2.999)])]
    _run_to_completion(skeleton)
    assert skeleton.sim.reached_destination()


def test_zones_are_projected_into_the_live_encounter_origin():
    from pipeline.captain_geo import project_zone_to_local_frame
    skeleton = _fresh_skeleton()
    zone = ExclusionZone(id="test_zone", type="dynamic_hazard",
                        polygon=[(52.05, 2.999), (52.05, 3.001), (52.06, 3.001), (52.06, 2.999)])
    origin = skeleton.order.waypoints[0]  # distance_travelled_nm == 0 at mission start
    projected = project_zone_to_local_frame(zone, origin)
    # The zone sits north + very slightly off the corridor's own due-north leg -- its
    # projected y should be positive (ahead) and x close to 0 (near the track).
    assert all(y > 0 for _x, y in projected.polygon)
    assert all(abs(x) < 500.0 for x, _y in projected.polygon)
