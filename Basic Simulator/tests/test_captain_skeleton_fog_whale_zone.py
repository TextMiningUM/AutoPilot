"""Captain walking-skeleton Phase 6 tests: the fog + whale_zone decision layers wired
end-to-end, including the speed-cap-stacking rule (design_captain_missions.md Sec
13.A.4) -- the two scripted windows deliberately overlap so the MOST restrictive active
cap must always win, regardless of firing order. No GPU/API key needed."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_skeleton import CaptainSkeleton  # noqa: E402

REPO_ROOT = APP_ROOT.parent
SCENARIO_PATH = REPO_ROOT / "Data" / "Captain" / "Scenarios" / "skeleton_fog_whale_zone_v1.json"


def _fresh_skeleton() -> CaptainSkeleton:
    return CaptainSkeleton.from_scenario_file(SCENARIO_PATH)


def _step_to_distance_nm(skeleton: CaptainSkeleton, target_nm: float) -> None:
    while skeleton.sim.state.distance_travelled_nm < target_nm and not skeleton.sim.reached_destination():
        skeleton.step_mission(1)


def test_fog_fires_first_and_caps_the_speed_to_six():
    skeleton = _fresh_skeleton()
    event = skeleton.run_to_next_event()
    assert event.event_id == "ev_fog"
    assert skeleton.sim.state.current_speed_kn == 6.0


def test_fog_cap_still_dominates_once_whale_zone_also_fires():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()  # ev_fog at nm 20
    event = skeleton.run_to_next_event()  # ev_whale at nm 35, inside fog's own window [20,45]
    assert event.event_id == "ev_whale"
    assert skeleton.sim.state.current_speed_kn == 6.0  # fog's tighter cap still wins


def test_fog_clears_and_the_looser_whale_zone_cap_takes_over():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()  # ev_fog
    skeleton.run_to_next_event()  # ev_whale
    _step_to_distance_nm(skeleton, 45.0)  # ev_fog's own window ends here
    assert skeleton.sim.state.current_speed_kn == 10.0


def test_whale_zone_clears_and_speed_returns_to_the_original_soa():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()
    skeleton.run_to_next_event()
    _step_to_distance_nm(skeleton, 55.0)  # ev_whale's own window ends here
    assert skeleton.sim.state.current_speed_kn == 12.0


def test_active_hazards_are_marked_cleared_once_their_window_ends():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()
    skeleton.run_to_next_event()
    _step_to_distance_nm(skeleton, 55.0)
    hazard_ids = {h["event_id"]: h for h in skeleton.state.active_hazards}
    assert "cleared_at_s" in hazard_ids["ev_fog"]
    assert "cleared_at_s" in hazard_ids["ev_whale"]


def test_full_mission_evaluates_cleanly_with_no_safety_violations():
    skeleton = _fresh_skeleton()
    worst_case_duration_s = (skeleton.sim.total_route_distance_nm / 6.0) * 3600.0
    total_steps = int(worst_case_duration_s // skeleton.sim.dt_mission_s) + 5
    skeleton.step_mission(total_steps)
    assert skeleton.sim.reached_destination()

    result = skeleton.evaluate()
    assert result.verdict == "PASS"
    assert result.safety_passed
    assert result.safety_violations == []
    assert result.mission_outcome_score == 1.0
