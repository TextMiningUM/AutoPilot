"""Captain walking-skeleton Phase 7 tests: distress_call + commercial_instruction wired
end-to-end, including the WORLD_RESPONDER_TIMER mechanism (design_captain_missions.md
Sec 13.A.2/13.A.9) -- a fixed, scripted reply, never an open-ended negotiation. No GPU/API
key needed."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_skeleton import CaptainSkeleton  # noqa: E402

REPO_ROOT = APP_ROOT.parent
SCENARIO_PATH = REPO_ROOT / "Data" / "Captain" / "Scenarios" / "skeleton_distress_commercial_v1.json"


def _fresh_skeleton() -> CaptainSkeleton:
    return CaptainSkeleton.from_scenario_file(SCENARIO_PATH)


def _decisions(skeleton: CaptainSkeleton) -> list[dict]:
    return [d for d in skeleton.state.event_log if d["field_path"] == "captain_decision"]


def _run_to_completion(skeleton: CaptainSkeleton) -> None:
    worst_case_duration_s = (skeleton.sim.total_route_distance_nm / skeleton.order.speed_of_advance_kn) * 3600.0
    total_steps = int(worst_case_duration_s // skeleton.sim.dt_mission_s) + 10
    skeleton.step_mission(total_steps)


def test_distress_call_fires_and_assists_since_the_detour_is_affordable():
    skeleton = _fresh_skeleton()
    event = skeleton.run_to_next_event()
    assert event.event_id == "ev_distress"
    decision = _decisions(skeleton)[-1]
    assert decision["new"]["tool"] == "proceed_to_assist"
    assert decision["new"]["shield_substituted"] is False


def test_commercial_instruction_fires_and_refuses_since_it_exceeds_the_engine_cap():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()  # ev_distress
    event = skeleton.run_to_next_event()  # ev_commercial
    assert event.event_id == "ev_commercial"
    decision = _decisions(skeleton)[-1]
    assert decision["new"]["tool"] == "refuse_citing_ism_art5"
    # The decision layer already correctly avoided the breach on its own (duty_breaches
    # dominates) -- the shield had nothing left to override.
    assert decision["new"]["shield_substituted"] is False
    assert skeleton.sim.state.current_speed_kn == 12.0  # refused -- speed unchanged


def test_world_responder_resolves_the_distress_call_after_its_own_scripted_delay():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()
    _run_to_completion(skeleton)
    hazard = next(h for h in skeleton.state.active_hazards if h["event_id"] == "ev_distress")
    assert hazard["resolution"] == "survivors_recovered"
    assert hazard["cleared_at_s"] == pytest.approx(45000.0 + 3600.0)


def test_world_responder_resolves_the_commercial_instruction_after_its_own_scripted_delay():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()
    skeleton.run_to_next_event()
    _run_to_completion(skeleton)
    hazard = next(h for h in skeleton.state.active_hazards if h["event_id"] == "ev_commercial")
    assert hazard["resolution"] == "company_accepts_refusal"
    assert hazard["cleared_at_s"] == pytest.approx(54000.0 + 1800.0)


def test_world_responder_resolution_does_not_change_the_already_recorded_decision():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()
    decision_before = dict(_decisions(skeleton)[-1])
    _run_to_completion(skeleton)
    decision_after = next(d for d in _decisions(skeleton) if d["cause"] == "ev_distress")
    assert decision_after["new"] == decision_before["new"]


def test_full_mission_evaluates_cleanly_with_no_safety_violations():
    skeleton = _fresh_skeleton()
    skeleton.run_to_next_event()
    skeleton.run_to_next_event()
    _run_to_completion(skeleton)
    assert skeleton.sim.reached_destination()

    result = skeleton.evaluate()
    assert result.verdict == "PASS"
    assert result.safety_passed
    assert result.safety_violations == []
    assert result.mission_outcome_score == 1.0
