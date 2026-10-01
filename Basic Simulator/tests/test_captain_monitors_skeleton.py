"""Captain walking-skeleton Phase 8 tests: the 3 discrete trigger monitors + the
monitor-engagement log wired end-to-end (design_captain_missions.md Sec 13.B.7/13.B.9),
over the existing engine-failure and fog/whale_zone scenarios (no new scenario file
needed -- this phase adds a cross-cutting mechanism, not a new event type). No GPU/API
key needed."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_skeleton import CaptainSkeleton  # noqa: E402

REPO_ROOT = APP_ROOT.parent
ENGINE_FAILURE_SCENARIO = REPO_ROOT / "Data" / "Captain" / "Scenarios" / "skeleton_engine_failure_v1.json"
DISTRESS_COMMERCIAL_SCENARIO = (REPO_ROOT / "Data" / "Captain" / "Scenarios"
                                / "skeleton_distress_commercial_v1.json")
FOG_WHALE_SCENARIO = REPO_ROOT / "Data" / "Captain" / "Scenarios" / "skeleton_fog_whale_zone_v1.json"


def _engagements(skeleton: CaptainSkeleton) -> list[dict]:
    return [d["new"] for d in skeleton.state.event_log if d["field_path"] == "monitor_engagement"]


def test_engine_failure_logs_a_monitor_engagement_meeting_its_mandatory_deadline():
    skeleton = CaptainSkeleton.from_scenario_file(ENGINE_FAILURE_SCENARIO)
    skeleton.run_to_next_event()
    engagements = _engagements(skeleton)
    assert len(engagements) == 1
    entry = engagements[0]
    assert entry["monitor"] == "machinery_fault_reported"
    assert entry["urgency"] == "mandatory"
    assert entry["met_deadline"] is True
    assert entry["responded_at_s"] == entry["trigger_fired_at_s"]  # synchronous baseline
    assert entry["deadline_s"] == entry["trigger_fired_at_s"] + skeleton.sim.dt_mission_s


def test_engine_failure_hazard_makes_machinery_fault_reported_active():
    skeleton = CaptainSkeleton.from_scenario_file(ENGINE_FAILURE_SCENARIO)
    assert skeleton.active_monitors() == []
    skeleton.run_to_next_event()
    assert "machinery_fault_reported" in skeleton.active_monitors()


def test_distress_call_and_commercial_instruction_each_log_their_own_engagement():
    skeleton = CaptainSkeleton.from_scenario_file(DISTRESS_COMMERCIAL_SCENARIO)
    skeleton.run_to_next_event()  # ev_distress
    skeleton.run_to_next_event()  # ev_commercial
    engagements = _engagements(skeleton)
    assert len(engagements) == 2
    assert {e["monitor"] for e in engagements} == {"distress_signal_received", "company_instruction_received"}
    assert all(e["met_deadline"] for e in engagements)


def test_distress_signal_received_clears_once_the_world_responder_resolves():
    skeleton = CaptainSkeleton.from_scenario_file(DISTRESS_COMMERCIAL_SCENARIO)
    skeleton.run_to_next_event()
    assert "distress_signal_received" in skeleton.active_monitors()
    worst_case_s = (skeleton.sim.total_route_distance_nm / skeleton.order.speed_of_advance_kn) * 3600.0
    skeleton.step_mission(int(worst_case_s // skeleton.sim.dt_mission_s) + 10)
    assert "distress_signal_received" not in skeleton.active_monitors()


def test_whale_zone_is_excluded_but_fog_still_logs_its_own_engagement():
    # whale_zone is correctly EXCLUDED from the mapping entirely (Sec 13.B.9's own
    # "whale zone resolved" case) -- no engagement logged for it. Fog DOES get a
    # monitor_engagement entry (its own trigger-fired/responded/deadline facts are
    # computable regardless of live ambient tracking) even though `active_monitors()`
    # doesn't surface it (that's reserved for the 3 DISCRETE monitors, Phase 8's scope).
    skeleton = CaptainSkeleton.from_scenario_file(FOG_WHALE_SCENARIO)
    skeleton.run_to_next_event()  # ev_fog
    skeleton.run_to_next_event()  # ev_whale
    engagements = _engagements(skeleton)
    assert len(engagements) == 1
    assert engagements[0]["monitor"] == "visibility_below"
    assert skeleton.active_monitors() == []


def test_show_mission_state_reports_active_monitors():
    skeleton = CaptainSkeleton.from_scenario_file(ENGINE_FAILURE_SCENARIO)
    skeleton.run_to_next_event()
    assert "machinery_fault_reported" in skeleton.show_mission_state()
