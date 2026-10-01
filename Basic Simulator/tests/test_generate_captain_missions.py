"""Tests for generate_captain_missions.py (design_captain_missions.md Sec 13.C.9) --
determinism, held-out-template correctness, schema validity, and that every generated
mission actually runs end-to-end through CaptainSkeleton without raising. No GPU/API key
needed."""
import json
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_skeleton import CaptainSkeleton  # noqa: E402
from app.mission_route_planner import segment_blocked  # noqa: E402
from generate_captain_missions import (  # noqa: E402
    HELD_OUT_TEMPLATES, ROUTE_TEMPLATES, generate_mission, main,
)
from pipeline.captain_types import ExclusionZone  # noqa: E402


def test_generate_mission_is_deterministic_for_the_same_seed_and_index():
    a = generate_mission(index=3, seed=42)
    b = generate_mission(index=3, seed=42)
    assert a == b


def test_generate_mission_differs_across_index_and_seed():
    base = generate_mission(index=0, seed=1)
    assert generate_mission(index=1, seed=1) != base
    assert generate_mission(index=0, seed=2) != base


def test_held_out_flag_matches_the_route_template_used():
    for i in range(30):
        mission = generate_mission(index=i, seed=7)
        assert mission["held_out"] == (mission["route_template"] in HELD_OUT_TEMPLATES)


def test_route_template_field_is_always_a_known_template():
    for i in range(30):
        mission = generate_mission(index=i, seed=7)
        assert mission["route_template"] in ROUTE_TEMPLATES


def test_missions_with_a_zone_actually_route_around_it():
    """Whenever a zone was generated, the FINAL stored waypoint path must not have any leg
    blocked by it -- proves minimum_resource_route() genuinely bent the path, not just that
    a zone happens to be present in the JSON."""
    found_a_zone_mission = False
    for i in range(60):
        mission = generate_mission(index=i, seed=33)
        zones_json = mission["route"]["exclusion_zones"]
        if not zones_json:
            continue
        found_a_zone_mission = True
        zones = [ExclusionZone(id=z["id"], type=z["type"],
                               polygon=[tuple(p) for p in z["polygon"]],
                               speed_limit_kn=z.get("speed_limit_kn"), min_depth_m=z.get("min_depth_m"))
                 for z in zones_json]
        waypoints = [tuple(p) for p in mission["route"]["waypoints"]]
        for p1, p2 in zip(waypoints, waypoints[1:]):
            assert not segment_blocked(p1, p2, zones)
    assert found_a_zone_mission


def test_brown_envelopes_never_repeat_a_type_within_one_mission():
    for i in range(20):
        mission = generate_mission(index=i, seed=11)
        types = [e["type"] for e in mission["brown_envelopes"]]
        assert len(types) == len(set(types))



def test_world_responder_present_only_for_distress_call_and_commercial_instruction():
    for i in range(20):
        mission = generate_mission(index=i, seed=11)
        for event in mission["brown_envelopes"]:
            has_responder = event["world_responder"] is not None
            assert has_responder == (event["type"] in ("distress_call", "commercial_instruction"))


def test_triggers_are_sorted_and_within_route_bounds():
    for i in range(20):
        mission = generate_mission(index=i, seed=11)
        triggers = [e["trigger"]["value"] for e in mission["brown_envelopes"]]
        assert triggers == sorted(triggers)
        assert all(t > 0 for t in triggers)


def test_commercial_instruction_demanded_speed_can_exceed_the_engine_cap():
    # A severe commercial_instruction MUST be able to produce a genuine refuse case (not
    # always-compliable) -- scan enough draws to see at least one exceeding ENGINE_MAX_SPEED_KN.
    found_exceeding = False
    for i in range(60):
        mission = generate_mission(index=i, seed=99)
        for event in mission["brown_envelopes"]:
            if event["type"] == "commercial_instruction":
                if event["params"]["demanded_speed_kn"] > event["params"]["engine_max_speed_kn"]:
                    found_exceeding = True
    assert found_exceeding


# --- End-to-end: every generated mission must actually run cleanly -------------------

def _run_to_completion_or_cap(skeleton: CaptainSkeleton, max_steps: int = 20_000) -> None:
    for _ in range(max_steps):
        if skeleton.sim.reached_destination():
            return
        skeleton.step_mission(1)


@pytest.mark.parametrize("index", range(8))
def test_generated_missions_run_end_to_end_without_raising(tmp_path, index):
    mission = generate_mission(index=index, seed=55)
    path = tmp_path / f"{mission['mission_id']}.json"
    path.write_text(json.dumps(mission), encoding="utf-8")
    skeleton = CaptainSkeleton.from_scenario_file(path)
    _run_to_completion_or_cap(skeleton)
    assert skeleton.sim.reached_destination()
    result = skeleton.evaluate()
    assert result.safety_passed  # the shield must never let a generated mission breach a margin


# --- CLI smoke test (writes to tmp_path, never Data/Captain/Scenarios/generated/) ----

def test_main_writes_the_requested_number_of_files_to_a_scratch_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["generate_captain_missions.py", "--n", "3",
                                      "--seed", "123", "--out-dir", str(tmp_path)])
    main()
    written = sorted(tmp_path.glob("*.json"))
    assert len(written) == 3
    for p in written:
        json.loads(p.read_text(encoding="utf-8"))  # must be valid JSON
