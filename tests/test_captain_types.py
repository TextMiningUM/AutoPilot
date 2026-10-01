"""Captain walking-skeleton Phase 0: schema-shape tests for pipeline/captain_types.py --
construct each new dataclass with a valid example, confirm MissionState derives correctly
from a MissionOrder, and confirm the stub region file (Data/Captain/Regions/stub_corridor_v1.json)
loads and has the shape later phases will depend on. No GPU/API key needed."""
from pathlib import Path

from pipeline.captain_types import (
    BrownEnvelopeEvent, ExclusionZone, MissionOrder, MissionState, Port, load_region_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STUB_REGION_PATH = REPO_ROOT / "Data" / "Captain" / "Regions" / "stub_corridor_v1.json"


def _example_mission_order() -> MissionOrder:
    return MissionOrder(
        mission_id="MSN-TEST-0001",
        issued_by="Fleet Operations Centre",
        issued_at="2026-10-12T14:00:00Z",
        vessel="MV Example Trader",
        t0_utc="2026-10-12T18:00:00Z",
        dt_mission_s=600.0,
        situation={"weather_forecast": "calm"},
        goal="Deliver cargo intact",
        success_criteria=["cargo delivered with zero damage", "no COLREG violations"],
        waypoints=[(51.95, 4.14), (52.3, 4.3), (52.96, 4.75)],
        speed_of_advance_kn=12.0,
        restricted_zones=[],
        rules_of_conduct="COLREG 1972 + company SMS",
        admin_logistics={"resources": {"fuel_tonnes": 850.0, "fuel_reserve_margin_pct": 15.0}},
        command_signal={"reporting_interval_hours": 24},
        events=[
            BrownEnvelopeEvent(
                event_id="ev1", type="engine_failure", severity="moderate",
                context_flags=["restricted_visibility_nearby"],
                trigger={"type": "distance_along_route_nm", "value": 40},
            )
        ],
    )


def test_exclusion_zone_constructs_with_minimal_fields():
    zone = ExclusionZone(id="whale_zone_1", type="dynamic_hazard",
                         polygon=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], speed_limit_kn=10.0)
    assert zone.min_depth_m is None


def test_port_constructs_with_all_fields():
    port = Port(id="port_stub_refuge", name="Stub Haven", position=(52.3, 4.3),
               services=["repair", "fuel_bunkering"], min_approach_depth_m=10.0, region_id="stub_corridor_v1")
    assert "repair" in port.services


def test_brown_envelope_event_defaults_are_empty_not_none():
    ev = BrownEnvelopeEvent(event_id="ev1", type="fog", severity="minor")
    assert ev.context_flags == []
    assert ev.trigger == {}
    assert ev.world_responder is None


def test_mission_order_round_trips_its_events():
    order = _example_mission_order()
    assert order.events[0].type == "engine_failure"
    assert len(order.waypoints) == 3


def test_mission_state_from_mission_order_seeds_open_goals():
    order = _example_mission_order()
    state = MissionState.from_mission_order(order)
    assert state.mission_id == order.mission_id
    assert len(state.goals) == len(order.success_criteria)
    assert all(g["status"] == "open" for g in state.goals)
    assert state.resources["fuel_tonnes"] == 850.0
    assert state.active_plan["waypoints"] == order.waypoints
    assert state.event_log == []


def test_mission_state_mutation_does_not_affect_the_frozen_mission_order():
    order = _example_mission_order()
    state = MissionState.from_mission_order(order)
    state.goals[0]["status"] = "abandoned"
    state.event_log.append({"t": 0, "field_path": "goals[0].status", "old": "open", "new": "abandoned"})
    assert order.success_criteria[0] == "cargo delivered with zero damage"  # order itself untouched


def test_stub_region_file_loads_and_has_expected_shape():
    region = load_region_json(STUB_REGION_PATH)
    assert region["region_id"] == "stub_corridor_v1"
    assert region["exclusion_zones"] == []
    assert len(region["ports"]) == 2  # Phase 0's port_stub_refuge + Phase 4's port_skeleton_refuge
    port_ids = {p["id"] for p in region["ports"]}
    assert port_ids == {"port_stub_refuge", "port_skeleton_refuge"}
    for port in region["ports"]:
        assert port["min_approach_depth_m"] > 0
        assert "repair" in port["services"]
