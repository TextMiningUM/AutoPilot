"""Captain walking-skeleton Phase 2 tests: route planner with exclusion zones
(design_captain_missions.md Sec 13.B.8/Sec 7). No GPU/API key needed."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.mission_route_planner import (  # noqa: E402
    RefugeCandidate, active_zones, check_route_feasibility, distance_to_refuge_nm,
    minimum_resource_route, plan_route, segment_blocked,
)
from app.mission_sim import calibrate_fuel_rate, haversine_nm  # noqa: E402
from pipeline.captain_types import ExclusionZone, Port  # noqa: E402

_SQUARE = ExclusionZone(id="sq1", type="dynamic_hazard",
                        polygon=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])


# --- segment_blocked / polygon geometry -------------------------------------------------

def test_segment_straight_through_square_is_blocked():
    assert segment_blocked((-3.0, 0.0), (3.0, 0.0), [_SQUARE])


def test_segment_entirely_outside_is_not_blocked():
    assert not segment_blocked((5.0, 5.0), (6.0, 6.0), [_SQUARE])


def test_segment_between_adjacent_square_vertices_is_not_blocked():
    # (-1,-1) -> (1,-1) is one of the square's own edges -- travelling along a zone's
    # boundary is allowed, only cutting through the interior is blocked.
    assert not segment_blocked((-1.0, -1.0), (1.0, -1.0), [_SQUARE])


def test_segment_diagonal_across_same_square_is_blocked():
    # Non-adjacent corners of a convex polygon -- the diagonal's midpoint lies strictly
    # inside, so this must be caught even though it crosses no single edge.
    assert segment_blocked((-1.0, -1.0), (1.0, 1.0), [_SQUARE])


def test_segment_touching_only_at_a_shared_vertex_is_not_blocked():
    # From a point outside, straight to one specific corner -- touches the polygon at
    # exactly one point (that corner), never crosses the interior.
    assert not segment_blocked((-3.0, -3.0), (-1.0, -1.0), [_SQUARE])


# --- active_zones (Sec 13.B.8's draft-dependent shallow-water filter) -------------------

def test_coastline_zone_always_active_regardless_of_draft():
    zone = ExclusionZone(id="coast", type="coastline", polygon=_SQUARE.polygon)
    assert active_zones([zone], draft_m=100.0) == [zone]


def test_shallow_water_zone_active_only_when_too_shallow_for_draft():
    deep_enough = ExclusionZone(id="shoal_ok", type="shallow_water", polygon=_SQUARE.polygon, min_depth_m=20.0)
    too_shallow = ExclusionZone(id="shoal_bad", type="shallow_water", polygon=_SQUARE.polygon, min_depth_m=5.0)
    result = active_zones([deep_enough, too_shallow], draft_m=10.0)
    assert result == [too_shallow]


# --- plan_route ---------------------------------------------------------------------

def test_plan_route_direct_when_no_zones_block_the_way():
    result = plan_route((0.0, 0.0), (1.0, 1.0), zones=[])
    assert result.path == [(0.0, 0.0), (1.0, 1.0)]
    assert result.distance_nm == pytest.approx(haversine_nm((0.0, 0.0), (1.0, 1.0)))


def test_plan_route_goes_around_a_blocking_zone():
    straight_nm = haversine_nm((-3.0, 0.0), (3.0, 0.0))
    result = plan_route((-3.0, 0.0), (3.0, 0.0), zones=[_SQUARE])
    assert len(result.path) > 2  # actually routes via zone corners, not a straight line
    assert result.distance_nm > straight_nm  # detour costs more than the blocked direct path
    assert not segment_blocked(result.path[0], result.path[1], [_SQUARE])  # every leg is clear
    for p1, p2 in zip(result.path, result.path[1:]):
        assert not segment_blocked(p1, p2, [_SQUARE])


def test_plan_route_raises_when_goal_is_fully_enclosed():
    enclosing = ExclusionZone(id="wall", type="coastline",
                              polygon=[(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)])
    with pytest.raises(ValueError):
        plan_route((-20.0, 0.0), (0.0, 0.0), zones=[enclosing])  # goal is inside a solid wall


# --- minimum_resource_route ------------------------------------------------------------

def test_minimum_resource_route_sums_legs_with_no_zones():
    waypoints = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    result = minimum_resource_route(waypoints, zones=[])
    expected = haversine_nm((0.0, 0.0), (1.0, 0.0)) + haversine_nm((1.0, 0.0), (1.0, 1.0))
    assert result.distance_nm == pytest.approx(expected)


# --- distance_to_refuge_nm --------------------------------------------------------------

def test_distance_to_refuge_picks_the_nearest_qualifying_port():
    near_port = Port(id="near", name="Near Haven", position=(0.1, 0.1), services=["repair"],
                     min_approach_depth_m=10.0, region_id="test")
    far_port = Port(id="far", name="Far Haven", position=(5.0, 5.0), services=["repair"],
                    min_approach_depth_m=10.0, region_id="test")
    too_shallow_port = Port(id="shallow", name="Shallow Haven", position=(0.05, 0.05), services=["repair"],
                           min_approach_depth_m=3.0, region_id="test")
    candidate = distance_to_refuge_nm((0.0, 0.0), [near_port, far_port, too_shallow_port],
                                      zones=[], required_services=["repair"], draft_m=8.0)
    assert isinstance(candidate, RefugeCandidate)
    assert candidate.port.id == "near"


def test_distance_to_refuge_returns_none_when_no_port_qualifies():
    port = Port(id="p1", name="P1", position=(0.1, 0.1), services=["medical"],
               min_approach_depth_m=10.0, region_id="test")
    assert distance_to_refuge_nm((0.0, 0.0), [port], zones=[],
                                 required_services=["repair"], draft_m=5.0) is None


# --- check_route_feasibility -------------------------------------------------------------

def test_check_route_feasibility_rejects_insufficient_fuel():
    model = calibrate_fuel_rate(850.0, 15.0, 600.0, 12.0)
    assert check_route_feasibility(distance_nm=600.0, soa_kn=12.0, fuel_model=model,
                                   fuel_tonnes_available=722.5)
    assert not check_route_feasibility(distance_nm=600.0, soa_kn=12.0, fuel_model=model,
                                       fuel_tonnes_available=100.0)
