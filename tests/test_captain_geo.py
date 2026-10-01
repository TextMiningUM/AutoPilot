"""Captain walking-skeleton Phase 9a tests: local equirectangular projection (
design_captain_missions.md Sec 13.A.1's handoff). Pure pipeline/ module -- no app/
dependency, no GPU/API key needed."""
import math

import pytest

from pipeline.captain_geo import from_local_frame, initial_bearing_deg, project_zone_to_local_frame, to_local_frame
from pipeline.captain_types import ExclusionZone


def test_origin_maps_to_zero_zero():
    origin = (52.0, 4.0)
    assert to_local_frame(52.0, 4.0, origin) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_one_degree_of_latitude_is_about_111_km_north():
    origin = (52.0, 4.0)
    x, y = to_local_frame(53.0, 4.0, origin)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(111195.0, rel=0.01)


def test_longitude_scales_by_cos_latitude():
    origin = (52.0, 4.0)
    x, _y = to_local_frame(52.0, 5.0, origin)
    expected_x = math.radians(1.0) * math.cos(math.radians(52.0)) * 6371000.0
    assert x == pytest.approx(expected_x, rel=1e-6)


def test_round_trips_through_both_directions():
    origin = (52.3, 4.3)
    lat, lon = 52.35, 4.45
    x, y = to_local_frame(lat, lon, origin)
    back_lat, back_lon = from_local_frame(x, y, origin)
    assert back_lat == pytest.approx(lat, abs=1e-9)
    assert back_lon == pytest.approx(lon, abs=1e-9)


def test_from_local_frame_zero_zero_returns_the_origin():
    origin = (52.3, 4.3)
    assert from_local_frame(0.0, 0.0, origin) == pytest.approx(origin, abs=1e-9)


def test_matches_haversine_nm_for_a_small_offset():
    # Cross-check against mission_sim.py's own independent great-circle distance, which
    # this projection must agree with closely over a SMALL (few-nm) offset (Sec 13.A.1's
    # own "accurate enough over an encounter's bounded window" claim).
    import sys
    from pathlib import Path
    app_root = Path(__file__).resolve().parent.parent / "Basic Simulator"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    from app.mission_sim import haversine_nm

    origin = (52.0, 4.0)
    target = (52.02, 4.03)  # roughly 2-3nm away
    x, y = to_local_frame(*target, origin)
    local_distance_m = math.hypot(x, y)
    local_distance_nm = local_distance_m / 1852.0
    great_circle_nm = haversine_nm(origin, target)
    assert local_distance_nm == pytest.approx(great_circle_nm, rel=0.01)


# --- initial_bearing_deg ------------------------------------------------------------------


def test_bearing_due_north_is_zero():
    assert initial_bearing_deg((52.0, 4.0), (53.0, 4.0)) == pytest.approx(0.0, abs=1e-6)


def test_bearing_due_east_is_ninety():
    assert initial_bearing_deg((52.0, 4.0), (52.0, 5.0)) == pytest.approx(90.0, abs=0.5)


def test_bearing_due_south_is_180():
    assert initial_bearing_deg((52.0, 4.0), (51.0, 4.0)) == pytest.approx(180.0, abs=1e-6)


def test_bearing_due_west_is_270():
    assert initial_bearing_deg((52.0, 4.0), (52.0, 3.0)) == pytest.approx(270.0, abs=0.5)


def test_bearing_is_always_in_0_360_range():
    assert 0.0 <= initial_bearing_deg((52.0, 4.0), (51.5, 3.5)) < 360.0


# --- project_zone_to_local_frame -----------------------------------------------------------


def test_project_zone_to_local_frame_preserves_non_geometry_fields():
    zone = ExclusionZone(id="whale_zone_1", type="dynamic_hazard",
                         polygon=[(52.0, 4.0), (52.0, 4.1), (52.1, 4.1), (52.1, 4.0)],
                         speed_limit_kn=10.0, min_depth_m=None)
    projected = project_zone_to_local_frame(zone, origin=(52.0, 4.0))
    assert projected.id == zone.id
    assert projected.type == zone.type
    assert projected.speed_limit_kn == zone.speed_limit_kn
    assert projected.min_depth_m == zone.min_depth_m


def test_project_zone_to_local_frame_projects_each_vertex():
    zone = ExclusionZone(id="z1", type="coastline", polygon=[(52.0, 4.0), (53.0, 4.0)])
    origin = (52.0, 4.0)
    projected = project_zone_to_local_frame(zone, origin)
    assert projected.polygon[0] == pytest.approx((0.0, 0.0), abs=1e-6)
    assert projected.polygon[1] == pytest.approx(to_local_frame(53.0, 4.0, origin), abs=1e-6)
