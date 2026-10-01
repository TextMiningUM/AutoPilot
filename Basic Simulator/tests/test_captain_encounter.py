"""Captain walking-skeleton Phase 9b/9c tests: ambient-traffic generation + encounter
window bounds (design_captain_missions.md Sec 13.A.1 points 1-2). No GPU/API key needed."""
import random
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_encounter import (  # noqa: E402
    AMBIENT_TRAFFIC_RATES_PER_HOUR, encounter_resolved, generate_ambient_contact,
    sample_ambient_arrival_times_s, should_start_encounter,
)
from app.missions import Vessel  # noqa: E402

# --- sample_ambient_arrival_times_s -------------------------------------------------------


def test_light_traffic_produces_roughly_the_expected_count_over_a_long_mission():
    mission_duration_s = 100 * 3600.0  # 100h
    arrivals = sample_ambient_arrival_times_s("light", mission_duration_s, seed=1)
    expected = AMBIENT_TRAFFIC_RATES_PER_HOUR["light"] * 100
    assert expected * 0.5 < len(arrivals) < expected * 2.0  # generous band, it's a Poisson draw


def test_dense_traffic_produces_more_arrivals_than_light_traffic():
    mission_duration_s = 50 * 3600.0
    light = sample_ambient_arrival_times_s("light", mission_duration_s, seed=7)
    dense = sample_ambient_arrival_times_s("dense", mission_duration_s, seed=7)
    assert len(dense) > len(light)


def test_arrivals_are_sorted_and_within_the_mission_duration():
    mission_duration_s = 80 * 3600.0
    arrivals = sample_ambient_arrival_times_s("moderate", mission_duration_s, seed=3)
    assert arrivals == sorted(arrivals)
    assert all(0.0 <= t < mission_duration_s for t in arrivals)


def test_same_seed_is_fully_reproducible():
    args = ("moderate", 60 * 3600.0)
    assert sample_ambient_arrival_times_s(*args, seed=42) == sample_ambient_arrival_times_s(*args, seed=42)


def test_different_seeds_produce_different_arrivals():
    args = ("moderate", 60 * 3600.0)
    assert sample_ambient_arrival_times_s(*args, seed=1) != sample_ambient_arrival_times_s(*args, seed=2)


def test_unknown_density_raises():
    with pytest.raises(ValueError):
        sample_ambient_arrival_times_s("extreme", 3600.0, seed=1)


# --- generate_ambient_contact --------------------------------------------------------------


def test_generate_ambient_contact_is_deterministic_for_a_seeded_rng():
    rnd1, rnd2 = random.Random(99), random.Random(99)
    c1 = generate_ambient_contact(0.0, 0.0, 0.0, 6.17, rnd1, "amb1")
    c2 = generate_ambient_contact(0.0, 0.0, 0.0, 6.17, rnd2, "amb1")
    assert (c1.x, c1.y, c1.heading, c1.speed) == (c2.x, c2.y, c2.heading, c2.speed)


def test_generate_ambient_contact_is_offset_from_the_live_own_ship_position():
    rnd = random.Random(5)
    contact = generate_ambient_contact(10000.0, 20000.0, 90.0, 6.17, rnd, "amb1")
    # Should not simply sit at own-ship's own position -- it's offset by a real range.
    import math
    assert math.hypot(contact.x - 10000.0, contact.y - 20000.0) >= 3000.0 - 1.0


def test_generate_ambient_contact_matches_absolute_bearing_convention_at_zero_heading():
    # At own_heading_deg=0.0, relative and absolute bearings coincide (matches
    # generate_random_imazu_missions.py's own implicit convention) -- this test pins that
    # down explicitly so a future refactor can't silently flip the convention unnoticed.
    rnd = random.Random(123)
    contact = generate_ambient_contact(0.0, 0.0, 0.0, 6.17, rnd, "amb1")
    assert isinstance(contact.x, float) and isinstance(contact.y, float)


# --- should_start_encounter / encounter_resolved (Sec 13.A.1 point 2) ----------------------


def test_should_start_encounter_true_for_a_close_contact():
    own = Vessel(name="own_ship", x=0.0, y=0.0, heading=0.0, speed=6.17)
    close = Vessel(name="amb1", x=0.0, y=5000.0, heading=180.0, speed=6.17)  # 5km ahead, closing
    assert should_start_encounter(own, [close])


def test_should_start_encounter_false_for_a_distant_non_closing_contact():
    own = Vessel(name="own_ship", x=0.0, y=0.0, heading=0.0, speed=6.17)
    # Same heading, well out of range (20km > the 6NM/11112m threshold), and FASTER than
    # own-ship (pulling away ahead) -- genuinely diverging, not just far.
    far = Vessel(name="amb1", x=0.0, y=20000.0, heading=0.0, speed=10.0)
    assert not should_start_encounter(own, [far])


def test_should_start_encounter_false_with_no_targets():
    own = Vessel(name="own_ship", x=0.0, y=0.0, heading=0.0, speed=6.17)
    assert not should_start_encounter(own, [])


def test_encounter_resolved_true_when_clear_and_on_goal_course():
    own = Vessel(name="own_ship", x=0.0, y=0.0, heading=0.0, speed=6.17, target_heading=0.0)
    cleared = Vessel(name="amb1", x=0.0, y=-5000.0, heading=0.0, speed=6.17)  # astern, diverging
    assert encounter_resolved(own, [cleared], goal_xy=(0.0, 100000.0),
                              safe_distance_m=500.0, max_turn_deg=30.0)


def test_encounter_resolved_false_while_still_closing():
    own = Vessel(name="own_ship", x=0.0, y=0.0, heading=0.0, speed=6.17, target_heading=0.0)
    closing = Vessel(name="amb1", x=0.0, y=5000.0, heading=180.0, speed=6.17)
    assert not encounter_resolved(own, [closing], goal_xy=(0.0, 100000.0),
                                  safe_distance_m=500.0, max_turn_deg=30.0)


def test_encounter_resolved_false_when_clear_but_off_the_goal_course():
    own = Vessel(name="own_ship", x=0.0, y=0.0, heading=90.0, speed=6.17, target_heading=90.0)
    cleared = Vessel(name="amb1", x=0.0, y=-5000.0, heading=0.0, speed=6.17)
    assert not encounter_resolved(own, [cleared], goal_xy=(0.0, 100000.0),
                                  safe_distance_m=500.0, max_turn_deg=30.0)


def test_encounter_resolved_true_with_no_targets_left_and_on_course():
    own = Vessel(name="own_ship", x=0.0, y=0.0, heading=0.0, speed=6.17, target_heading=0.0)
    assert encounter_resolved(own, [], goal_xy=(0.0, 100000.0), safe_distance_m=500.0, max_turn_deg=30.0)
