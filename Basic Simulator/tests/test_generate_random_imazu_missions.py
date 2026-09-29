"""Stap 2 plan (2026-09-29): sanity tests for the new random-geometry mission generator.
Verifies the reproducibility/held-out/schema-validity guarantees the rest of the plan
(oracle-planner, DAgger mining, RFT filter) depends on -- NOT a re-test of
app.geometry.compute_target/solve_intercept themselves (already covered elsewhere)."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.missions import mission_from_dict  # noqa: E402
from generate_random_imazu_missions import FAILURE_CATEGORIES, build_mission  # noqa: E402
import random  # noqa: E402


def _build(index: int, seed: int = 20260929) -> dict:
    rnd = random.Random(f"{seed}::RND{index:02d}")
    return build_mission(f"RND{index:02d}", index, rnd)


def test_generation_is_deterministic_per_index():
    m1 = _build(5)
    m2 = _build(5)
    assert m1 == m2


def test_different_indices_produce_different_geometry():
    m1 = _build(1)
    m2 = _build(2)
    assert m1["targets"] != m2["targets"]


def test_every_generated_mission_round_trips_through_mission_from_dict():
    for i in range(1, 21):
        mission_from_dict(_build(i))  # raises on a malformed schema


def test_target_count_is_always_one_to_three():
    for i in range(1, 31):
        assert 1 <= len(_build(i)["targets"]) <= 3


def test_held_out_flag_is_deterministic_and_nonzero_fraction():
    n = 60
    missions = [_build(i) for i in range(1, n + 1)]
    held = sum(m["held_out"] for m in missions)
    assert 0 < held < n  # a real, non-degenerate split
    # Same indices must be held out on a fresh regeneration (reproducibility guarantee
    # the DAgger/oracle-planner pipeline depends on for a stable eval split).
    missions_again = [_build(i) for i in range(1, n + 1)]
    assert [m["held_out"] for m in missions] == [m["held_out"] for m in missions_again]


def test_failure_category_is_always_one_of_the_registered_categories():
    for i in range(1, 21):
        assert _build(i)["failure_category"] in FAILURE_CATEGORIES
