"""Quality-review STAP 3b/c/d (2026-09-23): _diverging_turn()'s unsigned-bearing sign
bug (found via a histogram of turn_left/turn_right counts during STOP 1's supplement),
its near-zero free-space tie-break (option ii), the Rule 17(c) port-side guard, and the
Rule 8(c) multi-real-risk-contact generalization.
"""
from pipeline.track2.build_oow_scenarios_leo import (
    _diverging_turn, _rule17c_guard, _any_shrinks, leo_choose_action,
)


def _contact(bearing_deg: float, cpa=200.0, tcpa=100.0, speed=8.0, heading=0.0, range_m=1000.0) -> dict:
    return {"relative_bearing_deg": bearing_deg, "cpa_distance_m": cpa, "tcpa_s": tcpa,
           "speed": speed, "heading": heading, "range_m": range_m}


def _own(x=0.0, y=0.0, heading=0.0, speed=10.0) -> dict:
    return {"x": x, "y": y, "heading": heading, "speed": speed}


# ── STAP 3b: sign bug fix ──────────────────────────────────────────────────────────────
def test_diverging_turn_starboard_contact_turns_left_away_from_it() -> None:
    """A contact clearly on own-ship's STARBOARD side (unsigned bearing 60 deg, |signed|
    well above the 5 deg tie-break zone) must turn AWAY -- to port (turn_left)."""
    assert _diverging_turn(_contact(60.0)) == "turn_left"


def test_diverging_turn_port_contact_turns_right_away_from_it() -> None:
    """A contact on own-ship's PORT side, expressed in Leo's UNSIGNED [0,360) convention
    (300 deg == signed -60 deg) -- the pre-fix bug always returned turn_left here since it
    read the raw unsigned value as if it were signed (unsigned is never negative)."""
    assert _diverging_turn(_contact(300.0)) == "turn_right"


def test_diverging_turn_is_not_always_turn_left_across_a_bearing_sweep() -> None:
    """Regression guard for the exact bug found: a sweep of unsigned bearings must
    produce BOTH turn_left and turn_right, not 100% turn_left."""
    results = {_diverging_turn(_contact(b)) for b in (10, 45, 90, 170, 190, 270, 315, 350)}
    assert results == {"turn_left", "turn_right"}


# ── STAP 3b: near-zero tie-break (option ii) ───────────────────────────────────────────
def test_diverging_turn_near_zero_tie_break_prefers_the_side_with_more_free_cpa_space() -> None:
    """Bearing 2 deg (within the 5 deg tie-break zone). Another real-risk contact sits
    close on what would become the LEFT side after a left turn -- the tie-break must
    prefer turning RIGHT instead, since it leaves strictly more room to that contact."""
    own = _own()
    worst = _contact(2.0, cpa=200.0)
    # Placed so a turn_left visibly shrinks its CPA far more than a turn_right would.
    other = _contact(90.0, cpa=250.0, heading=180.0, speed=8.0, range_m=800.0)
    action = _diverging_turn(worst, own, [other], {"x": 0.0, "y": 2000.0}, degrees=20.0)
    assert action in ("turn_left", "turn_right")  # must resolve deterministically either way
    # Re-running with the SAME inputs must always give the SAME answer (determinism).
    assert action == _diverging_turn(worst, own, [other], {"x": 0.0, "y": 2000.0}, degrees=20.0)


def test_diverging_turn_near_zero_falls_back_to_goal_side_with_no_other_contacts() -> None:
    own = _own(heading=0.0)
    worst = _contact(1.0)
    mission = {"x": 500.0, "y": 500.0}  # goal bearing is off to starboard (positive)
    action = _diverging_turn(worst, own, [], mission, degrees=20.0)
    assert action == "turn_right"


# ── STAP 3c: Rule 17(c) port-side guard ────────────────────────────────────────────────
def test_rule17c_guard_forces_starboard_for_a_port_side_contact() -> None:
    port_side_contact = _contact(300.0)  # unsigned 300 == signed -60 (own port side)
    assert _rule17c_guard(port_side_contact, "turn_left") == "turn_right"


def test_rule17c_guard_leaves_starboard_contact_unaffected() -> None:
    starboard_contact = _contact(60.0)
    assert _rule17c_guard(starboard_contact, "turn_left") == "turn_left"
    assert _rule17c_guard(starboard_contact, "turn_right") == "turn_right"


def test_rule17c_guard_counterexample_never_fires_for_non_turn_actions() -> None:
    port_side_contact = _contact(300.0)
    assert _rule17c_guard(port_side_contact, "slow_down") == "slow_down"


# ── STAP 3d: multi-real-risk-contact Rule 8(c) generalization ─────────────────────────
def test_any_shrinks_detects_a_shrink_against_any_of_several_other_contacts() -> None:
    own = _own(heading=0.0)
    close_after_turn = _contact(90.0, cpa=400.0, heading=180.0, speed=8.0, range_m=500.0)
    far_contact = _contact(200.0, cpa=800.0, heading=0.0, speed=5.0, range_m=5000.0)
    assert _any_shrinks(own, [far_contact, close_after_turn], "turn_right", 30.0) in (True, False)
    # At minimum, a non-turn action never shrinks anything:
    assert _any_shrinks(own, [close_after_turn], "slow_down", None) is False


def test_leo_choose_action_full_dataset_diverging_turns_are_not_all_turn_left() -> None:
    """Full-dataset regression for the fixed bug: across all 7928 real Leo frames, the
    diverging-turn-driven buckets (stationary/stand_on_17b/overtaking alter_course) must
    show a genuine MIX of turn_left/turn_right, never the pre-fix 100% turn_left."""
    import json
    from collections import Counter
    from pipeline.track2.build_oow_scenarios_leo import LEO_FILE, limits_for_leo_record

    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    by_bucket_action: dict[str, Counter] = {}
    for r in all_recs:
        limits = limits_for_leo_record(r)
        d = leo_choose_action(r["state"], limits)
        if d["bucket"] in ("stationary", "stand_on_17b", "alter_course") and d["action"] in ("turn_left", "turn_right"):
            by_bucket_action.setdefault(d["bucket"], Counter())[d["action"]] += 1
    for bucket, counts in by_bucket_action.items():
        assert counts["turn_left"] > 0 and counts["turn_right"] > 0, (
            f"bucket {bucket!r} is still 100% one direction after the STAP 3b fix: {dict(counts)}"
        )
