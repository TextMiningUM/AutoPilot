"""Regression tests for app/evaluation.py's _ground_truth_at_checkpoint() -- the
compliance-rebuild STAP 2 structured (band/encounter/expected-rule) ground truth per
contact. All synthetic/deterministic geometry, no live model calls.
"""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.evaluation import _ground_truth_at_checkpoint  # noqa: E402

SAFE_DISTANCE_M = 500.0
MAX_TURN_DEG = 30.0


def _rows(own_xyhs: tuple, contacts: dict[str, tuple]) -> list[dict]:
    """own_xyhs/each contacts value: (x, y, heading, speed) at t=0."""
    ox, oy, ohd, ospd = own_xyhs
    rows = [{"time": 0.0, "vehicle": "own_ship", "x": ox, "y": oy, "heading": ohd, "speed": ospd}]
    for name, (x, y, hd, spd) in contacts.items():
        rows.append({"time": 0.0, "vehicle": name, "x": x, "y": y, "heading": hd, "speed": spd})
    return rows


def _gt(own_xyhs: tuple, contacts: dict[str, tuple]) -> dict:
    rows = _rows(own_xyhs, contacts)
    return _ground_truth_at_checkpoint(rows, 0.0, "own_ship", SAFE_DISTANCE_M, MAX_TURN_DEG)


def test_imazu01_t0_is_early_head_on_not_quiet() -> None:
    """Imazu01's exact t=0 geometry (CPA 0, TCPA 1800s, mission speed 6.1733 m/s ->
    derived horizon ~567s) -- compliance-rebuild STAP 1's plain real_risk() boolean could
    never flag this as anything but quiet (TCPA far exceeds any realistically-derived
    horizon); STAP 2's "early" band fixes this by classifying encounter/rule from geometry
    alone, independent of timing."""
    gt = _gt((0.0, -11112.0, 0.0, 6.173333333333333),
            {"ts1": (0.0, 11112.0, 180.0, 6.173333333333333)})
    c = gt["contacts"][0]
    assert c["band"] == "early"
    assert c["encounter"] == "head_on"
    assert c["own_role"] == "both_give_way"
    assert c["expected_encounter_rule"] == "Rule 14"
    assert c["expected_conduct_rule"] == "Rule 14"
    assert c["expected_direction"] == "starboard"
    assert "port_toward_contact" in c["forbidden"]
    assert "speed_up" in c["forbidden"]
    assert c["band"] != "safe" and c["encounter"] != "none"  # NIET "quiet"
    assert gt["decisive_contact"] == "ts1"


def test_um02_wide_cpa_is_safe_no_rule() -> None:
    """A target offset 6322m laterally on an identical parallel course (own's own UM02
    canary scenario) -- CPA stays 6322m regardless of TCPA, well outside the 500m default
    safe distance, so band must be "safe" with no rule cited at all."""
    gt = _gt((0.0, 0.0, 0.0, 5.0), {"ts1": (6322.0, 0.0, 0.0, 5.0)})
    c = gt["contacts"][0]
    assert c["cpa_m"] == 6322.0
    assert c["band"] == "safe"
    assert c["encounter"] == "none"
    assert c["expected_encounter_rule"] == "none"
    assert c["expected_conduct_rule"] == "none"
    assert gt["decisive_contact"] is None


def test_negative_tcpa_is_passed_band() -> None:
    """Two vessels already diverging (own heading 0 moving away from a contact behind it
    heading 180) -- cpa_tcpa() itself always clamps TCPA to >=0, so this band requires the
    separate signed/unclamped TCPA check; -30s here (own catches nothing, contact already
    receded past its closest point)."""
    gt = _gt((0.0, 0.0, 0.0, 5.0), {"ts1": (0.0, -300.0, 180.0, 5.0)})
    c = gt["contacts"][0]
    assert c["tcpa_s"] == -30.0
    assert c["band"] == "passed"
    assert c["expected_encounter_rule"] == "none"
    assert c["expected_conduct_rule"] == "none"


def test_crossing_port_stand_on_expects_hold() -> None:
    """Contact crossing from own's port bow (rel_bearing < 0), own is the STAND-ON vessel
    (Rule 17) -- expected_direction is "hold", not a turn, and turning left toward a
    contact on own's own port side is always forbidden regardless of role."""
    gt = _gt((0.0, 0.0, 0.0, 5.0), {"ts1": (-300.0, 100.0, 90.0, 5.0)})
    c = gt["contacts"][0]
    assert c["encounter"] == "crossing_port"
    assert c["own_role"] == "stand_on"
    assert c["expected_encounter_rule"] == "Rule 15"
    assert c["expected_conduct_rule"] == "Rule 17"
    assert c["expected_direction"] == "hold"
    assert "port_toward_contact" in c["forbidden"]
    assert c["band"] in ("early", "acute")  # not safe/passed -- a real close-quarters case


def test_deterministic_same_input_same_output() -> None:
    """Same input geometry -> byte-identical dict on repeat calls (no LLM, no randomness)."""
    own_xyhs = (0.0, -11112.0, 0.0, 6.173333333333333)
    contacts = {"ts1": (0.0, 11112.0, 180.0, 6.173333333333333)}
    assert _gt(own_xyhs, contacts) == _gt(own_xyhs, contacts)
