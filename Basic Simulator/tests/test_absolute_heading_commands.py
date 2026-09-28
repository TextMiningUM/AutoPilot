"""Regression tests for the 2026-09-28 absolute/idempotent steering command
(Simulation.steer_heading() + apply_action()'s optional "heading_deg" field) -- added to
fix baseline decision functions (app/baselines/*.py) re-issuing a RELATIVE turn every
decision while a prior order is still in progress, which stacks/overshoots under slow
(Nomoto) kinematics. Backward compatibility (existing turn_left/turn_right/action-only
callers, e.g. the LLM agent path) is the other half of what's verified here."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.missions import load_mission  # noqa: E402
from app.simulation import Simulation, VesselConstraints  # noqa: E402


def _sim(kinematics_model: str = "kinematics") -> Simulation:
    return Simulation(load_mission("Imazu01"), VesselConstraints(kinematics_model=kinematics_model))


def test_steer_heading_sets_absolute_target():
    sim = _sim()
    sim.steer_heading(123.4)
    assert sim.target_heading == pytest.approx(123.4)
    assert sim.own.target_heading == pytest.approx(123.4)


def test_steer_heading_is_idempotent_unlike_turn_right():
    """Repeating the SAME absolute order twice is a no-op; repeating the same RELATIVE
    turn_right(30) twice stacks to 60 -- exactly the bug heading_deg exists to avoid."""
    sim = _sim()
    start = sim.target_heading
    sim.steer_heading((start + 30) % 360)
    sim.steer_heading((start + 30) % 360)
    assert sim.target_heading == pytest.approx((start + 30) % 360)

    sim2 = _sim()
    start2 = sim2.target_heading
    sim2.turn_right(30.0)
    sim2.turn_right(30.0)
    assert sim2.target_heading == pytest.approx((start2 + 60) % 360)


def test_steer_heading_wraps_negative_and_over_360():
    sim = _sim()
    sim.steer_heading(-10.0)
    assert sim.target_heading == pytest.approx(350.0)
    sim.steer_heading(370.0)
    assert sim.target_heading == pytest.approx(10.0)


def test_apply_action_prefers_heading_deg_over_relative_action():
    sim = _sim()
    start = sim.target_heading
    sim.apply_action({"action": "turn_right", "degrees": 30.0, "heading_deg": (start + 90) % 360})
    assert sim.target_heading == pytest.approx((start + 90) % 360)


def test_apply_action_without_heading_deg_still_uses_relative_turn():
    """Backward compatibility: the LLM agent path never supplies heading_deg -- must
    behave EXACTLY as before this change."""
    sim = _sim()
    start = sim.target_heading
    sim.apply_action({"action": "turn_right", "degrees": 30.0})
    assert sim.target_heading == pytest.approx((start + 30) % 360)


def test_apply_action_heading_deg_does_not_block_speed_actions():
    sim = _sim()
    sim.apply_action({"action": "speed_up", "heading_deg": 45.0})
    assert sim.target_heading == pytest.approx(45.0)
    assert sim.target_speed > sim.own.speed - 1e-9  # speed_up still applied


def test_apply_action_none_heading_deg_falls_back_to_action():
    sim = _sim()
    start = sim.target_heading
    sim.apply_action({"action": "turn_left", "degrees": 15.0, "heading_deg": None})
    assert sim.target_heading == pytest.approx((start - 15.0) % 360)
