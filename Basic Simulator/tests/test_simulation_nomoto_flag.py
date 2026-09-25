"""Phase 2 verification: VesselConstraints.kinematics_model is opt-in and off by default --
switching it to "nomoto" must be the ONLY thing that changes own-ship's heading dynamics,
and the untouched default ("kinematics") must keep producing byte-identical trajectories
to before this flag existed (guards against the Nomoto wiring accidentally altering the
legacy path)."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.missions import load_mission  # noqa: E402
from app.simulation import Simulation, VesselConstraints  # noqa: E402


def test_default_kinematics_model_is_legacy_slew():
    assert VesselConstraints().kinematics_model == "kinematics"


def test_legacy_model_turn_rate_slew_unchanged():
    """Own-ship commanded to turn 90deg at turn_rate_deg_s=3.0/dt=10s must advance
    EXACTLY 30deg in one step (3.0*10), matching this project's pre-existing, already-
    verified turn-rate-slew behaviour (see basic_simulator.md 25th-pass note)."""
    mission = load_mission("Imazu01")
    sim = Simulation(mission, VesselConstraints())
    start_heading = sim.own.heading
    sim.turn_right(90.0)
    sim.step(dt=10.0)
    assert sim.own.heading == pytest.approx((start_heading + 30.0) % 360, abs=1e-6)


def test_nomoto_model_is_opt_in_and_produces_different_heading():
    mission = load_mission("Imazu01")
    legacy = Simulation(mission, VesselConstraints(kinematics_model="kinematics"))
    nomoto = Simulation(mission, VesselConstraints(kinematics_model="nomoto"))
    legacy.turn_right(90.0)
    nomoto.turn_right(90.0)
    legacy.step(dt=10.0)
    nomoto.step(dt=10.0)
    # Nomoto's rudder-servo lag means far LESS heading change in the same 10s than the
    # legacy slew's flat 30deg -- confirms the branch actually engages, not a no-op.
    assert nomoto.own.heading != pytest.approx(legacy.own.heading, abs=1e-6)
    assert abs(nomoto.own.heading - mission.own_ship.heading) < 30.0


def test_unrecognised_kinematics_model_value_falls_back_to_legacy():
    mission = load_mission("Imazu01")
    sim = Simulation(mission, VesselConstraints(kinematics_model="typo"))
    start_heading = sim.own.heading
    sim.turn_right(90.0)
    sim.step(dt=10.0)
    assert sim.own.heading == pytest.approx((start_heading + 30.0) % 360, abs=1e-6)
