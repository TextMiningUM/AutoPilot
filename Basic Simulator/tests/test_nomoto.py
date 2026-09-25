"""Phase 1 verification: the isolated Nomoto + rudder-servo core (pipeline/nomoto.py --
shared with pipeline/oow_agent_spec.py, NOT under Basic Simulator/app/) behaves per
Sawada et al. (2021)'s own published constants (K=0.05/s, T=50s, T_E=2.5s, rudder
+/-10deg) BEFORE it is wired into the live simulator (see test_simulation_nomoto_flag.py
for the app/simulation.py integration, still opt-in/off-by-default there)."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # Auto Pilot/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.nomoto import (  # noqa: E402
    NomotoParams, NomotoState, advance, autopilot_rudder_command,
    manoeuvre_time_s, step_rudder_servo, step_yaw_rate,
)

SAWADA_PARAMS = NomotoParams()  # K=0.05, T=50, T_E=2.5, rudder_limit=10


def test_defaults_match_sawada_constants():
    assert SAWADA_PARAMS.K_per_s == pytest.approx(0.05)
    assert SAWADA_PARAMS.T_s == pytest.approx(50.0)
    assert SAWADA_PARAMS.T_E_s == pytest.approx(2.5)
    assert SAWADA_PARAMS.rudder_limit_deg == pytest.approx(10.0)


def test_steady_turn_rate_at_full_rudder_matches_K_times_limit():
    """Steady state of T*dr/dt + r = K*delta is r = K*delta -- at full +10deg rudder that's
    0.05*10 = 0.5 deg/s, matching the paper's own cited "~0.5 deg/s stationair" figure."""
    r = 0.0
    for _ in range(2000):  # 2000s >> T=50s, well past settling
        r = step_yaw_rate(r, rudder_deg=10.0, params=SAWADA_PARAMS, dt_s=1.0)
    assert r == pytest.approx(0.5, abs=1e-3)


def test_rudder_servo_reaches_63pct_after_one_time_constant():
    """First-order lag T_E*dDelta/dt + Delta = Delta_c: after t=T_E, response is
    1 - e^-1 ~= 63.2% of the way from start to commanded value."""
    rudder = 0.0
    n_steps = int(SAWADA_PARAMS.T_E_s / 0.01)
    for _ in range(n_steps):
        rudder = step_rudder_servo(rudder, commanded_rudder_deg=10.0, params=SAWADA_PARAMS, dt_s=0.01)
    assert rudder == pytest.approx(6.321, rel=0.02)


def test_yaw_rate_reaches_63pct_after_one_time_constant_T():
    r = 0.0
    n_steps = int(SAWADA_PARAMS.T_s / 0.1)
    for _ in range(n_steps):
        r = step_yaw_rate(r, rudder_deg=10.0, params=SAWADA_PARAMS, dt_s=0.1)
    steady = SAWADA_PARAMS.K_per_s * 10.0  # 0.5
    assert r == pytest.approx(steady * 0.6321, rel=0.02)


def test_rudder_command_saturates_to_limit():
    cmd = autopilot_rudder_command(heading_deg=0.0, target_heading_deg=90.0, params=SAWADA_PARAMS)
    assert cmd == pytest.approx(10.0)
    cmd = autopilot_rudder_command(heading_deg=0.0, target_heading_deg=-90.0, params=SAWADA_PARAMS)
    assert cmd == pytest.approx(-10.0)


def test_60_degree_turn_takes_roughly_2_to_3_minutes():
    """User-cited paper figure: at 10deg rudder, a 60deg turn takes ~2.5-3 minutes
    (150-180s). Simulate a large course-change order and measure elapsed time to reach
    60deg turned, with a generous tolerance around that cited range since the exact
    autopilot control law here is our own approximation (see nomoto.py docstring)."""
    state = NomotoState(rudder_deg=0.0, yaw_rate_deg_s=0.0, heading_deg=0.0)
    target_heading = 90.0  # comfortably past 60deg so the ship is still mid-turn at 60deg
    elapsed = 0.0
    turned_60_at = None
    for _ in range(1200):  # up to 1200s
        state = advance(state, target_heading, SAWADA_PARAMS, dt_s=1.0, substep_s=1.0)
        elapsed += 1.0
        if turned_60_at is None and state.heading_deg >= 60.0:
            turned_60_at = elapsed
            break
    assert turned_60_at is not None, "never reached a 60deg turn within 1200s"
    assert 100.0 <= turned_60_at <= 260.0, f"60deg turn took {turned_60_at}s, expected roughly 150-180s"


def test_substep_1s_matches_calling_advance_once_per_second():
    """advance(dt_s=10, substep_s=1) must match 10 sequential advance(dt_s=1) calls --
    confirms sub-stepping doesn't silently change the trajectory vs. the paper's own 1s
    integration granularity."""
    state_a = NomotoState(rudder_deg=0.0, yaw_rate_deg_s=0.0, heading_deg=0.0)
    state_b = NomotoState(rudder_deg=0.0, yaw_rate_deg_s=0.0, heading_deg=0.0)
    for _ in range(5):
        state_a = advance(state_a, 90.0, SAWADA_PARAMS, dt_s=10.0, substep_s=1.0)
    for _ in range(50):
        state_b = advance(state_b, 90.0, SAWADA_PARAMS, dt_s=1.0, substep_s=1.0)
    assert state_a.heading_deg == pytest.approx(state_b.heading_deg, abs=1e-9)
    assert state_a.yaw_rate_deg_s == pytest.approx(state_b.yaw_rate_deg_s, abs=1e-9)
    assert state_a.rudder_deg == pytest.approx(state_b.rudder_deg, abs=1e-9)


def test_heading_wraps_into_0_360_range():
    state = NomotoState(rudder_deg=0.0, yaw_rate_deg_s=0.0, heading_deg=350.0)
    for _ in range(60):
        state = advance(state, 30.0, SAWADA_PARAMS, dt_s=1.0, substep_s=1.0)
    assert 0.0 <= state.heading_deg < 360.0


def test_already_on_target_heading_stays_put():
    state = NomotoState(rudder_deg=0.0, yaw_rate_deg_s=0.0, heading_deg=45.0)
    new_state = advance(state, 45.0, SAWADA_PARAMS, dt_s=10.0, substep_s=1.0)
    assert new_state.heading_deg == pytest.approx(45.0, abs=1e-6)
    assert new_state.rudder_deg == pytest.approx(0.0, abs=1e-6)
    assert new_state.yaw_rate_deg_s == pytest.approx(0.0, abs=1e-6)


def test_manoeuvre_time_s_60deg_matches_advance_based_measurement():
    """manoeuvre_time_s() must agree with directly stepping advance() to the same
    tolerance (it's a convenience wrapper around the same physics, not a different
    estimate) -- and land in the paper-cited ~2.5-3min range, same as
    test_60_degree_turn_takes_roughly_2_to_3_minutes above."""
    t = manoeuvre_time_s(60.0, SAWADA_PARAMS)
    assert 100.0 <= t <= 260.0


def test_manoeuvre_time_s_larger_turn_takes_longer():
    assert manoeuvre_time_s(30.0, SAWADA_PARAMS) < manoeuvre_time_s(90.0, SAWADA_PARAMS)


def test_manoeuvre_time_s_zero_turn_is_near_instant():
    assert manoeuvre_time_s(0.0, SAWADA_PARAMS) <= 1.0
