"""Nomoto first-order ship-manoeuvring model + rudder servo, per Sawada et al. (2021)'s
own constants (K=0.05 /s, T=50s, T_E=2.5s, rudder limit +/-10deg, 1s integration).

Lives in pipeline/ (not Basic Simulator/app/) so it's a single shared source usable by
BOTH the live simulator (Basic Simulator/app/simulation.py) AND pipeline/oow_agent_spec.py
(derive_risk_horizon_s_nomoto()) -- pipeline/ never depends on Basic Simulator/app/, only
the reverse, so this module must not move back under app/. Pure functions/dataclasses, no
project imports.

Governing equations (deg/deg-s convention throughout, not radians):
  Rudder servo:  T_E * dDelta/dt + Delta = Delta_c   (Delta_c = commanded/autopilot rudder, clipped to +/-limit)
  Nomoto (1st order): T * dr/dt + r = K * Delta        (r = yaw rate, deg/s)
  Heading:       dPsi/dt = r

The autopilot (heading-hold -> commanded rudder) is NOT part of Sawada's own published
Nomoto/servo constants -- it is a simple proportional-with-saturation controller added
here so a "steer toward this heading" caller has something concrete to drive the servo
with. Treat autopilot_kp as a tunable approximation, not a verified paper value.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class NomotoParams:
    """Sawada et al. (2021)'s own Nomoto + rudder-servo constants (defaults), plus one
    added autopilot gain (see module docstring)."""
    K_per_s: float = 0.05
    T_s: float = 50.0
    T_E_s: float = 2.5
    rudder_limit_deg: float = 10.0
    autopilot_kp: float = 1.0  # commanded_rudder_deg = clip(kp * heading_error_deg, +/-limit)


@dataclass(frozen=True)
class NomotoState:
    """Persistent per-vessel state the ODE integrates -- rudder angle and yaw rate are
    real physical state (not derivable from heading alone); heading is carried alongside
    for convenience so advance() is a pure state-in/state-out function."""
    rudder_deg: float = 0.0
    yaw_rate_deg_s: float = 0.0
    heading_deg: float = 0.0


def _signed_heading_diff(from_deg: float, to_deg: float) -> float:
    """Shortest signed difference to_deg - from_deg, wrapped to [-180, 180]."""
    return (to_deg - from_deg + 180.0) % 360.0 - 180.0


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def autopilot_rudder_command(heading_deg: float, target_heading_deg: float, params: NomotoParams) -> float:
    """Proportional heading-hold controller, saturated to +/-rudder_limit_deg -- see
    module docstring caveat: not a verified Sawada-paper control law, just a reasonable
    way to drive the rudder servo toward a commanded heading."""
    error = _signed_heading_diff(heading_deg, target_heading_deg)
    return _clip(params.autopilot_kp * error, params.rudder_limit_deg)


def step_rudder_servo(rudder_deg: float, commanded_rudder_deg: float, params: NomotoParams, dt_s: float) -> float:
    """One Euler step of T_E * dDelta/dt + Delta = Delta_c, clipped to the physical limit."""
    commanded = _clip(commanded_rudder_deg, params.rudder_limit_deg)
    new_rudder = rudder_deg + (commanded - rudder_deg) / params.T_E_s * dt_s
    return _clip(new_rudder, params.rudder_limit_deg)


def step_yaw_rate(yaw_rate_deg_s: float, rudder_deg: float, params: NomotoParams, dt_s: float) -> float:
    """One Euler step of T * dr/dt + r = K * Delta."""
    return yaw_rate_deg_s + (params.K_per_s * rudder_deg - yaw_rate_deg_s) / params.T_s * dt_s


def advance(state: NomotoState, target_heading_deg: float, params: NomotoParams,
            dt_s: float, substep_s: float = 1.0) -> NomotoState:
    """Advances rudder/yaw-rate/heading by dt_s of wall-clock time, internally sub-stepped
    at substep_s (default 1s, matching Sawada's own integration step) so the outer
    decision-cadence dt (this project's 1-60s, unrelated to the physics' own required
    integration granularity) doesn't affect numerical accuracy."""
    n_steps = max(1, round(dt_s / substep_s))
    step_dt = dt_s / n_steps
    rudder, r, heading = state.rudder_deg, state.yaw_rate_deg_s, state.heading_deg
    for _ in range(n_steps):
        commanded = autopilot_rudder_command(heading, target_heading_deg, params)
        rudder = step_rudder_servo(rudder, commanded, params, step_dt)
        r = step_yaw_rate(r, rudder, params, step_dt)
        heading = (heading + r * step_dt) % 360.0
    return NomotoState(rudder_deg=rudder, yaw_rate_deg_s=r, heading_deg=heading)


def manoeuvre_time_s(turn_deg: float, params: NomotoParams | None = None,
                     substep_s: float = 1.0, max_seconds: float = 1200.0,
                     tolerance_deg: float = 0.5) -> float:
    """Real (simulated) elapsed time for the Nomoto+servo model to complete a turn of
    `turn_deg` degrees from rest, starting the manoeuvre at heading 0 with a commanded
    target heading of `turn_deg` -- e.g. manoeuvre_time_s(60.0) with the paper's own
    defaults returns ~155s, matching Sawada et al. (2021)'s own cited "60deg at 10deg
    rudder takes ~2.5-3 minutes" figure. Used by
    pipeline.oow_agent_spec.derive_risk_horizon_s_nomoto() to replace the old
    instant-turn analytic estimate with a real simulated manoeuvre time. Returns
    max_seconds if the turn never completes within that budget (should not happen for
    any realistic turn_deg/params combination)."""
    p = params or NomotoParams()
    state = NomotoState(rudder_deg=0.0, yaw_rate_deg_s=0.0, heading_deg=0.0)
    target = abs(turn_deg)
    elapsed = 0.0
    while elapsed < max_seconds:
        state = advance(state, target, p, dt_s=substep_s, substep_s=substep_s)
        elapsed += substep_s
        if state.heading_deg >= target - tolerance_deg:
            return elapsed
    return max_seconds
