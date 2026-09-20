"""Pure collision geometry helpers, ported from
Brain Storming/generate_moos_scenarios.py so the simulator can compute the
same exact intercept geometry without importing that script (which has
import-time side effects: it deletes/recreates a hardcoded Linux output
directory as soon as it's imported)."""
from __future__ import annotations
import math

# Defaults mirror app.simulation.VesselConstraints's own defaults (turn_rate_deg_s=3.0)
# -- kept as plain floats here rather than importing VesselConstraints so this module
# stays dependency-free; build_missions.py passes the LIVE values explicitly instead of
# relying on these matching by coincidence.
DEFAULT_TURN_RATE_DEG_S = 3.0
DEFAULT_AVOIDANCE_TURN_DEG = 30.0  # "early and substantial" COLREG action, readily apparent (Rule 8)
DEFAULT_SAFETY_FACTOR = 3.0  # the turn itself may only eat this fraction (1/N) of the unavoided TCPA


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def bearing_range_to_xy(bearing_deg: float, rng_m: float,
                         origin: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Relative bearing (0=north, clockwise) + range from origin -> local (x, y) metres."""
    b = deg2rad(bearing_deg)
    return (origin[0] + rng_m * math.sin(b), origin[1] + rng_m * math.cos(b))


def min_feasible_time_s(turn_rate_deg_s: float = DEFAULT_TURN_RATE_DEG_S,
                        avoidance_turn_deg: float = DEFAULT_AVOIDANCE_TURN_DEG,
                        safety_factor: float = DEFAULT_SAFETY_FACTOR) -> float:
    """Minimum unavoided time-to-collision a scenario must have for own-ship to actually be
    able to complete a realistic avoidance turn (avoidance_turn_deg) at turn_rate_deg_s well
    before impact -- the turn itself is only allowed to eat 1/safety_factor of that time, so
    the rest is genuine margin for the new course to open up CPA before the original impact
    time. See app.simulation.VesselConstraints -- turn_rate_deg_s is the only kinematic limit
    the live simulator actually enforces, so that's the one this checks against."""
    return (avoidance_turn_deg / turn_rate_deg_s) * safety_factor


def solve_intercept(ts_start: tuple[float, float], v_ts: float, v_os: float,
                     heading_os_deg: float = 0.0) -> tuple[float, float]:
    """Smallest positive time T at which a target starting at ts_start, moving
    straight at v_ts, collides with own-ship (starting at origin, moving straight
    at v_os along heading_os_deg). Returns (T, heading_ts_deg)."""
    xs, ys = ts_start
    h = deg2rad(heading_os_deg)
    xr = xs * math.cos(h) - ys * math.sin(h)
    yr = xs * math.sin(h) + ys * math.cos(h)
    a = (v_os ** 2 - v_ts ** 2)
    b = -2.0 * yr * v_os
    c = (xr ** 2 + yr ** 2)
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            raise ValueError("No intercept solution (degenerate).")
        T = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            raise ValueError("No real intercept solution for given speeds/geometry.")
        sq = math.sqrt(disc)
        cands = [t for t in ((-b + sq) / (2 * a), (-b - sq) / (2 * a)) if t > 1.0]
        if not cands:
            raise ValueError("No positive intercept time.")
        T = min(cands)
    os_xr, os_yr = 0.0, v_os * T
    dx, dy = os_xr - xr, os_yr - yr
    heading_rot = math.degrees(math.atan2(dx, dy)) % 360.0
    heading_ts = (heading_rot + heading_os_deg) % 360.0
    return T, heading_ts


def _grow_range_until_feasible(bearing_from_os_deg: float, range_m: float, v_ts: float, v_os: float,
                                heading_os_deg: float, min_required_s: float,
                                max_iters: int = 200, growth: float = 1.05) -> float:
    """Grows range_m (bearing/speeds held fixed) until the resulting unavoided TCPA meets
    min_required_s. T isn't exactly linear in range (solve_intercept's quadratic), so this
    just iterates rather than solving for it directly -- cheap, this only runs at mission-
    generation time, never during live simulation."""
    r = range_m
    for _ in range(max_iters):
        try:
            T, _ = solve_intercept(bearing_range_to_xy(bearing_from_os_deg, r), v_ts, v_os, heading_os_deg)
        except ValueError:
            r *= growth
            continue
        if T >= min_required_s:
            return r
        r *= growth
    return r  # best effort -- shouldn't be reached for realistic inputs


def compute_target(bearing_from_os_deg: float, range_m: float, v_ts: float, v_os: float,
                    heading_os_deg: float = 0.0,
                    turn_rate_deg_s: float = DEFAULT_TURN_RATE_DEG_S,
                    avoidance_turn_deg: float = DEFAULT_AVOIDANCE_TURN_DEG,
                    safety_factor: float = DEFAULT_SAFETY_FACTOR) -> dict:
    """Full pipeline: desired initial relative bearing/range + target speed -> target
    start position, heading, and unavoided time-to-collision. If the requested range_m
    wouldn't leave own-ship enough time to complete a realistic avoidance turn at
    turn_rate_deg_s (see min_feasible_time_s), range_m is grown (bearing/speeds unchanged)
    until it does -- every generated scenario is guaranteed maneuverable, not just
    geometrically valid."""
    min_required_s = min_feasible_time_s(turn_rate_deg_s, avoidance_turn_deg, safety_factor)
    ts_start = bearing_range_to_xy(bearing_from_os_deg, range_m)
    T, heading_ts = solve_intercept(ts_start, v_ts, v_os, heading_os_deg)
    if T < min_required_s:
        range_m = _grow_range_until_feasible(bearing_from_os_deg, range_m, v_ts, v_os,
                                             heading_os_deg, min_required_s)
        ts_start = bearing_range_to_xy(bearing_from_os_deg, range_m)
        T, heading_ts = solve_intercept(ts_start, v_ts, v_os, heading_os_deg)
        print(f"  [feasibility] grew range to {range_m:.0f}m (bearing {bearing_from_os_deg}\u00b0) "
             f"so own-ship has {T:.0f}s >= {min_required_s:.0f}s to turn before impact")
    return {
        "start_x": round(ts_start[0], 1), "start_y": round(ts_start[1], 1),
        "heading": round(heading_ts, 1), "speed": v_ts,
        "bearing_from_os_deg": bearing_from_os_deg, "range_m": round(range_m, 1),
        "t_collision_s": round(T, 1),
    }


def same_line_target(range_ahead_m: float, v_ts: float, v_os: float,
                      heading_os_deg: float = 0.0, astern: bool = False,
                      turn_rate_deg_s: float = DEFAULT_TURN_RATE_DEG_S,
                      avoidance_turn_deg: float = DEFAULT_AVOIDANCE_TURN_DEG,
                      safety_factor: float = DEFAULT_SAFETY_FACTOR) -> dict:
    """Overtaking-family construction: target on own-ship's track, ahead (own-ship
    overtakes) or astern (own-ship is overtaken). Like compute_target, range_ahead_m is
    grown (never shrunk) if needed so own-ship has enough time to turn before impact --
    here t_collision_s is linear in range so the required range is solved directly."""
    closing = (v_ts - v_os) if astern else (v_os - v_ts)
    if closing > 0:
        min_required_s = min_feasible_time_s(turn_rate_deg_s, avoidance_turn_deg, safety_factor)
        required_range = min_required_s * closing
        if required_range > range_ahead_m:
            print(f"  [feasibility] grew range_ahead_m to {required_range:.0f}m (astern={astern}) "
                 f"so own-ship has >= {min_required_s:.0f}s to turn before impact")
            range_ahead_m = required_range
    x, y = bearing_range_to_xy(heading_os_deg if not astern else (heading_os_deg + 180) % 360,
                                range_ahead_m)
    heading_ts = heading_os_deg
    t_coll = range_ahead_m / closing if closing > 0 else float("inf")
    return {
        "start_x": round(x, 1), "start_y": round(y, 1),
        "heading": round(heading_ts, 1), "speed": v_ts,
        "bearing_from_os_deg": heading_os_deg if not astern else (heading_os_deg + 180) % 360,
        "range_m": round(range_ahead_m, 1), "t_collision_s": round(t_coll, 1),
    }

