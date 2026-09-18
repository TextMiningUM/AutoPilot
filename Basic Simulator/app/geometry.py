"""Pure collision geometry helpers, ported from
Brain Storming/generate_moos_scenarios.py so the simulator can compute the
same exact intercept geometry without importing that script (which has
import-time side effects: it deletes/recreates a hardcoded Linux output
directory as soon as it's imported)."""
from __future__ import annotations
import math


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def bearing_range_to_xy(bearing_deg: float, rng_m: float,
                         origin: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Relative bearing (0=north, clockwise) + range from origin -> local (x, y) metres."""
    b = deg2rad(bearing_deg)
    return (origin[0] + rng_m * math.sin(b), origin[1] + rng_m * math.cos(b))


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


def compute_target(bearing_from_os_deg: float, range_m: float, v_ts: float, v_os: float,
                    heading_os_deg: float = 0.0) -> dict:
    """Full pipeline: desired initial relative bearing/range + target speed -> target
    start position, heading, and unavoided time-to-collision."""
    ts_start = bearing_range_to_xy(bearing_from_os_deg, range_m)
    T, heading_ts = solve_intercept(ts_start, v_ts, v_os, heading_os_deg)
    return {
        "start_x": round(ts_start[0], 1), "start_y": round(ts_start[1], 1),
        "heading": round(heading_ts, 1), "speed": v_ts,
        "bearing_from_os_deg": bearing_from_os_deg, "range_m": range_m,
        "t_collision_s": round(T, 1),
    }


def same_line_target(range_ahead_m: float, v_ts: float, v_os: float,
                      heading_os_deg: float = 0.0, astern: bool = False) -> dict:
    """Overtaking-family construction: target on own-ship's track, ahead (own-ship
    overtakes) or astern (own-ship is overtaken)."""
    x, y = bearing_range_to_xy(heading_os_deg if not astern else (heading_os_deg + 180) % 360,
                                range_ahead_m)
    heading_ts = heading_os_deg
    closing = (v_ts - v_os) if astern else (v_os - v_ts)
    t_coll = range_ahead_m / closing if closing > 0 else float("inf")
    return {
        "start_x": round(x, 1), "start_y": round(y, 1),
        "heading": round(heading_ts, 1), "speed": v_ts,
        "bearing_from_os_deg": heading_os_deg if not astern else (heading_os_deg + 180) % 360,
        "range_m": range_ahead_m, "t_collision_s": round(t_coll, 1),
    }
