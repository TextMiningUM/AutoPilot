"""Captain walking-skeleton Phase 9b: ambient-traffic generation (design_captain_missions.md
Sec 13.A.1 point 1) -- a Poisson-process contact-arrival schedule along the route, and each
contact's own geometry built with the SAME construction already used for Imazu-style
missions (`app.geometry.compute_target()`, `generate_random_imazu_missions.py`'s own
pattern) -- reused unchanged, not reinvented, just called at the ship's live
position/heading/speed at spawn time instead of at a fixed mission start.

Pure Python, no GPU/API key, safe to run locally. Lives in app/ (not pipeline/) because it
constructs `app.missions.Vessel` objects and calls `app.geometry.compute_target()`, both
app/-only -- consistent with this project's pipeline-never-depends-on-app/ rule.

Deliberately NOT built here (Phase 9b's own stated scope -- the actual encounter-sim
splice is Phase 9c): nothing in this module is wired into `app/captain_skeleton.py`'s
stepping loop yet; it only generates contacts, it does not decide when an encounter should
start/end or run the encounter-sim itself.
"""
from __future__ import annotations
import random

from app.geometry import compute_target
from app.missions import Vessel

# Sec 13.A.1 point 1's own illustrative v1 defaults (tunable, not fixed forever).
AMBIENT_TRAFFIC_RATES_PER_HOUR = {"light": 0.1, "moderate": 0.3, "dense": 1.0}
_SPAWN_RANGE_BAND_M = (3000.0, 12000.0)  # same illustrative band as generate_random_imazu_missions.py


def sample_ambient_arrival_times_s(density: str, mission_duration_s: float, seed: int) -> list[float]:
    """Poisson-process contact-arrival times (seconds from mission start) over the whole
    mission duration, for a given traffic density category -- deterministic from `seed`
    (Sec 13.C.9's own reproducibility convention: one seed per mission, never a bare
    unseeded random call)."""
    if density not in AMBIENT_TRAFFIC_RATES_PER_HOUR:
        raise ValueError(f"unknown ambient traffic density: {density!r}")
    rate_per_s = AMBIENT_TRAFFIC_RATES_PER_HOUR[density] / 3600.0
    rnd = random.Random(seed)
    arrivals: list[float] = []
    t = 0.0
    while rate_per_s > 0:
        t += rnd.expovariate(rate_per_s)
        if t >= mission_duration_s:
            break
        arrivals.append(t)
    return arrivals


def generate_ambient_contact(own_local_x: float, own_local_y: float, own_heading_deg: float,
                             own_speed_mps: float, rnd: random.Random, name: str) -> Vessel:
    """One ambient contact's geometry at the ship's live position/heading/speed (Sec
    13.A.1 point 1). `relative_bearing_deg` is relative to own-ship's OWN heading (not an
    absolute compass bearing) -- converted to absolute before calling `compute_target()`,
    whose own `bearing_from_os_deg` parameter is interpreted as absolute (confirmed via
    `bearing_range_to_xy()`'s own implementation; `generate_random_imazu_missions.py`'s
    calls only look equivalent to "relative" because THAT generator always uses
    own_heading_deg=0, where absolute and relative bearings coincide). Not every random
    (bearing, range, speed) draw has a real `compute_target()` solution (the same
    degenerate case `generate_random_imazu_missions.py`'s own `_sample_target()` already
    retries past) -- retried here with a fresh draw rather than letting it raise."""
    for _attempt in range(30):
        relative_bearing_deg = rnd.uniform(-180.0, 180.0)
        absolute_bearing_deg = (own_heading_deg + relative_bearing_deg) % 360.0
        range_m = rnd.uniform(*_SPAWN_RANGE_BAND_M)
        speed_mps = own_speed_mps * rnd.uniform(0.5, 1.3)
        try:
            t = compute_target(absolute_bearing_deg, range_m, speed_mps, own_speed_mps,
                               heading_os_deg=own_heading_deg)
        except ValueError:
            continue
        return Vessel(name=name, x=own_local_x + t["start_x"], y=own_local_y + t["start_y"],
                     heading=t["heading"], speed=t["speed"])
    raise RuntimeError(f"no solvable ambient-contact geometry found after 30 attempts (name={name!r})")
