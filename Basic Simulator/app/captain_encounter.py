"""Captain walking-skeleton Phase 9b/9c: ambient-traffic generation + encounter window
bounds (design_captain_missions.md Sec 13.A.1 points 1-2) -- a Poisson-process
contact-arrival schedule along the route (each contact's own geometry built with the SAME
construction already used for Imazu-style missions, `app.geometry.compute_target()`,
`generate_random_imazu_missions.py`'s own pattern, reused unchanged), plus the pure
start/end detection functions the actual splice (wired in `app/captain_skeleton.py`) uses.

Pure Python, no GPU/API key, safe to run locally. Lives in app/ (not pipeline/) because it
constructs `app.missions.Vessel` objects and calls `app.geometry.compute_target()`/
`app.narrate.*`, all app/-only -- consistent with this project's
pipeline-never-depends-on-app/ rule.
"""
from __future__ import annotations
import random

from app.geometry import compute_target
from app.missions import Vessel
from app.narrate import contact_line
from pipeline.oow_agent_spec import goal_course_action

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
                             own_speed_mps: float, rnd: random.Random, name: str,
                             max_intercept_s: float = 3000.0) -> Vessel:
    """One ambient contact's geometry at the ship's live position/heading/speed (Sec
    13.A.1 point 1). `relative_bearing_deg` is relative to own-ship's OWN heading (not an
    absolute compass bearing) -- converted to absolute before calling `compute_target()`,
    whose own `bearing_from_os_deg` parameter is interpreted as absolute (confirmed via
    `bearing_range_to_xy()`'s own implementation; `generate_random_imazu_missions.py`'s
    calls only look equivalent to "relative" because THAT generator always uses
    own_heading_deg=0, where absolute and relative bearings coincide). Not every random
    (bearing, range, speed) draw has a real `compute_target()` solution (the same
    degenerate case `generate_random_imazu_missions.py`'s own `_sample_target()` already
    retries past) -- retried here with a fresh draw rather than letting it raise.

    `max_intercept_s` (Phase 9c, found via a real non-convergent splice): `compute_target()`
    GUARANTEES an eventual intercept by construction, but some (bearing, speed) draws --
    e.g. a near-parallel course with only a tiny closing component -- produce a genuinely
    enormous unavoided `t_collision_s`, which then takes the live encounter splice far
    longer than any reasonable budget to resolve (own-ship correctly perceives no acute
    risk and holds course the whole time, since there genuinely isn't one on any
    sane timescale). A draw whose own `t_collision_s` exceeds this budget is retried, same
    as the degenerate-geometry case above -- this is a generation-time constraint on what
    counts as a useful AMBIENT contact for a bounded splice, not a change to
    `compute_target()` itself (whole-mission Imazu-style generation has no such budget)."""
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
        if t["t_collision_s"] > max_intercept_s:
            continue
        return Vessel(name=name, x=own_local_x + t["start_x"], y=own_local_y + t["start_y"],
                     heading=t["heading"], speed=t["speed"])
    raise RuntimeError(f"no solvable ambient-contact geometry found after 30 attempts (name={name!r})")


# --- Encounter window bounds (Sec 13.A.1 point 2) -----------------------------------------

# "A deliberately earlier/more generous threshold than the encounter-sim's own internal
# quiet/risk classification... so the OOW gets control well before real risk" (Sec 13.A.1).
START_TCPA_THRESHOLD_S = 1800.0  # 30 min
START_RANGE_THRESHOLD_M = 11112.0  # 6 NM (1852m/NM)


def should_start_encounter(own: Vessel, targets: list[Vessel],
                          tcpa_threshold_s: float = START_TCPA_THRESHOLD_S,
                          range_threshold_m: float = START_RANGE_THRESHOLD_M) -> bool:
    """Sec 13.A.1 point 2's encounter START condition: ANY contact with TCPA <= 30 min OR
    range <= 6 NM. Uses `contact_line()`'s own `closing` fact (not a bare TCPA threshold
    check) to correctly exclude a perfectly parallel, non-closing contact -- `cpa_tcpa()`
    itself returns `tcpa=0.0` for zero relative velocity (its own degenerate-case
    convention, "closest approach is NOW" is indistinguishable from "never gets any
    closer or farther" from the bare number alone)."""
    for t in targets:
        c = contact_line(own, t)
        if c["range_m"] <= range_threshold_m:
            return True
        if c["closing"] and c["tcpa_s"] <= tcpa_threshold_s:
            return True
    return False


def encounter_resolved(own: Vessel, targets: list[Vessel], goal_xy: tuple[float, float],
                       safe_distance_m: float, max_turn_deg: float) -> bool:
    """Sec 13.A.1 point 2's encounter END condition: EVERY contact is already "past and
    clear" (`contact_line()`'s own `closing` fact is False) AND the Goal Course Check
    reports back on the planned track within its own deadband -- both facts already exist
    in the OOW pipeline today, so ending an encounter needs no new detection logic."""
    if any(contact_line(own, t, safe_distance_m, max_turn_deg)["closing"] for t in targets):
        return False
    action, _ = goal_course_action(own.x, own.y, own.heading, goal_xy[0], goal_xy[1],
                                   target_heading=own.target_heading)
    return action == "hold_course"
