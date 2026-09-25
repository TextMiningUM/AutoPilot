"""Dynamic Window Approach (DWA) baseline (Fox, Burgard & Thrun, 1997) -- samples
(heading, speed) pairs kinematically reachable within a short prediction horizon (given
own-ship's turn_rate_deg_s/accel/decel limits -- the "dynamic window"), forward-simulates
each candidate against every target's straight-line projection, and scores candidates by
goal-heading alignment + obstacle clearance + speed, picking the best-scoring one that
keeps clearance above the safe distance (or, if none can, the least-bad clearance).

ADAPTATION NOTE: classic DWA scores velocity commands reachable within ONE control
interval for a robot with direct velocity actuation. This project's own-ship instead
receives a persistent target_heading/target_speed that the simulator rate-limits toward
over MANY future steps (see app/simulation.py) -- so "reachable" here means reachable
within the PREDICTION HORIZON below, not one simulation step; the resulting single
turn_left/turn_right/speed_up/slow_down/hold_course action is whichever one action moves
own-ship toward the winning (heading, speed) pair fastest (heading correction takes
priority over a speed change, matching this project's other baselines).
"""
from __future__ import annotations
import math

from app.missions import Mission, Vessel
from app.narrate import narrate
from app.simulation import VesselConstraints
from pipeline.oow_agent_spec import bearing_and_range, relative_bearing

from ._rule_citation import infer_rule_citation

HEADING_DEADBAND_DEG = 3.0
SPEED_DEADBAND_MPS = 0.05
HORIZON_S = 60.0
N_SUBSTEPS = 6
N_HEADING_SAMPLES = 25
W_HEADING = 0.7
W_VELOCITY = 0.3


def _project(x: float, y: float, heading_deg: float, speed: float, t: float) -> tuple[float, float]:
    h = math.radians(heading_deg)
    return x + speed * math.sin(h) * t, y + speed * math.cos(h) * t


def _clearance(own: Vessel, heading: float, speed: float, targets: list[Vessel]) -> float:
    """Minimum own-target distance across N_SUBSTEPS samples over HORIZON_S, assuming own
    holds (heading, speed) constant and every target continues straight at its current
    heading/speed (matches how the Imazu-style missions' targets actually behave)."""
    if not targets:
        return float("inf")
    best = float("inf")
    for i in range(N_SUBSTEPS + 1):
        t = HORIZON_S * i / N_SUBSTEPS
        ox, oy = _project(own.x, own.y, heading, speed, t)
        for tgt in targets:
            tx, ty = _project(tgt.x, tgt.y, tgt.heading, tgt.speed, t)
            best = min(best, math.hypot(ox - tx, oy - ty))
    return best


def decide(mission: Mission, own: Vessel, targets: list[Vessel],
          constraints: VesselConstraints) -> tuple[dict, dict]:
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    goal_brg, _ = bearing_and_range(own.x, own.y, mission.goal[0], mission.goal[1])

    max_swing_deg = max(5.0, constraints.turn_rate_deg_s * HORIZON_S)
    heading_candidates = [
        (own.heading + max_swing_deg * (2 * i / (N_HEADING_SAMPLES - 1) - 1)) % 360
        for i in range(N_HEADING_SAMPLES)
    ]
    speed_candidates = sorted({
        own.speed,
        min(constraints.max_speed_mps, own.speed + constraints.max_acceleration_mps2 * HORIZON_S),
        max(0.0, own.speed - constraints.max_deceleration_mps2 * HORIZON_S),
    })

    best, best_score, best_clearance = None, float("-inf"), float("-inf")
    best_unsafe, best_unsafe_clearance = None, float("-inf")
    for heading in heading_candidates:
        for speed in speed_candidates:
            clearance = _clearance(own, heading, speed, targets)
            heading_score = 1.0 - abs(relative_bearing(heading, goal_brg)) / 180.0
            velocity_score = speed / constraints.max_speed_mps if constraints.max_speed_mps > 0 else 0.0
            score = W_HEADING * heading_score + W_VELOCITY * velocity_score
            if clearance >= safe_distance_m:
                if score > best_score:
                    best, best_score, best_clearance = (heading, speed), score, clearance
            elif clearance > best_unsafe_clearance:
                best_unsafe, best_unsafe_clearance = (heading, speed), clearance

    if best is None:  # every candidate violates the safe distance -- least-bad fallback
        best, best_clearance = best_unsafe, best_unsafe_clearance
    winning_heading, winning_speed = best

    off_from_current = relative_bearing(own.heading, winning_heading)
    if abs(off_from_current) > HEADING_DEADBAND_DEG:
        action, degrees = ("turn_right" if off_from_current > 0 else "turn_left"), round(abs(off_from_current), 1)
    elif winning_speed > own.speed + SPEED_DEADBAND_MPS:
        action, degrees = "speed_up", None
    elif winning_speed < own.speed - SPEED_DEADBAND_MPS:
        action, degrees = "slow_down", None
    else:
        action, degrees = "hold_course", None

    encounter_rule, conduct_rule, decisive = infer_rule_citation(own, targets, constraints, action)
    if decisive is None:
        reasoning = (f"No contact meets real_risk() -- DWA winning heading "
                    f"{winning_heading:.0f} deg/speed {winning_speed:.2f} m/s "
                    f"(clearance {best_clearance:.0f}m over {HORIZON_S:.0f}s horizon).")
    else:
        reasoning = (f"DWA: winning heading {winning_heading:.0f} deg/speed {winning_speed:.2f} m/s "
                    f"(clearance {best_clearance:.0f}m vs contact {decisive['name']!r} over "
                    f"{HORIZON_S:.0f}s horizon) -> action={action}.")

    decision = {
        "action": action, "degrees": degrees,
        "encounter_rule": encounter_rule, "conduct_rule": conduct_rule,
        "reasoning": reasoning,
    }
    situation = narrate(mission, own, targets, cruise_speed_mps=constraints.cruise_speed_mps,
                        safe_distance_m=safe_distance_m, max_turn_deg=max_turn_deg)
    debug = {"situation": situation, "config": "baseline_dwa", "raw_response": reasoning}
    return decision, debug
