"""Artificial Potential Field (APF) baseline -- own-ship is pulled toward an attractive
potential at the goal and pushed away from a repulsive potential around every target
(Khatib, 1986). The resulting force vector's direction becomes the new steering heading;
speed is held at own-ship's current speed, same convention as the other baselines here.
"""
from __future__ import annotations
import math

from app.missions import Mission, Vessel
from app.narrate import narrate
from app.simulation import VesselConstraints
from pipeline.oow_agent_spec import bearing_and_range, relative_bearing

from ._rule_citation import infer_rule_citation

HEADING_DEADBAND_DEG = 3.0
ATTRACTIVE_GAIN = 1.0
REPULSIVE_GAIN = 1.0
# Repulsion only acts within this multiple of the safe distance -- beyond it a target
# contributes zero force, same "only react to what's actually close" principle as the
# other baselines' real_risk()/contact_line() gating.
INFLUENCE_RADIUS_FACTOR = 3.0


def decide(mission: Mission, own: Vessel, targets: list[Vessel],
          constraints: VesselConstraints) -> tuple[dict, dict]:
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    influence_radius_m = safe_distance_m * INFLUENCE_RADIUS_FACTOR

    goal_brg, goal_rng = bearing_and_range(own.x, own.y, mission.goal[0], mission.goal[1])
    # Attractive force: unit vector toward the goal, scaled by gain (capped at goal_rng so
    # it never explodes right at the goal itself).
    gh = math.radians(goal_brg)
    fx = ATTRACTIVE_GAIN * math.sin(gh)
    fy = ATTRACTIVE_GAIN * math.cos(gh)

    repelled_by: list[str] = []
    for t in targets:
        dx, dy = own.x - t.x, own.y - t.y  # points AWAY from the target
        dist = math.hypot(dx, dy)
        if dist <= 1e-6 or dist >= influence_radius_m:
            continue
        # Standard Khatib repulsive gradient magnitude, eta*(1/dist - 1/rho0)/dist^2 -- but
        # that raw form is in units of 1/metres^2, vanishingly small (~1e-8) next to the
        # unit-magnitude attractive term above at metre-scale dist/rho0 (100s-1000s).
        # Rescaled by safe_distance_m**2 (this domain's own characteristic length) so the
        # repulsion's magnitude is directly comparable to (and can dominate) the attraction
        # once a target closes to well inside the safe distance -- confirmed via a real run
        # this rescale was needed: without it, repulsion was ~7 orders of magnitude too
        # weak and the ship sailed straight into every target regardless of how close it got.
        strength = (REPULSIVE_GAIN * (1.0 / dist - 1.0 / influence_radius_m)
                   / dist ** 2 * safe_distance_m ** 2)
        fx += strength * (dx / dist)
        fy += strength * (dy / dist)
        repelled_by.append(t.name)

    if abs(fx) < 1e-9 and abs(fy) < 1e-9:
        desired_heading = goal_brg  # net force vanished (rare) -- fall back to the goal
    else:
        desired_heading = math.degrees(math.atan2(fx, fy)) % 360

    off_from_current = relative_bearing(own.heading, desired_heading)
    if abs(off_from_current) <= HEADING_DEADBAND_DEG:
        action, degrees = "hold_course", None
    else:
        action = "turn_right" if off_from_current > 0 else "turn_left"
        degrees = round(abs(off_from_current), 1)

    encounter_rule, conduct_rule, _decisive = infer_rule_citation(own, targets, constraints, action)
    if not repelled_by:
        reasoning = (f"No contact within the influence radius -- net potential-field "
                    f"heading {desired_heading:.0f} deg (attractive goal pull only).")
    else:
        reasoning = (f"Potential field: net force heading {desired_heading:.0f} deg "
                    f"(attractive goal pull + repulsion from {', '.join(map(repr, repelled_by))}) "
                    f"-> action={action}.")

    decision = {
        "action": action, "degrees": degrees,
        "encounter_rule": encounter_rule, "conduct_rule": conduct_rule,
        "reasoning": reasoning,
    }
    situation = narrate(mission, own, targets, cruise_speed_mps=constraints.cruise_speed_mps,
                        safe_distance_m=safe_distance_m, max_turn_deg=max_turn_deg)
    debug = {"situation": situation, "config": "baseline_apf", "raw_response": reasoning}
    return decision, debug
