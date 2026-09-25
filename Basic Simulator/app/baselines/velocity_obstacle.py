"""Velocity Obstacle (VO) baseline (Fiorini & Shiller, 1998) -- own-ship samples candidate
headings around its goal-directed preferred heading and picks the one closest to preferred
that is collision-free against every target's velocity-obstacle cone. Speed is held at
own-ship's current speed -- only heading is varied, matching this project's other
baselines and the Imazu missions' own "alter course" pass criteria.
"""
from __future__ import annotations
import math

from app.missions import Mission, Vessel
from app.narrate import narrate
from app.simulation import VesselConstraints
from pipeline.oow_agent_spec import bearing_and_range, relative_bearing

from ._rule_citation import infer_rule_citation

HEADING_DEADBAND_DEG = 3.0
# Candidate offsets from the preferred (goal) heading -- wide enough to route around a
# multi-contact Imazu-style encounter without an expensive continuous search.
_CANDIDATE_OFFSETS_DEG = list(range(-150, 151, 5))


def _velocity(speed: float, heading_deg: float) -> tuple[float, float]:
    h = math.radians(heading_deg)
    return speed * math.sin(h), speed * math.cos(h)


def _vo_margin(own: Vessel, tgt: Vessel, candidate_heading: float, radius_m: float) -> float:
    """Positive == candidate heading is OUTSIDE tgt's velocity-obstacle cone (clear),
    negative == inside (a collision course) -- own keeps its CURRENT speed, only heading
    varies. The margin's magnitude is the angular clearance/violation in degrees."""
    dx, dy = tgt.x - own.x, tgt.y - own.y
    dist = math.hypot(dx, dy)
    if dist <= radius_m:
        return -1.0  # already inside the safety radius -- maximally unsafe
    ovx, ovy = _velocity(own.speed, candidate_heading)
    tvx, tvy = _velocity(tgt.speed, tgt.heading)
    rvx, rvy = ovx - tvx, ovy - tvy
    closing = (dx * rvx + dy * rvy) < 0  # relative velocity points toward the target
    if not closing or (rvx == 0 and rvy == 0):
        return 1.0  # diverging, or no relative motion at all -- never inside a VO cone
    los_angle = math.atan2(dx, dy)
    rv_angle = math.atan2(rvx, rvy)
    angle_off = abs((math.degrees(rv_angle - los_angle) + 180) % 360 - 180)
    half_angle = math.degrees(math.asin(min(1.0, radius_m / dist)))
    return angle_off - half_angle


def decide(mission: Mission, own: Vessel, targets: list[Vessel],
          constraints: VesselConstraints) -> tuple[dict, dict]:
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    goal_brg, _ = bearing_and_range(own.x, own.y, mission.goal[0], mission.goal[1])

    best_heading, best_margin, best_cost = own.heading, float("-inf"), float("inf")
    for off in _CANDIDATE_OFFSETS_DEG:
        candidate = (goal_brg + off) % 360
        margin = min((_vo_margin(own, t, candidate, safe_distance_m) for t in targets),
                    default=1.0)
        cost = abs(relative_bearing(goal_brg, candidate))  # deviation from goal heading
        is_clear, best_is_clear = margin >= 0, best_margin >= 0
        # Prefer any collision-free candidate closest to the goal heading; if NONE are
        # collision-free, fall back to whichever has the LARGEST margin (least-bad cone
        # violation) regardless of its deviation from the goal.
        if is_clear and not best_is_clear:
            best_heading, best_margin, best_cost = candidate, margin, cost
        elif is_clear == best_is_clear:
            if is_clear and cost < best_cost:
                best_heading, best_margin, best_cost = candidate, margin, cost
            elif not is_clear and margin > best_margin:
                best_heading, best_margin, best_cost = candidate, margin, cost

    off_from_current = relative_bearing(own.heading, best_heading)
    if abs(off_from_current) <= HEADING_DEADBAND_DEG:
        action, degrees = "hold_course", None
    else:
        action = "turn_right" if off_from_current > 0 else "turn_left"
        degrees = round(abs(off_from_current), 1)

    encounter_rule, conduct_rule, decisive = infer_rule_citation(own, targets, constraints, action)
    if decisive is None:
        reasoning = (f"No contact meets real_risk() -- steering candidate heading "
                    f"{best_heading:.0f} deg (goal-directed, VO-clear).")
    else:
        reasoning = (f"Velocity Obstacle: chosen heading {best_heading:.0f} deg "
                    f"(margin={best_margin:.1f} deg vs contact {decisive['name']!r}'s cone) "
                    f"-> action={action}.")

    decision = {
        "action": action, "degrees": degrees,
        "encounter_rule": encounter_rule, "conduct_rule": conduct_rule,
        "reasoning": reasoning,
    }
    situation = narrate(mission, own, targets, cruise_speed_mps=constraints.cruise_speed_mps,
                        safe_distance_m=safe_distance_m, max_turn_deg=max_turn_deg)
    debug = {"situation": situation, "config": "baseline_vo", "raw_response": reasoning}
    return decision, debug
