"""Model Predictive Control (MPC) baseline -- a rollout-based, receding-horizon
controller: samples first-step heading offsets, forward-simulates each for a short
horizon using app.baselines.ruletree.decide_action() as the internal "what happens next"
base policy (a rollout/base-policy simplification of full MPC -- this project has no
numerical optimizer dependency, so optimizing over a control SEQUENCE is done by
sampling+simulating candidate plans rather than solving a QP/NLP), scores each rollout on
final goal-heading deviation + minimum clearance + control effort, and -- as in real MPC --
applies only the FIRST action of the winning plan, re-planning from scratch every step.
"""
from __future__ import annotations
import math
from dataclasses import replace

from app.missions import Mission, Vessel
from app.narrate import narrate
from app.simulation import VesselConstraints
from pipeline.oow_agent_spec import bearing_and_range, relative_bearing

from . import ruletree
from ._rule_citation import infer_rule_citation

HEADING_DEADBAND_DEG = 3.0
HORIZON_STEPS = 6
STEP_S = 10.0
_CANDIDATE_OFFSETS_DEG = list(range(-90, 91, 15))
# W_EFFORT must stay far smaller than W_GOAL -- the internal ruletree base policy
# actively re-aligns toward the goal within the FIRST internal rollout step regardless of
# which first_offset was chosen (see _rollout_cost's docstring), so the average-deviation
# signal that differentiates candidates is only ever a few tenths of a degree-fraction
# large; an effort weight anywhere near that size would let "cheapest to do nothing" win
# even when a real course correction is clearly needed (confirmed via a real run: with
# W_GOAL=0.5/W_EFFORT=0.1 own-ship took one avoidance turn then held that heading dead
# straight for the rest of the mission, sailing thousands of metres past the goal).
W_GOAL = 2.0
W_CLEARANCE = 5.0
W_EFFORT = 0.02


def _advance(v: Vessel, dt: float) -> Vessel:
    h = math.radians(v.heading)
    return replace(v, x=v.x + v.speed * math.sin(h) * dt, y=v.y + v.speed * math.cos(h) * dt)


def _turn_toward(heading: float, target_heading: float, max_turn_deg: float) -> float:
    """One rate-limited turn-rate step toward target_heading -- mirrors
    app/simulation.py's Simulation._advance_own_kinematics() heading update exactly, so
    the internal rollout's own-ship dynamics match what the real simulator will actually
    do with the same command, not an idealised instant-turn model."""
    diff = relative_bearing(heading, target_heading)
    if abs(diff) <= max_turn_deg:
        return target_heading % 360
    return (heading + math.copysign(max_turn_deg, diff)) % 360


def _rollout_cost(mission: Mission, own0: Vessel, first_offset: float, targets: list[Vessel],
                  constraints: VesselConstraints) -> tuple[float, float]:
    """(cost, min_clearance) for committing to a target heading of own0.heading +
    first_offset as the first control, then letting ruletree.decide_action() steer
    (by adjusting that target heading further) for the rest of the horizon. Lower cost is
    better.

    CRITICAL: own-ship's ACTUAL heading here is rate-limited toward its commanded target
    heading via turn_rate_deg_s, exactly like the real Simulation -- NOT snapped
    instantly. An earlier version snapped instantly, which let the internal model believe
    ANY turn (even 90 deg) completes for free within one 10s virtual step; since that made
    almost every candidate's rollout converge to an equally-good-looking outcome, the
    real, rate-limited plant then received a control sequence that never matched what the
    (unrealistic) internal prediction had promised, causing a persistent left/right
    chatter (confirmed via a real run: alternating turn_left/turn_right every single
    step, never converging).

    Goal deviation is the MEAN |bearing-to-goal offset| sampled at every step of the
    rollout, not just the final one -- the internal base policy actively steers back
    toward the goal bearing whenever no contact is at risk, which would otherwise
    converge every candidate to nearly the same outcome and wash out real differences
    between them; averaging keeps a candidate that starts off badly aligned penalized
    even though the base policy would eventually correct it too."""
    max_turn_per_step = constraints.turn_rate_deg_s * STEP_S
    own = replace(own0)
    target_heading = (own0.heading + first_offset) % 360
    virt_targets = [replace(t) for t in targets]
    min_clearance = min((math.hypot(own.x - t.x, own.y - t.y) for t in virt_targets), default=float("inf"))
    goal_brg0, _ = bearing_and_range(own.x, own.y, mission.goal[0], mission.goal[1])
    deviations = [abs(relative_bearing(own.heading, goal_brg0)) / 180.0]

    for step in range(HORIZON_STEPS):
        own = replace(own, heading=_turn_toward(own.heading, target_heading, max_turn_per_step))
        own = _advance(own, STEP_S)
        virt_targets = [_advance(t, STEP_S) for t in virt_targets]
        if virt_targets:
            min_clearance = min(min_clearance, min(math.hypot(own.x - t.x, own.y - t.y) for t in virt_targets))
        if step < HORIZON_STEPS - 1:  # no need to re-plan after the last advance
            decision = ruletree.decide_action(mission, own, virt_targets, constraints)
            if decision["action"] == "turn_left":
                target_heading = (target_heading - decision["degrees"]) % 360
            elif decision["action"] == "turn_right":
                target_heading = (target_heading + decision["degrees"]) % 360
        goal_brg, _ = bearing_and_range(own.x, own.y, mission.goal[0], mission.goal[1])
        deviations.append(abs(relative_bearing(own.heading, goal_brg)) / 180.0)

    goal_deviation = sum(deviations) / len(deviations)
    safe_distance_m = constraints.min_cpa_m
    clearance_penalty = max(0.0, (safe_distance_m - min_clearance) / safe_distance_m) if targets else 0.0
    effort = abs(first_offset) / 180.0
    cost = W_GOAL * goal_deviation + W_CLEARANCE * clearance_penalty + W_EFFORT * effort
    return cost, min_clearance


def decide(mission: Mission, own: Vessel, targets: list[Vessel],
          constraints: VesselConstraints) -> tuple[dict, dict]:
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg

    best_offset, best_cost, best_clearance = 0.0, float("inf"), float("-inf")
    for offset in _CANDIDATE_OFFSETS_DEG:
        cost, clearance = _rollout_cost(mission, own, offset, targets, constraints)
        if cost < best_cost:
            best_offset, best_cost, best_clearance = offset, cost, clearance

    if abs(best_offset) <= HEADING_DEADBAND_DEG:
        action, degrees = "hold_course", None
    else:
        action = "turn_right" if best_offset > 0 else "turn_left"
        degrees = round(abs(best_offset), 1)

    encounter_rule, conduct_rule, decisive = infer_rule_citation(own, targets, constraints, action)
    if decisive is None:
        reasoning = (f"No contact meets real_risk() -- MPC rollout best first-step offset "
                    f"{best_offset:.0f} deg (cost={best_cost:.3f}).")
    else:
        reasoning = (f"MPC: best {HORIZON_STEPS}-step rollout commits to a first-step offset "
                    f"of {best_offset:.0f} deg (cost={best_cost:.3f}, min clearance "
                    f"{best_clearance:.0f}m vs contact {decisive['name']!r}) -> action={action}.")

    decision = {
        "action": action, "degrees": degrees,
        "encounter_rule": encounter_rule, "conduct_rule": conduct_rule,
        "reasoning": reasoning,
    }
    situation = narrate(mission, own, targets, cruise_speed_mps=constraints.cruise_speed_mps,
                        safe_distance_m=safe_distance_m, max_turn_deg=max_turn_deg)
    debug = {"situation": situation, "config": "baseline_mpc", "raw_response": reasoning}
    return decision, debug
