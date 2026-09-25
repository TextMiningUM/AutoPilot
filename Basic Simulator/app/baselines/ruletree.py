"""Rule-based COLREG decision tree -- the reference/deterministic baseline. Reuses the
EXACT same geometry/classification functions the live LLM agent's ground truth and
compliance scoring are built from (pipeline.oow_agent_spec, app.narrate.contact_line), so
this baseline's own decisions and the audit tooling can never silently disagree about what
"correct" means.

Per-step decision only (no RAG/CoT/model calls) -- see app/baselines/__init__.py's
docstring for the shared decide() signature every baseline in this package implements.
"""
from __future__ import annotations

from app.missions import Mission, Vessel
from app.narrate import contact_line, narrate
from app.simulation import VesselConstraints
from pipeline.oow_agent_spec import (
    classify_rules, derive_risk_horizon_s, goal_course_action, real_risk, risk_band,
)

# Fixed avoidance-turn size for any collision-avoidance manoeuvre this tree orders --
# matches the same 30 deg convention already used elsewhere in this project (e.g.
# geometry.py's min_feasible_time_s default avoidance_turn_deg).
AVOIDANCE_TURN_DEG = 30.0

# classify_encounter()'s return strings -> classify_rules()'s role vocabulary.
_ENCOUNTER_TO_ROLE = {
    "head_on": "mutual",
    "crossing_target_on_starboard": "give_way",
    "crossing_target_on_port": "stand_on",
    "we_are_overtaking_target": "overtaking_give_way",
    "target_is_overtaking_us": "overtaking_stand_on",
}


def decide_action(mission: Mission, own: Vessel, targets: list[Vessel],
                  constraints: VesselConstraints) -> dict:
    """The core decision, without narrate()/debug -- factored out so app/baselines/mpc.py
    can reuse this exact logic as its internal rollout policy without paying for a
    situation-report string on every simulated future step."""
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    risk_horizon_s = derive_risk_horizon_s(safe_distance_m, max_turn_deg, own.speed)

    contacts = [contact_line(own, t, safe_distance_m, max_turn_deg) for t in targets]
    at_risk = [c for c in contacts
              if real_risk(c["cpa_m"], c["tcpa_s"], safe_distance_m, risk_horizon_s)]
    at_risk.sort(key=lambda c: (c["tcpa_s"], c["cpa_m"]))  # most urgent (soonest TCPA) first
    decisive = at_risk[0] if at_risk else None

    if decisive is None:
        action, degrees = goal_course_action(own.x, own.y, own.heading, mission.goal[0], mission.goal[1])
        encounter_rule, conduct_rule = "none", "none"
        reasoning = ("No contact meets real_risk() (CPA below safe distance AND TCPA "
                    "inside the risk horizon) -- following GOAL COURSE CHECK.")
    else:
        band = risk_band(decisive["cpa_m"], decisive["tcpa_s"], safe_distance_m, risk_horizon_s)
        role = _ENCOUNTER_TO_ROLE[decisive["encounter"]]
        if role in ("stand_on", "overtaking_stand_on"):
            if band == "acute" and decisive["stand_on_deadline_passed"]:
                # Rule 17(a)(ii)/(b): the stand-on vessel must take its own avoiding
                # action once it's clear the give-way vessel isn't keeping clear --
                # turn away from whichever side the contact is actually on.
                action = "turn_right" if decisive["rel_bearing_deg"] < 0 else "turn_left"
                degrees = AVOIDANCE_TURN_DEG
            else:
                action, degrees = "hold_course", None
        else:
            # mutual (head-on), give_way (crossing), overtaking_give_way: own-ship is
            # the one required to act -- alter course to starboard (COLREG's default
            # preference across all three of these roles).
            action, degrees = "turn_right", AVOIDANCE_TURN_DEG
        encounter_rule, conduct_rule = classify_rules(role, action)
        reasoning = (f"Contact {decisive['name']!r}: {decisive['encounter']} at CPA "
                    f"{decisive['cpa_m']:.0f}m/TCPA {decisive['tcpa_s']:.0f}s (band={band}). "
                    f"Rule-based tree: role={role} -> action={action}.")

    return {
        "action": action, "degrees": degrees,
        "encounter_rule": encounter_rule, "conduct_rule": conduct_rule,
        "reasoning": reasoning,
    }


def decide(mission: Mission, own: Vessel, targets: list[Vessel],
          constraints: VesselConstraints) -> tuple[dict, dict]:
    decision = decide_action(mission, own, targets, constraints)
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    situation = narrate(mission, own, targets, cruise_speed_mps=constraints.cruise_speed_mps,
                        safe_distance_m=safe_distance_m, max_turn_deg=max_turn_deg)
    debug = {"situation": situation, "config": "baseline_ruletree", "raw_response": decision["reasoning"]}
    return decision, debug
