"""Sawada et al. (2021)'s own conventional (non-RL) collision-avoidance method -- the
comparison baseline their paper used against their deep-reinforcement-learning agent on
the same 22 Imazu problems this project's Imazu01..Imazu22 missions reproduce exactly
(see generate_imazu_missions.py).

*** BEST-EFFORT RECONSTRUCTION, NOT THE LITERAL PUBLISHED ALGORITHM ***
Only the paper's own Table 4 (exact scenario start geometry: positions/headings/speeds)
was available to transcribe into this repo -- the full text describing their
conventional method's own decision logic was NOT available when this module was written.
This is instead a reconstruction of a Collision Risk Index (CRI) based method: the
standard family of "conventional"/expert-system avoidance method widely published by the
same Japanese ship-manoeuvring research community (Kobe University: Sawada, Sato, Majima,
and related work by Kuwada, Ohtsu, Wakabayashi and others on collision-risk indices for
automatic ship handling) as the non-learning comparison baseline in exactly this kind of
DRL-vs-conventional study. It combines normalized DCPA and TCPA into a single continuous
risk score (0-1), then scales the avoidance turn's MAGNITUDE by that score -- a small,
early risk gets a small early correction, a severe/imminent risk gets a large one -- unlike
app/baselines/ruletree.py's fixed-size avoidance turn. The COLREGS-consistent avoidance
DIRECTION (which side to turn, when a stand-on vessel may still hold) reuses the exact
same classify_encounter()/classify_rules() mapping every other baseline in this package
uses, since that part is dictated by COLREGS itself, not by which paper's conventional
method is being approximated.

If the actual paper text/algorithm ever becomes available, this module should be revised
to match it exactly and this docstring updated to drop the reconstruction caveat.
"""
from __future__ import annotations
import math

from app.missions import Mission, Vessel
from app.narrate import contact_line, narrate
from app.simulation import VesselConstraints
from pipeline.oow_agent_spec import classify_rules, derive_risk_horizon_s, goal_course_action

MIN_AVOIDANCE_TURN_DEG = 10.0
MAX_AVOIDANCE_TURN_DEG = 45.0
# CRI above which even a stand-on vessel must act -- the continuous-CRI counterpart of
# ruletree.py's binary "band=='acute' and stand_on_deadline_passed" gate (Rule 17(a)(ii)/(b)).
STAND_ON_EMERGENCY_CRI = 0.8
CRI_ACT_THRESHOLD = 0.15  # below this, treated as no real risk -- follow GOAL COURSE CHECK

_ENCOUNTER_TO_ROLE = {
    "head_on": "mutual",
    "crossing_target_on_starboard": "give_way",
    "crossing_target_on_port": "stand_on",
    "we_are_overtaking_target": "overtaking_give_way",
    "target_is_overtaking_us": "overtaking_stand_on",
}


def _cri(cpa_m: float | None, tcpa_s: float | None, safe_distance_m: float,
        risk_horizon_s: float) -> float:
    """Collision Risk Index in [0, 1] -- geometric mean of a normalized DCPA-closeness
    term and a normalized TCPA-urgency term, so BOTH must be significant for a high score
    (the continuous counterpart of real_risk()'s hard AND-gate). TCPA<0 (already past
    closest approach) or missing CPA/TCPA -> 0 (no risk)."""
    if cpa_m is None or tcpa_s is None or tcpa_s < 0:
        return 0.0
    dcpa_term = max(0.0, min(1.0, 1.0 - cpa_m / safe_distance_m))
    tcpa_term = max(0.0, min(1.0, 1.0 - tcpa_s / risk_horizon_s))
    return math.sqrt(dcpa_term * tcpa_term)


def decide(mission: Mission, own: Vessel, targets: list[Vessel],
          constraints: VesselConstraints) -> tuple[dict, dict]:
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    risk_horizon_s = derive_risk_horizon_s(safe_distance_m, max_turn_deg, own.speed)

    contacts = [contact_line(own, t, safe_distance_m, max_turn_deg) for t in targets]
    scored = [(c, _cri(c["cpa_m"], c["tcpa_s"], safe_distance_m, risk_horizon_s)) for c in contacts]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    decisive, cri = scored[0] if scored else (None, 0.0)

    if decisive is None or cri < CRI_ACT_THRESHOLD:
        action, degrees = goal_course_action(own.x, own.y, own.heading, mission.goal[0], mission.goal[1])
        encounter_rule, conduct_rule = "none", "none"
        reasoning = (f"Collision Risk Index below the acting threshold ({cri:.2f} < "
                    f"{CRI_ACT_THRESHOLD}) -- following GOAL COURSE CHECK.")
    else:
        role = _ENCOUNTER_TO_ROLE[decisive["encounter"]]
        if role in ("stand_on", "overtaking_stand_on") and cri < STAND_ON_EMERGENCY_CRI:
            action, degrees = "hold_course", None
        else:
            if role in ("stand_on", "overtaking_stand_on"):
                # Rule 17(a)(ii)/(b) emergency action -- turn away from the contact's side.
                action = "turn_right" if decisive["rel_bearing_deg"] < 0 else "turn_left"
            else:
                action = "turn_right"  # give-way roles: COLREGS' default starboard preference
            degrees = round(MIN_AVOIDANCE_TURN_DEG
                           + (MAX_AVOIDANCE_TURN_DEG - MIN_AVOIDANCE_TURN_DEG) * cri, 1)
        encounter_rule, conduct_rule = classify_rules(role, action)
        reasoning = (f"Collision Risk Index {cri:.2f} vs contact {decisive['name']!r} "
                    f"({decisive['encounter']}) -- action={action}"
                    + (f" ({degrees} deg, CRI-scaled)" if degrees else "") + ".")

    decision = {
        "action": action, "degrees": degrees,
        "encounter_rule": encounter_rule, "conduct_rule": conduct_rule,
        "reasoning": reasoning,
    }
    situation = narrate(mission, own, targets, cruise_speed_mps=constraints.cruise_speed_mps,
                        safe_distance_m=safe_distance_m, max_turn_deg=max_turn_deg)
    debug = {"situation": situation, "config": "baseline_sawada", "raw_response": reasoning}
    return decision, debug
