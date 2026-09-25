"""Shared helper: derive (encounter_rule, conduct_rule) post-hoc from a chosen action and
the geometrically decisive contact -- for baselines whose own decision logic (Velocity
Obstacle, Artificial Potential Field, Dynamic Window Approach, MPC) doesn't reason about
COLREG rule citations directly. Reuses the EXACT SAME classify_encounter()/classify_rules()
mapping app/baselines/ruletree.py and the LLM agent's own ground truth are scored against,
so citation inference is never a second, independently-drifting definition."""
from __future__ import annotations

from app.missions import Vessel
from app.narrate import contact_line
from app.simulation import VesselConstraints
from pipeline.oow_agent_spec import classify_rules, derive_risk_horizon_s, real_risk

# classify_encounter()'s return strings -> classify_rules()'s role vocabulary -- same
# mapping as app/baselines/ruletree.py.
_ENCOUNTER_TO_ROLE = {
    "head_on": "mutual",
    "crossing_target_on_starboard": "give_way",
    "crossing_target_on_port": "stand_on",
    "we_are_overtaking_target": "overtaking_give_way",
    "target_is_overtaking_us": "overtaking_stand_on",
}


def decisive_contact(own: Vessel, targets: list[Vessel], constraints: VesselConstraints) -> dict | None:
    """The most urgent (soonest-TCPA) contact currently meeting real_risk(), or None if no
    contact does -- the SAME selection rule app/baselines/ruletree.py uses, factored out
    here so every baseline picks "the" decisive contact identically."""
    safe_distance_m, max_turn_deg = constraints.min_cpa_m, constraints.max_rudder_angle_deg
    risk_horizon_s = derive_risk_horizon_s(safe_distance_m, max_turn_deg, own.speed)
    contacts = [contact_line(own, t, safe_distance_m, max_turn_deg) for t in targets]
    at_risk = [c for c in contacts
              if real_risk(c["cpa_m"], c["tcpa_s"], safe_distance_m, risk_horizon_s)]
    at_risk.sort(key=lambda c: (c["tcpa_s"], c["cpa_m"]))
    return at_risk[0] if at_risk else None


def infer_rule_citation(own: Vessel, targets: list[Vessel], constraints: VesselConstraints,
                        action: str) -> tuple[str, str, dict | None]:
    """(encounter_rule, conduct_rule, decisive_contact_or_None) for a chosen action,
    inferred post-hoc from geometry -- used by baselines that don't reason about rule
    citations themselves (VO/APF/DWA/MPC)."""
    decisive = decisive_contact(own, targets, constraints)
    if decisive is None:
        return "none", "none", None
    role = _ENCOUNTER_TO_ROLE[decisive["encounter"]]
    encounter_rule, conduct_rule = classify_rules(role, action)
    return encounter_rule, conduct_rule, decisive
