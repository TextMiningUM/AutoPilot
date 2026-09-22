"""Deterministic, read-only measurement layer around OOW-agent decisions.

Pure Python, zero GPU/LLM calls. Counts three specific, deterministically-detectable
error patterns identified in OOW_Mission_Sim_Analysis_V1.md Sec.2 (#1 fabricated risk,
#2 wrong turn direction, #4 physically-impossible turn requests) -- and does nothing else.

THIS LAYER NEVER CORRECTS, CLIPS, OR OVERRIDES A DECISION. The research goal is to see
whether the model itself learns to derive the correct COLREG rule/action from the
situation report -- correcting a wrong decision here would make the ship do the right
thing for the wrong reason, destroying the very signal (a genuine model mistake) this
layer exists to surface. There is no "apply" mode and no safety-net flag, on or off,
optional or otherwise. It only counts and logs, always.

See run_llm_scenario.py's checkpoint-building loop for the one call site (attaches a
"measurement" field to each checkpoint dict, alongside -- never inside -- "decision").

TODO (explicitly NOT built here, later improvement): every checkpoint where Check B
fires is a natural DPO "rejected" example (the decision the model actually made) against
a deterministic "chosen" (the correct give-way direction); Check A similarly pairs a
"rejected" fabricated rule_applied against a "chosen" of "none" + the goal-course action.
`details` carries every underlying value (min_cpa_m, safe_distance_m, cited_rule,
requested_degrees, limit_degrees) specifically so that future mining doesn't need to
re-derive anything -- only the mining step itself is out of scope for now.
"""
from __future__ import annotations
import copy
import re

from app.simulation import VesselConstraints

_RULE_NUM_RE = re.compile(r"rule\s*(\d+)", re.IGNORECASE)


def _cited_rule_number(rule_applied) -> int | None:
    """Leading rule number cited in `rule_applied`, or None if no rule was cited at all
    (covers the literal "none", empty string, and missing-field cases)."""
    if not rule_applied:
        return None
    text = str(rule_applied).strip()
    if text.lower() == "none":
        return None
    m = _RULE_NUM_RE.search(text)
    return int(m.group(1)) if m else None


def measure_decision_quality(decision: dict, situation: list[dict],
                              constraints: VesselConstraints) -> dict:
    """Read-only measurement of one decision against its situation/constraints.

    decision: the dict returned by agents._parse_json_action() -- {"action", "degrees",
        "rule_applied", "reasoning", ...}. Never mutated (deep-copied before inspection;
        see test_measurement.py's test_never_mutates_decision).
    situation: list of per-contact dicts as returned by narrate.contact_line() for every
        live contact this decision was made against -- each must carry at least "cpa_m".
        Pass [] for a contact-free situation (no targets at all).
    constraints: the live VesselConstraints used for this mission/step -- supplies the
        safe-passing-distance threshold (min_cpa_m) and the physical per-step turn limit
        (turn_rate_deg_s * time_step_s).

    Returns {"checks_fired": [...], "details": {...}}. `details` only contains an entry
    for a check that either fired (counts as an error) or is purely informational
    (B_suspect_rule17, never counted as an error).
    """
    decision = copy.deepcopy(decision)  # read-only: never mutate the caller's dict
    checks_fired: list[str] = []
    details: dict = {}

    cited_rule_raw = decision.get("rule_applied")
    rule_num = _cited_rule_number(cited_rule_raw)
    action = decision.get("action")

    # Check A -- fabricated risk: a rule was cited even though NO contact has a real
    # collision risk (CPA below this mission's safe passing distance). CPA ONLY, never
    # TCPA -- TCPA=0 also fires once the closest point has already passed, which is not a
    # live risk (see narrate.contact_line()'s "closing" field / SYSTEM_OOW_AGENT's own
    # definition of real risk -- this measurement must use the same definition the model
    # was asked to use, or it measures something else).
    min_cpa_m = min((c["cpa_m"] for c in situation), default=float("inf"))
    real_risk = min_cpa_m < constraints.min_cpa_m
    a_fired = (not real_risk) and rule_num is not None
    if a_fired:
        checks_fired.append("A_fabricated_risk")
        details["A"] = {
            "min_cpa_m": min_cpa_m if situation else None,
            "safe_distance_m": constraints.min_cpa_m,
            "cited_rule": cited_rule_raw,
        }

    # Check B -- wrong turn direction: only meaningful once a REAL risk exists (Check A
    # did not fire) and the cited rule is one of the give-way rules mandating a starboard
    # alteration (14 head-on / 15+16 crossing give-way). Rule 13 (overtaking) may
    # legitimately pass either side -- never counted. Rule 17 (stand-on) has its own
    # 17(c) nuance -- logged separately as informational only, never counted as an error.
    # Rule 19 (restricted visibility) has different port-turn restrictions and is simply
    # out of scope -- it is not in the {14,15,16} set below, so it can never fire B.
    if not a_fired:
        if rule_num in (14, 15, 16) and action == "turn_left":
            checks_fired.append("B_wrong_direction")
            details["B"] = {"cited_rule": cited_rule_raw, "action": action}
        elif rule_num == 17 and action == "turn_left":
            details["B_suspect_rule17"] = {"cited_rule": cited_rule_raw, "action": action}

    # Check C -- physically impossible turn request: degrees requested above what the
    # ship can actually turn in one decision step (turn_rate_deg_s * time_step_s, read
    # from the live constraints -- never hardcoded). Not clipped here -- the simulator
    # already silently caps it elsewhere; measuring that the model ASKED for more than
    # physically possible is the whole point.
    degrees = decision.get("degrees")
    if action in ("turn_left", "turn_right") and isinstance(degrees, (int, float)):
        limit_degrees = constraints.turn_rate_deg_s * constraints.time_step_s
        if degrees > limit_degrees:
            checks_fired.append("C_degrees_over_limit")
            details["C"] = {"requested_degrees": degrees, "limit_degrees": limit_degrees}

    return {"checks_fired": checks_fired, "details": details}
