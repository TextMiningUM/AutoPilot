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
"rejected" fabricated (encounter_rule, conduct_rule) against a "chosen" of "none"/"none"
+ the goal-course action. `details` carries every underlying value (min_cpa_m,
safe_distance_m, cited rules, requested_degrees, limit_degrees) specifically so that
future mining doesn't need to re-derive anything -- only the mining step itself is out
of scope for now.
"""
from __future__ import annotations
import copy

from app.simulation import VesselConstraints
from pipeline.oow_agent_spec import real_risk as _real_risk_fn

_GIVE_WAY_TURN_CONDUCT_RULES = ("Rule 14", "Rule 16")
_GIVE_WAY_TURN_ENCOUNTER_RULES = ("Rule 14", "Rule 15")


def measure_decision_quality(decision: dict, situation: list[dict],
                              constraints: VesselConstraints) -> dict:
    """Read-only measurement of one decision against its situation/constraints.

    decision: the dict returned by agents._parse_json_action() -- {"action", "degrees",
        "encounter_rule", "conduct_rule", "reasoning", ...}. Never mutated (deep-copied
        before inspection; see test_measurement.py's test_never_mutates_decision).
    situation: list of per-contact dicts as returned by narrate.contact_line() for every
        live contact this decision was made against -- each must carry at least "cpa_m"
        and "tcpa_s". Pass [] for a contact-free situation (no targets at all).
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

    # Fase C0 (RAG-rebuild-v2 plan): the 147 archived units_v1 checkpoints predate the
    # encounter_rule/conduct_rule schema split and only ever recorded a single
    # "rule_applied" field. Falling back to it here (as BOTH fields at once) is the
    # closest faithful reading of that old data -- the old system genuinely used one
    # field for both concerns, so this is a backward-compat reinterpretation, not a
    # fabrication -- and never changes behaviour for any live/new-schema decision, which
    # always has its own encounter_rule/conduct_rule keys already.
    encounter_rule = decision.get("encounter_rule") or decision.get("rule_applied") or "none"
    conduct_rule = decision.get("conduct_rule") or decision.get("rule_applied") or "none"
    action = decision.get("action")

    # Check A -- fabricated risk: a rule was cited even though NO contact has a real
    # collision risk (quality-review STAP 1, 2026-09-23: the ONE shared real_risk() --
    # CPA below this mission's safe passing distance AND TCPA within the real-risk time
    # horizon, pipeline.oow_agent_spec -- same function build_oow_scenarios_leo.py's
    # leo_choose_action and build_oow_scenarios.py's to_unified_action use, so this
    # measurement can never silently drift from what the training-data labelers treat as
    # real risk. TCPA alone is never enough either -- TCPA=0 also fires once the closest
    # point has already passed, which is not a live risk; see narrate.contact_line()'s
    # "closing" field / SYSTEM_OOW_AGENT's own definition of real risk).
    any_real_risk = any(_real_risk_fn(c["cpa_m"], c["tcpa_s"], constraints.min_cpa_m) for c in situation)
    min_cpa_m = min((c["cpa_m"] for c in situation), default=float("inf"))
    a_fired = (not any_real_risk) and (encounter_rule != "none" or conduct_rule != "none")
    if a_fired:
        checks_fired.append("A_fabricated_risk")
        details["A"] = {
            "min_cpa_m": min_cpa_m if situation else None,
            "safe_distance_m": constraints.min_cpa_m,
            "cited_encounter_rule": encounter_rule, "cited_conduct_rule": conduct_rule,
        }

    # Check B -- wrong turn direction: only meaningful once a REAL risk exists (Check A
    # did not fire) and the decision is a give-way vessel's turn mandating starboard
    # (conduct_rule Rule 14 head-on / Rule 16 crossing give-way, or encounter_rule
    # Rule 14/15 as a fallback). conduct_rule=="Rule 17" (stand-on) or "Rule 19"
    # (restricted visibility, different port-turn restrictions) are checked FIRST and are
    # mutually exclusive with the give-way branch -- either could otherwise ALSO match
    # the encounter_rule Rule 14/15 fallback and false-positive as "wrong direction",
    # which is exactly the case the original single-rule_applied version's if/elif
    # exclusivity already avoided. Rule 13 (overtaking) may legitimately pass either
    # side -- never counted, and never needs an exclusivity carve-out since its
    # encounter_rule is always "Rule 13" too, never 14/15.
    if not a_fired:
        if conduct_rule == "Rule 17":
            if action == "turn_left":
                details["B_suspect_rule17"] = {"cited_encounter_rule": encounter_rule,
                                               "cited_conduct_rule": conduct_rule, "action": action}
        elif conduct_rule == "Rule 19":
            pass  # out of scope -- different port-turn restrictions, never measured here
        else:
            give_way_turn = (conduct_rule in _GIVE_WAY_TURN_CONDUCT_RULES
                            or encounter_rule in _GIVE_WAY_TURN_ENCOUNTER_RULES)
            if give_way_turn and action == "turn_left":
                checks_fired.append("B_wrong_direction")
                details["B"] = {"cited_encounter_rule": encounter_rule,
                                "cited_conduct_rule": conduct_rule, "action": action}

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

