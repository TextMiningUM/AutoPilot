"""Captain walking-skeleton Phase 5: minimal composite evaluation -- only the Safety,
Mission outcome, and Resource efficiency axes (design_captain_missions.md Sec 10.1/10.2,
Sec 13.C.11's ground-truth table) are implemented. The other 5 axes (Regulatory/procedural
compliance, Procedure/regime selection accuracy, Explanation/citation accuracy, Decision
timeliness/calling discipline, Crew welfare) each need multiple live event types/trigger
monitors the walking skeleton doesn't have yet (only engine_failure is fully wired, Sec
14) -- explicitly deferred, not silently dropped; adding one later is "wire its own axis
function + fold its weight back into the composite", not a redesign.

Pure Python, no GPU/API key, safe to run locally. Pure pipeline/ module -- operates only
on plain scalars/the event log (never a live MissionSim/route-planner object), mirroring
`Evaluation Functions/evaluate_run.py`'s own style (axis functions take extracted
primitives, not simulator objects) and keeping pipeline/ decoupled from
Basic Simulator/app/ (same rule as captain_agent_spec.py's EngineFailureContext). A caller
in app/ (needed for Sec 13.B.8's route-planner oracle baseline) extracts the facts and
passes them in -- see `app/captain_skeleton.py`'s own `evaluate()`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from pipeline.captain_agent_spec import check_safety_margins
from pipeline.captain_types import BrownEnvelopeEvent, CaptainAction

# Sec 10.2's decided starting weights, restricted to the 3 axes this phase implements --
# NOT renormalised to 100%: the remaining 35% (Crew welfare 20% + 4 others at 7% each)
# belongs to axes this skeleton doesn't compute yet, and silently stretching this reduced
# set to fill the full scale would misrepresent an unimplemented axis as a perfect score.
PHASE5_WEIGHTS = {"mission_outcome": 0.25, "safety": 0.20, "resource_efficiency": 0.07}

# Every v1 event with a fully-wired decision layer (Phase 3/6) names its own mandated speed
# under a different, event-specific key (engine_failure's shield cap vs fog's Rule-19 safe
# speed vs whale_zone's posted limit) -- checked in this fixed order, first match wins.
_CAP_PARAM_NAMES = ("capped_speed_kn", "safe_speed_kn", "speed_limit_kn")


def _declared_cap_kn(event: BrownEnvelopeEvent) -> float | None:
    """The speed this event's own procedure mandates, regardless of which param name its
    event type happens to use -- None if this event type declares no speed cap at all."""
    for name in _CAP_PARAM_NAMES:
        if name in event.params:
            return float(event.params[name])
    return None


# --- Safety (Sec 10.1 hard gate 1) -------------------------------------------------------

def safety_axis_captain(event_log: list[dict[str, Any]],
                        events_by_id: dict[str, BrownEnvelopeEvent]) -> tuple[bool, float, list[str]]:
    """Sec 13.C.11's counterfactual safety ground truth, specialised to this skeleton's
    recorded decisions: independently RE-CHECKS every logged `captain_decision` entry's
    applied action against its own event's declared speed cap -- never trusts the
    recorded `shield_substituted` flag at face value (Sec 10.3's own "recompute ground
    truth independently" pattern). Returns (passed, score, violation_descriptions);
    passed=False means Hard Gate 1 applies (composite = 0), mirroring
    `evaluate_run.py`'s `safety_axis()` hard-gate contract."""
    violations: list[str] = []
    for entry in event_log:
        if entry["field_path"] != "captain_decision":
            continue
        event = events_by_id.get(entry["cause"])
        cap = _declared_cap_kn(event) if event is not None else None
        if cap is None:
            continue
        applied = entry["new"]
        action = CaptainAction(tool=applied["tool"], params=applied["params"])
        for v in check_safety_margins(action, engine_max_speed_kn=cap):
            violations.append(f"{entry['cause']}: {v.detail}")
    passed = not violations
    return passed, (1.0 if passed else 0.0), violations


# --- Mission outcome (Sec 10.1, Sec 10.2 hard gate 2) ------------------------------------

@dataclass(frozen=True)
class MissionOutcomeFacts:
    """Independently recomputed facts needed for Sec 10.1's Mission-outcome axis -- never
    trusts a self-reported claim, mirrors Sec 10.3's own ground-truth pattern."""
    reached_destination: bool
    fuel_tonnes_remaining: float
    elapsed_h: float
    eta_deadline_h: float | None


def mission_outcome_axis(facts: MissionOutcomeFacts) -> tuple[float, list[str]]:
    """Sec 13.C.11: recomputed resource margin vs the Mission Order's own success
    criteria. Each of the skeleton's 2 success criteria (Sec 14's scenario: "cargo
    delivered intact", "arrive within the stated ETA deadline") is independently
    re-verified; the axis score is the fraction actually met -- BOTH criteria require
    `reached_destination` first (an undelivered cargo can't be "intact", and there is no
    real arrival time to judge against a deadline), so a ship that never arrives scores
    0.0 here by construction, triggering Hard Gate 2 (the caller applies the cap, this
    function only reports the raw fraction + which criteria were unmet)."""
    met: list[str] = []
    unmet: list[str] = []
    if facts.reached_destination and facts.fuel_tonnes_remaining > 0.0:
        met.append("cargo delivered intact")
    else:
        unmet.append("cargo delivered intact")
    if facts.reached_destination and (facts.eta_deadline_h is None or facts.elapsed_h <= facts.eta_deadline_h):
        met.append("arrive within the stated ETA deadline")
    else:
        unmet.append("arrive within the stated ETA deadline")
    total = len(met) + len(unmet)
    score = len(met) / total if total else 1.0
    return score, unmet


# --- Resource efficiency (Sec 10.1) -------------------------------------------------------

@dataclass(frozen=True)
class ResourceEfficiencyFacts:
    """Sec 13.C.11: the Sec 13.B.8 route planner's own unconstrained minimum-resource run
    (ignoring the engine failure entirely), taken once per mission as the reference -- the
    actual usage below legitimately costs more (or, for a cubic fuel-rate curve, can even
    cost LESS fuel at a reduced speed despite costing more time) when a real brown envelope
    forced a detour/speed change; this mirrors OOW's own temporal/spatial efficiency axes,
    where real COLREG manoeuvring always costs a bit more than the straight-line ideal."""
    reference_fuel_t: float
    reference_time_h: float
    actual_fuel_t: float
    actual_time_h: float


def resource_efficiency_axis(facts: ResourceEfficiencyFacts) -> dict[str, float]:
    """Returns `{"fuel_score", "time_score", "score"}`, each in [0, 1] -- 1.0 at or better
    than the unconstrained reference, decaying as actual usage exceeds it. No extra credit
    for using LESS than the reference (clamped at 1.0), same "no reward for
    over-conservatism" convention as `evaluate_run.py`'s `safety_axis()`."""
    fuel_ratio = facts.reference_fuel_t / facts.actual_fuel_t if facts.actual_fuel_t > 0 else 1.0
    time_ratio = facts.reference_time_h / facts.actual_time_h if facts.actual_time_h > 0 else 1.0
    fuel_score = max(0.0, min(1.0, fuel_ratio))
    time_score = max(0.0, min(1.0, time_ratio))
    return {"fuel_score": fuel_score, "time_score": time_score, "score": (fuel_score + time_score) / 2.0}


# --- Composite (Sec 10.2's two-hard-gate architecture, reduced to 3 axes) ----------------

@dataclass(frozen=True)
class CaptainMissionEvaluation:
    composite_score: float
    verdict: str
    safety_passed: bool
    safety_score: float
    safety_violations: list[str]
    mission_outcome_score: float
    mission_outcome_unmet: list[str]
    resource_efficiency: dict[str, float]


def evaluate_captain_mission(
    event_log: list[dict[str, Any]], events_by_id: dict[str, BrownEnvelopeEvent],
    outcome_facts: MissionOutcomeFacts, resource_facts: ResourceEfficiencyFacts,
) -> CaptainMissionEvaluation:
    """Sec 10.2's two-hard-gate composite, restricted to the 3 axes Phase 5 implements --
    same gate ORDER as `evaluate_run.py` (safety first, then mission-incomplete), just a
    Captain-specific ground truth for each gate's own condition."""
    safety_passed, safety_score, safety_violations = safety_axis_captain(event_log, events_by_id)
    outcome_score, unmet = mission_outcome_axis(outcome_facts)
    resource = resource_efficiency_axis(resource_facts)

    if not safety_passed:
        composite = 0.0
        verdict = "FAIL -- safety margin breached"
    elif not outcome_facts.reached_destination:
        composite = min(0.2, PHASE5_WEIGHTS["mission_outcome"] * outcome_score)
        verdict = "FAIL -- did not reach the destination"
    else:
        composite = (
            PHASE5_WEIGHTS["mission_outcome"] * outcome_score
            + PHASE5_WEIGHTS["safety"] * safety_score
            + PHASE5_WEIGHTS["resource_efficiency"] * resource["score"]
        )
        verdict = "PASS" if not unmet else "PASS_WITH_UNMET_CRITERIA"

    return CaptainMissionEvaluation(
        composite_score=round(composite, 4), verdict=verdict,
        safety_passed=safety_passed, safety_score=safety_score, safety_violations=safety_violations,
        mission_outcome_score=round(outcome_score, 4), mission_outcome_unmet=unmet,
        resource_efficiency=resource,
    )
