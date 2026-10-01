"""Captain walking-skeleton Phase 3: procedure library, shield, and cost()/regret decision
layer (design_captain_missions.md Sec 13.A.4/13.A.6/13.A.8).

Pure Python, no GPU/API key, safe to run locally. Mirrors `pipeline/oow_agent_spec.py`'s
role for OOW -- the Captain's own classify/validate/decide logic -- but lives in its own
file since Captain is a separately-trained, separate-domain agent (not a shared model with
OOW), per the project's own confirmed architecture decision.

Only `engine_failure`'s decision layer is fully wired here, per the walking skeleton's
scope (Sec 14): the other 4 v1 event types (fog, distress_call, whale_zone,
commercial_instruction) have their mandatory-duty/shield METADATA recorded in
PROCEDURE_LIBRARY (genuinely rule-like, cheap to specify up front), but no
candidates()/rollout() yet -- building a generic polymorphic dispatch mechanism for a
single wired event would be speculative; add the next event's own concrete functions
when it is actually needed, following this file's existing shape.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from pipeline.captain_types import CaptainAction

# --- Mandatory duties + shield metadata (Sec 13.A.4's layers 1-2, genuinely rule-like) ---


@dataclass(frozen=True)
class MandatoryDuty:
    """Layer 1 (Sec 13.A.4): a reporting duty and its deadline, independent of which
    candidate action is chosen."""
    description: str
    deadline_s: float  # relative to the event's own trigger time; 0.0 = immediate


@dataclass(frozen=True)
class ProcedureEntry:
    """One event type's layer-1/layer-2 entry (Sec 13.A.4) -- `shield_description` is
    documentation for humans/prompts; the actual enforcement is `check_safety_margins()`
    (general-purpose, not keyed to one event type) plus any event-specific hard checks a
    caller adds on top."""
    event_type: str
    mandatory_duties: list[MandatoryDuty]
    shield_description: str


PROCEDURE_LIBRARY: dict[str, ProcedureEntry] = {
    "engine_failure": ProcedureEntry(
        event_type="engine_failure",
        mandatory_duties=[MandatoryDuty("fault log entry", deadline_s=0.0)],
        shield_description="never exceed the Chief-Engineer-declared safe speed",
    ),
    "fog": ProcedureEntry(
        event_type="fog",
        mandatory_duties=[],
        shield_description="never exceed the Rule-19 safe speed for the conditions",
    ),
    "distress_call": ProcedureEntry(
        event_type="distress_call",
        mandatory_duties=[MandatoryDuty("notify DPA and flag state", deadline_s=0.0)],
        shield_description="never ignore the call without logging a valid reason",
    ),
    "whale_zone": ProcedureEntry(
        event_type="whale_zone",
        mandatory_duties=[],
        shield_description="never exceed the zone's posted speed limit inside the polygon",
    ),
    "commercial_instruction": ProcedureEntry(
        event_type="commercial_instruction",
        mandatory_duties=[MandatoryDuty("log entry + DPA notification of the outcome", deadline_s=0.0)],
        shield_description="never comply with an instruction that breaches a safety margin (Sec 13.A.6)",
    ),
}


# --- Shield: check_safety_margins() (Sec 13.A.6) ----------------------------------------


@dataclass(frozen=True)
class MarginViolation:
    margin: str  # "engine_speed_cap" | "exclusion_zone" | "rest_hours" | "fuel_reserve"
    detail: str


def check_safety_margins(
    action: CaptainAction, *,
    engine_max_speed_kn: float | None = None,
    candidate_zone_ids_entered: list[str] | None = None,
    authorized_zone_ids: set[str] | None = None,
    oow_rest_breach: bool = False,
    captain_rest_breach: bool = False,
    candidate_fuel_consumption_t: float | None = None,
    fuel_tonnes_available: float | None = None,
    fuel_reserve_margin_pct: float | None = None,
) -> list[MarginViolation]:
    """Sec 13.A.6's 4 generic margin checks -- general-purpose, usable by the shield for
    ANY Captain decision, not hardcoded to one event type. Every param is optional: a check
    is simply skipped (never reports a violation) when its own inputs aren't supplied, so a
    caller only passes the facts relevant to its specific candidate action."""
    violations: list[MarginViolation] = []

    if engine_max_speed_kn is not None:
        speed_kn = action.params.get("speed_kn")
        if speed_kn is not None and speed_kn > engine_max_speed_kn:
            violations.append(MarginViolation(
                "engine_speed_cap", f"{speed_kn}kn exceeds the declared cap of {engine_max_speed_kn}kn"))

    if candidate_zone_ids_entered:
        authorized = authorized_zone_ids or set()
        for zone_id in candidate_zone_ids_entered:
            if zone_id not in authorized:
                violations.append(MarginViolation("exclusion_zone", f"enters unauthorised zone {zone_id}"))

    if oow_rest_breach:
        violations.append(MarginViolation("rest_hours", "OOW watch ledger is already below its STCW minimum"))
    if captain_rest_breach:
        violations.append(MarginViolation("rest_hours", "Captain's own rest ledger is already below its STCW minimum"))

    if (candidate_fuel_consumption_t is not None and fuel_tonnes_available is not None
            and fuel_reserve_margin_pct is not None):
        remaining_after = fuel_tonnes_available - candidate_fuel_consumption_t
        reserve_required = fuel_tonnes_available * (fuel_reserve_margin_pct / 100.0)
        if remaining_after < reserve_required:
            violations.append(MarginViolation(
                "fuel_reserve",
                f"would leave {remaining_after:.1f}t, below the required {reserve_required:.1f}t reserve"))

    return violations


def resolve_captain_decision(proposed: CaptainAction, violations: list[MarginViolation],
                              mandatory_fallback: CaptainAction) -> tuple[CaptainAction, bool]:
    """Sec 13.A.6's missing-outcome-branch, NORMAL (shield-enforced) simulation: a proposed
    decision that breaches a real safety margin is forcibly substituted with
    `mandatory_fallback` (e.g. refuse/defer, citing ISM Art. 5) before it ever reaches the
    Mission State -- the ship never experiences the bad outcome. Returns
    `(applied_action, was_substituted)`; the caller is responsible for still recording the
    ORIGINAL `proposed` action as a shield-violation DPO-rejected sample (Sec 13.C.10) when
    `was_substituted` is True."""
    if violations:
        return mandatory_fallback, True
    return proposed, False


# --- Decision layer: cost()/regret (Sec 13.A.8) -----------------------------------------


@dataclass(frozen=True)
class CostDimensions:
    """The 5 cost dimensions (Sec 13.A.8a), each a plain non-negative scalar in its own
    stated unit. `commercial` is deliberately kept in the SAME small (tonnes + hours,
    effectively $1-per-unit) units `rollout_engine_failure()` computes it in below, rather
    than a realistic bunker-price/day-rate $ conversion -- Sec 13.A.8a's own "exact numbers
    aren't fixed, the order-of-magnitude gap is the actual design decision" caveat applies
    directly here: a real $ conversion could easily produce commercial differences in the
    thousands, which would swamp a single tier-4 (goal) point and break the intended
    lexicographic property for no benefit at this abstraction level."""
    life_risk: float = 0.0
    ship_env_risk: float = 0.0
    duty_breaches: float = 0.0
    goal_shortfall: float = 0.0  # in [0, 1]: 1 - achievable fraction of success criteria
    commercial: float = 0.0


_TIER_WEIGHTS = {
    "life_risk": 1e8, "ship_env_risk": 1e6, "duty_breaches": 1e4, "goal_shortfall": 1e2, "commercial": 1.0,
}


def cost(dims: CostDimensions) -> float:
    """Sec 13.A.8a's lexicographic-by-construction weighted sum -- the 100x-per-tier gap
    means no realistic amount of tier-N savings can ever change a ranking already decided
    at tier-(N-1), for dimension values of the magnitude this skeleton actually produces."""
    return (dims.life_risk * _TIER_WEIGHTS["life_risk"]
            + dims.ship_env_risk * _TIER_WEIGHTS["ship_env_risk"]
            + dims.duty_breaches * _TIER_WEIGHTS["duty_breaches"]
            + dims.goal_shortfall * _TIER_WEIGHTS["goal_shortfall"]
            + dims.commercial * _TIER_WEIGHTS["commercial"])


@dataclass(frozen=True)
class RolloutResult:
    dims: CostDimensions
    notes: dict[str, float] = field(default_factory=dict)  # for explanation/debugging only


def oracle_best(candidates: list[CaptainAction],
                 rollout_fn: Callable[[CaptainAction], RolloutResult]) -> tuple[CaptainAction, float]:
    """argmin cost over candidates (Sec 13.A.4) -- returns (best_action, its own cost)."""
    scored = [(a, cost(rollout_fn(a).dims)) for a in candidates]
    return min(scored, key=lambda pair: pair[1])


def regret(chosen: CaptainAction, candidates: list[CaptainAction],
           rollout_fn: Callable[[CaptainAction], RolloutResult]) -> float:
    """regret = cost(chosen) - cost(oracle_best) (Sec 13.A.4) -- `chosen` need not be a
    member of `candidates`; `rollout_fn` only needs to be able to evaluate it, so a
    genuinely novel Captain proposal is scored exactly like any candidate, never penalised
    just for not matching the candidate generator's own list."""
    _, best_cost = oracle_best(candidates, rollout_fn)
    return cost(rollout_fn(chosen).dims) - best_cost


# --- Engine-failure decision layer (the ONE event fully wired for the skeleton) ---------


@dataclass(frozen=True)
class EngineFailureContext:
    """All facts the engine-failure decision layer needs (Sec 13.A.2/13.A.4), computed
    once at event-fire time and passed into candidates()/rollout() unchanged. Deliberately
    decoupled from the simulator's own `FuelModel`/`RefugeCandidate` classes (pipeline/ must
    never depend on Basic Simulator/app/, only the reverse, per this project's own
    established dependency rule -- e.g. pipeline/nomoto.py's own docstring) -- a caller in
    app/ passes `fuel_model.fuel_rate_tonnes_per_h` and the refuge's own distance/port id
    as plain values instead."""
    capped_speed_kn: float          # the Chief Engineer's declared safe speed (shield cap)
    original_soa_kn: float          # the Mission Order's planned speed, resumed after repair
    remaining_distance_nm: float    # distance left on the planned route at the moment of failure
    fuel_tonnes_available: float
    fuel_rate_tonnes_per_h: Callable[[float], float]  # a ship's own fuel-rate function of speed_kn
    deadline_slack_h: float | None  # hours of slack vs. the Mission Order's own deadline; None = no deadline
    refuge_distance_nm: float | None  # Sec 13.B.8's nearest qualifying port's route distance; None = none reachable
    refuge_port_id: str | None = None
    repair_duration_h: float = 24.0  # illustrative fixed time-in-port estimate, not paper-cited


def candidates_engine_failure(ctx: EngineFailureContext) -> list[CaptainAction]:
    """Sec 13.A.4's engine-failure candidates: continue at the capped speed (always
    available), plus divert to the port of refuge (only if one is actually reachable) --
    with no refuge nearby, this collapses to a single candidate, matching the design doc's
    own worked example."""
    candidates = [CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": ctx.capped_speed_kn})]
    if ctx.refuge_distance_nm is not None:
        candidates.append(CaptainAction(tool="request_place_of_refuge", params={"port_id": ctx.refuge_port_id}))
    return candidates


def _goal_shortfall(time_h: float, reference_time_h: float, deadline_slack_h: float | None) -> float:
    """Sec 13.A.8a's goal_cost, simplified to a binary step for the walking skeleton
    (v1): the Mission Order's deadline-related success criterion is either still
    achievable (0.0) or not (1.0), based on whether this action's own elapsed-time overrun
    beyond the unconstrained reference stays within the stated slack. No deadline
    (`deadline_slack_h is None`) means no criterion is at risk from this delay."""
    if deadline_slack_h is None:
        return 0.0
    overrun_h = time_h - reference_time_h
    return 1.0 if overrun_h > deadline_slack_h else 0.0


def rollout_engine_failure(action: CaptainAction, ctx: EngineFailureContext) -> RolloutResult:
    """Sec 13.A.8b's mission-level rollout, specialised to the engine-failure decision.
    `life_risk`/`ship_env_risk`/`duty_breaches` are 0.0 for both candidates here -- a plain
    engine failure with no further evidence is not itself a life-threatening emergency, and
    both candidates satisfy their own mandatory fault-log duty equally (Sec 13.A.5's
    evidence-based risk weighting is explicitly out of scope for this deterministic
    skeleton). The real differentiator is `goal_shortfall` (tier 4) and `commercial` (tier
    5): continuing travels the WHOLE remaining distance at the reduced speed; diverting
    pays a detour + fixed repair cost but then resumes the ORIGINAL speed for whatever
    distance remains beyond the port (assumed to lie roughly on the route)."""
    reference_time_h = ctx.remaining_distance_nm / ctx.original_soa_kn
    reference_fuel_t = ctx.fuel_rate_tonnes_per_h(ctx.original_soa_kn) * reference_time_h

    if action.tool == "continue_at_capped_speed":
        time_h = ctx.remaining_distance_nm / ctx.capped_speed_kn
        fuel_t = ctx.fuel_rate_tonnes_per_h(ctx.capped_speed_kn) * time_h
    elif action.tool == "request_place_of_refuge":
        if ctx.refuge_distance_nm is None:
            raise ValueError("request_place_of_refuge proposed but no refuge candidate is available")
        distance_to_port_nm = ctx.refuge_distance_nm
        time_to_port_h = distance_to_port_nm / ctx.capped_speed_kn
        fuel_to_port_t = ctx.fuel_rate_tonnes_per_h(ctx.capped_speed_kn) * time_to_port_h
        remaining_after_port_nm = max(0.0, ctx.remaining_distance_nm - distance_to_port_nm)
        time_after_repair_h = remaining_after_port_nm / ctx.original_soa_kn
        fuel_after_repair_t = ctx.fuel_rate_tonnes_per_h(ctx.original_soa_kn) * time_after_repair_h
        time_h = time_to_port_h + ctx.repair_duration_h + time_after_repair_h
        fuel_t = fuel_to_port_t + fuel_after_repair_t
    else:
        raise ValueError(f"unknown engine-failure candidate action: {action.tool!r}")

    delay_h = max(0.0, time_h - reference_time_h)
    fuel_excess_t = max(0.0, fuel_t - reference_fuel_t)
    dims = CostDimensions(
        goal_shortfall=_goal_shortfall(time_h, reference_time_h, ctx.deadline_slack_h),
        commercial=fuel_excess_t + delay_h,
    )
    return RolloutResult(dims=dims, notes={"time_h": time_h, "fuel_t": fuel_t, "delay_h": delay_h})
