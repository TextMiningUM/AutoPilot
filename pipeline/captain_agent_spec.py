"""Captain walking-skeleton Phase 3/4/6/7: procedure library, shield, cost()/regret
decision layer, the Sec 13.B.6 output schema + instruction-precedence mechanism, and all
5 v1 events' decision layers (design_captain_missions.md Sec 13.A.4/13.A.6/13.A.8/13.B.6).

Pure Python, no GPU/API key, safe to run locally. Mirrors `pipeline/oow_agent_spec.py`'s
role for OOW -- the Captain's own classify/validate/decide logic -- but lives in its own
file since Captain is a separately-trained, separate-domain agent (not a shared model with
OOW), per the project's own confirmed architecture decision.

All 5 v1 event types now have fully wired decision layers (Sec 14/Phase 6/Phase 7):
engine_failure/fog/whale_zone (Phase 3/6), and distress_call/commercial_instruction
(Phase 7 -- their world-responder resolutions are scheduled/applied by
`app/captain_skeleton.py`, not here; this module only computes candidates/cost, pure
decision logic has no scheduler dependency). Building a generic polymorphic dispatch
mechanism for the wired events would be speculative; each event type gets its own concrete
functions, following this file's existing shape.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

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


# --- Fog / whale-zone decision layers (Phase 6) -- degenerate, one candidate each --------
# Sec 13.A.4: both are "genuinely table-determined" (Rule 19's safe speed / a charted
# posted limit leave no real tradeoff space), unlike engine failure's real cost tradeoff --
# but still routed through cost()/oracle_best() so zero regret is DEMONSTRATED, not just
# asserted, and so a future shield substitution is scored the same way as any other event.

def _single_candidate_speed_rollout(speed_kn: float, original_soa_kn: float, affected_distance_nm: float,
                                    fuel_rate_tonnes_per_h: Callable[[float], float]) -> RolloutResult:
    """Shared cost arithmetic for a degenerate (one mandated speed) candidate -- used by
    both fog and whale_zone, which differ only in WHY the speed is mandated (Rule 19 vs a
    charted/posted limit), never in how the commercial cost of complying is computed."""
    reference_time_h = affected_distance_nm / original_soa_kn
    reference_fuel_t = fuel_rate_tonnes_per_h(original_soa_kn) * reference_time_h
    time_h = affected_distance_nm / speed_kn
    fuel_t = fuel_rate_tonnes_per_h(speed_kn) * time_h
    delay_h = max(0.0, time_h - reference_time_h)
    fuel_excess_t = max(0.0, fuel_t - reference_fuel_t)
    dims = CostDimensions(commercial=fuel_excess_t + delay_h)
    return RolloutResult(dims=dims, notes={"time_h": time_h, "fuel_t": fuel_t, "delay_h": delay_h})


@dataclass(frozen=True)
class FogContext:
    """Facts the fog decision layer needs (Sec 13.A.2/13.A.4) -- Rule 19 mandates the safe
    speed directly, so there is only ever ONE legitimate candidate."""
    safe_speed_kn: float          # the Rule-19 mandated safe speed for current visibility
    original_soa_kn: float
    affected_distance_nm: float   # distance travelled at reduced speed before visibility clears
    fuel_rate_tonnes_per_h: Callable[[float], float]


def candidates_fog(ctx: FogContext) -> list[CaptainAction]:
    """Sec 13.A.4: fog's one legitimate candidate -- reduce to the Rule-19 safe speed."""
    return [CaptainAction(tool="reduce_to_safe_speed", params={"speed_kn": ctx.safe_speed_kn})]


def rollout_fog(action: CaptainAction, ctx: FogContext) -> RolloutResult:
    """The only candidate's cost -- a real commercial cost (time/fuel lost to the mandatory
    reduction) but zero regret BY CONSTRUCTION (no cheaper legitimate alternative exists)."""
    if action.tool != "reduce_to_safe_speed":
        raise ValueError(f"unknown fog candidate action: {action.tool!r}")
    return _single_candidate_speed_rollout(ctx.safe_speed_kn, ctx.original_soa_kn,
                                           ctx.affected_distance_nm, ctx.fuel_rate_tonnes_per_h)


@dataclass(frozen=True)
class WhaleZoneContext:
    """Facts the whale-zone decision layer needs (Sec 13.A.2/13.A.4) -- a charted/posted
    zone speed limit mandates the speed directly, so there is only ever ONE candidate."""
    speed_limit_kn: float
    original_soa_kn: float
    zone_transit_distance_nm: float  # the zone's own extent along the route
    fuel_rate_tonnes_per_h: Callable[[float], float]


def candidates_whale_zone(ctx: WhaleZoneContext) -> list[CaptainAction]:
    """Sec 13.A.4: the zone's one legitimate candidate -- reduce to its posted limit."""
    return [CaptainAction(tool="reduce_to_zone_speed_limit", params={"speed_kn": ctx.speed_limit_kn})]


def rollout_whale_zone(action: CaptainAction, ctx: WhaleZoneContext) -> RolloutResult:
    """The only candidate's cost -- same shape as `rollout_fog()`, zero regret by
    construction."""
    if action.tool != "reduce_to_zone_speed_limit":
        raise ValueError(f"unknown whale_zone candidate action: {action.tool!r}")
    return _single_candidate_speed_rollout(ctx.speed_limit_kn, ctx.original_soa_kn,
                                           ctx.zone_transit_distance_nm, ctx.fuel_rate_tonnes_per_h)


# --- Distress call decision layer (Phase 7) -- a genuine judgement case (Sec 13.A.4) -----
# "ignoring it costs at least one duty_cost point (tier 3, weight 1e4) -- no fuel saving
# (commercial, weight 1) can ever close that gap" (Sec 13.A.8a's own worked answer to
# "the oracle could ignore a distress call because it's cheaper"). Declining is NOT always
# wrong, though: Sec 13.A.4 frames it as "assist vs. document a VALID reason not to" -- the
# one valid reason modeled here is that assisting would itself breach a safety margin
# (Sec 13.A.6), computed by the caller via `check_safety_margins()`, same as Sec 13.A.6's
# own worked example of that function's general-purpose use.

@dataclass(frozen=True)
class DistressCallContext:
    """Facts the distress-call decision layer needs (Sec 13.A.2/13.A.4)."""
    detour_distance_nm: float   # extra round-trip distance to reach + resume from the distress position
    original_soa_kn: float
    remaining_distance_nm: float  # own mission's remaining distance at the time of the call
    fuel_rate_tonnes_per_h: Callable[[float], float]
    deadline_slack_h: float | None
    assisting_breaches_safety_margin: bool  # Sec 13.A.6's computed condition -- the one valid reason to decline


def candidates_distress_call(ctx: DistressCallContext) -> list[CaptainAction]:
    """Sec 13.A.4: assist (discharge the SOLAS duty) vs. decline with a logged reason."""
    return [CaptainAction(tool="proceed_to_assist", params={}),
           CaptainAction(tool="decline_with_logged_reason", params={})]


def rollout_distress_call(action: CaptainAction, ctx: DistressCallContext) -> RolloutResult:
    """Assisting costs real time/fuel (commercial tier) but discharges the duty cleanly;
    declining is free commercially but costs a real `duty_breaches` point UNLESS assisting
    would itself have breached a safety margin -- Sec 13.A.8a's 100x-per-tier gap then
    makes 'assist' win whenever it's safe to do so, and 'decline' win when it genuinely
    isn't, regardless of how large the commercial difference happens to be."""
    reference_time_h = ctx.remaining_distance_nm / ctx.original_soa_kn
    reference_fuel_t = ctx.fuel_rate_tonnes_per_h(ctx.original_soa_kn) * reference_time_h

    if action.tool == "proceed_to_assist":
        total_distance_nm = ctx.remaining_distance_nm + ctx.detour_distance_nm
        time_h = total_distance_nm / ctx.original_soa_kn
        fuel_t = ctx.fuel_rate_tonnes_per_h(ctx.original_soa_kn) * time_h
        duty_breaches = 0.0
    elif action.tool == "decline_with_logged_reason":
        time_h = reference_time_h
        fuel_t = reference_fuel_t
        duty_breaches = 0.0 if ctx.assisting_breaches_safety_margin else 1.0
    else:
        raise ValueError(f"unknown distress-call candidate action: {action.tool!r}")

    delay_h = max(0.0, time_h - reference_time_h)
    fuel_excess_t = max(0.0, fuel_t - reference_fuel_t)
    dims = CostDimensions(
        duty_breaches=duty_breaches,
        goal_shortfall=_goal_shortfall(time_h, reference_time_h, ctx.deadline_slack_h),
        commercial=fuel_excess_t + delay_h,
    )
    return RolloutResult(dims=dims, notes={"time_h": time_h, "fuel_t": fuel_t, "delay_h": delay_h})


# --- Commercial instruction vs. safety decision layer (Phase 7) -------------------------
# Sec 13.A.2: "Commercial instruction's 'breaches a safety margin' becomes a COMPUTED
# condition against the live Mission State rather than a fixed always-refuse rule" -- the
# caller supplies that computed boolean (typically via `check_safety_margins()` against the
# ship's own engine/fuel limits), this module never re-derives it independently.

@dataclass(frozen=True)
class CommercialInstructionContext:
    """Facts the commercial-instruction decision layer needs (Sec 13.A.2/13.A.4)."""
    demanded_speed_kn: float   # what the company/DPA message is demanding
    original_soa_kn: float
    remaining_distance_nm: float
    fuel_rate_tonnes_per_h: Callable[[float], float]
    deadline_slack_h: float | None
    complying_breaches_safety_margin: bool  # Sec 13.A.6's computed condition


def candidates_commercial_instruction(ctx: CommercialInstructionContext) -> list[CaptainAction]:
    """Sec 13.A.4: comply with the instruction vs. refuse, citing ISM Art. 5."""
    return [CaptainAction(tool="comply_with_instruction", params={"speed_kn": ctx.demanded_speed_kn}),
           CaptainAction(tool="refuse_citing_ism_art5", params={})]


def rollout_commercial_instruction(action: CaptainAction, ctx: CommercialInstructionContext) -> RolloutResult:
    """Complying costs a real `duty_breaches` point when it would actually breach a
    safety margin -- the decision layer's own ranking then already agrees with the
    shield (Sec 13.A.6) rather than needing the shield to override a bad oracle pick.
    When it's SAFE, complying is not automatically commercially better: the cubic
    fuel-rate curve (Sec 13.A.3) means going faster can burn more fuel despite costing
    less time, so complying only wins when real deadline pressure (`deadline_slack_h`
    already tight/negative, e.g. from an earlier delay elsewhere in the mission) makes
    the extra speed actually necessary -- never assumed, always computed."""
    reference_time_h = ctx.remaining_distance_nm / ctx.original_soa_kn
    reference_fuel_t = ctx.fuel_rate_tonnes_per_h(ctx.original_soa_kn) * reference_time_h

    if action.tool == "comply_with_instruction":
        time_h = ctx.remaining_distance_nm / ctx.demanded_speed_kn
        fuel_t = ctx.fuel_rate_tonnes_per_h(ctx.demanded_speed_kn) * time_h
        duty_breaches = 1.0 if ctx.complying_breaches_safety_margin else 0.0
    elif action.tool == "refuse_citing_ism_art5":
        time_h = reference_time_h
        fuel_t = reference_fuel_t
        duty_breaches = 0.0
    else:
        raise ValueError(f"unknown commercial-instruction candidate action: {action.tool!r}")

    delay_h = max(0.0, time_h - reference_time_h)
    fuel_excess_t = max(0.0, fuel_t - reference_fuel_t)
    dims = CostDimensions(
        duty_breaches=duty_breaches,
        goal_shortfall=_goal_shortfall(time_h, reference_time_h, ctx.deadline_slack_h),
        commercial=fuel_excess_t + delay_h,
    )
    return RolloutResult(dims=dims, notes={"time_h": time_h, "fuel_t": fuel_t, "delay_h": delay_h})


# --- Output schema + precedence (Sec 13.B.6, Phase 4) ------------------------------------

HOLD_ACTION = CaptainAction(tool="hold", params={})


@dataclass(frozen=True)
class CaptainResponse:
    """Sec 13.B.6's full output schema -- `action` carries the machine-checkable
    tool/params, `plan`/`reasoning` are the free-text planning fields a prompt-driven
    Captain would also produce. The walking skeleton's deterministic baseline (Sec 14)
    fills these with simple literal values, never hand-written decision logic (Sec 13.B.5's
    facts-only rule applies to the RENDERER, not to what this schema can hold)."""
    action: CaptainAction
    plan: dict[str, Any]  # {"updated_goals": [...], "resource_note": <str>}
    reasoning: str


def validate_captain_response_json(obj: dict) -> list[str]:
    """Returns a list of validation-error strings (empty == valid) for a candidate Captain
    response dict -- mirrors `oow_agent_spec.validate_action_json()`'s pattern, per Sec
    13.B.6's own "mirrors OOW's validate_action_json() pattern" decision."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return [f"expected a dict, got {type(obj).__name__}"]
    for key in ("tool", "params", "plan", "reasoning"):
        if key not in obj:
            errors.append(f"missing required key {key!r}")
    if not isinstance(obj.get("tool"), str) or not obj.get("tool"):
        errors.append(f"'tool' must be a non-empty string, got {obj.get('tool')!r}")
    if not isinstance(obj.get("params"), dict):
        errors.append(f"'params' must be a dict, got {obj.get('params')!r}")
    plan = obj.get("plan")
    if not isinstance(plan, dict) or "updated_goals" not in plan or "resource_note" not in plan:
        errors.append("'plan' must be a dict with 'updated_goals' and 'resource_note' keys")
    elif not isinstance(plan.get("updated_goals"), list):
        errors.append(f"'plan.updated_goals' must be a list, got {plan.get('updated_goals')!r}")
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        errors.append("'reasoning' must be a non-empty string")
    return errors


def parse_captain_response(obj: dict) -> tuple[CaptainResponse, bool]:
    """Parses a candidate dict into a CaptainResponse; on ANY schema violation, falls back
    to a safe no-op (`hold`) response and flags `was_parse_error=True` -- Sec 13.B.6: "never
    silently apply a malformed instruction". Returns `(response, was_parse_error)`."""
    errors = validate_captain_response_json(obj)
    if errors:
        fallback = CaptainResponse(action=HOLD_ACTION, plan={"updated_goals": [], "resource_note": ""},
                                   reasoning=f"parse error, defaulted to hold: {'; '.join(errors)}")
        return fallback, True
    return CaptainResponse(
        action=CaptainAction(tool=obj["tool"], params=obj["params"]),
        plan=obj["plan"], reasoning=obj["reasoning"],
    ), False


@dataclass
class ActiveInstruction:
    """One scope's currently-active Captain instruction (Sec 13.B.6) -- `resolution` is a
    machine-checkable predicate dict, e.g. `{"type": "elapsed_minutes", "value": 120}`,
    never bare free text like "until clear of the TSS"."""
    scope: str  # "route" | "speed" | "regime"
    action: CaptainAction
    resolution: dict[str, Any]
    issued_at_s: float


class InstructionStack:
    """Sec 13.B.6's "at most one active instruction per scope, most recent instruction
    supersedes any earlier unresolved one" rule -- a dict keyed by scope, never a merge."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveInstruction] = {}

    def issue(self, instruction: ActiveInstruction) -> None:
        """The newest instruction for a scope always replaces any earlier unresolved one."""
        self._active[instruction.scope] = instruction

    def get(self, scope: str) -> ActiveInstruction | None:
        return self._active.get(scope)

    def resolve_if_due(self, scope: str, now_s: float) -> bool:
        """Evaluates the scope's active instruction's own resolution predicate against the
        current mission-sim time; clears it and returns True if resolved, False otherwise
        (including when nothing is active for this scope). Only `elapsed_minutes` is
        implemented for the walking skeleton (Sec 15.3) -- an `exit_polygon`-style
        predicate needs live zone-occupancy tracking, not yet needed by any scenario this
        skeleton actually runs (Sec 13.C.13's engine-failure scenario has no exclusion
        zones); add it alongside the first scenario that genuinely needs it."""
        instr = self._active.get(scope)
        if instr is None:
            return False
        pred = instr.resolution
        if pred.get("type") == "elapsed_minutes" and (now_s - instr.issued_at_s) / 60.0 >= pred["value"]:
            del self._active[scope]
            return True
        return False
