"""Captain walking-skeleton Phase 3/4/6/7 tests: procedure library, shield, cost()/regret
decision layer, the Sec 13.B.6 output schema + instruction-precedence mechanism, and all 5
v1 events' decision layers. Pure pipeline/ module -- no app/ dependency, no GPU/API key
needed."""
import pytest

from pipeline.captain_agent_spec import (
    HOLD_ACTION, PROCEDURE_LIBRARY, ActiveInstruction, CommercialInstructionContext, CostDimensions,
    DistressCallContext, EngineFailureContext, FogContext, InstructionStack, MarginViolation,
    WhaleZoneContext, candidates_commercial_instruction, candidates_distress_call,
    candidates_engine_failure, candidates_fog, candidates_whale_zone, check_safety_margins,
    cost, oracle_best, parse_captain_response, regret, resolve_captain_decision,
    rollout_commercial_instruction, rollout_distress_call, rollout_engine_failure, rollout_fog,
    rollout_whale_zone, validate_captain_response_json,
)
from pipeline.captain_types import CaptainAction

# --- cost() / lexicographic ordering -----------------------------------------------------


def test_cost_is_the_weighted_sum_of_dimensions():
    dims = CostDimensions(life_risk=0.0, ship_env_risk=0.0, duty_breaches=1.0, goal_shortfall=0.5, commercial=3.0)
    assert cost(dims) == pytest.approx(1 * 1e4 + 0.5 * 1e2 + 3.0)


def test_goal_tier_dominates_a_smaller_commercial_advantage():
    # Missing the goal (1.0) costs 100; a $99 commercial saving can't make up for it.
    worse_goal_cheaper_commercial = CostDimensions(goal_shortfall=1.0, commercial=0.0)
    better_goal_pricier_commercial = CostDimensions(goal_shortfall=0.0, commercial=99.0)
    assert cost(better_goal_pricier_commercial) < cost(worse_goal_cheaper_commercial)


def test_life_tier_difference_dominates_regardless_of_identical_lower_tier_costs():
    # The guarantee is per ADJACENT tier (a decision already fixed by the higher tier can't
    # be overturned by the one below it) -- not "any tiny tier-1 value beats the maximum
    # possible value of every lower tier simultaneously", which the 100x-per-tier weight
    # gap does not actually claim.
    shared_lower_tiers = dict(ship_env_risk=1.0, duty_breaches=5.0, goal_shortfall=1.0, commercial=500.0)
    worse_life_risk = CostDimensions(life_risk=0.001, **shared_lower_tiers)
    better_life_risk = CostDimensions(life_risk=0.0, **shared_lower_tiers)
    assert cost(better_life_risk) < cost(worse_life_risk)


# --- check_safety_margins() --------------------------------------------------------------


def test_no_violations_when_no_facts_are_supplied():
    assert check_safety_margins(CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": 8.0})) == []


def test_engine_speed_cap_violation():
    action = CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": 10.0})
    violations = check_safety_margins(action, engine_max_speed_kn=8.0)
    assert len(violations) == 1
    assert violations[0].margin == "engine_speed_cap"


def test_engine_speed_cap_respected_is_clean():
    action = CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": 8.0})
    assert check_safety_margins(action, engine_max_speed_kn=8.0) == []


def test_exclusion_zone_violation_for_unauthorised_zone():
    action = CaptainAction(tool="reroute", params={})
    violations = check_safety_margins(action, candidate_zone_ids_entered=["zone_1"], authorized_zone_ids={"zone_2"})
    assert any(v.margin == "exclusion_zone" for v in violations)


def test_exclusion_zone_authorised_is_clean():
    action = CaptainAction(tool="reroute", params={})
    violations = check_safety_margins(action, candidate_zone_ids_entered=["zone_1"], authorized_zone_ids={"zone_1"})
    assert violations == []


def test_rest_hours_breach_is_reported():
    action = CaptainAction(tool="hold", params={})
    violations = check_safety_margins(action, oow_rest_breach=True)
    assert any(v.margin == "rest_hours" for v in violations)


def test_fuel_reserve_violation():
    action = CaptainAction(tool="reroute", params={})
    violations = check_safety_margins(action, candidate_fuel_consumption_t=800.0,
                                      fuel_tonnes_available=850.0, fuel_reserve_margin_pct=15.0)
    assert any(v.margin == "fuel_reserve" for v in violations)  # leaves 50t, reserve needs 127.5t


def test_fuel_reserve_respected_is_clean():
    action = CaptainAction(tool="reroute", params={})
    violations = check_safety_margins(action, candidate_fuel_consumption_t=600.0,
                                      fuel_tonnes_available=850.0, fuel_reserve_margin_pct=15.0)
    assert violations == []  # leaves 250t, well above the 127.5t reserve


# --- resolve_captain_decision() -----------------------------------------------------------


def test_clean_decision_passes_through_unsubstituted():
    proposed = CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": 8.0})
    applied, substituted = resolve_captain_decision(proposed, violations=[], mandatory_fallback=proposed)
    assert applied is proposed
    assert not substituted


def test_violating_decision_is_substituted_with_the_fallback():
    proposed = CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": 10.0})
    fallback = CaptainAction(tool="slow_down", params={"speed_kn": 8.0})
    violations = [MarginViolation("engine_speed_cap", "too fast")]
    applied, substituted = resolve_captain_decision(proposed, violations, fallback)
    assert applied is fallback
    assert substituted


# --- PROCEDURE_LIBRARY --------------------------------------------------------------------


def test_procedure_library_has_all_five_v1_events():
    assert set(PROCEDURE_LIBRARY) == {"engine_failure", "fog", "distress_call", "whale_zone", "commercial_instruction"}


def test_every_procedure_entry_has_a_shield_description():
    for entry in PROCEDURE_LIBRARY.values():
        assert entry.shield_description


# --- Engine-failure decision layer --------------------------------------------------------

def _fuel_rate_fn(base_load: float = 1.0, k: float = 0.001):
    return lambda speed_kn: base_load + k * speed_kn ** 3


def test_candidates_collapse_to_continue_when_no_refuge_is_reachable():
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=_fuel_rate_fn(),
        deadline_slack_h=None, refuge_distance_nm=None,
    )
    candidates = candidates_engine_failure(ctx)
    assert len(candidates) == 1
    assert candidates[0].tool == "continue_at_capped_speed"


def test_candidates_include_divert_when_a_refuge_is_reachable():
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=_fuel_rate_fn(),
        deadline_slack_h=None, refuge_distance_nm=40.0, refuge_port_id="port_stub_refuge",
    )
    tools = {c.tool for c in candidates_engine_failure(ctx)}
    assert tools == {"continue_at_capped_speed", "request_place_of_refuge"}


def test_rollout_continue_matches_hand_computed_arithmetic():
    fuel_fn = _fuel_rate_fn()
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=fuel_fn,
        deadline_slack_h=None, refuge_distance_nm=None,
    )
    result = rollout_engine_failure(CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": 8.0}), ctx)
    expected_time_h = 100.0 / 8.0
    reference_time_h = 100.0 / 12.0
    expected_fuel_t = fuel_fn(8.0) * expected_time_h
    reference_fuel_t = fuel_fn(12.0) * reference_time_h
    assert result.notes["time_h"] == pytest.approx(expected_time_h)
    assert result.notes["fuel_t"] == pytest.approx(expected_fuel_t)
    assert result.dims.commercial == pytest.approx(
        max(0.0, expected_fuel_t - reference_fuel_t) + max(0.0, expected_time_h - reference_time_h))
    assert result.dims.goal_shortfall == 0.0  # no deadline supplied


def test_rollout_divert_matches_hand_computed_arithmetic():
    fuel_fn = _fuel_rate_fn()
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=fuel_fn,
        deadline_slack_h=None, refuge_distance_nm=40.0, refuge_port_id="port_stub_refuge",
        repair_duration_h=24.0,
    )
    result = rollout_engine_failure(CaptainAction(tool="request_place_of_refuge", params={"port_id": "port_stub_refuge"}), ctx)
    time_to_port_h = 40.0 / 8.0
    remaining_after_h = (100.0 - 40.0) / 12.0
    expected_time_h = time_to_port_h + 24.0 + remaining_after_h
    assert result.notes["time_h"] == pytest.approx(expected_time_h)


def test_rollout_raises_for_divert_with_no_refuge():
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=_fuel_rate_fn(),
        deadline_slack_h=None, refuge_distance_nm=None,
    )
    with pytest.raises(ValueError):
        rollout_engine_failure(CaptainAction(tool="request_place_of_refuge", params={}), ctx)


def test_rollout_raises_for_an_unknown_action():
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=_fuel_rate_fn(),
        deadline_slack_h=None, refuge_distance_nm=None,
    )
    with pytest.raises(ValueError):
        rollout_engine_failure(CaptainAction(tool="abort_mission", params={}), ctx)


def test_rollout_is_deterministic():
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=_fuel_rate_fn(),
        deadline_slack_h=None, refuge_distance_nm=40.0, refuge_port_id="port_stub_refuge",
    )
    action = CaptainAction(tool="request_place_of_refuge", params={"port_id": "port_stub_refuge"})
    r1 = rollout_engine_failure(action, ctx)
    r2 = rollout_engine_failure(action, ctx)
    assert r1.dims == r2.dims
    assert r1.notes == r2.notes


def test_goal_shortfall_flips_from_met_to_missed_as_slack_tightens():
    fuel_fn = _fuel_rate_fn()
    base = dict(capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
               fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=fuel_fn, refuge_distance_nm=None)
    action = CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": 8.0})
    # delay_h for this scenario is 100/8 - 100/12 = 4.1666...h
    generous = rollout_engine_failure(action, EngineFailureContext(**base, deadline_slack_h=10.0))
    tight = rollout_engine_failure(action, EngineFailureContext(**base, deadline_slack_h=1.0))
    assert generous.dims.goal_shortfall == 0.0
    assert tight.dims.goal_shortfall == 1.0


def test_oracle_best_picks_continue_when_diverting_is_clearly_worse():
    # Short remaining distance, mild cap, a distant/slow detour with a long repair --
    # diverting is never worth it here (matches the design doc's own worked example).
    fuel_fn = _fuel_rate_fn()
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=fuel_fn,
        deadline_slack_h=None, refuge_distance_nm=40.0, refuge_port_id="port_stub_refuge",
        repair_duration_h=24.0,
    )
    candidates = candidates_engine_failure(ctx)
    best, _ = oracle_best(candidates, lambda a: rollout_engine_failure(a, ctx))
    assert best.tool == "continue_at_capped_speed"


def test_oracle_best_prefers_diverting_when_the_cap_is_severe_and_distance_is_long():
    # Severe cap over a long remaining distance -- continuing the whole way is far worse
    # than a short detour + repair + resuming the original speed for the rest.
    fuel_fn = _fuel_rate_fn()
    ctx = EngineFailureContext(
        capped_speed_kn=4.0, original_soa_kn=12.0, remaining_distance_nm=2000.0,
        fuel_tonnes_available=10000.0, fuel_rate_tonnes_per_h=fuel_fn,
        deadline_slack_h=None, refuge_distance_nm=40.0, refuge_port_id="port_stub_refuge",
        repair_duration_h=24.0,
    )
    candidates = candidates_engine_failure(ctx)
    best, _ = oracle_best(candidates, lambda a: rollout_engine_failure(a, ctx))
    assert best.tool == "request_place_of_refuge"


def test_regret_is_zero_for_the_oracle_best_action_itself():
    fuel_fn = _fuel_rate_fn()
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=fuel_fn,
        deadline_slack_h=None, refuge_distance_nm=40.0, refuge_port_id="port_stub_refuge",
    )
    candidates = candidates_engine_failure(ctx)
    rollout_fn = lambda a: rollout_engine_failure(a, ctx)  # noqa: E731
    best, _ = oracle_best(candidates, rollout_fn)
    assert regret(best, candidates, rollout_fn) == pytest.approx(0.0)


def test_regret_is_positive_for_the_clearly_worse_candidate():
    fuel_fn = _fuel_rate_fn()
    ctx = EngineFailureContext(
        capped_speed_kn=8.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
        fuel_tonnes_available=850.0, fuel_rate_tonnes_per_h=fuel_fn,
        deadline_slack_h=None, refuge_distance_nm=40.0, refuge_port_id="port_stub_refuge",
        repair_duration_h=24.0,
    )
    candidates = candidates_engine_failure(ctx)
    rollout_fn = lambda a: rollout_engine_failure(a, ctx)  # noqa: E731
    worse = next(c for c in candidates if c.tool == "request_place_of_refuge")
    assert regret(worse, candidates, rollout_fn) > 0.0


# --- Output schema (Sec 13.B.6) -----------------------------------------------------------


def test_validate_captain_response_accepts_a_well_formed_response():
    obj = {"tool": "continue_at_capped_speed", "params": {"speed_kn": 8.0},
          "plan": {"updated_goals": [], "resource_note": "fuel on track"}, "reasoning": "deadline is tight"}
    assert validate_captain_response_json(obj) == []


def test_validate_captain_response_reports_missing_keys():
    errors = validate_captain_response_json({"tool": "hold"})
    assert any("params" in e for e in errors)
    assert any("plan" in e for e in errors)
    assert any("reasoning" in e for e in errors)


def test_validate_captain_response_rejects_a_non_dict_plan():
    obj = {"tool": "hold", "params": {}, "plan": "not a dict", "reasoning": "x"}
    assert any("plan" in e for e in validate_captain_response_json(obj))


def test_validate_captain_response_rejects_empty_reasoning():
    obj = {"tool": "hold", "params": {}, "plan": {"updated_goals": [], "resource_note": ""}, "reasoning": "   "}
    assert any("reasoning" in e for e in validate_captain_response_json(obj))


def test_parse_captain_response_round_trips_a_valid_response():
    obj = {"tool": "request_place_of_refuge", "params": {"port_id": "port_stub_refuge"},
          "plan": {"updated_goals": [], "resource_note": "diverting"}, "reasoning": "severe cap"}
    response, was_parse_error = parse_captain_response(obj)
    assert not was_parse_error
    assert response.action == CaptainAction(tool="request_place_of_refuge", params={"port_id": "port_stub_refuge"})
    assert response.reasoning == "severe cap"


def test_parse_captain_response_falls_back_to_hold_on_a_malformed_response():
    response, was_parse_error = parse_captain_response({"tool": "hold"})
    assert was_parse_error
    assert response.action == HOLD_ACTION


# --- Instruction precedence (Sec 13.B.6) ---------------------------------------------------


def test_instruction_stack_returns_none_for_an_empty_scope():
    stack = InstructionStack()
    assert stack.get("speed") is None


def test_instruction_stack_the_newest_instruction_supersedes_the_earlier_one():
    stack = InstructionStack()
    first = ActiveInstruction(scope="speed", action=CaptainAction(tool="slow_down", params={"speed_kn": 8.0}),
                              resolution={"type": "elapsed_minutes", "value": 60}, issued_at_s=0.0)
    second = ActiveInstruction(scope="speed", action=CaptainAction(tool="slow_down", params={"speed_kn": 6.0}),
                               resolution={"type": "elapsed_minutes", "value": 30}, issued_at_s=10.0)
    stack.issue(first)
    stack.issue(second)
    assert stack.get("speed") is second


def test_instruction_stack_resolve_if_due_clears_an_elapsed_minutes_instruction():
    stack = InstructionStack()
    instr = ActiveInstruction(scope="speed", action=CaptainAction(tool="slow_down", params={}),
                              resolution={"type": "elapsed_minutes", "value": 30}, issued_at_s=0.0)
    stack.issue(instr)
    assert not stack.resolve_if_due("speed", now_s=1000.0)  # 16.6 min < 30 min
    assert stack.resolve_if_due("speed", now_s=1800.0)  # exactly 30 min
    assert stack.get("speed") is None


def test_instruction_stack_resolve_if_due_is_false_for_an_unoccupied_scope():
    stack = InstructionStack()
    assert not stack.resolve_if_due("route", now_s=1000.0)


# --- Fog / whale-zone decision layers (Phase 6) --------------------------------------------


def test_candidates_fog_has_exactly_one_candidate():
    ctx = FogContext(safe_speed_kn=6.0, original_soa_kn=12.0, affected_distance_nm=25.0,
                     fuel_rate_tonnes_per_h=_fuel_rate_fn())
    candidates = candidates_fog(ctx)
    assert len(candidates) == 1
    assert candidates[0] == CaptainAction(tool="reduce_to_safe_speed", params={"speed_kn": 6.0})


def test_rollout_fog_matches_hand_computed_arithmetic():
    fuel_fn = _fuel_rate_fn()
    ctx = FogContext(safe_speed_kn=6.0, original_soa_kn=12.0, affected_distance_nm=25.0,
                     fuel_rate_tonnes_per_h=fuel_fn)
    result = rollout_fog(CaptainAction(tool="reduce_to_safe_speed", params={"speed_kn": 6.0}), ctx)
    expected_time_h = 25.0 / 6.0
    reference_time_h = 25.0 / 12.0
    assert result.notes["time_h"] == pytest.approx(expected_time_h)
    expected_fuel_t = fuel_fn(6.0) * expected_time_h
    reference_fuel_t = fuel_fn(12.0) * reference_time_h
    assert result.dims.commercial == pytest.approx(
        max(0.0, expected_fuel_t - reference_fuel_t) + max(0.0, expected_time_h - reference_time_h))


def test_rollout_fog_raises_for_an_unknown_action():
    ctx = FogContext(safe_speed_kn=6.0, original_soa_kn=12.0, affected_distance_nm=25.0,
                     fuel_rate_tonnes_per_h=_fuel_rate_fn())
    with pytest.raises(ValueError):
        rollout_fog(CaptainAction(tool="hold", params={}), ctx)


def test_oracle_best_for_fog_has_zero_regret_by_construction():
    fuel_fn = _fuel_rate_fn()
    ctx = FogContext(safe_speed_kn=6.0, original_soa_kn=12.0, affected_distance_nm=25.0,
                     fuel_rate_tonnes_per_h=fuel_fn)
    candidates = candidates_fog(ctx)
    rollout_fn = lambda a: rollout_fog(a, ctx)  # noqa: E731
    best, _ = oracle_best(candidates, rollout_fn)
    assert regret(best, candidates, rollout_fn) == pytest.approx(0.0)


def test_candidates_whale_zone_has_exactly_one_candidate():
    ctx = WhaleZoneContext(speed_limit_kn=10.0, original_soa_kn=12.0, zone_transit_distance_nm=20.0,
                          fuel_rate_tonnes_per_h=_fuel_rate_fn())
    candidates = candidates_whale_zone(ctx)
    assert len(candidates) == 1
    assert candidates[0] == CaptainAction(tool="reduce_to_zone_speed_limit", params={"speed_kn": 10.0})


def test_rollout_whale_zone_matches_hand_computed_arithmetic():
    fuel_fn = _fuel_rate_fn()
    ctx = WhaleZoneContext(speed_limit_kn=10.0, original_soa_kn=12.0, zone_transit_distance_nm=20.0,
                          fuel_rate_tonnes_per_h=fuel_fn)
    result = rollout_whale_zone(CaptainAction(tool="reduce_to_zone_speed_limit", params={"speed_kn": 10.0}), ctx)
    expected_time_h = 20.0 / 10.0
    assert result.notes["time_h"] == pytest.approx(expected_time_h)


def test_rollout_whale_zone_raises_for_an_unknown_action():
    ctx = WhaleZoneContext(speed_limit_kn=10.0, original_soa_kn=12.0, zone_transit_distance_nm=20.0,
                          fuel_rate_tonnes_per_h=_fuel_rate_fn())
    with pytest.raises(ValueError):
        rollout_whale_zone(CaptainAction(tool="hold", params={}), ctx)


# --- Distress call decision layer (Phase 7) ------------------------------------------------


def test_candidates_distress_call_offers_assist_and_decline():
    ctx = DistressCallContext(detour_distance_nm=20.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                              fuel_rate_tonnes_per_h=_fuel_rate_fn(), deadline_slack_h=None,
                              assisting_breaches_safety_margin=False)
    tools = {c.tool for c in candidates_distress_call(ctx)}
    assert tools == {"proceed_to_assist", "decline_with_logged_reason"}


def test_distress_call_oracle_best_assists_when_it_is_safe_to_do_so():
    fuel_fn = _fuel_rate_fn()
    ctx = DistressCallContext(detour_distance_nm=20.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                              fuel_rate_tonnes_per_h=fuel_fn, deadline_slack_h=None,
                              assisting_breaches_safety_margin=False)
    candidates = candidates_distress_call(ctx)
    best, _ = oracle_best(candidates, lambda a: rollout_distress_call(a, ctx))
    assert best.tool == "proceed_to_assist"  # the SOLAS duty_cost dominates any commercial saving


def test_distress_call_oracle_best_declines_when_assisting_would_breach_a_safety_margin():
    fuel_fn = _fuel_rate_fn()
    ctx = DistressCallContext(detour_distance_nm=20.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                              fuel_rate_tonnes_per_h=fuel_fn, deadline_slack_h=None,
                              assisting_breaches_safety_margin=True)
    candidates = candidates_distress_call(ctx)
    best, _ = oracle_best(candidates, lambda a: rollout_distress_call(a, ctx))
    assert best.tool == "decline_with_logged_reason"


def test_rollout_distress_call_assist_matches_hand_computed_arithmetic():
    fuel_fn = _fuel_rate_fn()
    ctx = DistressCallContext(detour_distance_nm=20.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                              fuel_rate_tonnes_per_h=fuel_fn, deadline_slack_h=None,
                              assisting_breaches_safety_margin=False)
    result = rollout_distress_call(CaptainAction(tool="proceed_to_assist", params={}), ctx)
    expected_time_h = 120.0 / 12.0
    assert result.notes["time_h"] == pytest.approx(expected_time_h)
    assert result.dims.duty_breaches == 0.0


def test_rollout_distress_call_decline_with_no_valid_reason_costs_a_duty_breach():
    ctx = DistressCallContext(detour_distance_nm=20.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                              fuel_rate_tonnes_per_h=_fuel_rate_fn(), deadline_slack_h=None,
                              assisting_breaches_safety_margin=False)
    result = rollout_distress_call(CaptainAction(tool="decline_with_logged_reason", params={}), ctx)
    assert result.dims.duty_breaches == 1.0


def test_rollout_distress_call_decline_with_a_valid_reason_costs_no_duty_breach():
    ctx = DistressCallContext(detour_distance_nm=20.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                              fuel_rate_tonnes_per_h=_fuel_rate_fn(), deadline_slack_h=None,
                              assisting_breaches_safety_margin=True)
    result = rollout_distress_call(CaptainAction(tool="decline_with_logged_reason", params={}), ctx)
    assert result.dims.duty_breaches == 0.0


def test_rollout_distress_call_raises_for_an_unknown_action():
    ctx = DistressCallContext(detour_distance_nm=20.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                              fuel_rate_tonnes_per_h=_fuel_rate_fn(), deadline_slack_h=None,
                              assisting_breaches_safety_margin=False)
    with pytest.raises(ValueError):
        rollout_distress_call(CaptainAction(tool="hold", params={}), ctx)


# --- Commercial instruction vs. safety decision layer (Phase 7) ----------------------------


def test_candidates_commercial_instruction_offers_comply_and_refuse():
    ctx = CommercialInstructionContext(demanded_speed_kn=16.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                                       fuel_rate_tonnes_per_h=_fuel_rate_fn(), deadline_slack_h=None,
                                       complying_breaches_safety_margin=True)
    tools = {c.tool for c in candidates_commercial_instruction(ctx)}
    assert tools == {"comply_with_instruction", "refuse_citing_ism_art5"}


def test_commercial_instruction_oracle_best_complies_when_deadline_pressure_demands_it():
    # Note: with NO deadline pressure, going faster is actually commercially WORSE here
    # (the cubic fuel-rate curve means more speed can burn more fuel despite costing less
    # time, same insight as Phase 5's eval) -- refusing a pointless speed-up is correct.
    # Complying only wins when an already-accumulated deficit (deadline_slack_h < 0, e.g.
    # from an earlier fog/engine-failure delay) means ONLY the extra speed can still make
    # the deadline.
    fuel_fn = _fuel_rate_fn()
    ctx = CommercialInstructionContext(demanded_speed_kn=16.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                                       fuel_rate_tonnes_per_h=fuel_fn, deadline_slack_h=-1.0,
                                       complying_breaches_safety_margin=False)
    candidates = candidates_commercial_instruction(ctx)
    best, _ = oracle_best(candidates, lambda a: rollout_commercial_instruction(a, ctx))
    assert best.tool == "comply_with_instruction"


def test_commercial_instruction_oracle_best_refuses_when_complying_would_breach_a_safety_margin():
    fuel_fn = _fuel_rate_fn()
    ctx = CommercialInstructionContext(demanded_speed_kn=16.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                                       fuel_rate_tonnes_per_h=fuel_fn, deadline_slack_h=None,
                                       complying_breaches_safety_margin=True)
    candidates = candidates_commercial_instruction(ctx)
    best, _ = oracle_best(candidates, lambda a: rollout_commercial_instruction(a, ctx))
    assert best.tool == "refuse_citing_ism_art5"


def test_rollout_commercial_instruction_comply_matches_hand_computed_arithmetic():
    fuel_fn = _fuel_rate_fn()
    ctx = CommercialInstructionContext(demanded_speed_kn=16.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                                       fuel_rate_tonnes_per_h=fuel_fn, deadline_slack_h=None,
                                       complying_breaches_safety_margin=False)
    result = rollout_commercial_instruction(CaptainAction(tool="comply_with_instruction", params={"speed_kn": 16.0}), ctx)
    expected_time_h = 100.0 / 16.0
    assert result.notes["time_h"] == pytest.approx(expected_time_h)
    assert result.dims.duty_breaches == 0.0


def test_rollout_commercial_instruction_raises_for_an_unknown_action():
    ctx = CommercialInstructionContext(demanded_speed_kn=16.0, original_soa_kn=12.0, remaining_distance_nm=100.0,
                                       fuel_rate_tonnes_per_h=_fuel_rate_fn(), deadline_slack_h=None,
                                       complying_breaches_safety_margin=False)
    with pytest.raises(ValueError):
        rollout_commercial_instruction(CaptainAction(tool="hold", params={}), ctx)
