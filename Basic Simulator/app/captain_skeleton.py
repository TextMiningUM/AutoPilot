"""Captain walking-skeleton Phase 4/6/7: wires MissionSim (Phase 1) + the route planner
(Phase 2) + the procedure library/shield/decision layer (Phase 3/6/7) into one runnable,
non-Streamlit skeleton (design_captain_missions.md Sec 14/14.1/15.3) -- one mission per
scenario file, all 5 v1 event types wired, a deterministic baseline Captain (a direct
`oracle_best` lookup, Sec 14's own "proves cost()/regret runs end-to-end, not just that a
mission can run end-to-end").

Pure Python, no GPU/API key, safe to run locally. Lives in app/ (not pipeline/) because it
is the integration point depending on BOTH layers (MissionSim/route planner from app/, the
procedure library/shield/cost model from pipeline/) -- consistent with this project's own
pipeline-never-depends-on-app/ rule (see `pipeline/captain_agent_spec.py`'s own
`EngineFailureContext` docstring) and with how `pipeline/oow_agent_spec.py`'s own live
wiring lives in app/agents.py, not inside pipeline/ itself.

Phase 7 also wires Sec 13.A.9's `WORLD_RESPONDER_TIMER` (the scheduler entry existed since
Phase 1 but nothing populated/processed it until now) -- a fixed, scripted reply (never an
open-ended negotiation) for distress_call/commercial_instruction, scheduled in the
mission-sim's own `EventScheduler` and polled alongside the brown-envelope trigger scan.

Deliberately NOT built here (explicit scope reduction, consistent with Sec 13.A.1's own "5
concretising points" already deferred in Phase 1): NO encounter-sim splice / ambient COLREG
traffic -- a scenario scripts none yet, so there is nothing to splice into; add it once a
scenario actually needs it. Also NOT built: the general Sec 13.B.7 trigger-monitor
mechanism (Phase 8) and full evaluation beyond Phase 5's 3 axes.
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # Auto Pilot/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.mission_route_planner import distance_to_refuge_nm, minimum_resource_route
from app.mission_sim import EventType, MissionSim, ScheduledEvent, position_along_route_nm
from pipeline.captain_agent_spec import (
    PROCEDURE_LIBRARY, CommercialInstructionContext, DistressCallContext, EngineFailureContext,
    FogContext, WhaleZoneContext, candidates_commercial_instruction, candidates_distress_call,
    candidates_engine_failure, candidates_fog, candidates_whale_zone, check_safety_margins,
    oracle_best, resolve_captain_decision, rollout_commercial_instruction, rollout_distress_call,
    rollout_engine_failure, rollout_fog, rollout_whale_zone,
)
from pipeline.captain_eval import (
    CaptainMissionEvaluation, MissionOutcomeFacts, ResourceEfficiencyFacts, evaluate_captain_mission,
)
from pipeline.captain_types import (
    BrownEnvelopeEvent, CaptainAction, EngineStatus, ExclusionZone, MissionOrder, MissionState,
    Port, load_region_json, ports_from_region, zones_from_region,
)
from pipeline.nomoto import SHIP_PROFILES

REGIONS_DIR = REPO_ROOT / "Data" / "Captain" / "Regions"


def load_scenario(path: Path) -> dict[str, Any]:
    """Loads a Sec 13.C.13 ground-truth scenario file as a plain dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_mission_order_from_scenario(scenario: dict[str, Any]) -> MissionOrder:
    """Builds a MissionOrder from a scenario's `mission_order` sub-dict, injecting
    `waypoints` (from `route.waypoints`) and `events` (from `brown_envelopes`) -- kept out
    of `mission_order` itself in the JSON so the route/events are stated exactly once, not
    duplicated across two sibling keys."""
    order_fields = dict(scenario["mission_order"])
    order_fields["mission_id"] = scenario["mission_id"]
    order_fields["waypoints"] = [tuple(p) for p in scenario["route"]["waypoints"]]
    order_fields["events"] = [BrownEnvelopeEvent(**e) for e in scenario["brown_envelopes"]]
    return MissionOrder(**order_fields)


def build_region_data(scenario: dict[str, Any]) -> tuple[list[ExclusionZone], list[Port]]:
    """Sec 13.B.8's obstacle set: the union of the mission's own dynamic exclusion zones
    (`route.exclusion_zones`) and the static geographic layer for `route.region_id`;
    ports are always drawn from the shared region library, never invented per mission."""
    region = load_region_json(REGIONS_DIR / f"{scenario['route']['region_id']}.json")
    dynamic_zones = zones_from_region({"exclusion_zones": scenario["route"].get("exclusion_zones", [])})
    static_zones = zones_from_region(region)
    return dynamic_zones + static_zones, ports_from_region(region)


@dataclass
class CaptainSkeleton:
    """The walking skeleton's own small debug-driven runner (Sec 15.3) -- a direct
    function-call control set, mirroring how OOW's own `app/baselines/*.py` were first
    exercised before any Streamlit UI existed for them."""
    order: MissionOrder
    state: MissionState
    sim: MissionSim
    zones: list[ExclusionZone]
    ports: list[Port]
    draft_m: float
    pending_events: list[BrownEnvelopeEvent] = field(default_factory=list)
    fired_event_ids: set[str] = field(default_factory=set)
    _active_speed_caps: dict[str, float] = field(default_factory=dict)  # regime key -> capped speed_kn
    _requested_soa_kn: float = 0.0  # the Captain's own current baseline speed instruction

    @classmethod
    def from_scenario_file(cls, path: Path) -> "CaptainSkeleton":
        scenario = load_scenario(path)
        order = build_mission_order_from_scenario(scenario)
        state = MissionState.from_mission_order(order)
        sim = MissionSim.from_mission_order(order)
        zones, ports = build_region_data(scenario)
        draft_m = float(order.admin_logistics.get("resources", {}).get("draft_m", 0.0))
        return cls(order=order, state=state, sim=sim, zones=zones, ports=ports,
                   draft_m=draft_m, pending_events=list(order.events),
                   _requested_soa_kn=order.speed_of_advance_kn)

    # --- internal: delta log + trigger/dispatch -----------------------------------------

    def _log_delta(self, field_path: str, old: Any, new: Any, cause: str) -> None:
        """Appends one append-only delta entry (Sec 13.B.5) -- never overwrites a prior one."""
        self.state.event_log.append({"t": self.sim.state.elapsed_s, "field_path": field_path,
                                     "old": old, "new": new, "cause": cause})

    def _pending_untriggered(self) -> list[BrownEnvelopeEvent]:
        return [e for e in self.pending_events if e.event_id not in self.fired_event_ids]

    def _apply_speed_cap(self, key: str, speed_kn: float) -> None:
        """Sec 13.A.4: multiple regimes (engine failure, fog, whale zone) can be active at
        once -- the effective speed is always the MOST restrictive cap, never whichever
        fired last."""
        self._active_speed_caps[key] = speed_kn
        self._recompute_effective_speed()

    def _clear_speed_cap(self, key: str) -> None:
        self._active_speed_caps.pop(key, None)
        self._recompute_effective_speed()

    def _set_requested_speed(self, speed_kn: float) -> None:
        """Changes the Captain's own baseline speed instruction (e.g. complying with a
        commercial demand to go FASTER than the Mission Order's planned SOA, Sec 13.A.4) --
        distinct from a safety-mandated cap, which always still applies as a ceiling on top
        of whatever is requested here (the effective speed is always the min of the two)."""
        self._requested_soa_kn = speed_kn
        self._recompute_effective_speed()

    def _recompute_effective_speed(self) -> None:
        caps = [self._requested_soa_kn, *self._active_speed_caps.values()]
        self.sim.state.current_speed_kn = min(caps)

    def _schedule_regime_clear(self, event: BrownEnvelopeEvent, affected_distance_nm: float) -> None:
        """A minimal 'duration' mechanic for a temporary regime (fog/whale zone, Phase 6) --
        enqueues a synthetic companion event that reverts this event's own speed cap once
        its own scripted window has elapsed. NOT yet the general live position/time monitor
        Sec 13.B.7 will eventually provide (Phase 8) -- the zone/window extent here is a
        distance-along-route approximation, not live lat/lon-vs-polygon containment."""
        clear_at_nm = self.sim.state.distance_travelled_nm + affected_distance_nm
        self.pending_events.append(BrownEnvelopeEvent(
            event_id=f"{event.event_id}_clear", type=f"{event.type}_clear", severity=event.severity,
            trigger={"type": "distance_along_route_nm", "value": clear_at_nm},
            params={"source_event_id": event.event_id},
        ))

    def _deadline_slack_h(self, remaining_nm: float) -> float | None:
        """Hours of slack left vs. the Mission Order's own `eta_deadline_h`, assuming the
        mission continues at the original SOA from here -- shared by every decision layer
        that needs `deadline_slack_h` (Sec 13.A.4)."""
        eta_deadline_h = self.order.admin_logistics.get("eta_deadline_h")
        if eta_deadline_h is None:
            return None
        elapsed_h = self.sim.state.elapsed_s / 3600.0
        reference_finish_h = elapsed_h + remaining_nm / self.order.speed_of_advance_kn
        return eta_deadline_h - reference_finish_h

    def _schedule_world_responder(self, event: BrownEnvelopeEvent) -> None:
        """Sec 13.A.2's fixed, scripted world-responder reply (never an open-ended
        negotiation) -- schedules a WORLD_RESPONDER_TIMER entry in the mission-sim's own
        scheduler (Sec 13.A.9) at `elapsed_s + world_responder.delay_s`."""
        responder = event.world_responder
        if responder is None:
            return
        self.sim.scheduler.schedule(
            self.sim.state.elapsed_s + float(responder["delay_s"]), EventType.WORLD_RESPONDER_TIMER,
            kind=f"world_responder:{event.event_id}",
            data={"source_event_id": event.event_id, "resolution": responder["resolution"]},
        )

    def _check_scheduler(self) -> None:
        """Pops and processes every scheduler entry (Sec 13.A.9) whose timestamp has
        already been reached -- separate from `_check_triggers()`'s own brown-envelope
        scan, since these live in the EventScheduler's own priority queue (Phase 1), not
        `pending_events`. Only WORLD_RESPONDER_TIMER has a real handler (Phase 7's own
        scope); other types (e.g. ROUTINE_REPORT) are popped but produce no effect yet --
        MPR auto-emission is a separate, future piece of work."""
        while True:
            next_ts = self.sim.scheduler.peek_next_timestamp()
            if next_ts is None or next_ts > self.sim.state.elapsed_s:
                return
            scheduled = self.sim.scheduler.pop_next()
            if scheduled.event_type == EventType.WORLD_RESPONDER_TIMER:
                self._handle_world_responder(scheduled)

    def _handle_world_responder(self, scheduled: ScheduledEvent) -> None:
        """Applies a scripted, fixed world-responder resolution (Sec 13.A.2) -- closes out
        the originating hazard and logs the fixed outcome text. Does NOT retroactively
        change the Captain's already-recorded decision/cost: that was computed from the
        facts known AT DECISION TIME, same principle as a real captain deciding before
        learning how it actually turns out."""
        source_event_id = scheduled.data["source_event_id"]
        resolution = scheduled.data["resolution"]
        for hazard in self.state.active_hazards:
            if hazard["event_id"] == source_event_id:
                hazard["cleared_at_s"] = self.sim.state.elapsed_s
                hazard["resolution"] = resolution
        self._log_delta("active_hazards", old=source_event_id, new=resolution, cause=scheduled.kind)

    def _check_triggers(self) -> BrownEnvelopeEvent | None:
        """Checks every still-pending event's (fixed-time) trigger condition against the
        current mission-sim clock/position; fires and returns the first one that is due,
        or None. Only `distance_along_route_nm`/`elapsed_time_s` triggers are implemented
        (Sec 13.C.13's own worked examples) -- condition-based triggers (position doubt,
        etc.) belong to the trigger-monitor mechanism (Sec 13.B.7), out of this skeleton's
        scope (Sec 14)."""
        for event in self._pending_untriggered():
            trigger = event.trigger
            due = ((trigger.get("type") == "distance_along_route_nm"
                    and self.sim.state.distance_travelled_nm >= trigger["value"])
                   or (trigger.get("type") == "elapsed_time_s"
                       and self.sim.state.elapsed_s >= trigger["value"]))
            if due:
                self._fire_event(event)
                return event
        return None

    def _fire_event(self, event: BrownEnvelopeEvent) -> None:
        self.fired_event_ids.add(event.event_id)
        if event.type.endswith("_clear"):
            self._handle_regime_clear(event)
            return
        self.state.active_hazards.append({"event_id": event.event_id, "type": event.type,
                                          "severity": event.severity})
        self._log_delta("active_hazards", old=None, new=event.event_id, cause=event.event_id)
        if event.type == "engine_failure":
            self._handle_engine_failure(event)
        elif event.type == "fog":
            self._handle_fog(event)
        elif event.type == "whale_zone":
            self._handle_whale_zone(event)
        elif event.type == "distress_call":
            self._handle_distress_call(event)
        elif event.type == "commercial_instruction":
            self._handle_commercial_instruction(event)
        else:
            # Metadata-only events (Sec 14's stated scope): record the procedure-library
            # lookup so it's visible in the log, but apply no decision layer yet.
            entry = PROCEDURE_LIBRARY[event.type]
            self._log_delta("event_log", old=None,
                            new=f"{event.type} fired (shield: {entry.shield_description})",
                            cause=event.event_id)

    def _handle_regime_clear(self, event: BrownEnvelopeEvent) -> None:
        """Reverts a temporary speed-cap regime (fog/whale zone) once its own scripted
        window has elapsed (Phase 6's minimal 'duration' mechanic, see
        `_schedule_regime_clear()`)."""
        source_event_id = event.params["source_event_id"]
        self._clear_speed_cap(source_event_id)
        for hazard in self.state.active_hazards:
            if hazard["event_id"] == source_event_id:
                hazard["cleared_at_s"] = self.sim.state.elapsed_s
        self._log_delta("active_hazards", old=source_event_id, new=f"{source_event_id}_cleared",
                        cause=event.event_id)

    def _handle_engine_failure(self, event: BrownEnvelopeEvent) -> None:
        """The ONE fully-wired decision layer (Sec 13.A.4/13.A.8): builds the
        EngineFailureContext from live Mission-sim/route-planner facts, applies the
        deterministic baseline Captain (`oracle_best`, Sec 14), runs the shield, and
        applies the result to the sim."""
        capped_speed_kn = float(event.params["capped_speed_kn"])
        repair_duration_h = float(event.params.get("repair_duration_h", 24.0))
        fault = event.params.get("fault")
        engine_status = EngineStatus(max_speed_kn=capped_speed_kn, fault=fault,
                                     reported_at=self.sim.state.elapsed_s)
        self.state.resources["engine_status"] = {"max_speed_kn": engine_status.max_speed_kn,
                                                  "fault": engine_status.fault}

        remaining_nm = self.sim.remaining_distance_nm()
        current_pos = position_along_route_nm(self.order.waypoints, self.sim.state.distance_travelled_nm)
        refuge = distance_to_refuge_nm(current_pos, self.ports, self.zones,
                                       required_services=["repair"], draft_m=self.draft_m)
        deadline_slack_h = self._deadline_slack_h(remaining_nm)

        ctx = EngineFailureContext(
            capped_speed_kn=capped_speed_kn, original_soa_kn=self.order.speed_of_advance_kn,
            remaining_distance_nm=remaining_nm, fuel_tonnes_available=self.sim.state.fuel_tonnes,
            fuel_rate_tonnes_per_h=self.sim.fuel_model.fuel_rate_tonnes_per_h,
            deadline_slack_h=deadline_slack_h,
            refuge_distance_nm=refuge.route.distance_nm if refuge is not None else None,
            refuge_port_id=refuge.port.id if refuge is not None else None,
            repair_duration_h=repair_duration_h,
        )
        candidates = candidates_engine_failure(ctx)
        proposed, _ = oracle_best(candidates, lambda a: rollout_engine_failure(a, ctx))

        violations = check_safety_margins(proposed, engine_max_speed_kn=capped_speed_kn)
        fallback = CaptainAction(tool="continue_at_capped_speed", params={"speed_kn": capped_speed_kn})
        applied, substituted = resolve_captain_decision(proposed, violations, fallback)

        self._apply_speed_cap("engine_failure", float(applied.params.get("speed_kn", capped_speed_kn)))
        self._log_delta("captain_decision", old=None,
                        new={"tool": applied.tool, "params": applied.params, "shield_substituted": substituted},
                        cause=event.event_id)

    def _handle_fog(self, event: BrownEnvelopeEvent) -> None:
        """Sec 13.A.4's degenerate fog decision layer: Rule 19 mandates the safe speed
        directly -- one candidate, still run through cost()/oracle_best() so zero regret
        is demonstrated, not hardcoded."""
        safe_speed_kn = float(event.params["safe_speed_kn"])
        affected_distance_nm = float(event.params["affected_distance_nm"])
        ctx = FogContext(safe_speed_kn=safe_speed_kn, original_soa_kn=self.order.speed_of_advance_kn,
                        affected_distance_nm=affected_distance_nm,
                        fuel_rate_tonnes_per_h=self.sim.fuel_model.fuel_rate_tonnes_per_h)
        candidates = candidates_fog(ctx)
        proposed, _ = oracle_best(candidates, lambda a: rollout_fog(a, ctx))

        violations = check_safety_margins(proposed, engine_max_speed_kn=safe_speed_kn)
        fallback = CaptainAction(tool="reduce_to_safe_speed", params={"speed_kn": safe_speed_kn})
        applied, substituted = resolve_captain_decision(proposed, violations, fallback)

        self._apply_speed_cap(event.event_id, float(applied.params["speed_kn"]))
        self._log_delta("captain_decision", old=None,
                        new={"tool": applied.tool, "params": applied.params, "shield_substituted": substituted},
                        cause=event.event_id)
        self._schedule_regime_clear(event, affected_distance_nm)

    def _handle_whale_zone(self, event: BrownEnvelopeEvent) -> None:
        """Sec 13.A.4's degenerate whale-zone decision layer: a charted/posted zone speed
        limit mandates the speed directly -- one candidate. The zone's own extent is
        approximated as a distance-along-route window (Sec 13.A.2's 'position inside the
        polygon' condition becomes a live monitor in Phase 8, Sec 13.B.7 -- this skeleton
        doesn't yet re-check live lat/lon against the zone polygon)."""
        speed_limit_kn = float(event.params["speed_limit_kn"])
        affected_distance_nm = float(event.params["affected_distance_nm"])
        ctx = WhaleZoneContext(speed_limit_kn=speed_limit_kn, original_soa_kn=self.order.speed_of_advance_kn,
                              zone_transit_distance_nm=affected_distance_nm,
                              fuel_rate_tonnes_per_h=self.sim.fuel_model.fuel_rate_tonnes_per_h)
        candidates = candidates_whale_zone(ctx)
        proposed, _ = oracle_best(candidates, lambda a: rollout_whale_zone(a, ctx))

        violations = check_safety_margins(proposed, engine_max_speed_kn=speed_limit_kn)
        fallback = CaptainAction(tool="reduce_to_zone_speed_limit", params={"speed_kn": speed_limit_kn})
        applied, substituted = resolve_captain_decision(proposed, violations, fallback)

        self._apply_speed_cap(event.event_id, float(applied.params["speed_kn"]))
        self._log_delta("captain_decision", old=None,
                        new={"tool": applied.tool, "params": applied.params, "shield_substituted": substituted},
                        cause=event.event_id)
        self._schedule_regime_clear(event, affected_distance_nm)

    def _handle_distress_call(self, event: BrownEnvelopeEvent) -> None:
        """Sec 13.A.4's distress-call decision layer -- a genuine judgement case: assist
        vs. decline with a logged reason, the latter only valid when assisting would
        itself breach a safety margin (Sec 13.A.6, computed here via
        `check_safety_margins()`'s own fuel-reserve check)."""
        detour_distance_nm = float(event.params["detour_distance_nm"])
        remaining_nm = self.sim.remaining_distance_nm()
        fuel_rate = self.sim.fuel_model.fuel_rate_tonnes_per_h

        assist_time_h = (remaining_nm + detour_distance_nm) / self.order.speed_of_advance_kn
        assist_fuel_t = fuel_rate(self.order.speed_of_advance_kn) * assist_time_h
        resources = self.order.admin_logistics.get("resources", {})
        # Sec 13.A.6's computed condition -- the fuel-reserve check is action-independent
        # (it only looks at the kwargs below, Sec 13.A.6's own table), so this is really
        # asking "would ASSISTING specifically breach the reserve", needed by the cost
        # model below regardless of which candidate the oracle actually picks.
        assist_violations = check_safety_margins(
            CaptainAction(tool="proceed_to_assist", params={}),
            candidate_fuel_consumption_t=assist_fuel_t, fuel_tonnes_available=self.sim.state.fuel_tonnes,
            fuel_reserve_margin_pct=float(resources.get("fuel_reserve_margin_pct", 0.0)),
        )

        ctx = DistressCallContext(
            detour_distance_nm=detour_distance_nm, original_soa_kn=self.order.speed_of_advance_kn,
            remaining_distance_nm=remaining_nm, fuel_rate_tonnes_per_h=fuel_rate,
            deadline_slack_h=self._deadline_slack_h(remaining_nm),
            assisting_breaches_safety_margin=bool(assist_violations),
        )
        candidates = candidates_distress_call(ctx)
        proposed, _ = oracle_best(candidates, lambda a: rollout_distress_call(a, ctx))

        # The shield only ever enforces against assisting's OWN violation, and only when
        # that's actually what was proposed -- a decision layer that already correctly
        # declined (because assisting would breach the margin) must never be falsely
        # flagged as "substituted" just because the hypothetical check found a violation.
        violations = assist_violations if proposed.tool == "proceed_to_assist" else []
        fallback = CaptainAction(tool="decline_with_logged_reason", params={})
        applied, substituted = resolve_captain_decision(proposed, violations, fallback)

        self._log_delta("captain_decision", old=None,
                        new={"tool": applied.tool, "params": applied.params, "shield_substituted": substituted},
                        cause=event.event_id)
        self._schedule_world_responder(event)

    def _handle_commercial_instruction(self, event: BrownEnvelopeEvent) -> None:
        """Sec 13.A.4's commercial-instruction decision layer: whether complying breaches
        a safety margin is a COMPUTED condition (Sec 13.A.6) against the ship's own engine
        speed limit, never a fixed always-refuse rule."""
        demanded_speed_kn = float(event.params["demanded_speed_kn"])
        remaining_nm = self.sim.remaining_distance_nm()
        resources = self.order.admin_logistics.get("resources", {})
        ship_profile = SHIP_PROFILES[resources.get("ship_profile", "sawada2021")]

        # Sec 13.A.6's computed condition, needed by the cost model below -- `speed_kn` IS
        # part of the action's own params here, so this naturally returns [] for whatever
        # candidate doesn't carry a speed_kn at all (refuse_citing_ism_art5).
        hypothetical_comply = CaptainAction(tool="comply_with_instruction", params={"speed_kn": demanded_speed_kn})
        comply_violations = check_safety_margins(hypothetical_comply, engine_max_speed_kn=ship_profile.max_speed_kn)

        ctx = CommercialInstructionContext(
            demanded_speed_kn=demanded_speed_kn, original_soa_kn=self.order.speed_of_advance_kn,
            remaining_distance_nm=remaining_nm, fuel_rate_tonnes_per_h=self.sim.fuel_model.fuel_rate_tonnes_per_h,
            deadline_slack_h=self._deadline_slack_h(remaining_nm),
            complying_breaches_safety_margin=bool(comply_violations),
        )
        candidates = candidates_commercial_instruction(ctx)
        proposed, _ = oracle_best(candidates, lambda a: rollout_commercial_instruction(a, ctx))

        # Re-checks WHATEVER was actually proposed (naturally [] when proposed has no
        # speed_kn, i.e. refuse) -- never the fixed hypothetical above, so a decision layer
        # that already correctly refused is never falsely flagged as "substituted".
        violations = check_safety_margins(proposed, engine_max_speed_kn=ship_profile.max_speed_kn)
        fallback = CaptainAction(tool="refuse_citing_ism_art5", params={})
        applied, substituted = resolve_captain_decision(proposed, violations, fallback)

        if applied.tool == "comply_with_instruction":
            self._set_requested_speed(demanded_speed_kn)
        self._log_delta("captain_decision", old=None,
                        new={"tool": applied.tool, "params": applied.params, "shield_substituted": substituted},
                        cause=event.event_id)
        self._schedule_world_responder(event)

    # --- debug control set (Sec 15.3) ---------------------------------------------------

    def step_mission(self, n: int = 1) -> None:
        """Advances the mission-sim by `n` steps (each `dt_mission_s` seconds), checking
        for a due trigger/scheduled world-responder resolution after every step."""
        for _ in range(n):
            if self.sim.reached_destination():
                return
            self.sim.step()
            self._check_triggers()
            self._check_scheduler()

    def run_to_next_event(self) -> BrownEnvelopeEvent | None:
        """Advances until the next scripted brown envelope fires or the mission ends --
        returns the fired event, or None once the destination is reached with nothing
        left pending. Scheduled world-responder resolutions (Sec 13.A.9) are processed
        alongside but don't interrupt the advance (Sec 15.3's own control is only ever
        for brown envelopes)."""
        while not self.sim.reached_destination():
            self.sim.step()
            event = self._check_triggers()
            self._check_scheduler()
            if event is not None:
                return event
        return None

    def force_event(self, event_id: str) -> BrownEnvelopeEvent:
        """Manually triggers a specific scripted event out of turn (for testing, Sec 15.3)."""
        event = next((e for e in self.pending_events if e.event_id == event_id), None)
        if event is None:
            raise ValueError(f"no pending event with id {event_id!r}")
        if event_id in self.fired_event_ids:
            raise ValueError(f"event {event_id!r} has already fired")
        self._fire_event(event)
        return event

    def show_mission_state(self) -> str:
        """Dumps the current Mission State (Sec 13.B.5) as text."""
        lines = [
            f"Mission {self.state.mission_id}",
            f"  elapsed: {self.sim.state.elapsed_s / 3600.0:.2f} h, "
            f"distance travelled: {self.sim.state.distance_travelled_nm:.1f} / "
            f"{self.sim.total_route_distance_nm:.1f} nm",
            f"  fuel remaining: {self.sim.state.fuel_tonnes:.1f} t, "
            f"current speed: {self.sim.state.current_speed_kn:.1f} kn",
            f"  active hazards: {[h['type'] for h in self.state.active_hazards]}",
            f"  goals: {[(g['goal'], g['status']) for g in self.state.goals]}",
            f"  event log entries: {len(self.state.event_log)}",
        ]
        return "\n".join(lines)

    def show_procedure_lookup(self, event: BrownEnvelopeEvent) -> str:
        """Shows what the procedure library (Sec 13.A.4) returns for `event`, without
        applying it."""
        entry = PROCEDURE_LIBRARY[event.type]
        duties = "; ".join(f"{d.description} (deadline {d.deadline_s}s)" for d in entry.mandatory_duties) or "none"
        return f"Procedure lookup for {event.type}:\n  mandatory duties: {duties}\n  shield: {entry.shield_description}"

    def show_mpr(self) -> str:
        """Renders the Mission Progress Report (Sec 8.3) at the current point -- facts
        only, no decision logic (Sec 13.B.5)."""
        remaining_nm = self.sim.remaining_distance_nm()
        eta_s = self.sim.eta_s()
        eta_line = f"Speed: {self.sim.state.current_speed_kn:.1f} kn, ETA: {eta_s / 3600.0:.1f} h"
        if eta_s == float("inf"):
            eta_line = "Speed: 0.0 kn (stopped), ETA: n/a"
        lines = [
            f"MISSION PROGRESS REPORT -- {self.state.mission_id}",
            f"As of: {self.sim.current_utc().isoformat()}",
            f"Position: {remaining_nm:.1f} nm remaining of {self.sim.total_route_distance_nm:.1f} nm total",
            eta_line,
            f"Fuel remaining: {self.sim.state.fuel_tonnes:.1f} t",
            f"Active hazards: {', '.join(h['type'] for h in self.state.active_hazards) or 'none'}",
        ]
        return "\n".join(lines)

    # --- minimal evaluation (Sec 10, Phase 5: Safety/Mission-outcome/Resource-efficiency) -

    def evaluate(self) -> CaptainMissionEvaluation:
        """Sec 10's minimal composite evaluation (Phase 5 scope: only the Safety,
        Mission-outcome, and Resource-efficiency axes). Computes the Sec 13.B.8 oracle
        baseline (the unconstrained minimum-resource route, ignoring the engine failure
        entirely) and the actual recorded fuel/time usage here (needs the route planner),
        then delegates all SCORING to the pure `pipeline.captain_eval` module."""
        events_by_id = {e.event_id: e for e in self.order.events}
        oracle = minimum_resource_route(self.order.waypoints, self.zones)
        reference_time_h = oracle.distance_nm / self.order.speed_of_advance_kn
        reference_fuel_t = (self.sim.fuel_model.fuel_rate_tonnes_per_h(self.order.speed_of_advance_kn)
                            * reference_time_h)

        fuel_at_departure = float(self.order.admin_logistics.get("resources", {}).get("fuel_tonnes", 0.0))
        actual_fuel_t = fuel_at_departure - self.sim.state.fuel_tonnes
        actual_time_h = self.sim.state.elapsed_s / 3600.0

        outcome_facts = MissionOutcomeFacts(
            reached_destination=self.sim.reached_destination(),
            fuel_tonnes_remaining=self.sim.state.fuel_tonnes,
            elapsed_h=actual_time_h,
            eta_deadline_h=self.order.admin_logistics.get("eta_deadline_h"),
        )
        resource_facts = ResourceEfficiencyFacts(
            reference_fuel_t=reference_fuel_t, reference_time_h=reference_time_h,
            actual_fuel_t=actual_fuel_t, actual_time_h=actual_time_h,
        )
        return evaluate_captain_mission(self.state.event_log, events_by_id, outcome_facts, resource_facts)
