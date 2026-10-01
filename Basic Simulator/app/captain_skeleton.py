"""Captain walking-skeleton Phase 4: wires MissionSim (Phase 1) + the route planner
(Phase 2) + the procedure library/shield/decision layer (Phase 3) into one runnable,
non-Streamlit skeleton (design_captain_missions.md Sec 14/14.1/15.3) -- one mission, one
scenario file, ONE fully-wired event type (engine_failure), a deterministic baseline
Captain (a direct `oracle_best` lookup, Sec 14's own "proves cost()/regret runs
end-to-end, not just that a mission can run end-to-end").

Pure Python, no GPU/API key, safe to run locally. Lives in app/ (not pipeline/) because it
is the integration point depending on BOTH layers (MissionSim/route planner from app/, the
procedure library/shield/cost model from pipeline/) -- consistent with this project's own
pipeline-never-depends-on-app/ rule (see `pipeline/captain_agent_spec.py`'s own
`EngineFailureContext` docstring) and with how `pipeline/oow_agent_spec.py`'s own live
wiring lives in app/agents.py, not inside pipeline/ itself.

Deliberately NOT built here (explicit scope reduction, consistent with Sec 13.A.1's own "5
concretising points" already deferred in Phase 1): NO encounter-sim splice / ambient COLREG
traffic -- this scenario scripts none, so there is nothing to splice into yet; add it once
a scenario actually needs it. Also NOT built: the other 4 v1 event types' live triggering
(only engine_failure fires in this scenario) and full evaluation (Sec 10, Phase 5's job) --
this module only runs the mission and renders text, it does not score it.
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
from app.mission_sim import MissionSim, position_along_route_nm
from pipeline.captain_agent_spec import (
    PROCEDURE_LIBRARY, EngineFailureContext, candidates_engine_failure, check_safety_margins,
    oracle_best, resolve_captain_decision, rollout_engine_failure,
)
from pipeline.captain_eval import (
    CaptainMissionEvaluation, MissionOutcomeFacts, ResourceEfficiencyFacts, evaluate_captain_mission,
)
from pipeline.captain_types import (
    BrownEnvelopeEvent, CaptainAction, EngineStatus, ExclusionZone, MissionOrder, MissionState,
    Port, load_region_json, ports_from_region, zones_from_region,
)

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

    @classmethod
    def from_scenario_file(cls, path: Path) -> "CaptainSkeleton":
        scenario = load_scenario(path)
        order = build_mission_order_from_scenario(scenario)
        state = MissionState.from_mission_order(order)
        sim = MissionSim.from_mission_order(order)
        zones, ports = build_region_data(scenario)
        draft_m = float(order.admin_logistics.get("resources", {}).get("draft_m", 0.0))
        return cls(order=order, state=state, sim=sim, zones=zones, ports=ports,
                   draft_m=draft_m, pending_events=list(order.events))

    # --- internal: delta log + trigger/dispatch -----------------------------------------

    def _log_delta(self, field_path: str, old: Any, new: Any, cause: str) -> None:
        """Appends one append-only delta entry (Sec 13.B.5) -- never overwrites a prior one."""
        self.state.event_log.append({"t": self.sim.state.elapsed_s, "field_path": field_path,
                                     "old": old, "new": new, "cause": cause})

    def _pending_untriggered(self) -> list[BrownEnvelopeEvent]:
        return [e for e in self.pending_events if e.event_id not in self.fired_event_ids]

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
        self.state.active_hazards.append({"event_id": event.event_id, "type": event.type,
                                          "severity": event.severity})
        self._log_delta("active_hazards", old=None, new=event.event_id, cause=event.event_id)
        if event.type == "engine_failure":
            self._handle_engine_failure(event)
        else:
            # Metadata-only events (Sec 14's stated scope): record the procedure-library
            # lookup so it's visible in the log, but apply no decision layer yet.
            entry = PROCEDURE_LIBRARY[event.type]
            self._log_delta("event_log", old=None,
                            new=f"{event.type} fired (shield: {entry.shield_description})",
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

        elapsed_h = self.sim.state.elapsed_s / 3600.0
        reference_finish_h = elapsed_h + remaining_nm / self.order.speed_of_advance_kn
        eta_deadline_h = self.order.admin_logistics.get("eta_deadline_h")
        deadline_slack_h = (eta_deadline_h - reference_finish_h) if eta_deadline_h is not None else None

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

        self.sim.state.current_speed_kn = float(applied.params.get("speed_kn", capped_speed_kn))
        self._log_delta("captain_decision", old=None,
                        new={"tool": applied.tool, "params": applied.params, "shield_substituted": substituted},
                        cause=event.event_id)

    # --- debug control set (Sec 15.3) ---------------------------------------------------

    def step_mission(self, n: int = 1) -> None:
        """Advances the mission-sim by `n` steps (each `dt_mission_s` seconds), checking
        for a due trigger after every step."""
        for _ in range(n):
            if self.sim.reached_destination():
                return
            self.sim.step()
            self._check_triggers()

    def run_to_next_event(self) -> BrownEnvelopeEvent | None:
        """Advances until the next scripted brown envelope fires or the mission ends --
        returns the fired event, or None once the destination is reached with nothing
        left pending."""
        while not self.sim.reached_destination():
            self.sim.step()
            event = self._check_triggers()
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
