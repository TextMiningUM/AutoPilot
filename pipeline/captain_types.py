"""Captain-domain data structures (design_captain_missions.md Sec 7/8.2/13.B.5/13.C.13).

Walking-skeleton Phase 0 (2026-10-01): pure dataclasses + a tiny JSON loader, no decision
logic, no simulator wiring yet -- safe to run locally, no GPU/API key required. Deeply
nested, still-evolving subsections (Mission Order's situation/admin_logistics/command_signal
blocks) are deliberately kept as plain dict[str, Any] rather than fully specified
dataclasses, per Sec 15.1's own guidance ("small and mechanical, usually finalised while
writing the walking skeleton, not before") -- only the fields later phases actually compute
against (waypoints, success criteria, resources, events) are typed here.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExclusionZone:
    """A no-go polygon (Sec 7) -- dynamic (weather/security/wildlife) or static
    (coastline/shallow_water/fixed_installation), same schema for both, distinguished only
    by `type`."""
    id: str
    type: str  # "dynamic_hazard" | "coastline" | "shallow_water" | "fixed_installation"
    polygon: list[tuple[float, float]]
    speed_limit_kn: float | None = None
    min_depth_m: float | None = None  # only meaningful for type == "shallow_water"


@dataclass(frozen=True)
class Port:
    """A candidate port of refuge / scheduled port call (Sec 7) -- missions reference ports
    by `id` into a shared, named region's library, never invent one per mission."""
    id: str
    name: str
    position: tuple[float, float]  # (lat, lon)
    services: list[str]  # drawn from a fixed vocabulary, e.g. "repair"/"fuel_bunkering"/"medical"
    min_approach_depth_m: float
    region_id: str


@dataclass(frozen=True)
class BrownEnvelopeEvent:
    """One scripted mission event (Sec 8.2 `events[]` / Sec 13.C.13 `brown_envelopes[]`).
    `type` is one of the 5 v1-specified events (Sec 13.A.2: engine_failure, fog,
    distress_call, whale_zone, commercial_instruction) for anything the walking skeleton
    (Sec 14) actually computes against -- the remaining ~65 Sec 4 categories stay backlog
    and would need their own state-delta table entry before use here."""
    event_id: str
    type: str
    severity: str  # "minor" | "moderate" | "serious" | "catastrophic"
    context_flags: list[str] = field(default_factory=list)
    trigger: dict[str, Any] = field(default_factory=dict)
    world_responder: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)  # Sec 13.A.5


@dataclass(frozen=True)
class EngineStatus:
    """A subordinate fact container (Sec 2.5/15.2) -- the Chief Engineer's own report, fed
    into the Captain's prompt/shield as a plain fact, never a separate fine-tuned agent."""
    max_speed_kn: float
    fault: str | None = None
    reported_at: float | None = None  # mission-sim elapsed_s


@dataclass(frozen=True)
class CaptainAction:
    """One Captain decision (Sec 13.B.6's output schema's `tool`/`params` subset) -- the
    unit both the shield (Sec 13.A.6) and the decision layer (Sec 13.A.4/13.A.8) operate
    on, whether it's a `candidates()` member or a genuinely novel proposal."""
    tool: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MissionOrder:
    """The Captain's input contract (Sec 8.2), issued once at mission start and never
    mutated afterward -- see MissionState for the live, mutable counterpart (Sec 8.4)."""
    mission_id: str
    issued_by: str
    issued_at: str  # ISO-8601
    vessel: str
    t0_utc: str  # ISO-8601 departure wall-clock (Sec 13.A.1)
    dt_mission_s: float
    situation: dict[str, Any]
    goal: str
    success_criteria: list[str]
    waypoints: list[tuple[float, float]]
    speed_of_advance_kn: float
    restricted_zones: list[str]
    rules_of_conduct: str
    admin_logistics: dict[str, Any]
    command_signal: dict[str, Any]
    events: list[BrownEnvelopeEvent] = field(default_factory=list)


@dataclass
class MissionState:
    """The live, mutable counterpart of a MissionOrder (Sec 8.4/13.B.5) -- split into an
    immutable snapshot (copied once at mission start) and the fields every brown envelope/
    Captain decision actually mutates. `event_log` is append-only (never overwritten), so the
    state at any past time is reconstructable by replaying deltas up to that point."""
    mission_id: str
    success_criteria: list[str]
    restricted_zones: list[str]
    resources: dict[str, Any]
    goals: list[dict[str, Any]] = field(default_factory=list)
    active_hazards: list[dict[str, Any]] = field(default_factory=list)
    active_plan: dict[str, Any] = field(default_factory=dict)
    event_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mission_order(cls, order: MissionOrder) -> "MissionState":
        """Initialises a fresh MissionState from a MissionOrder's own frozen fields --
        goals start all-open, resources/hazards seeded from admin_logistics/situation."""
        return cls(
            mission_id=order.mission_id,
            success_criteria=list(order.success_criteria),
            restricted_zones=list(order.restricted_zones),
            resources=dict(order.admin_logistics.get("resources", {})),
            goals=[{"goal": c, "status": "open"} for c in order.success_criteria],
            active_hazards=[],
            active_plan={"waypoints": list(order.waypoints), "speed_of_advance_kn": order.speed_of_advance_kn},
            event_log=[],
        )


def load_region_json(path: Path) -> dict[str, Any]:
    """Loads a named geographic region file (Sec 7/13.C.13 `region_id` library) -- a plain
    dict with `region_id`/`exclusion_zones`/`ports` keys, not yet parsed into dataclasses
    here since callers want different subsets (route planner vs. port lookup)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def zones_from_region(region: dict[str, Any]) -> list[ExclusionZone]:
    """Parses a loaded region dict's `exclusion_zones` list into typed instances (Sec 7)."""
    return [
        ExclusionZone(id=z["id"], type=z["type"], polygon=[tuple(p) for p in z["polygon"]],
                      speed_limit_kn=z.get("speed_limit_kn"), min_depth_m=z.get("min_depth_m"))
        for z in region.get("exclusion_zones", [])
    ]


def ports_from_region(region: dict[str, Any]) -> list[Port]:
    """Parses a loaded region dict's `ports` list into typed instances (Sec 7)."""
    return [
        Port(id=p["id"], name=p["name"], position=tuple(p["position"]), services=list(p["services"]),
             min_approach_depth_m=p["min_approach_depth_m"], region_id=p["region_id"])
        for p in region.get("ports", [])
    ]

