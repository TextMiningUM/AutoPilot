"""Captain walking-skeleton Phase 1: discrete-event scheduler + mission-sim clock
(design_captain_missions.md Sec 13.A.1/13.A.3/13.A.7/13.A.9).

Pure Python, no GPU/API key, safe to run locally. Deliberately standalone and NOT wired
into `app.missions.Mission`/`app.simulation.Simulation` yet (that wiring is Phase 4) -- so
every existing mission/run is completely unaffected by this module's mere existence; the
"opt-in only" backward-compatibility guarantee (Sec 13.A.1) holds trivially because nothing
in the existing codebase constructs a `MissionSim` unless explicitly asked to.

Units follow Sec 13.A.7 throughout: anything driving the clock/events/deadlines is in
SECONDS (suffixed `_s`); speeds are kn, distances are NM, fuel is tonnes. The one named
exception is rest-hours (STCW's own convention, Sec 13.A.3), kept in hours and converted
at the comparison point only.
"""
from __future__ import annotations
import heapq
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # Auto Pilot/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.captain_geo import initial_bearing_deg
from pipeline.captain_types import MissionOrder

_EARTH_RADIUS_NM = 3440.065  # mean Earth radius in nautical miles


class EventType(IntEnum):
    """The closed 7-type scheduler alphabet (Sec 13.A.9.a) -- the int VALUE is also the
    tie-break rank (lower = higher priority = popped first at an identical timestamp)."""
    EMERGENCY_TRIGGER = 1
    MANDATORY_TRIGGER = 2
    WORLD_RESPONDER_TIMER = 3
    ENCOUNTER_BOUNDARY = 4
    WAYPOINT_ARRIVAL = 5
    ROUTINE_REPORT = 6
    MONITOR_TICK = 7


@dataclass(frozen=True)
class ScheduledEvent:
    """One scheduler queue entry's payload (Sec 13.A.9.a) -- `kind` is a free-form
    identifier (e.g. "routine_report_3", "world_responder:ev2") for whichever specific
    event this is, `event_type` is its fixed rank-bearing category."""
    kind: str
    event_type: EventType
    timestamp_s: float
    data: dict[str, Any] = field(default_factory=dict)


class EventScheduler:
    """A single priority queue for every mission-sim event (Sec 13.A.9) -- replaces "two
    loops with an if-statement" with one min-heap ordered by (timestamp_s, rank,
    sequence_no), the total order deterministic replay (Sec 13.C.12) depends on. Only
    FIXED-TIME events (Sec 13.A.9.b) are ever scheduled here -- condition events (trigger
    monitors) have no closed-form timestamp and are detected by the recurring mission-sim
    step itself, not pre-scheduled."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, ScheduledEvent]] = []
        self._next_seq = 0

    def schedule(self, timestamp_s: float, event_type: EventType, kind: str,
                 data: dict[str, Any] | None = None) -> ScheduledEvent:
        """Enqueues one fixed-time event; `sequence_no` is assigned here, at enqueue time,
        so two same-timestamp/same-rank entries still resolve in a fixed, deterministic
        (insertion) order."""
        event = ScheduledEvent(kind=kind, event_type=event_type, timestamp_s=timestamp_s, data=data or {})
        heapq.heappush(self._heap, (timestamp_s, int(event_type), self._next_seq, event))
        self._next_seq += 1
        return event

    def peek_next_timestamp(self) -> float | None:
        """The timestamp of whatever would be popped next, without popping it."""
        return self._heap[0][0] if self._heap else None

    def pop_next(self) -> ScheduledEvent | None:
        """Pops the single highest-priority (earliest timestamp, then lowest rank value,
        then earliest enqueued) event, or None if the queue is empty."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[3]

    def __len__(self) -> int:
        return len(self._heap)


def schedule_routine_reports(scheduler: EventScheduler, reporting_interval_hours: float,
                              mission_duration_s: float) -> list[ScheduledEvent]:
    """Pre-computes every ROUTINE_REPORT timestamp for the whole mission up front, in closed
    form from t0 (Sec 8.3/13.A.9.a) -- clock-based, never revalidated against live state."""
    interval_s = reporting_interval_hours * 3600.0
    events: list[ScheduledEvent] = []
    n = 1
    t = n * interval_s
    while t <= mission_duration_s:
        events.append(scheduler.schedule(t, EventType.ROUTINE_REPORT, kind=f"routine_report_{n}"))
        n += 1
        t = n * interval_s
    return events


def haversine_nm(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points in nautical miles -- the
    whole-route distance convention (Sec 13.A.1): great-circle/rhumb-line nm, never the
    local equirectangular metric frame reserved for inside an encounter window."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_NM * 2 * math.asin(math.sqrt(a))


def route_distance_nm(waypoints: list[tuple[float, float]]) -> float:
    """Total great-circle distance along an ordered waypoint list."""
    return sum(haversine_nm(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1))


def position_along_route_nm(waypoints: list[tuple[float, float]], distance_nm: float) -> tuple[float, float]:
    """Linear lat/lon interpolation between whichever two waypoints bracket `distance_nm`
    along the route -- the SAME equirectangular-approximation abstraction level already
    accepted for the encounter-sim's local frame (Sec 13.A.1), not a true great-circle
    interpolation. Needed because MissionSim only tracks cumulative distance travelled
    (Sec 13.A.1), not a live lat/lon, so the route planner (which needs a real point) has
    to derive one. Clamps to the first/last waypoint for distance_nm outside the route's
    own [0, total] range."""
    if distance_nm <= 0:
        return waypoints[0]
    cumulative_nm = 0.0
    for i in range(len(waypoints) - 1):
        leg_start, leg_end = waypoints[i], waypoints[i + 1]
        leg_len_nm = haversine_nm(leg_start, leg_end)
        if cumulative_nm + leg_len_nm >= distance_nm:
            fraction = (distance_nm - cumulative_nm) / leg_len_nm if leg_len_nm > 0 else 0.0
            lat = leg_start[0] + fraction * (leg_end[0] - leg_start[0])
            lon = leg_start[1] + fraction * (leg_end[1] - leg_start[1])
            return (lat, lon)
        cumulative_nm += leg_len_nm
    return waypoints[-1]


def current_leg_bearing_deg(waypoints: list[tuple[float, float]], distance_nm: float) -> float:
    """The current route leg's own initial bearing (true, degrees) at `distance_nm` along
    the route -- own-ship is assumed to track exactly along the route when not spliced
    into an encounter (Sec 13.A.1), so this doubles as 'own-ship's current heading' for
    ambient-contact generation and the encounter-sim handoff (Phase 9c)."""
    if distance_nm <= 0:
        return initial_bearing_deg(waypoints[0], waypoints[1])
    cumulative_nm = 0.0
    for i in range(len(waypoints) - 1):
        leg_start, leg_end = waypoints[i], waypoints[i + 1]
        leg_len_nm = haversine_nm(leg_start, leg_end)
        if cumulative_nm + leg_len_nm >= distance_nm:
            return initial_bearing_deg(leg_start, leg_end)
        cumulative_nm += leg_len_nm
    return initial_bearing_deg(waypoints[-2], waypoints[-1])


def next_waypoint_nm(waypoints: list[tuple[float, float]], distance_nm: float) -> tuple[float, float]:
    """The upcoming waypoint (current leg's own end point) at `distance_nm` along the
    route -- Sec 13.C.14's "an encounter's goal is only the projected leg endpoint", used
    as the encounter-sim's own Mission.goal (Phase 9c)."""
    if distance_nm <= 0:
        return waypoints[1]
    cumulative_nm = 0.0
    for i in range(len(waypoints) - 1):
        leg_len_nm = haversine_nm(waypoints[i], waypoints[i + 1])
        if cumulative_nm + leg_len_nm >= distance_nm:
            return waypoints[i + 1]
        cumulative_nm += leg_len_nm
    return waypoints[-1]


@dataclass(frozen=True)
class FuelModel:
    """`fuel_rate(t) = base_load + k * speed_kn(t)**3` (Sec 13.A.3), both constants
    derived once per mission by `calibrate_fuel_rate()` -- never independently guessed."""
    base_load_tonnes_per_h: float
    k: float
    f_hotel: float

    def fuel_rate_tonnes_per_h(self, speed_kn: float) -> float:
        return self.base_load_tonnes_per_h + self.k * speed_kn ** 3


def calibrate_fuel_rate(fuel_tonnes_at_departure: float, fuel_reserve_margin_pct: float,
                         total_route_distance_nm: float, soa_kn: float,
                         f_hotel: float = 0.10) -> FuelModel:
    """Sec 13.A.3's two-step calibration: fix a hotel-load fraction, then solve the Mission
    Order's own fuel-budget equation for `k` at constant SOA over the whole route."""
    fuel_budget = fuel_tonnes_at_departure * (1 - fuel_reserve_margin_pct / 100.0)
    t_h = total_route_distance_nm / soa_kn  # hours -- nm / kn is unavoidably hours
    k = fuel_budget / (soa_kn ** 3 * (1 + f_hotel) * t_h)
    base_load = f_hotel * k * soa_kn ** 3
    return FuelModel(base_load_tonnes_per_h=base_load, k=k, f_hotel=f_hotel)


_WATCH_CYCLE_S = 12 * 3600.0
_WATCH_ON_S = 4 * 3600.0


def watch_roster_off_hours(start_s: float, end_s: float) -> float:
    """Hours within [start_s, end_s] (seconds, phase anchored at t=0) that fall in the
    OOW's 4-hours-on/8-hours-off roster's off-watch (resting) phase (Sec 13.A.3)."""
    if end_s <= start_s:
        return 0.0
    total_off_s = 0.0
    t = start_s
    while t < end_s:
        phase = t % _WATCH_CYCLE_S
        if phase < _WATCH_ON_S:
            t = min(end_s, t - phase + _WATCH_ON_S)  # skip to the end of the on-watch sub-interval
            continue
        cycle_start = t - phase
        off_seg_end = min(end_s, cycle_start + _WATCH_CYCLE_S)
        total_off_s += off_seg_end - t
        t = off_seg_end
    return total_off_s / 3600.0


def always_resting_hours(start_s: float, end_s: float) -> float:
    """The Captain's own nominal baseline (Sec 13.A.3): "fully resting" unless disrupted --
    i.e. the whole window counts as rest before any disruption is subtracted."""
    return max(0.0, end_s - start_s) / 3600.0


@dataclass
class RestHoursLedger:
    """STCW A-VIII/1 rest-hours bookkeeping (10h/24h, 77h/7d minimums, Sec 13.A.3) against a
    nominal roster (`nominal_rest_fn`), reduced by logged disruptions. Use
    `for_oow_watch_roster()`/`for_captain()` rather than constructing directly."""
    nominal_rest_fn: Callable[[float, float], float]
    disruptions: list[tuple[float, float]] = field(default_factory=list)  # (at_s, hours_lost)

    @classmethod
    def for_oow_watch_roster(cls) -> "RestHoursLedger":
        return cls(nominal_rest_fn=watch_roster_off_hours)

    @classmethod
    def for_captain(cls) -> "RestHoursLedger":
        return cls(nominal_rest_fn=always_resting_hours)

    def add_disruption(self, at_s: float, hours_lost: float) -> None:
        """Logs a Mandatory/Emergency engagement that ate into what would otherwise have
        been a rest period -- disruptions are never added back later (Sec 13.A.3)."""
        self.disruptions.append((at_s, hours_lost))

    def rest_hours_in_trailing_window(self, now_s: float, window_h: float) -> float:
        window_s = window_h * 3600.0
        start_s = max(0.0, now_s - window_s)
        nominal_h = self.nominal_rest_fn(start_s, now_s)
        lost_h = sum(h for (t, h) in self.disruptions if start_s <= t <= now_s)
        return max(0.0, nominal_h - lost_h)

    def breaches_stcw_minimum(self, now_s: float) -> bool:
        return (self.rest_hours_in_trailing_window(now_s, 24.0) < 10.0
                or self.rest_hours_in_trailing_window(now_s, 24.0 * 7) < 77.0)


@dataclass
class MissionSimState:
    """Mutable mission-sim state -- position is tracked as distance travelled along the
    route (nm), not lat/lon, since nothing in Phase 1 needs the encounter-sim's local
    metric frame yet (Sec 13.A.1)."""
    elapsed_s: float = 0.0
    distance_travelled_nm: float = 0.0
    fuel_tonnes: float = 0.0
    current_speed_kn: float = 0.0
    oow_rest_ledger: RestHoursLedger = field(default_factory=RestHoursLedger.for_oow_watch_roster)
    captain_rest_ledger: RestHoursLedger = field(default_factory=RestHoursLedger.for_captain)


class MissionSim:
    """The mission-sim stepping engine (Sec 13.A.1) -- strictly opt-in: callers construct
    one explicitly from a route/resource budget (or a MissionOrder via
    `from_mission_order()`); no existing `Mission`/`Simulation` code path creates one."""

    def __init__(self, waypoints: list[tuple[float, float]], soa_kn: float,
                 fuel_tonnes_at_departure: float, fuel_reserve_margin_pct: float,
                 t0_utc: str, dt_mission_s: float = 600.0,
                 reporting_interval_hours: float | None = None) -> None:
        if len(waypoints) < 2:
            raise ValueError("MissionSim needs at least 2 waypoints to form a route")
        self.waypoints = waypoints
        self.soa_kn = soa_kn
        self.dt_mission_s = dt_mission_s
        self.t0_utc = t0_utc
        self.total_route_distance_nm = route_distance_nm(waypoints)
        self.fuel_model = calibrate_fuel_rate(fuel_tonnes_at_departure, fuel_reserve_margin_pct,
                                               self.total_route_distance_nm, soa_kn)
        self.state = MissionSimState(fuel_tonnes=fuel_tonnes_at_departure, current_speed_kn=soa_kn)
        self.scheduler = EventScheduler()
        self.nominal_duration_s = (self.total_route_distance_nm / soa_kn) * 3600.0
        if reporting_interval_hours is not None:
            schedule_routine_reports(self.scheduler, reporting_interval_hours, self.nominal_duration_s)

    @classmethod
    def from_mission_order(cls, order: MissionOrder) -> "MissionSim":
        resources = order.admin_logistics.get("resources", {})
        return cls(
            waypoints=order.waypoints, soa_kn=order.speed_of_advance_kn,
            fuel_tonnes_at_departure=float(resources.get("fuel_tonnes", 0.0)),
            fuel_reserve_margin_pct=float(resources.get("fuel_reserve_margin_pct", 0.0)),
            t0_utc=order.t0_utc, dt_mission_s=order.dt_mission_s,
            reporting_interval_hours=order.command_signal.get("reporting_interval_hours"),
        )

    def remaining_distance_nm(self) -> float:
        return max(0.0, self.total_route_distance_nm - self.state.distance_travelled_nm)

    def eta_s(self) -> float:
        """Remaining time at the CURRENT speed (Sec 13.A.3), recomputed live -- `inf` if
        stopped."""
        if self.state.current_speed_kn <= 0:
            return float("inf")
        return (self.remaining_distance_nm() / self.state.current_speed_kn) * 3600.0

    def current_utc(self) -> datetime:
        """`t0_utc + elapsed_s` (Sec 13.A.1) -- the real calendar/clock reference."""
        t0 = datetime.fromisoformat(self.t0_utc.replace("Z", "+00:00"))
        return t0 + timedelta(seconds=self.state.elapsed_s)

    def reached_destination(self) -> bool:
        return self.state.distance_travelled_nm >= self.total_route_distance_nm

    def step(self, dt_s: float | None = None) -> None:
        """One mission-sim tick (Sec 13.A.9.b: "MONITOR_TICK *is* the mission-sim step") --
        advances position/fuel/elapsed time by dt_s (default dt_mission_s). When the
        requested dt would carry distance_travelled_nm past the route's total distance,
        the step is clipped to arrival -- distance, elapsed time, AND fuel burn are all
        scaled down together, so the final partial step never over-consumes fuel for
        travel that didn't actually happen."""
        dt = self.dt_mission_s if dt_s is None else dt_s
        speed_kn = self.state.current_speed_kn
        requested_distance_nm = speed_kn * (dt / 3600.0)
        new_distance_nm = min(self.total_route_distance_nm,
                               self.state.distance_travelled_nm + requested_distance_nm)
        actual_distance_nm = new_distance_nm - self.state.distance_travelled_nm
        actual_dt = (actual_distance_nm / speed_kn * 3600.0) if speed_kn > 0 else dt
        fuel_burned = self.fuel_model.fuel_rate_tonnes_per_h(speed_kn) * (actual_dt / 3600.0)
        self.state.fuel_tonnes = max(0.0, self.state.fuel_tonnes - fuel_burned)
        self.state.distance_travelled_nm = new_distance_nm
        self.state.elapsed_s += actual_dt

    def advance_to(self, target_s: float) -> None:
        """Steps in `dt_mission_s` increments (the final partial step clipped exactly to
        `target_s`) so the clock lands precisely on a scheduled event's timestamp without
        overshooting it."""
        while self.state.elapsed_s < target_s:
            remaining = target_s - self.state.elapsed_s
            self.step(min(self.dt_mission_s, remaining))

    def run_to_next_scheduled(self) -> ScheduledEvent | None:
        """Advances the clock to the next scheduled fixed-time event and pops it -- None
        if the queue is empty."""
        next_ts = self.scheduler.peek_next_timestamp()
        if next_ts is None:
            return None
        self.advance_to(next_ts)
        return self.scheduler.pop_next()
