"""Captain walking-skeleton Phase 8: the Sec 13.B.7 trigger monitors + the Sec 13.B.9
event-to-monitor mapping and urgency-class deadlines.

Pure Python, no GPU/API key, safe to run locally. Pure pipeline/ module -- every monitor
is a plain boolean function over already-known facts (never a live MissionSim/route-
planner object), same decoupling convention as captain_agent_spec.py/captain_eval.py.

Sec 13.B.7 distinguishes CONTINUOUS monitors (visibility/position-doubt/track-deviation/
alarm -- evaluate a live Mission State value against a threshold, can become true from
ambient causes as well as a scripted event) from DISCRETE monitors (machinery-fault/
distress-signal/company-instruction -- true only because a specific event type fired, by
construction). The 4 continuous monitors are fully implemented and tested here as PURE
functions, but this phase does NOT wire them to a live ambient-state feed in
`app/captain_skeleton.py` -- no ambient weather/position-uncertainty schedule exists yet
(that's Sec 13.A.1 point 1's own job, deferred to the encounter-sim-splice phase). The 3
discrete monitors ARE wired end-to-end (Sec 13.B.9's mapping only needs them plus the
already-recorded hazard/decision log, no new live-state feed required).
"""
from __future__ import annotations
from dataclasses import dataclass


# --- The 4 continuous monitors (Sec 13.B.7) -- pure functions, not yet live-wired --------

def visibility_below(current_visibility_m: float, threshold_m: float) -> bool:
    """True when visibility has dropped below the Rule-19 safe-speed threshold --
    continuous, can fire from ambient weather or a scripted fog event."""
    return current_visibility_m < threshold_m


def position_uncertainty_above(current_uncertainty_m: float, threshold_m: float) -> bool:
    """True when the vessel's own position-fix uncertainty exceeds the threshold --
    continuous."""
    return current_uncertainty_m > threshold_m


def track_deviation_above(deviation_nm: float, elapsed_since_deviation_s: float,
                          threshold_nm: float = 20.0, within_s: float = 21600.0) -> bool:
    """True when the vessel has been off-track by more than `threshold_nm` for at least
    `within_s` seconds (renamed from `within_h` to `within_s`, Sec 13.A.7's units
    convention) -- continuous; a brief transient deviation does not trip it."""
    return deviation_nm > threshold_nm and elapsed_since_deviation_s >= within_s


def alarm_unresolved(alarm_active_since_s: float | None, now_s: float, timeout_s: float) -> bool:
    """An 'alarm' is self-referentially defined (Sec 13.B.7) as any OTHER monitor whose own
    condition has been true for longer than `timeout_s` without an OOW/Captain action
    addressing it -- continuous. The caller supplies whichever other monitor's own
    active-since timestamp is relevant; `alarm_active_since_s=None` means no alarm is
    currently active (never unresolved)."""
    if alarm_active_since_s is None:
        return False
    return (now_s - alarm_active_since_s) >= timeout_s


# --- The 3 discrete monitors (Sec 13.B.7) -- wired end-to-end in app/captain_skeleton.py -

def machinery_fault_reported(engine_failure_active: bool) -> bool:
    """Discrete -- true only when the Chief Engineer's own fact reports a fault (Sec
    2.5/13.A.5 `EngineStatus`), i.e. an engine_failure event is currently active."""
    return engine_failure_active


def distress_signal_received(distress_call_active: bool) -> bool:
    """Discrete -- true only once a distress-call event has fired and is not yet resolved
    by its world-responder (Sec 13.A.2)."""
    return distress_call_active


def company_instruction_received(commercial_instruction_active: bool) -> bool:
    """Discrete -- true only once a commercial-instruction event has fired and is not yet
    resolved by its world-responder (Sec 13.A.2)."""
    return commercial_instruction_active


# --- Event -> monitor mapping (Sec 13.B.9) + urgency-class deadlines (Sec 13.B.7) --------

@dataclass(frozen=True)
class MonitorMapping:
    """One v1 event type's own Sec 13.B.9 mapping. `monitor_name`/`urgency` are both None
    for an event correctly EXCLUDED from the recall/precision metric entirely -- Sec
    13.B.9's own 'whale zone resolved' case: a pre-authorised Standing Order, not a live
    monitor, never forced into an ill-fitting urgency class."""
    monitor_name: str | None
    urgency: str | None  # "emergency" | "mandatory" | "routine" | None


EVENT_MONITOR_MAP: dict[str, MonitorMapping] = {
    "engine_failure": MonitorMapping("machinery_fault_reported", "mandatory"),
    "fog": MonitorMapping("visibility_below", "mandatory"),
    "distress_call": MonitorMapping("distress_signal_received", "mandatory"),
    "whale_zone": MonitorMapping(None, None),
    "commercial_instruction": MonitorMapping("company_instruction_received", "mandatory"),
}


def urgency_deadline_s(urgency: str, trigger_fired_at_s: float, dt_mission_s: float,
                       next_report_cycle_s: float | None = None) -> float:
    """The ABSOLUTE mission-sim timestamp by which the Captain must have engaged, given a
    trigger that fired at `trigger_fired_at_s` (Sec 13.B.7): Emergency = the same instant
    (zero elapsed sim time); Mandatory = one `dt_mission_s` later; Routine = the next
    scheduled report cycle's own absolute timestamp (independent of exactly when the
    trigger fired, since report cycles are clock-based, Sec 8.3)."""
    if urgency == "emergency":
        return trigger_fired_at_s
    if urgency == "mandatory":
        return trigger_fired_at_s + dt_mission_s
    if urgency == "routine":
        if next_report_cycle_s is None:
            raise ValueError("urgency_deadline_s('routine', ...) needs next_report_cycle_s")
        return next_report_cycle_s
    raise ValueError(f"unknown urgency class: {urgency!r}")


def engaged_within_deadline(trigger_fired_at_s: float, responded_at_s: float, urgency: str,
                            dt_mission_s: float, next_report_cycle_s: float | None = None) -> bool:
    """True iff the Captain's own response landed at or before the urgency class's own
    deadline (Sec 13.B.7) -- Sec 10.3's recall/precision metric's own ground truth,
    computable now even though this phase doesn't build the metric itself yet."""
    deadline_s = urgency_deadline_s(urgency, trigger_fired_at_s, dt_mission_s, next_report_cycle_s)
    return responded_at_s <= deadline_s
