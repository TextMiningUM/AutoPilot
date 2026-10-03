"""Chief Engineer condition-monitoring core (design_chief_engineer.md Sec 3.4/3.5/3.8).

Pure stdlib, dependency-free dataclasses + deterministic functions -- mirrors the role
pipeline/oow_agent_spec.py plays for OOW and pipeline/captain_agent_spec.py plays for
Captain: a single source of truth for condition evaluation / RUL estimation / maintenance
recommendation, importable by BOTH the live interface
(Basic Simulator/app/pages/2_Engine_Room.py) and any future training-data generator, so the
two never drift out of sync (project convention: "a hand-copied second version WILL drift").

Known limits (`KNOWN_LIMITS`) are grounded in the reference installation (MAN B&W
S50ME-C10.7/S60ME-C10.7, Sawada et al. 2021 106m cargo vessel) where a real, extracted
manual fact is available (fuel oil viscosity, see citation below); the rest are
industry-typical illustrative baselines from design_chief_engineer.md Sec 3.3, explicitly
labelled as such in `source_citation` -- never silently presented as vendor-specific.

Safe to run LOCALLY (pure stdlib, no GPU/API key, no model loading).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

SEVERITY_ORDER: tuple[str, ...] = ("nominal", "watch", "warning", "critical")

_URGENCY_BY_SEVERITY = {
    "watch": "scheduled",
    "warning": "priority",
    "critical": "immediate",
}
_ACTION_BY_SEVERITY = {
    "watch": "Log the trend at the next watch round; no work order required yet.",
    "warning": "Raise a work order and inspect at the next safe opportunity{due_clause}.",
    "critical": "Stop/isolate the affected system and notify the Captain immediately; "
                "manual intervention required now.",
}

# Fraction of rated speed the engine can safely sustain at each worst-case severity tier,
# used by compute_engine_status() -- same fact shape captain_skeleton.py's
# _handle_engine_failure() already builds from scenario-scripted params (pipeline.captain_types.EngineStatus).
_SEVERITY_TO_SPEED_FRACTION = {"nominal": 1.0, "watch": 1.0, "warning": 0.75, "critical": 0.4}


@dataclass(frozen=True)
class ConditionLimit:
    """A single parameter's normal/warning/critical bounds for one engine-room component.

    `direction` is one of:
      "above" -- bad if the value rises past the threshold (e.g. bearing temperature)
      "below" -- bad if the value falls past the threshold (e.g. lube-oil TBN depletion)
      "range" -- bad if the value exits a two-sided band (e.g. fuel oil viscosity),
                 in which case `warning_threshold`/`critical_threshold` are (low, high) tuples.
    """
    component_id: str
    parameter: str
    unit: str
    direction: str
    warning_threshold: float | tuple[float, float]
    critical_threshold: float | tuple[float, float]
    source_citation: str


@dataclass(frozen=True)
class ConditionReading:
    """One timestamped sensor/log reading for a (component_id, parameter) pair."""
    component_id: str
    parameter: str
    value: float
    unit: str
    timestamp_s: float


@dataclass(frozen=True)
class ConditionVerdict:
    """The evaluated severity + trend + remaining-useful-life estimate for one reading,
    given its ConditionLimit and the reading history it was computed from."""
    component_id: str
    parameter: str
    severity: str  # one of SEVERITY_ORDER
    value: float
    trend_slope_per_h: float
    rul_hours: float | None  # None if the trend isn't moving toward the critical bound
    limit: ConditionLimit


@dataclass(frozen=True)
class MaintenanceRecommendation:
    component_id: str
    severity: str
    action: str
    urgency: str  # "routine" | "scheduled" | "priority" | "immediate"
    due_by_hours: float | None
    source_citation: str


# ── § 1  Known limits registry ──────────────────────────────────────────────────────────
# Real, extracted (manual_extract provenance, see Data/ChiefEngineer/ChiefEngineer_Agents_Training/
# chief_engineer_known_issues_traces.jsonl row CE-MAN-S50ME-C10.7-...-Components_for_Fuel_Oil_S-0):
_REAL_FUEL_VISCOSITY_CITATION = (
    "S50ME-C10.7_project_guide.pdf, '198 39 51-2.10 Components for Fuel Oil System' "
    "(real, extracted 2026-10-03): viscosity at engine inlet 2 cSt (abs. min) - 20 cSt (abs. max), "
    "normal 3-15 cSt."
)
_ILLUSTRATIVE_CITATION = "design_chief_engineer.md Sec 3.3 (illustrative industry-typical baseline, not vendor-specific)."

KNOWN_LIMITS: dict[tuple[str, str], ConditionLimit] = {
    ("fuel_injection", "viscosity_cst"): ConditionLimit(
        component_id="fuel_injection", parameter="viscosity_cst", unit="cSt", direction="range",
        warning_threshold=(3.0, 15.0), critical_threshold=(2.0, 20.0),
        source_citation=_REAL_FUEL_VISCOSITY_CITATION,
    ),
    ("main_bearing", "temperature_c"): ConditionLimit(
        component_id="main_bearing", parameter="temperature_c", unit="\u00b0C", direction="above",
        warning_threshold=85.0, critical_threshold=100.0, source_citation=_ILLUSTRATIVE_CITATION,
    ),
    ("cylinder_unit", "exhaust_temp_spread_c"): ConditionLimit(
        component_id="cylinder_unit", parameter="exhaust_temp_spread_c", unit="\u00b0C", direction="above",
        warning_threshold=50.0, critical_threshold=80.0, source_citation=_ILLUSTRATIVE_CITATION,
    ),
    ("turbocharger", "exhaust_gas_temp_c"): ConditionLimit(
        component_id="turbocharger", parameter="exhaust_gas_temp_c", unit="\u00b0C", direction="above",
        warning_threshold=450.0, critical_threshold=500.0, source_citation=_ILLUSTRATIVE_CITATION,
    ),
    ("lubricating_oil", "tbn"): ConditionLimit(
        component_id="lubricating_oil", parameter="tbn", unit="mg KOH/g", direction="below",
        warning_threshold=20.0, critical_threshold=15.0, source_citation=_ILLUSTRATIVE_CITATION,
    ),
    ("lubricating_oil", "iron_content_ppm"): ConditionLimit(
        component_id="lubricating_oil", parameter="iron_content_ppm", unit="ppm", direction="above",
        warning_threshold=50.0, critical_threshold=100.0, source_citation=_ILLUSTRATIVE_CITATION,
    ),
    ("cooling_water", "temperature_c"): ConditionLimit(
        component_id="cooling_water", parameter="temperature_c", unit="\u00b0C", direction="above",
        warning_threshold=80.0, critical_threshold=90.0, source_citation=_ILLUSTRATIVE_CITATION,
    ),
}


def all_known_limits() -> list[ConditionLimit]:
    """Every registered ConditionLimit, in a stable (insertion) order."""
    return list(KNOWN_LIMITS.values())


# ── § 2  Severity evaluation ─────────────────────────────────────────────────────────────
def severity_for(value: float, limit: ConditionLimit) -> str:
    """Deterministic threshold classification into SEVERITY_ORDER. "watch" is a soft band
    within 10% (of the relevant span) of the warning threshold -- an early heads-up tier
    with no limit-crossing yet."""
    if limit.direction == "range":
        warn_lo, warn_hi = limit.warning_threshold  # type: ignore[misc]
        crit_lo, crit_hi = limit.critical_threshold  # type: ignore[misc]
        if value <= crit_lo or value >= crit_hi:
            return "critical"
        if value <= warn_lo or value >= warn_hi:
            return "warning"
        span = warn_hi - warn_lo
        if span > 0 and (value <= warn_lo + 0.1 * span or value >= warn_hi - 0.1 * span):
            return "watch"
        return "nominal"

    warn = limit.warning_threshold  # type: ignore[assignment]
    crit = limit.critical_threshold  # type: ignore[assignment]
    assert isinstance(warn, (int, float)) and isinstance(crit, (int, float))
    if limit.direction == "above":
        if value >= crit:
            return "critical"
        if value >= warn:
            return "warning"
        if value >= warn * 0.9:
            return "watch"
        return "nominal"
    if limit.direction == "below":
        if value <= crit:
            return "critical"
        if value <= warn:
            return "warning"
        if value <= warn * 1.1:
            return "watch"
        return "nominal"
    raise ValueError(f"Unknown ConditionLimit.direction: {limit.direction!r}")


def _linear_trend_per_hour(history: list[ConditionReading]) -> float:
    """Least-squares slope of value vs. time (hours). 0.0 if fewer than 2 points."""
    if len(history) < 2:
        return 0.0
    xs = [r.timestamp_s / 3600.0 for r in history]
    ys = [r.value for r in history]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0


def estimate_rul_hours(current: ConditionReading, slope_per_h: float, limit: ConditionLimit) -> float | None:
    """Hours until the linear trend crosses the relevant critical bound, assuming it
    continues unchanged. Returns None if the trend isn't moving toward failure (flat or
    improving) -- callers must treat None as "no estimate", never as "infinite time"."""
    if limit.direction == "above":
        crit = limit.critical_threshold  # type: ignore[assignment]
        assert isinstance(crit, (int, float))
        if current.value >= crit:
            return 0.0
        if slope_per_h <= 0:
            return None
        return (crit - current.value) / slope_per_h
    if limit.direction == "below":
        crit = limit.critical_threshold  # type: ignore[assignment]
        assert isinstance(crit, (int, float))
        if current.value <= crit:
            return 0.0
        if slope_per_h >= 0:
            return None
        return (current.value - crit) / (-slope_per_h)
    if limit.direction == "range":
        crit_lo, crit_hi = limit.critical_threshold  # type: ignore[misc]
        if current.value <= crit_lo or current.value >= crit_hi:
            return 0.0
        if slope_per_h > 0:
            return (crit_hi - current.value) / slope_per_h
        if slope_per_h < 0:
            return (current.value - crit_lo) / (-slope_per_h)
        return None
    raise ValueError(f"Unknown ConditionLimit.direction: {limit.direction!r}")


def evaluate_condition(
    reading: ConditionReading, limit: ConditionLimit, history: list[ConditionReading] | None = None,
) -> ConditionVerdict:
    """The single call-site both the live page and any future training generator should use:
    reading + its limit (+ optional reading history for trend/RUL) -> one ConditionVerdict."""
    hist = history if history else [reading]
    slope = _linear_trend_per_hour(hist)
    severity = severity_for(reading.value, limit)
    rul = estimate_rul_hours(reading, slope, limit)
    return ConditionVerdict(
        component_id=limit.component_id, parameter=limit.parameter, severity=severity,
        value=reading.value, trend_slope_per_h=slope, rul_hours=rul, limit=limit,
    )


def recommend_maintenance(verdict: ConditionVerdict) -> MaintenanceRecommendation | None:
    """Deterministic severity -> action/urgency lookup. None for "nominal" (nothing to do)."""
    if verdict.severity == "nominal":
        return None
    due_clause = ""
    if verdict.severity == "warning" and verdict.rul_hours is not None:
        due_clause = f", within {round(verdict.rul_hours)}h if the trend continues"
    action = _ACTION_BY_SEVERITY[verdict.severity].format(due_clause=due_clause)
    due_by = 0.0 if verdict.severity == "critical" else verdict.rul_hours
    return MaintenanceRecommendation(
        component_id=verdict.component_id, severity=verdict.severity, action=action,
        urgency=_URGENCY_BY_SEVERITY[verdict.severity], due_by_hours=due_by,
        source_citation=verdict.limit.source_citation,
    )


# ── § 3  Synthetic degradation traces + incidents (design_chief_engineer.md Sec 3.8) ───
def _nominal_start(limit: ConditionLimit) -> float:
    """A plausible healthy starting value for a fresh trace."""
    if limit.direction == "range":
        lo, hi = limit.warning_threshold  # type: ignore[misc]
        return (lo + hi) / 2
    if limit.direction == "above":
        return limit.warning_threshold * 0.6  # type: ignore[operator]
    if limit.direction == "below":
        return limit.warning_threshold * 1.3  # type: ignore[operator]
    raise ValueError(f"Unknown ConditionLimit.direction: {limit.direction!r}")


def generate_degradation_trace(
    limit: ConditionLimit, seed: int, n_points: int = 24, dt_hours: float = 2.0, will_fail: bool = True,
) -> list[ConditionReading]:
    """Seeded, deterministic synthetic reading sequence for one component/parameter.

    If `will_fail`, drifts linearly from a healthy start toward just past the critical
    bound over `n_points * dt_hours` hours (direction chosen deterministically from `seed`
    when the limit is two-sided); otherwise stays flat/nominal with small jitter only.
    Same seed + same args always reproduces the identical trace (project convention:
    "determinism by construction" -- no bare unseeded random call).
    """
    rng = random.Random(seed)
    start = _nominal_start(limit)

    if limit.direction == "range":
        lo_c, hi_c = limit.critical_threshold  # type: ignore[misc]
        drift_up = rng.random() < 0.5
        target = hi_c * 1.05 if drift_up else lo_c * 0.95
    elif limit.direction == "above":
        target = limit.critical_threshold * 1.1  # type: ignore[operator]
    else:  # "below"
        target = limit.critical_threshold * 0.9  # type: ignore[operator]

    total_drift = (target - start) if will_fail else 0.0
    noise_amp = abs(total_drift) * 0.03 if will_fail else max(abs(start) * 0.01, 0.1)

    readings: list[ConditionReading] = []
    for i in range(n_points):
        frac = i / (n_points - 1) if n_points > 1 else 1.0
        base = start + total_drift * frac
        value = base + rng.uniform(-noise_amp, noise_amp)
        readings.append(ConditionReading(
            component_id=limit.component_id, parameter=limit.parameter,
            value=value, unit=limit.unit, timestamp_s=i * dt_hours * 3600.0,
        ))
    return readings


@dataclass(frozen=True)
class SyntheticIncident:
    """A seeded, reproducible "degradation -> (caught | missed)" narrative, built directly
    on top of generate_degradation_trace()/evaluate_condition() so the live condition-model
    and any future training-data generator never drift apart (same underlying functions)."""
    incident_id: str
    component_id: str
    parameter: str
    caught_in_time: bool
    detection_index: int | None
    narrative: str
    readings: list[ConditionReading]
    verdicts: list[ConditionVerdict] = field(default_factory=list)


def generate_synthetic_incident(
    limit: ConditionLimit, seed: int, caught_in_time: bool, n_points: int = 24, dt_hours: float = 2.0,
) -> SyntheticIncident:
    """Generate one degradation trace and evaluate it step-by-step; if `caught_in_time`,
    truncate shortly after the first "warning"/"critical" verdict (watchkeeper intervened),
    otherwise let it run to the end of the trace (missed/late response)."""
    readings = generate_degradation_trace(limit, seed, n_points=n_points, dt_hours=dt_hours, will_fail=True)
    verdicts: list[ConditionVerdict] = []
    detection_index: int | None = None
    for i, r in enumerate(readings):
        v = evaluate_condition(r, limit, readings[: i + 1])
        verdicts.append(v)
        if detection_index is None and v.severity in ("warning", "critical"):
            detection_index = i

    if caught_in_time and detection_index is not None:
        cutoff = min(detection_index + 2, len(readings) - 1)
        readings = readings[: cutoff + 1]
        verdicts = verdicts[: cutoff + 1]
        narrative = (
            f"Watchkeeper caught the {limit.parameter} trend on {limit.component_id} at "
            f"{verdicts[-1].severity} severity (t={readings[-1].timestamp_s / 3600:.1f}h) and "
            f"intervened before it reached critical."
        )
    else:
        narrative = (
            f"{limit.component_id}'s {limit.parameter} reached {verdicts[-1].severity} severity "
            f"(t={readings[-1].timestamp_s / 3600:.1f}h) with no intervention recorded."
        )

    return SyntheticIncident(
        incident_id=f"CE-SYN-{limit.component_id}-{limit.parameter}-{seed}",
        component_id=limit.component_id, parameter=limit.parameter, caught_in_time=caught_in_time,
        detection_index=detection_index, narrative=narrative, readings=readings, verdicts=verdicts,
    )


# ── § 4  EngineStatus aggregation (design_chief_engineer.md Sec 3.6 / design_captain_missions.md Sec 2.5) ──
def compute_engine_status(verdicts: list[ConditionVerdict], rated_speed_kn: float, reported_at: float | None = None):
    """Aggregate per-component verdicts into one pipeline.captain_types.EngineStatus fact
    (worst severity across all components caps max_speed_kn) -- the SAME fact shape
    Basic Simulator/app/captain_skeleton.py's _handle_engine_failure() already builds from
    scenario-scripted event.params, so this can replace that scripted path later with zero
    consumer-side changes to EngineFailureContext/CaptainAction. Imports EngineStatus lazily
    to avoid a hard import-time dependency from this pure-stdlib module."""
    from pipeline.captain_types import EngineStatus

    if not verdicts:
        return EngineStatus(max_speed_kn=rated_speed_kn, fault=None, reported_at=reported_at)
    worst = max(verdicts, key=lambda v: SEVERITY_ORDER.index(v.severity))
    if worst.severity in ("nominal", "watch"):
        return EngineStatus(max_speed_kn=rated_speed_kn, fault=None, reported_at=reported_at)
    fraction = _SEVERITY_TO_SPEED_FRACTION[worst.severity]
    fault = f"{worst.component_id} {worst.parameter} at {worst.severity} severity"
    return EngineStatus(max_speed_kn=round(rated_speed_kn * fraction, 1), fault=fault, reported_at=reported_at)
