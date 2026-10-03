"""Tests for pipeline/chief_engineer_agent_spec.py -- pure stdlib, no GPU/API key required."""
from __future__ import annotations

import pytest

from pipeline.chief_engineer_agent_spec import (
    ConditionLimit, ConditionReading,
    severity_for, estimate_rul_hours, evaluate_condition, recommend_maintenance,
    generate_degradation_trace, generate_synthetic_incident, compute_engine_status,
    all_known_limits, KNOWN_LIMITS,
)

ABOVE = ConditionLimit("bearing", "temp_c", "C", "above", warning_threshold=80.0, critical_threshold=100.0,
                        source_citation="test")
BELOW = ConditionLimit("oil", "tbn", "mg KOH/g", "below", warning_threshold=20.0, critical_threshold=15.0,
                        source_citation="test")
RANGE = ConditionLimit("fuel", "viscosity_cst", "cSt", "range", warning_threshold=(3.0, 15.0),
                        critical_threshold=(2.0, 20.0), source_citation="test")


def test_severity_for_above():
    assert severity_for(50.0, ABOVE) == "nominal"
    assert severity_for(73.0, ABOVE) == "watch"   # >= 0.9*80
    assert severity_for(85.0, ABOVE) == "warning"
    assert severity_for(105.0, ABOVE) == "critical"


def test_severity_for_below():
    assert severity_for(30.0, BELOW) == "nominal"
    assert severity_for(21.5, BELOW) == "watch"   # <= 1.1*20
    assert severity_for(18.0, BELOW) == "warning"
    assert severity_for(10.0, BELOW) == "critical"


def test_severity_for_range():
    assert severity_for(9.0, RANGE) == "nominal"
    assert severity_for(3.5, RANGE) == "watch"
    assert severity_for(2.5, RANGE) == "warning"
    assert severity_for(1.0, RANGE) == "critical"
    assert severity_for(25.0, RANGE) == "critical"


def test_estimate_rul_hours_above_rising():
    r = ConditionReading("bearing", "temp_c", 90.0, "C", timestamp_s=0.0)
    rul = estimate_rul_hours(r, slope_per_h=2.0, limit=ABOVE)
    assert rul == pytest.approx(5.0)  # (100-90)/2


def test_estimate_rul_hours_none_when_not_worsening():
    r = ConditionReading("bearing", "temp_c", 90.0, "C", timestamp_s=0.0)
    assert estimate_rul_hours(r, slope_per_h=-1.0, limit=ABOVE) is None
    assert estimate_rul_hours(r, slope_per_h=0.0, limit=ABOVE) is None


def test_estimate_rul_hours_already_past_critical():
    r = ConditionReading("bearing", "temp_c", 101.0, "C", timestamp_s=0.0)
    assert estimate_rul_hours(r, slope_per_h=1.0, limit=ABOVE) == 0.0


def test_evaluate_condition_uses_history_trend():
    history = [
        ConditionReading("bearing", "temp_c", 70.0, "C", timestamp_s=0.0),
        ConditionReading("bearing", "temp_c", 80.0, "C", timestamp_s=3600.0),
        ConditionReading("bearing", "temp_c", 90.0, "C", timestamp_s=7200.0),
    ]
    verdict = evaluate_condition(history[-1], ABOVE, history)
    assert verdict.severity == "warning"
    assert verdict.trend_slope_per_h == pytest.approx(10.0)
    assert verdict.rul_hours == pytest.approx(1.0)  # (100-90)/10


def test_recommend_maintenance_none_for_nominal():
    r = ConditionReading("bearing", "temp_c", 50.0, "C", timestamp_s=0.0)
    verdict = evaluate_condition(r, ABOVE)
    assert recommend_maintenance(verdict) is None


def test_recommend_maintenance_critical_is_immediate_and_due_now():
    r = ConditionReading("bearing", "temp_c", 105.0, "C", timestamp_s=0.0)
    verdict = evaluate_condition(r, ABOVE)
    rec = recommend_maintenance(verdict)
    assert rec is not None
    assert rec.urgency == "immediate"
    assert rec.due_by_hours == 0.0


def test_generate_degradation_trace_deterministic_and_reaches_critical():
    t1 = generate_degradation_trace(ABOVE, seed=7, n_points=12, dt_hours=1.0, will_fail=True)
    t2 = generate_degradation_trace(ABOVE, seed=7, n_points=12, dt_hours=1.0, will_fail=True)
    assert [r.value for r in t1] == [r.value for r in t2]
    assert severity_for(t1[-1].value, ABOVE) == "critical"


def test_generate_degradation_trace_stable_when_not_will_fail():
    trace = generate_degradation_trace(ABOVE, seed=3, n_points=12, dt_hours=1.0, will_fail=False)
    assert all(severity_for(r.value, ABOVE) == "nominal" for r in trace)


def test_generate_synthetic_incident_caught_truncates_before_end():
    full = generate_synthetic_incident(ABOVE, seed=11, caught_in_time=False, n_points=20, dt_hours=1.0)
    caught = generate_synthetic_incident(ABOVE, seed=11, caught_in_time=True, n_points=20, dt_hours=1.0)
    assert caught.detection_index is not None
    assert len(caught.readings) <= len(full.readings)
    assert caught.verdicts[-1].severity in ("watch", "warning", "critical")


def test_compute_engine_status_nominal_full_speed():
    r = ConditionReading("bearing", "temp_c", 50.0, "C", timestamp_s=0.0)
    verdict = evaluate_condition(r, ABOVE)
    status = compute_engine_status([verdict], rated_speed_kn=14.0)
    assert status.max_speed_kn == 14.0
    assert status.fault is None


def test_compute_engine_status_critical_caps_speed_and_sets_fault():
    r = ConditionReading("bearing", "temp_c", 110.0, "C", timestamp_s=0.0)
    verdict = evaluate_condition(r, ABOVE)
    status = compute_engine_status([verdict], rated_speed_kn=14.0)
    assert status.max_speed_kn == pytest.approx(14.0 * 0.4)
    assert status.fault is not None and "bearing" in status.fault


def test_known_limits_registry_nonempty_and_consistent_keys():
    limits = all_known_limits()
    assert len(limits) >= 5
    for (component_id, parameter), limit in KNOWN_LIMITS.items():
        assert limit.component_id == component_id
        assert limit.parameter == parameter
        assert limit.source_citation
