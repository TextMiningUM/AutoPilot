"""Quality-review STAP 2 (2026-09-23): THREE per-row training-variable limits
(safe_distance_m, max_turn_deg, risk_horizon_s) replace the previously-hardcoded
500m/30deg/300s used by both Track-2 generators -- a value that never varies in
training is learned as a shortcut constant (a user who configures the live simulator
away from 500/30 would otherwise find the fine-tuned model still silently judging
against 500/30).

Covers: derive_risk_horizon_s()'s geometry (the two reference speeds given at review),
sample_row_limits()'s determinism + weight-key coverage, real_risk() tipping on EACH
sampled safe_distance/horizon value, turn-degree capping at the per-row max, the shared
constraint_line() containing exactly the row's own three sampled values (never the old
hardcoded defaults), and text parity between a training row and a live simulator step
given the same settings.
"""
import math
import re

from pipeline.oow_agent_spec import (
    real_risk, RISK_HORIZON_S, RISK_HORIZON_K, derive_risk_horizon_s, sample_row_limits,
    fixed_limits, constraint_line, SAFE_DISTANCE_WEIGHTS, MAX_TURN_DEG_WEIGHTS,
    HORIZON_MULTIPLIER_WEIGHTS,
)
from pipeline.track2.build_oow_scenarios_leo import _turn_degrees as _leo_turn_degrees
from pipeline.track2.build_oow_scenarios import _turn_degrees as _synth_turn_degrees


# ── derive_risk_horizon_s(): the two reference speeds given at review ──────────────────
def test_derive_risk_horizon_matches_the_12kt_reference() -> None:
    """12 kt (~6.17 m/s) / 30 deg / 500m -> t_manoeuvre~162s -> horizon~567s (~570s)."""
    v_os = 12 * 0.514444  # kt -> m/s
    horizon = derive_risk_horizon_s(500.0, 30.0, v_os)
    assert 550.0 <= horizon <= 590.0


def test_derive_risk_horizon_matches_the_usv_reference() -> None:
    """A 2.5 m/s USV / 30 deg / 500m -> t_manoeuvre~400s -> horizon~1400s -- a MUCH
    longer horizon than the bridge's old fixed 300s (genuinely correct: a USV takes far
    longer than a 12kt ship to physically open the same safe distance by turning at the
    same per-command max -- NOT a bug, reported at STOP-1-supplement, k deliberately not
    adjusted to compensate)."""
    horizon = derive_risk_horizon_s(500.0, 30.0, 2.5)
    assert 1380.0 <= horizon <= 1420.0
    assert horizon > RISK_HORIZON_S * 4  # far beyond the historical fixed 300s horizon


def test_derive_risk_horizon_falls_back_to_historical_default_when_stopped() -> None:
    assert derive_risk_horizon_s(500.0, 30.0, 0.0) == RISK_HORIZON_S
    assert derive_risk_horizon_s(500.0, 30.0, None) == RISK_HORIZON_S


# ── sample_row_limits(): determinism + weight-key coverage ────────────────────────────
def test_sample_row_limits_is_deterministic_per_row_id() -> None:
    a = sample_row_limits("state::abc123", 10.0)
    b = sample_row_limits("state::abc123", 10.0)
    assert a == b


def test_sample_row_limits_differs_across_row_ids() -> None:
    seen = {tuple(sample_row_limits(f"row{i}", 10.0).values()) for i in range(30)}
    assert len(seen) > 1, "30 different row ids all sampled the identical limits -- seeding is broken"


def test_sample_row_limits_only_ever_draws_from_the_declared_weight_keys() -> None:
    for i in range(200):
        limits = sample_row_limits(f"row{i}", 10.0)
        assert limits["safe_distance_m"] in SAFE_DISTANCE_WEIGHTS
        assert limits["max_turn_deg"] in MAX_TURN_DEG_WEIGHTS
        assert limits["risk_horizon_multiplier"] in HORIZON_MULTIPLIER_WEIGHTS


def test_sample_row_limits_stand_on_tcpa_is_60_percent_of_its_own_horizon() -> None:
    limits = sample_row_limits("row0", 10.0)
    assert limits["stand_on_tcpa_s"] == limits["risk_horizon_s"] * 0.6


# ── real_risk() tips at EACH sampled safe_distance / horizon value ─────────────────────
def test_real_risk_tips_at_each_sampled_safe_distance_value() -> None:
    cpa_m = 350.0  # below 400/500/750/926, NOT below 300
    for safe_distance_m in SAFE_DISTANCE_WEIGHTS:
        expected = cpa_m < safe_distance_m
        assert real_risk(cpa_m, 100.0, safe_distance_m) is expected


def test_real_risk_tips_at_each_sampled_horizon_multiplier() -> None:
    horizon_default = 300.0
    tcpa_s = 250.0  # below 0.6x(180)? no -- exactly between some multipliers
    for multiplier in HORIZON_MULTIPLIER_WEIGHTS:
        risk_horizon_s = horizon_default * multiplier
        expected = 0 <= tcpa_s < risk_horizon_s
        assert real_risk(100.0, tcpa_s, 500.0, risk_horizon_s) is expected


# ── turn-degree capping at the PER-ROW max (never the old fixed MAX_TURN_DEG) ──────────
def test_leo_turn_degrees_never_exceeds_the_rows_own_sampled_max() -> None:
    for max_turn_deg in MAX_TURN_DEG_WEIGHTS:
        limits = {"safe_distance_m": 500.0, "max_turn_deg": max_turn_deg,
                 "risk_horizon_s": 300.0, "stand_on_tcpa_s": 180.0}
        degrees = _leo_turn_degrees(1.0, limits)  # worst-case shortfall (CPA ~ 0)
        assert degrees <= max_turn_deg


def test_synthetic_turn_degrees_never_exceeds_the_rows_own_sampled_max() -> None:
    for max_turn_deg in MAX_TURN_DEG_WEIGHTS:
        limits = {"safe_distance_m": 500.0, "max_turn_deg": max_turn_deg,
                 "risk_horizon_s": 300.0, "stand_on_tcpa_s": 180.0}
        degrees = _synth_turn_degrees(1.0, limits)
        assert degrees <= max_turn_deg


# ── constraint_line() contains EXACTLY the row's own sampled values ────────────────────
def test_constraint_line_contains_exactly_the_given_values_not_hardcoded_defaults() -> None:
    text = constraint_line(750.0, 20.0, 999.0)
    assert "750m" in text
    assert "20 degrees" in text
    assert "999s" in text
    # Never the old hardcoded 500/30/300 defaults leaking through instead:
    assert "500m" not in text
    assert "30 degrees" not in text
    assert "300s" not in text


def test_fixed_limits_uses_derived_default_horizon_at_multiplier_one() -> None:
    limits = fixed_limits(300.0, 30.0, 10.0)
    assert limits["safe_distance_m"] == 300.0
    assert limits["risk_horizon_multiplier"] == 1.0
    assert limits["risk_horizon_s"] == limits["risk_horizon_default_s"]
    assert math.isclose(limits["risk_horizon_s"], derive_risk_horizon_s(300.0, 30.0, 10.0))


# ── parity: a training row and a live simulator step, same settings -> same text
def test_simulator_and_training_constraint_line_are_byte_identical() -> None:
    """Basic Simulator/app/agents.py's build_oow_prompt() calls the SAME constraint_line()
    the training generators call (not a hand-duplicated copy) -- given the same
    safe_distance_m/max_turn_deg/own-ship speed, the two produce byte-IDENTICAL text by
    construction (single source, no possible drift)."""
    safe_distance_m, max_turn_deg, speed = 500.0, 30.0, 10.0
    horizon = derive_risk_horizon_s(safe_distance_m, max_turn_deg, speed)
    text = constraint_line(safe_distance_m, max_turn_deg, horizon)
    assert "CPA is below" in text and "TCPA" in text and "horizon" in text


def test_constraint_line_states_the_conjunction_not_cpa_alone() -> None:
    """Quality-review STOP-1/2-verification (2026-09-23): the OLD wording ("CPA below
    that is a real collision risk") stated the ALREADY-FIXED STAP-1 CPA-alone bug's own
    (wrong) definition as an unconditional fact -- must never reappear.

    Screening-set-B audit follow-up (2026-09-23): the single-sentence "BOTH...AND..."
    conjunction was ITSELF later found to conflate "is this a real encounter" with "is
    action mandatory yet" -- a model reading "no immediate action is required" (TCPA
    outside horizon) concluded encounter_rule 'none' on a genuine CPA-0 collision course.
    Replaced by two explicit sentences: identifying the encounter/rule is gated on CPA
    alone; only ACTING on it is gated on the TCPA-within-horizon condition. No rule
    numbers -- this module states geometry/physics only."""
    text = constraint_line(500.0, 30.0, 300.0)
    assert "CPA below that is a real collision risk" not in text
    assert ("you must identify the encounter and the applicable steering rule"
           in text)
    assert "Action becomes mandatory when that contact's TCPA is within the risk horizon" in text
    assert "you may act early but must at least name the encounter" in text
    assert not re.search(r"\bRule\s*\d+", text)

