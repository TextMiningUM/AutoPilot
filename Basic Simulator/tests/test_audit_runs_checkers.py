"""Regression tests for the 2026-09-23 audit_runs.py checker fixes:
1. A_fabricated_risk re-scoped (early-action no longer counted as fabrication) +
   new E_unclassified_encounter.
2. G_gate_b_number_fabricated: rule-citation numbers excluded, constraint/rounded/
   unit-converted values treated as known.
3. G_gate_c_risk_mismatch: explicit risk/no_risk/unknown extractor, early-action zone
   exempted.
Evidence/test cases are the ones given directly in the 2026-09-23 bug report.
"""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
ANALYSIS_DIR = APP_ROOT / "_analysis"
for p in (APP_ROOT, ANALYSIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import audit_runs as ar  # noqa: E402
from app.simulation import VesselConstraints  # noqa: E402

CONSTRAINTS = VesselConstraints()  # min_cpa_m=500.0, turn_rate_deg_s=3.0, time_step_s=10.0


def contact(name: str, cpa_m: float, tcpa_s: float, rel_bearing_deg: float = 10.0) -> dict:
    return {"name": name, "range_m": cpa_m, "rel_bearing_deg": rel_bearing_deg,
            "cpa_m": cpa_m, "tcpa_s": tcpa_s, "closing": True, "stationary": False}


def make_cp(step: int, situation_report: str, action: str, degrees, encounter_rule: str,
           conduct_rule: str, reasoning: str) -> dict:
    return {"step": step, "situation_report": situation_report,
            "decision": {"action": action, "degrees": degrees, "encounter_rule": encounter_rule,
                        "conduct_rule": conduct_rule, "reasoning": reasoning}}


# ── 1. A_fabricated_risk re-scoping + E_unclassified_encounter ───────────────
def test_imazu01_t0_no_a_but_unclassified_encounter() -> None:
    """CPA 0, TCPA 1800 (beyond the 300s horizon), model cited none/Rule 8 -> no
    A_fabricated_risk, but E_unclassified_encounter (a real, if distant, collision
    course the model failed to recognize as any encounter at all)."""
    situation = [contact("ts1", cpa_m=0.0, tcpa_s=1800.0)]
    cp = make_cp(0, "...", "hold_course", None, "none", "Rule 8", "No immediate risk detected.")
    codes = [f["code"] for f in ar.check_2_4_direction(cp, situation, CONSTRAINTS)]
    assert "A_fabricated_risk" not in codes
    e_codes = [f["code"] for f in ar.check_2_1b_unclassified_encounter(cp, situation, CONSTRAINTS)]
    assert "E_unclassified_encounter" in e_codes


def test_quiet_mission_with_rule_cited_still_fires_a() -> None:
    situation = [contact("ts1", cpa_m=9999.0, tcpa_s=100.0)]
    cp = make_cp(0, "...", "turn_right", 10.0, "Rule 15", "Rule 16", "Giving way to ts1.")
    codes = [f["code"] for f in ar.check_2_4_direction(cp, situation, CONSTRAINTS)]
    assert "A_fabricated_risk" in codes


def test_unclassified_encounter_does_not_fire_when_clear() -> None:
    situation = [contact("ts1", cpa_m=9999.0, tcpa_s=100.0)]
    cp = make_cp(0, "...", "hold_course", None, "none", "none", "No risk, holding course.")
    assert ar.check_2_1b_unclassified_encounter(cp, situation, CONSTRAINTS) == []


# ── 2. gate_b: number-fabrication false positives ─────────────────────────────
def test_gate_b_zero_flags_on_grounded_rounded_and_converted_numbers() -> None:
    report = ('Own-ship at (0.000, 0.000) NM, heading 0.0, speed 10.00 kt.\n'
             '1 other ship:\n'
             '  - Ship named "ts1": range 1.234 NM, rel.bearing 10.0 deg, heading 0.0, '
             'speed 6.80 kt, CPA 1.234 NM, TCPA 100s')
    reasoning = ("Contact ts1: CPA 1.2 NM (2200 m), Rule 15 applies, closing at 6.8 kt, "
                "well inside the 500 m safe distance.")
    cp = make_cp(0, report, "turn_right", 10.0, "Rule 15", "Rule 16", reasoning)
    situation = [contact("ts1", cpa_m=1.234 * 1852.0, tcpa_s=100.0)]
    codes = [f["code"] for f in ar.check_2_7_reasoning_vs_decision(cp, situation, CONSTRAINTS)]
    assert "G_gate_b_number_fabricated" not in codes


def test_gate_b_flags_a_genuinely_fabricated_number() -> None:
    report = ('Own-ship at (0.000, 0.000) NM, heading 0.0, speed 10.00 kt.\n'
             '1 other ship:\n'
             '  - Ship named "ts1": range 0.486 NM, rel.bearing 10.0 deg, heading 0.0, '
             'speed 6.80 kt, CPA 0.486 NM, TCPA 100s')
    reasoning = "Contact ts1 has a CPA of 350 m, well inside the safe distance."
    cp = make_cp(0, report, "turn_right", 10.0, "Rule 15", "Rule 16", reasoning)
    situation = [contact("ts1", cpa_m=0.486 * 1852.0, tcpa_s=100.0)]
    findings = ar.check_2_7_reasoning_vs_decision(cp, situation, CONSTRAINTS)
    codes = [f["code"] for f in findings]
    assert "G_gate_b_number_fabricated" in codes
    detail = next(f for f in findings if f["code"] == "G_gate_b_number_fabricated")["details"]
    assert 350.0 in detail["numbers"]


# ── 3. gate_c: risk-conclusion extractor ──────────────────────────────────────
def test_gate_c_zero_mismatches_on_quiet_no_risk() -> None:
    situation = [contact("ts1", cpa_m=9999.0, tcpa_s=100.0)]
    cp = make_cp(0, "...", "hold_course", None, "none", "none",
                "No risk from any contact; holding course.")
    codes = [f["code"] for f in ar.check_2_7_reasoning_vs_decision(cp, situation, CONSTRAINTS)]
    assert "G_gate_c_risk_mismatch" not in codes


def test_gate_c_no_mismatch_on_early_action_collision_course() -> None:
    situation = [contact("ts1", cpa_m=0.0, tcpa_s=1800.0)]
    cp = make_cp(0, "...", "hold_course", None, "none", "Rule 8",
                "This is a collision course requiring early action.")
    codes = [f["code"] for f in ar.check_2_7_reasoning_vs_decision(cp, situation, CONSTRAINTS)]
    assert "G_gate_c_risk_mismatch" not in codes


def test_gate_c_flags_a_real_mismatch() -> None:
    situation = [contact("ts1", cpa_m=100.0, tcpa_s=100.0)]  # real risk
    cp = make_cp(0, "...", "hold_course", None, "none", "none", "No risk at all, all clear.")
    codes = [f["code"] for f in ar.check_2_7_reasoning_vs_decision(cp, situation, CONSTRAINTS)]
    assert "G_gate_c_risk_mismatch" in codes
