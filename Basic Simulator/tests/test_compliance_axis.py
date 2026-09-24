"""Regression tests for compliance-rebuild STAP 3 -- the deterministic compliance score
(Evaluation Functions/evaluate_run.py's compliance_axis()) and its run-level
P_wrong_side_pass check (app/evaluation.py's _check_wrong_side_pass()).
"""
import importlib.util
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

_EVAL_RUN_PATH = APP_ROOT / "Evaluation Functions" / "evaluate_run.py"
_spec = importlib.util.spec_from_file_location("evaluate_run", _EVAL_RUN_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compliance_axis = _mod.compliance_axis
explanation_axis = _mod.explanation_axis
COMPLIANCE_WEIGHTS = _mod.COMPLIANCE_WEIGHTS
COMPLIANCE_LABELS = _mod.COMPLIANCE_LABELS
COMPLIANCE_CATEGORY = _mod.COMPLIANCE_CATEGORY

from app.evaluation import _check_wrong_side_pass  # noqa: E402


def _row(vehicle: str, x: float, y: float, heading: float) -> dict:
    return {"time": 0.0, "vehicle": vehicle, "x": x, "y": y, "heading": heading, "speed": 5.0}


def test_wrong_side_pass_fires_on_starboard_to_starboard_head_on() -> None:
    """Head-on encounter where the contact ends up on own's own STARBOARD side at closest
    approach (rel_bearing +3deg, within classify_encounter()'s head-on window) -- the
    Rule-14 convention (both alter to starboard) requires a PORT-to-port passage, so this
    must fire."""
    rows = [_row("own_ship", 0.0, 0.0, 0.0), _row("ts1", 5.24, 100.0, 180.0)]
    findings = _check_wrong_side_pass(rows, "own_ship")
    assert ("P_wrong_side_pass", "ts1") in findings


def test_wrong_side_pass_does_not_fire_on_port_to_port_head_on() -> None:
    """Same head-on encounter, contact instead on own's own PORT side (rel_bearing
    -3deg) -- the correct, Rule-14-compliant pass side -- must NOT fire."""
    rows = [_row("own_ship", 0.0, 0.0, 0.0), _row("ts1", -5.24, 100.0, 180.0)]
    findings = _check_wrong_side_pass(rows, "own_ship")
    assert findings == []


def test_collision_gates_score_to_zero_regardless_of_findings() -> None:
    """collided=True must force 0.0 no matter what other codes are present, for BOTH axes."""
    checkpoint_codes = [(0.0, ["B_wrong_direction", "E_role_fabrication"])]
    run_level_codes = [("P_wrong_side_pass", "ts1"), ("cpa_violation", None)]
    score, breakdown = compliance_axis(checkpoint_codes, run_level_codes, collided=True)
    assert score == 0.0
    assert breakdown == [{"code": "collision", "label": "Collision", "at": None, "deduction": -1.0}]
    expl_score, expl_breakdown = explanation_axis(checkpoint_codes, run_level_codes, collided=True)
    assert expl_score == 0.0
    assert expl_breakdown == [{"code": "collision", "label": "Collision", "at": None, "deduction": -1.0}]


def test_breakdown_sums_to_the_score() -> None:
    """Every deduction in the breakdown must account for the full drop from 1.0."""
    checkpoint_codes = [(0.0, ["B_wrong_direction"]), (10.0, ["A_fabricated_risk", "C_degrees_over_limit"])]
    run_level_codes = [("P_wrong_side_pass", "ts1")]
    score, breakdown = compliance_axis(checkpoint_codes, run_level_codes, collided=False)
    total_deduction = sum(f["deduction"] for f in breakdown)
    assert round(1.0 + total_deduction, 6) == round(score, 6)
    # A_fabricated_risk is an "explanation"-category code -- compliance_axis() (manoeuvre
    # only) must not count it; only B_wrong_direction/C_degrees_over_limit/P_wrong_side_pass.
    assert len(breakdown) == 3
    assert {f["code"] for f in breakdown} == {"B_wrong_direction", "C_degrees_over_limit", "P_wrong_side_pass"}


def test_explanation_axis_scores_only_explanation_codes() -> None:
    """explanation_axis() is compliance_axis()'s counterpart -- same input shape, but only
    deducts "explanation"-category codes (citation/reasoning accuracy), ignoring manoeuvre
    codes entirely."""
    checkpoint_codes = [(0.0, ["B_wrong_direction"]), (10.0, ["A_fabricated_risk", "C_degrees_over_limit"])]
    run_level_codes = [("P_wrong_side_pass", "ts1")]
    score, breakdown = explanation_axis(checkpoint_codes, run_level_codes, collided=False)
    assert len(breakdown) == 1
    assert breakdown[0]["code"] == "A_fabricated_risk"
    assert round(score, 6) == round(1.0 - COMPLIANCE_WEIGHTS["A_fabricated_risk"], 6)


def test_every_weighted_code_has_a_category() -> None:
    for code in COMPLIANCE_WEIGHTS:
        assert code in COMPLIANCE_CATEGORY, f"{code} is weighted but has no COMPLIANCE_CATEGORY entry"
        assert COMPLIANCE_CATEGORY[code] in ("manoeuvre", "explanation")


def test_breakdown_entries_carry_a_human_label() -> None:
    _, breakdown = compliance_axis([(0.0, ["B_wrong_direction"])], [], collided=False)
    assert breakdown == [{"code": "B_wrong_direction", "label": "Wrong Turn Direction",
                         "at": 0.0, "deduction": -0.15}]


def test_every_weighted_code_has_a_label() -> None:
    for code in COMPLIANCE_WEIGHTS:
        assert code in COMPLIANCE_LABELS, f"{code} is weighted but has no COMPLIANCE_LABELS entry"
    assert "collision" in COMPLIANCE_LABELS  # the hard-gate code, not itself weighted


def test_score_clips_at_zero_never_negative() -> None:
    """Enough deductions to overshoot 1.0 in magnitude must clip to 0.0, not go negative."""
    checkpoint_codes = [(t, ["B_wrong_direction", "D_no_action_when_required", "E_role_fabrication"])
                        for t in range(10)]
    score, _ = compliance_axis(checkpoint_codes, [], collided=False)
    assert score == 0.0
