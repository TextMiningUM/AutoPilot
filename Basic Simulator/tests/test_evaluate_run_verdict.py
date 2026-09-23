"""Regression test for the 2026-09-23 verdict-propagation fix in
"Evaluation Functions/evaluate_run.py": a run that reaches the goal without a literal
collision, but whose minimum separation to a target dips below the mission's own
safe_distance_m, must never be silently labelled a bare "PASS" -- see
_analysis/audit_runs.py's BLOCKER_1_5_verdict_inconsistent check, which this fix
satisfies for future runs (already-generated runs keep their old, now-known-wrong
"PASS" and are still correctly flagged by that check)."""
import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

_EVAL_RUN_PATH = APP_ROOT / "Evaluation Functions" / "evaluate_run.py"
_spec = importlib.util.spec_from_file_location("evaluate_run", _EVAL_RUN_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
evaluate_run = _mod.evaluate_run

SAFE_DISTANCE_M = 500.0


def _write_csv(path: Path, min_sep_m: float) -> None:
    """own_ship travels straight along y=0 from x=0 to x=1000 (reaching the goal); a
    stationary target sits at (500, min_sep_m) so the closest approach is exactly
    min_sep_m -- well clear of the 15m collision radius."""
    rows = [{"time": t, "vehicle": "own_ship", "x": t * 10.0, "y": 0.0,
            "heading": 90.0, "speed": 10.0} for t in range(0, 101, 10)]
    rows += [{"time": t, "vehicle": "ts1", "x": 500.0, "y": min_sep_m,
             "heading": 0.0, "speed": 0.0} for t in range(0, 101, 10)]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "vehicle", "x", "y", "heading", "speed"])
        w.writeheader()
        w.writerows(rows)


def test_verdict_flags_cpa_violation_short_of_a_collision() -> None:
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "run.csv"
        _write_csv(csv_path, min_sep_m=380.0)  # < 500 safe distance, > 15 collision radius
        result = evaluate_run(csv_path, "own_ship", start_xy=(0.0, 0.0), goal_xy=(1000.0, 0.0),
                              nominal_speed=10.0, safe_distance_m=SAFE_DISTANCE_M, verbose=False)
        assert result["safety"]["passed"] is True  # not a literal collision
        assert result["verdict"] == "PASS_WITH_CPA_VIOLATION"


def test_verdict_is_plain_pass_when_clear_of_the_safe_distance() -> None:
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "run.csv"
        _write_csv(csv_path, min_sep_m=600.0)  # >= 500 safe distance
        result = evaluate_run(csv_path, "own_ship", start_xy=(0.0, 0.0), goal_xy=(1000.0, 0.0),
                              nominal_speed=10.0, safe_distance_m=SAFE_DISTANCE_M, verbose=False)
        assert result["verdict"] == "PASS"


def test_verdict_still_fail_on_a_literal_collision() -> None:
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "run.csv"
        _write_csv(csv_path, min_sep_m=5.0)  # < 15 collision radius
        result = evaluate_run(csv_path, "own_ship", start_xy=(0.0, 0.0), goal_xy=(1000.0, 0.0),
                              nominal_speed=10.0, safe_distance_m=SAFE_DISTANCE_M, verbose=False)
        assert result["verdict"] == "FAIL -- collision occurred"
