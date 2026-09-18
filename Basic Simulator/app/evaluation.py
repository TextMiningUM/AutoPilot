"""Thin wrapper around Evaluation Functions/evaluate_run.py (folder name has a
space, so it isn't a plain importable package -- load it by file path instead).
Reused UNCHANGED: writes the in-memory trajectory to a temp CSV in the exact
schema evaluate_run.py already expects (time,vehicle,x,y,heading,speed)."""
from __future__ import annotations
import csv
import importlib.util
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
EVAL_RUN_PATH = ROOT / "Evaluation Functions" / "evaluate_run.py"

_spec = importlib.util.spec_from_file_location("evaluate_run", EVAL_RUN_PATH)
_evaluate_run_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluate_run_mod)
evaluate_run = _evaluate_run_mod.evaluate_run


def score_trajectory(trajectory_rows: list[dict], start_xy: tuple[float, float],
                      goal_xy: tuple[float, float], nominal_speed: float,
                      own_vehicle: str = "own_ship", collision_radius_m: float = 15.0,
                      safe_distance_m: float = 50.0) -> dict:
    """trajectory_rows: list of {time, vehicle, x, y, heading, speed} dicts
    (Simulation.trajectory) -> evaluate_run.py's full result dict (verdict,
    composite_score, safety/compliance/temporal/spatial/manoeuvre breakdown)."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["time", "vehicle", "x", "y", "heading", "speed"])
        writer.writeheader()
        writer.writerows(trajectory_rows)
        tmp_path = f.name
    try:
        return evaluate_run(
            tmp_path, own_vehicle=own_vehicle, start_xy=start_xy, goal_xy=goal_xy,
            nominal_speed=nominal_speed, collision_radius_m=collision_radius_m,
            safe_distance_m=safe_distance_m, verbose=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
