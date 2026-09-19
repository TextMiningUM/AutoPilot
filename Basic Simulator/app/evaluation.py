"""Thin wrapper around Evaluation Functions/evaluate_run.py (folder name has a
space, so it isn't a plain importable package -- load it by file path instead).
Reused UNCHANGED: writes the in-memory trajectory to a temp CSV in the exact
schema evaluate_run.py already expects (time,vehicle,x,y,heading,speed)."""
from __future__ import annotations
import csv
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent               # Basic Simulator/
WORKSPACE_ROOT = ROOT.parent        # Auto Pilot/ (.env lives here)
EVAL_RUN_PATH = ROOT / "Evaluation Functions" / "evaluate_run.py"

_spec = importlib.util.spec_from_file_location("evaluate_run", EVAL_RUN_PATH)
_evaluate_run_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluate_run_mod)
evaluate_run = _evaluate_run_mod.evaluate_run


def score_trajectory(trajectory_rows: list[dict], start_xy: tuple[float, float],
                      goal_xy: tuple[float, float], nominal_speed: float,
                      own_vehicle: str = "own_ship", collision_radius_m: float = 15.0,
                      safe_distance_m: float = 50.0,
                      llm_violations: list[str] | None = None) -> dict:
    """trajectory_rows: list of {time, vehicle, x, y, heading, speed} dicts
    (Simulation.trajectory) -> evaluate_run.py's full result dict (verdict,
    composite_score, safety/compliance/temporal/spatial/manoeuvre breakdown).
    `llm_violations`, if given (from llm_compliance_check(), called separately
    and on demand -- see its docstring for why), feeds real COLREG violations
    into the compliance axis instead of the default always-empty check list."""
    violation_checks = ()
    if llm_violations:
        violation_checks = (lambda _own, _v=list(llm_violations): _v,)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["time", "vehicle", "x", "y", "heading", "speed"])
        writer.writeheader()
        writer.writerows(trajectory_rows)
        tmp_path = f.name
    try:
        return evaluate_run(
            tmp_path, own_vehicle=own_vehicle, start_xy=start_xy, goal_xy=goal_xy,
            nominal_speed=nominal_speed, collision_radius_m=collision_radius_m,
            safe_distance_m=safe_distance_m, violation_checks=violation_checks, verbose=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


LLM_COMPLIANCE_SYSTEM = """You are a COLREG compliance auditor reviewing a completed vessel \
trajectory. You are given the full time-series track (time, vehicle, x, y, heading, speed) for \
own-ship and every target vessel it encountered, in metres and degrees (heading 0=north, \
clockwise, matching compass bearings). Determine whether own-ship's manoeuvres complied with the \
International Regulations for Preventing Collisions at Sea (COLREG): the correct give-way/ \
stand-on role for each encounter type (head-on, crossing, overtaking), early and substantial \
action by the give-way vessel (normally to starboard), never altering to port toward a vessel on \
own-ship's own port side, and the stand-on vessel holding course/speed unless it became clearly \
necessary to act. \
Reply with ONLY a JSON object, no other text:
{"violations": ["<rule number + one-sentence description of what went wrong>", ...]}
If own-ship's manoeuvres were fully compliant, return an empty violations list."""


def _format_trajectory_csv(trajectory_rows: list[dict]) -> str:
    lines = ["time,vehicle,x,y,heading,speed"]
    for r in sorted(trajectory_rows, key=lambda r: (r["time"], r["vehicle"])):
        lines.append(f"{r['time']:.0f},{r['vehicle']},{r['x']:.1f},{r['y']:.1f},"
                     f"{r['heading']:.1f},{r['speed']:.2f}")
    return "\n".join(lines)


def llm_compliance_check(trajectory_rows: list[dict], own_vehicle: str = "own_ship",
                         model: str = "claude-sonnet-4-5") -> list[str]:
    """One-shot LLM judge of full-trajectory COLREG compliance, using Anthropic Claude.

    Deliberately NOT called during live stepping -- it's a single network round-trip
    (real latency), so it must only run on demand, once, after a run is complete (or
    paused), triggered by an explicit UI button. score_trajectory()'s normal local
    scoring never calls this -- compliance defaults to "no violations found" (score 1.0)
    until this is explicitly run and its result is passed back in as `llm_violations`.

    Returns a list of violation description strings (empty list = compliant)."""
    import anthropic
    from core.io import load_env

    load_env(WORKSPACE_ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env -- cannot run the LLM compliance check.")

    client = anthropic.Anthropic(api_key=key)
    user_msg = f"Trajectory (own_vehicle={own_vehicle}):\n\n{_format_trajectory_csv(trajectory_rows)}"
    resp = client.messages.create(
        model=model, max_tokens=800, system=LLM_COMPLIANCE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [f"[LLM compliance check -- could not parse response] {text[:200]}"]
    return [str(v) for v in parsed.get("violations", [])]
