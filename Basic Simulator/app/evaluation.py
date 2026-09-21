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
clockwise, matching compass bearings). Audit EVERY course/speed change own-ship made, and every \
encounter (head-on, crossing, overtaking) it was involved in, against the International \
Regulations for Preventing Collisions at Sea (COLREG): the correct give-way/stand-on role for \
each encounter type, early and substantial action by the give-way vessel (normally to \
starboard), never altering to port toward a vessel on own-ship's own port side, and the \
stand-on vessel holding course/speed unless it became clearly necessary to act. \
This is a full audit, not just a list of mistakes -- classify EVERY manoeuvre/encounter you \
review as either a violation or correctly handled, and explain BOTH kinds fully so a reader \
understands the whole encounter without re-reading the raw trajectory themselves. \
Each violation string must cover, in this order, as one or two sentences: (1) WHEN it happened \
(approximate time in seconds), (2) WHAT own-ship actually did at that moment (heading/course \
change or lack of one, relative to the target(s) involved), (3) WHY that violates COLREG (name \
the rule number and the specific requirement it breaches), and (4) WHAT the COLREG-compliant \
manoeuvre would have been instead (concrete: which direction to turn, or to hold course/speed, \
and why that resolves the encounter correctly). \
Each compliant-action string must cover, in this order, as one or two sentences: (1) WHEN it \
happened, (2) WHAT own-ship did, (3) WHY that was the CORRECT thing to do under COLREG (name \
the rule number and the specific requirement it satisfies). \
Reply with ONLY a JSON object, no other text -- no preamble, no analysis, no summary before or \
after it:
{"violations": ["t=<seconds>s: <what own-ship did> -- violates Rule <n> because <reason>; the \
COLREG-compliant action would have been <concrete correct manoeuvre>.", ...],
 "compliant_actions": ["t=<seconds>s: <what own-ship did> -- correctly satisfies Rule <n> \
because <reason>.", ...]}
If own-ship made no manoeuvres/encounters worth auditing at all, return both as empty lists."""



def _extract_json_objects(text: str) -> list[str]:
    """Balanced-brace scan for every top-level {...} object in `text`, in order of
    appearance -- Claude's reply, despite LLM_COMPLIANCE_SYSTEM's "ONLY a JSON object"
    instruction, sometimes still prefixes it with a sentence or two of prose analysis
    (e.g. "Looking at this trajectory, I need to analyze..."); a naive whole-text
    json.loads() then fails outright even though a valid JSON object is sitting right
    there. Mirrors app.agents._extract_json_objects (duplicated, not imported, since this
    module must stay import-light -- app.agents pulls in torch/transformers)."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None
    return objs


def _format_trajectory_csv(trajectory_rows: list[dict]) -> str:
    lines = ["time,vehicle,x,y,heading,speed"]
    for r in sorted(trajectory_rows, key=lambda r: (r["time"], r["vehicle"])):
        lines.append(f"{r['time']:.0f},{r['vehicle']},{r['x']:.1f},{r['y']:.1f},"
                     f"{r['heading']:.1f},{r['speed']:.2f}")
    return "\n".join(lines)


def llm_compliance_check(trajectory_rows: list[dict], own_vehicle: str = "own_ship",
                         model: str = "claude-sonnet-4-5") -> dict:
    """One-shot LLM judge of full-trajectory COLREG compliance, using Anthropic Claude -- a
    full two-sided AUDIT (what was done wrong AND what was done right, each explained), not
    just a list of mistakes.

    Deliberately NOT called during live stepping -- it's a single network round-trip
    (real latency), so it must only run on demand, once, after a run is complete (or
    paused), triggered by an explicit UI button. score_trajectory()'s normal local
    scoring never calls this -- compliance defaults to "no violations found" (score 1.0)
    until this is explicitly run and its ["violations"] is passed back in as `llm_violations`.

    Returns {"violations": [...], "compliant_actions": [...]} (both lists of explanation
    strings; empty violations = fully compliant per Claude's audit)."""
    import anthropic
    from core.io import load_env

    load_env(WORKSPACE_ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env -- cannot run the LLM compliance check.")

    client = anthropic.Anthropic(api_key=key)
    user_msg = f"Trajectory (own_vehicle={own_vehicle}):\n\n{_format_trajectory_csv(trajectory_rows)}"
    resp = client.messages.create(
        # 3072 (not the old 800) -- every manoeuvre now gets a full when/what/why explanation
        # (violation OR compliant), not just a short sentence per mistake.
        model=model, max_tokens=3072, system=LLM_COMPLIANCE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    # Prefer the LAST complete {...} object that actually parses AND has a "violations" key
    # (rather than requiring the ENTIRE reply to be pure JSON) -- see _extract_json_objects.
    for candidate in reversed(_extract_json_objects(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "violations" in parsed:
            return {
                "violations": [str(v) for v in parsed["violations"]],
                "compliant_actions": [str(v) for v in parsed.get("compliant_actions", [])],
            }
    return {"violations": [f"[LLM compliance check -- could not parse response] {text[:200]}"],
           "compliant_actions": []}
