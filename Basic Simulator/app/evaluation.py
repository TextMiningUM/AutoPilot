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
                      llm_violations: list[str] | None = None,
                      llm_compliance_score: float | None = None) -> dict:
    """trajectory_rows: list of {time, vehicle, x, y, heading, speed} dicts
    (Simulation.trajectory) -> evaluate_run.py's full result dict (verdict,
    composite_score, safety/compliance/temporal/spatial/manoeuvre breakdown).
    `llm_violations`, if given (from llm_compliance_check(), called separately and on
    demand -- see its docstring for why), is only used to surface violation TEXT in the
    result -- the compliance SCORE itself comes from `llm_compliance_score` (Claude's own
    0-1 audit judgement, same call). Compliance defaults to 0.0 (unaudited, not
    innocent-until-proven) until that on-demand check has actually been run and its
    result passed in here."""
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
            llm_compliance_score=llm_compliance_score,
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
If own-ship's own self-reported decisions (with each one's cited COLREG rule) are provided \
below the trajectory, cross-check EVERY citation against the actual encounter geometry in \
effect at that time -- a fabricated citation, the wrong rule number for that encounter type, or \
claiming 'none' when a rule clearly applied, is ITSELF a violation, even when the resulting \
manoeuvre happened to be independently safe: an accidentally-safe action reached through \
incorrect COLREG reasoning is not true compliance. Word this kind of violation as: 't=<seconds>\
s: own-ship cited Rule <n> (or "none") but the actual encounter required <correct rule or "no \
rule"> because <reason>.' A single isolated wrong-but-safe citation is a minor/technical \
shortcoming (anchor 0.75 below); citations that are wrong at MOST decision points are systemic \
non-compliance (anchor 0.0) even if every resulting action happened to be safe. \
Finally, give ONE overall compliance_score for the whole trajectory, a float from 0.0 to 1.0, \
using these anchors (pick the closest, or interpolate between two if the situation is a genuine \
in-between case) -- judge by SEVERITY AND CONSEQUENCE, not just by counting violations: \
1.0 = fully compliant, every applicable rule followed correctly, zero violations. \
0.75 = materially compliant with only a minor/technical shortcoming (e.g. a correct-direction \
manoeuvre that was slightly late or slightly less than "early and substantial"), but it never \
created a real close-quarters situation. \
0.5 = at least one genuine rule violation (wrong give-way response, or a prohibited port \
alteration) occurred, but it did NOT create an unsafe close-quarters situation -- a safe CPA was \
maintained throughout despite the improper manoeuvre. \
0.25 = one or more violations that DID create a real close-quarters/unsafe-CPA situation (a \
near-miss), though no actual collision occurred. \
0.0 = repeated or serious violations that directly caused (or were the proximate cause of) an \
actual collision, or such systematic non-compliance that own-ship's behaviour cannot be \
considered COLREG-aware at all. \
If violations is empty, compliance_score MUST be 1.0. If violations is non-empty, \
compliance_score MUST be less than 1.0, chosen using the anchors above. \
Reply with ONLY a JSON object, no other text -- no preamble, no analysis, no summary before or \
after it:
{"violations": ["t=<seconds>s: <what own-ship did> -- violates Rule <n> because <reason>; the \
COLREG-compliant action would have been <concrete correct manoeuvre>.", ...],
 "compliant_actions": ["t=<seconds>s: <what own-ship did> -- correctly satisfies Rule <n> \
because <reason>.", ...],
 "compliance_score": <float 0.0-1.0, see anchors above>}
If own-ship made no manoeuvres/encounters worth auditing at all, return both lists empty and \
compliance_score 1.0."""




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


def _format_checkpoint_citations(checkpoints: list[dict] | None) -> str:
    """own-ship's own self-reported action + COLREG rule citation at each decision point
    (app.agents.ask_oow's `rule_applied` field) -- without this, the audit only ever sees
    raw positions/headings and can't tell a genuinely rule-based manoeuvre from an
    accidentally-safe one reached via a fabricated/wrong rule citation."""
    if not checkpoints:
        return ""
    lines = ["\n\nOwn-ship's own self-reported decisions (verify each rule_applied citation "
            "against the actual encounter geometry above):"]
    for cp in checkpoints:
        decision = cp.get("decision") or {}
        lines.append(f"t={cp.get('time', 0):.0f}s: action={decision.get('action', '?')}, "
                     f"rule_applied={decision.get('rule_applied', 'none')}")
    return "\n".join(lines)


def llm_compliance_check(trajectory_rows: list[dict], own_vehicle: str = "own_ship",
                         model: str = "claude-sonnet-4-5",
                         checkpoints: list[dict] | None = None) -> dict:
    """One-shot LLM judge of full-trajectory COLREG compliance, using Anthropic Claude -- a
    full two-sided AUDIT (what was done wrong AND what was done right, each explained), not
    just a list of mistakes.

    Deliberately NOT called during live stepping -- it's a single network round-trip
    (real latency), so it must only run on demand, once, after a run is complete (or
    paused), triggered by an explicit UI button. score_trajectory()'s normal local
    scoring never calls this -- compliance defaults to 0.0 (unaudited, NOT
    innocent-until-proven) until this is explicitly run and its ["compliance_score"] is
    passed back in as `llm_compliance_score`.

    `checkpoints`, if given (run_llm_scenario.py's/a precomputed run log's own checkpoint
    list), lets the audit ALSO cross-check own-ship's SELF-REPORTED rule_applied citation at
    each decision against the actual geometry -- catching a fabricated/wrong-but-safe
    citation that pure trajectory geometry alone can't reveal. Optional: omitted for the
    live "Agent Real-Time" mode, which only tracks the single most recent decision.

    Returns {"violations": [...], "compliant_actions": [...], "compliance_score": float}
    (compliance_score is Claude's own 0.0-1.0 severity-weighted judgement, see
    LLM_COMPLIANCE_SYSTEM's anchors -- 1.0 only when violations is empty)."""
    import anthropic
    from core.io import load_env

    load_env(WORKSPACE_ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env -- cannot run the LLM compliance check.")

    client = anthropic.Anthropic(api_key=key)
    user_msg = (f"Trajectory (own_vehicle={own_vehicle}):\n\n{_format_trajectory_csv(trajectory_rows)}"
               f"{_format_checkpoint_citations(checkpoints)}")
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
            violations = [str(v) for v in parsed["violations"]]
            score = parsed.get("compliance_score")
            # Fall back to a safe binary reading (1.0/0.0) if Claude omitted the field or
            # returned something that isn't a plain number -- never silently treat an
            # un-scored response as perfect.
            if not isinstance(score, (int, float)):
                score = 1.0 if not violations else 0.0
            return {
                "violations": violations,
                "compliant_actions": [str(v) for v in parsed.get("compliant_actions", [])],
                "compliance_score": max(0.0, min(1.0, float(score))),
            }
    return {"violations": [f"[LLM compliance check -- could not parse response] {text[:200]}"],
           "compliant_actions": [], "compliance_score": 0.0}
