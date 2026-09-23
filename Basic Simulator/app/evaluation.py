"""Thin wrapper around Evaluation Functions/evaluate_run.py (folder name has a
space, so it isn't a plain importable package -- load it by file path instead).
Reused UNCHANGED: writes the in-memory trajectory to a temp CSV in the exact
schema evaluate_run.py already expects (time,vehicle,x,y,heading,speed)."""
from __future__ import annotations
import csv
import importlib.util
import json
import math
import os
import re
import tempfile
from pathlib import Path

from app.simulation import GOAL_RADIUS_M

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
                      safe_distance_m: float = 50.0, reached_radius_m: float = GOAL_RADIUS_M,
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
            safe_distance_m=safe_distance_m, reached_radius_m=reached_radius_m,
            violation_checks=violation_checks, verbose=False,
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
If own-ship's own self-reported decisions are provided below the trajectory, each one is paired \
with a FIXED, pre-computed ground-truth encounter classification (deterministic CPA/TCPA + \
relative-bearing rule classification, including whether real risk of collision existed at all \
-- NOT your own judgement call). Treat that ground truth as established fact: do NOT \
re-derive or second-guess whether a rule applied or whether risk of collision existed -- only \
judge whether the agent's OWN citation/action matches the given ground truth. A citation that \
contradicts the ground truth (a fabricated citation, the wrong rule number, or claiming 'none' \
when the ground truth says a rule applied -- or the reverse: citing a rule when the ground \
truth says 'no rule applies (quiet)') is ITSELF a violation, even when the resulting manoeuvre \
happened to be independently safe: an accidentally-safe action reached through incorrect \
COLREG reasoning is not true compliance. Word this kind of violation as: 't=<seconds>s: \
own-ship cited Rule <n> (or "none") but the ground truth says <correct rule or "no rule \
applies"> because <reason from the ground truth>.' A single isolated wrong-but-safe citation is \
a minor/technical shortcoming (anchor 0.75 below); citations that contradict the ground truth \
at MOST decision points are systemic non-compliance (anchor 0.0) even if every resulting action \
happened to be safe. \
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


def _rows_at_time(trajectory_rows: list[dict], t: float, tol: float = 0.5) -> dict[str, dict]:
    """{vehicle: row} for whichever recorded trajectory instant is closest to `t` (simulation
    records at whole dt steps via round(self.t, 2); checkpoints store the unrounded sim.t at
    the same instant, so an exact-equality lookup can miss by float noise -- nearest-within-
    tolerance is robust to that without needing both call sites to agree on rounding)."""
    times = sorted({r["time"] for r in trajectory_rows}, key=lambda rt: abs(rt - t))
    if not times or abs(times[0] - t) > tol:
        return {}
    nearest = times[0]
    return {r["vehicle"]: r for r in trajectory_rows if r["time"] == nearest}


# Compliance-rebuild STAP 2 (2026-09-23): classify_encounter()'s raw return string ->
# (this dict's `encounter` name, own_role, pipeline.oow_agent_spec.classify_rules()'s role
# key) -- "stationary" is NOT one of classify_encounter()'s own return values (it has no
# notion of speed, only geometry), detected separately below from the contact's own
# recorded speed.
_ENCOUNTER_MAP = {
    "head_on": ("head_on", "both_give_way", "mutual"),
    "crossing_target_on_starboard": ("crossing_stbd", "give_way", "give_way"),
    "crossing_target_on_port": ("crossing_port", "stand_on", "stand_on"),
    "we_are_overtaking_target": ("we_overtake", "give_way", "overtaking_give_way"),
    "target_is_overtaking_us": ("overtaken", "stand_on", "overtaking_stand_on"),
}
_STATIONARY_SPEED_EPS = 0.05  # m/s -- matches classify_rules()'s "stationary" role

# Expected manoeuvre direction + forbidden actions per `encounter` name (STAP 3 checks these
# against the agent's actual action; STAP 2 only populates them). "speed_up" is forbidden in
# EVERY early/acute encounter regardless of type -- appended separately below, not listed here.
_ENCOUNTER_EXPECTATIONS = {
    "head_on":       {"expected_direction": "starboard", "forbidden": ["port_toward_contact"]},
    "crossing_stbd": {"expected_direction": "starboard", "forbidden": ["port_toward_contact", "cross_ahead"]},
    "crossing_port": {"expected_direction": "hold", "forbidden": ["port_toward_contact"]},
    "we_overtake":   {"expected_direction": "either", "forbidden": ["cross_ahead_close"]},
    "overtaken":     {"expected_direction": "hold", "forbidden": []},
    "stationary":    {"expected_direction": "away_from_contact", "forbidden": []},
}


def _contact_ground_truth(own: dict, vname: str, row: dict, safe_distance_m: float,
                          risk_horizon_s: float) -> dict:
    """One contact's structured ground truth (see _ground_truth_at_checkpoint's docstring
    for the band/expectation rules this implements)."""
    from app.narrate import cpa_tcpa
    from pipeline.oow_agent_spec import classify_encounter, classify_rules

    cpa, _ = cpa_tcpa(own["x"], own["y"], own["heading"], own["speed"],
                      row["x"], row["y"], row["heading"], row["speed"])
    # Signed/unclamped TCPA -- cpa_tcpa() itself always clamps to t>=0, so "already past
    # the closest point" can never be seen through its return value alone. Same dot-
    # product sign check narrate.contact_line() already duplicates for its own "closing"
    # flag, reused here so a genuinely diverging encounter can be banded "passed" instead
    # of being forced into "acute"/"early" forever.
    oh, th = math.radians(own["heading"]), math.radians(row["heading"])
    vox, voy = own["speed"] * math.sin(oh), own["speed"] * math.cos(oh)
    vtx, vty = row["speed"] * math.sin(th), row["speed"] * math.cos(th)
    dx, dy = row["x"] - own["x"], row["y"] - own["y"]
    dvx, dvy = vtx - vox, vty - voy
    rel_sq = dvx ** 2 + dvy ** 2
    tcpa = 0.0 if rel_sq < 1e-6 else -(dx * dvx + dy * dvy) / rel_sq
    enc_raw, _, rel = classify_encounter(own["x"], own["y"], own["heading"],
                                         row["x"], row["y"], row["heading"])

    if tcpa < 0:
        band = "passed"
    elif cpa >= safe_distance_m:
        band = "safe"
    elif tcpa >= risk_horizon_s:
        band = "early"
    else:
        band = "acute"

    if band in ("safe", "passed"):
        encounter, own_role = "none", "none"
        expected_encounter_rule, expected_conduct_rule = "none", "none"
        expected_direction, forbidden = "none", []
    else:
        if row["speed"] < _STATIONARY_SPEED_EPS:
            encounter, own_role, role_key = "stationary", "none", "stationary"
        else:
            encounter, own_role, role_key = _ENCOUNTER_MAP.get(enc_raw, ("none", "none", "none"))
        # "hold_course" (never "stop") -- the CANONICAL expected rule pair for this role,
        # independent of whatever action the agent actually took; classify_rules()'s own
        # action=="stop" special case (Rule 8) is a scoring LENIENCY, not a ground-truth fact.
        expected_encounter_rule, expected_conduct_rule = classify_rules(role_key, "hold_course")
        exp = _ENCOUNTER_EXPECTATIONS.get(encounter, {"expected_direction": "none", "forbidden": []})
        expected_direction = exp["expected_direction"]
        forbidden = [*exp["forbidden"], "speed_up"]

    return {
        "contact": vname, "cpa_m": cpa, "tcpa_s": tcpa, "rel_bearing_deg": rel,
        "band": band, "encounter": encounter, "own_role": own_role,
        "expected_encounter_rule": expected_encounter_rule,
        "expected_conduct_rule": expected_conduct_rule,
        "expected_direction": expected_direction, "forbidden": forbidden,
    }


def _ground_truth_at_checkpoint(trajectory_rows: list[dict], t: float, own_vehicle: str,
                                safe_distance_m: float, max_turn_deg: float) -> dict:
    """Deterministic encounter classification (app.narrate's own CPA/TCPA + relative-bearing
    rule classifier + pipeline.oow_agent_spec's real_risk()/derive_risk_horizon_s() -- the
    SAME machinery the live agent's own situation report/constraint line already rely on)
    computed independently of the LLM, at the exact instant a decision was made.
    `safe_distance_m`/`max_turn_deg` MUST be the run's own VesselConstraints values
    (min_cpa_m/max_rudder_angle_deg).

    Compliance-rebuild STAP 2 (2026-09-23): returns a STRUCTURED dict, not a string --
    STAP 1's plain real_risk() boolean (used as a single "quiet" gate) could not
    distinguish a certain-but-distant collision course (e.g. Imazu01 t=0: CPA 0, TCPA
    1800s -- a real head-on encounter, just not yet urgent) from genuinely no risk at all,
    since a fixed geometry-derived horizon (~567s for that mission) will always be smaller
    than a large enough TCPA. Splitting into THREE bands (safe/early/acute, plus "passed"
    for an already-closed encounter) fixes this: "early" still cites the geometrically
    correct encounter/rule (classify_encounter() works independently of timing) even when
    not yet "acute" -- only "safe" (CPA already outside safe_distance_m) and "passed" (TCPA
    already negative) report no rule at all. Each contact dict:
      {"contact", "cpa_m", "tcpa_s", "rel_bearing_deg", "band",
       "encounter" (head_on/crossing_stbd/crossing_port/we_overtake/overtaken/stationary/none),
       "own_role" (give_way/stand_on/both_give_way/none),
       "expected_encounter_rule", "expected_conduct_rule" (from oow_agent_spec.classify_rules,
       the SAME table the Track-2 labelers use), "expected_direction"
       (starboard/hold/either/away_from_contact/none), "forbidden" (list of action-name
       strings STAP 3 checks the agent's actual action against)}.
    The returned dict also carries "decisive_contact": the acute contact with the smallest
    CPA, or (if none acute) the early contact with the smallest CPA, or None.

    Without this, llm_compliance_check() only ever saw a raw trajectory CSV and had to
    freehand-judge from scratch whether a rule applied -- a genuinely non-deterministic
    judgement call for a borderline-distance encounter, confirmed to flip between identical
    repeat calls on the exact same scenario (see repo memory / session notes: q01, 4 configs,
    identical CPA=6322m, 3 of 4 calls disagreed on whether Rule 15 applied at all). Injecting
    this FIXED fact means the LLM only ever has to judge whether the agent's citation/action
    matches it, not re-derive "was there risk of collision" itself each time."""
    from pipeline.oow_agent_spec import derive_risk_horizon_s

    rows = _rows_at_time(trajectory_rows, t)
    own = rows.get(own_vehicle)
    if not own:
        return {"contacts": [], "decisive_contact": None}
    risk_horizon_s = derive_risk_horizon_s(safe_distance_m, max_turn_deg, own["speed"])
    contacts = [_contact_ground_truth(own, vname, row, safe_distance_m, risk_horizon_s)
               for vname, row in sorted(rows.items()) if vname != own_vehicle]
    acute = [c for c in contacts if c["band"] == "acute"]
    early = [c for c in contacts if c["band"] == "early"]
    decisive = min(acute, key=lambda c: c["cpa_m"]) if acute else (
        min(early, key=lambda c: c["cpa_m"]) if early else None)
    return {"contacts": contacts, "decisive_contact": decisive["contact"] if decisive else None}


def _render_ground_truth_text(gt: dict) -> str:
    """Human-readable rendering of _ground_truth_at_checkpoint()'s dict, for the LLM audit
    prompt (llm_compliance_check() still needs TEXT until STAP 4 removes the LLM's scoring
    role) -- kept as a thin, separate rendering step so the dict itself stays the one
    structured source of truth STAP 3's deterministic scoring will consume directly."""
    from app.units import m_to_nm

    if not gt["contacts"]:
        return "(no ground truth available -- no trajectory sample at this time)"
    parts = []
    for c in gt["contacts"]:
        if c["band"] == "safe":
            verdict = "no rule applies (safe -- CPA already outside the safe-passing distance)"
        elif c["band"] == "passed":
            verdict = "no rule applies (passed -- already past closest point of approach)"
        else:
            verdict = (f"{c['expected_encounter_rule']}/{c['expected_conduct_rule']} applies "
                      f"({c['encounter']}, band={c['band']}, expected direction={c['expected_direction']})")
        parts.append(f"{c['contact']} rel_bearing={c['rel_bearing_deg']:.0f}deg "
                    f"cpa={m_to_nm(c['cpa_m']):.3f}NM tcpa={c['tcpa_s']:.0f}s -> {verdict}")
    return "; ".join(parts) if parts else "(no other vessels)"


def _format_checkpoint_citations(checkpoints: list[dict] | None,
                                 trajectory_rows: list[dict] | None,
                                 own_vehicle: str, safe_distance_m: float, max_turn_deg: float) -> str:
    """own-ship's own self-reported action + COLREG rule citation at each decision point
    (app.agents.ask_oow's `encounter_rule`/`conduct_rule` fields), each paired with a DETERMINISTIC
    ground-truth encounter classification (see _ground_truth_at_checkpoint) computed the same way
    the rest of this project already does -- without this, the audit only ever saw raw positions/
    headings and had to freehand-judge from scratch whether a rule applied at all, which is
    non-deterministic for a borderline-distance encounter (see that function's docstring).
    `safe_distance_m`/`max_turn_deg` MUST be the run's own VesselConstraints values, threaded
    straight through to _ground_truth_at_checkpoint."""
    if not checkpoints:
        return ""
    lines = ["\n\nOwn-ship's own self-reported decisions, each paired with a FIXED, "
            "pre-computed ground-truth encounter classification -- treat the ground truth "
            "as established fact, do not re-derive or second-guess whether risk of collision "
            "existed; only judge whether the agent's citation/action matches it:"]
    for cp in checkpoints:
        decision = cp.get("decision") or {}
        t = cp.get("time", 0)
        ground_truth = (_render_ground_truth_text(_ground_truth_at_checkpoint(
                            trajectory_rows, t, own_vehicle, safe_distance_m, max_turn_deg))
                       if trajectory_rows else "(no trajectory provided)")
        lines.append(f"t={t:.0f}s: action={decision.get('action', '?')}, "
                     f"encounter_rule={decision.get('encounter_rule', 'none')}, "
                     f"conduct_rule={decision.get('conduct_rule', 'none')} | "
                     f"ground truth: {ground_truth}")
    return "\n".join(lines)


def llm_compliance_check(trajectory_rows: list[dict], own_vehicle: str = "own_ship",
                         model: str = "claude-sonnet-4-5",
                         checkpoints: list[dict] | None = None,
                         safe_distance_m: float = 500.0, max_turn_deg: float = 30.0) -> dict:
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
    list), lets the audit ALSO cross-check own-ship's SELF-REPORTED encounter_rule/conduct_rule
    citations at each decision against the actual geometry -- catching a fabricated/wrong-but-safe
    citation that pure trajectory geometry alone can't reveal. Optional: omitted for the
    live "Agent Real-Time" mode, which only tracks the single most recent decision.
    `safe_distance_m`/`max_turn_deg` should always be the run's own VesselConstraints values
    (min_cpa_m/max_rudder_angle_deg) -- only used for the checkpoint ground-truth text above,
    defaults here match VesselConstraints()'s own defaults for callers with no live run
    constraints available.

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
               f"{_format_checkpoint_citations(checkpoints, trajectory_rows, own_vehicle, safe_distance_m, max_turn_deg)}")
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
