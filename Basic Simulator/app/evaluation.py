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
COMPLIANCE_LABELS = _evaluate_run_mod.COMPLIANCE_LABELS


def compliance_finding_parts(entry) -> tuple[str, str, object, float]:
    """(code, label, at, deduction) for one compliance.breakdown entry -- entry is either
    the current dict shape ({"code","label","at","deduction"}, 2026-09-24 on) or an older
    bare [code, step_or_detail, deduction] list/tuple from a run generated before that
    (never migrated, see evaluate_run.py's COMPLIANCE_LABELS docstring) -- old entries get
    their label looked up here, at display time, from the same COMPLIANCE_LABELS dict."""
    if isinstance(entry, dict):
        return entry["code"], entry.get("label", entry["code"]), entry.get("at"), entry["deduction"]
    code, at, deduction = entry
    return code, COMPLIANCE_LABELS.get(code, code), at, deduction


def _auditor_codes_at_checkpoint(decision: dict, ground_truth: dict) -> list[str]:
    """Compliance-rebuild STAP 3 (2026-09-23): per-checkpoint auditor codes, computed
    against _ground_truth_at_checkpoint()'s STAP-2 dict (never real_risk() alone).
    `decision` is the agent's own self-reported {"action","encounter_rule","conduct_rule"}.

      E_role_fabrication:       a rule cited with no decisive contact at all (band
                                 safe/passed for every contact) -- STAP-2-band-aware
                                 counterpart to measurement.py Check A's CPA-only gate.
      E_encounter_mismatch:     a rule WAS cited but doesn't match the expected one for
                                 the decisive contact's geometry.
      E_unclassified_encounter: NO rule cited even though the decisive contact's geometry
                                 expects one (failed to recognise a real encounter).
      B_17c:                    own-ship is the STAND-ON vessel and acted (anything other
                                 than hold_course) before the encounter was even "acute" --
                                 Rule 17(a)(i) requires holding course/speed until it is
                                 acute (17(a)(ii)); early action here is itself the error,
                                 independent of which direction was chosen.
      E_8c:                     conduct_rule "Rule 8" cited for neither a genuine give-way
                                 emergency stop (action=="stop") nor a stationary/non-
                                 vessel encounter -- a fabricated/misapplied Rule 8 citation.
      P_port_toward_contact:    turn_left in band early/acute, decisive contact on own's
                                 own port side (rel_bearing_deg < 0), under Rule 14/15.
    """
    codes: list[str] = []
    encounter_rule = decision.get("encounter_rule") or decision.get("rule_applied") or "none"
    conduct_rule = decision.get("conduct_rule") or decision.get("rule_applied") or "none"
    action = decision.get("action")

    contacts = {c["contact"]: c for c in ground_truth.get("contacts", [])}
    decisive_name = ground_truth.get("decisive_contact")
    decisive = contacts.get(decisive_name) if decisive_name else None

    if decisive is None:
        if encounter_rule != "none" or conduct_rule != "none":
            codes.append("E_role_fabrication")
        return codes

    band = decisive["band"]
    exp_encounter_rule = decisive["expected_encounter_rule"]
    exp_direction = decisive["expected_direction"]

    if encounter_rule == "none":
        if exp_encounter_rule != "none":
            codes.append("E_unclassified_encounter")
    elif encounter_rule != exp_encounter_rule:
        codes.append("E_encounter_mismatch")

    if decisive["own_role"] == "stand_on" and band == "early" and action != "hold_course":
        codes.append("B_17c")

    if conduct_rule == "Rule 8" and action != "stop" and decisive["encounter"] != "stationary":
        codes.append("E_8c")

    if (action == "turn_left" and band in ("early", "acute")
            and decisive["rel_bearing_deg"] < 0 and exp_encounter_rule in ("Rule 14", "Rule 15")):
        codes.append("P_port_toward_contact")

    return codes


def _check_wrong_side_pass(trajectory_rows: list[dict], own_vehicle: str) -> list[tuple[str, str]]:
    """Compliance-rebuild STAP 3 (2026-09-23): run-level P_wrong_side_pass. For each
    contact, finds its closest-approach instant in the REALIZED trajectory (not a single
    instant's projected CPA) and checks the pass-side convention for whichever encounter
    type classify_encounter() reports THERE:
      - head_on (Rule 14): both vessels alter to starboard -> must end port-to-port, i.e.
        the contact must be on own's own PORT side (rel_bearing < 0) at closest approach.
      - crossing, own give-way (Rule 15, contact on own's starboard side): own must pass
        BEHIND the contact -- projecting the contact's position onto own's own course
        vector at closest approach must be non-positive (not still ahead along-track)."""
    from pipeline.oow_agent_spec import classify_encounter

    by_vehicle: dict[str, list[dict]] = {}
    for row in trajectory_rows:
        by_vehicle.setdefault(row["vehicle"], []).append(row)
    own_rows = sorted(by_vehicle.get(own_vehicle, []), key=lambda r: r["time"])
    if not own_rows:
        return []

    findings: list[tuple[str, str]] = []
    for vname, rows in by_vehicle.items():
        if vname == own_vehicle:
            continue
        best_own, best_tgt, best_d = None, None, float("inf")
        for tgt_row in rows:
            nearest_own = min(own_rows, key=lambda o: abs(o["time"] - tgt_row["time"]))
            d = math.hypot(nearest_own["x"] - tgt_row["x"], nearest_own["y"] - tgt_row["y"])
            if d < best_d:
                best_d, best_own, best_tgt = d, nearest_own, tgt_row
        if best_own is None:
            continue
        enc, _, rel = classify_encounter(best_own["x"], best_own["y"], best_own["heading"],
                                         best_tgt["x"], best_tgt["y"], best_tgt["heading"])
        if enc == "head_on":
            if rel >= 0:  # contact ended up on own's STARBOARD side at closest approach
                findings.append(("P_wrong_side_pass", vname))
        elif enc == "crossing_target_on_starboard":  # own is the give-way vessel
            oh = math.radians(best_own["heading"])
            course_x, course_y = math.sin(oh), math.cos(oh)
            dx, dy = best_tgt["x"] - best_own["x"], best_tgt["y"] - best_own["y"]
            if dx * course_x + dy * course_y > 0:  # contact still ahead along own's track
                findings.append(("P_wrong_side_pass", vname))
    return findings


def _compliance_findings(trajectory_rows: list[dict], checkpoints: list[dict] | None,
                         own_vehicle: str, safe_distance_m: float, max_turn_deg: float) -> list[dict]:
    """One entry per (checkpoint, code) occurrence -- the single source both
    score_trajectory()'s deterministic score AND llm_compliance_check()'s STAP-4
    plain-language explanations are built from, so neither can silently diverge from the
    other. Each finding: {"code", "step" (time), "situation_report" (rendered STAP-2
    ground truth text), "decision" (own-ship's own self-reported action/rules), "band",
    "expected_encounter_rule", "expected_conduct_rule", "expected_direction"}."""
    from app.measurement import measure_decision_quality
    from app.narrate import cpa_tcpa
    from app.simulation import VesselConstraints

    constraints = VesselConstraints(min_cpa_m=safe_distance_m, max_rudder_angle_deg=max_turn_deg)
    scored_measurement_codes = ("A_fabricated_risk", "B_wrong_direction",
                               "C_degrees_over_limit", "D_no_action_when_required")
    findings: list[dict] = []
    for cp in (checkpoints or []):
        t = cp.get("time", 0)
        decision = cp.get("decision") or {}
        gt = _ground_truth_at_checkpoint(trajectory_rows, t, own_vehicle, safe_distance_m, max_turn_deg)
        rows = _rows_at_time(trajectory_rows, t)
        own_row = rows.get(own_vehicle)
        situation = []
        if own_row:
            for vname, row in rows.items():
                if vname == own_vehicle:
                    continue
                cpa, tcpa = cpa_tcpa(own_row["x"], own_row["y"], own_row["heading"], own_row["speed"],
                                     row["x"], row["y"], row["heading"], row["speed"])
                situation.append({"cpa_m": cpa, "tcpa_s": tcpa})
        measured = measure_decision_quality(decision, situation, constraints, ground_truth=gt)
        codes = [c for c in measured["checks_fired"] if c in scored_measurement_codes]
        codes.extend(_auditor_codes_at_checkpoint(decision, gt))
        if not codes:
            continue

        contacts = {c["contact"]: c for c in gt.get("contacts", [])}
        decisive = contacts.get(gt.get("decisive_contact")) if gt.get("decisive_contact") else None
        situation_text = _render_ground_truth_text(gt)
        for code in codes:
            findings.append({
                "code": code, "step": t, "situation_report": situation_text, "decision": decision,
                "band": decisive["band"] if decisive else "safe",
                "expected_encounter_rule": decisive["expected_encounter_rule"] if decisive else "none",
                "expected_conduct_rule": decisive["expected_conduct_rule"] if decisive else "none",
                "expected_direction": decisive["expected_direction"] if decisive else "none",
            })
    return findings


def score_trajectory(trajectory_rows: list[dict], start_xy: tuple[float, float],
                      goal_xy: tuple[float, float], nominal_speed: float,
                      own_vehicle: str = "own_ship", collision_radius_m: float = 15.0,
                      safe_distance_m: float = 50.0, max_turn_deg: float = 30.0,
                      reached_radius_m: float = GOAL_RADIUS_M,
                      checkpoints: list[dict] | None = None) -> dict:
    """trajectory_rows: list of {time, vehicle, x, y, heading, speed} dicts
    (Simulation.trajectory) -> evaluate_run.py's full result dict (verdict,
    composite_score, safety/compliance/temporal/spatial/manoeuvre breakdown).

    Compliance-rebuild STAP 3 (2026-09-23): compliance is now ALWAYS a deterministic
    score computed from `checkpoints` (own-ship's own self-reported decisions -- see
    run_llm_scenario.py's checkpoint-building loop), never an LLM audit result -- there is
    no more "unaudited, defaults to 0.0" state. `safe_distance_m`/`max_turn_deg` MUST be
    the run's own VesselConstraints values, feeding both measurement.py's checks and
    _ground_truth_at_checkpoint()'s STAP-2 bands. The result's "compliance"."findings" is
    the SAME list llm_compliance_check() (STAP 4) can turn into plain-language
    explanations, on demand -- never recomputed a second, differently, way."""
    findings = _compliance_findings(trajectory_rows, checkpoints, own_vehicle, safe_distance_m, max_turn_deg)
    codes_by_step: dict[float, list[str]] = {}
    for f in findings:
        codes_by_step.setdefault(f["step"], []).append(f["code"])
    checkpoint_codes = list(codes_by_step.items())
    run_level_codes = _check_wrong_side_pass(trajectory_rows, own_vehicle)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["time", "vehicle", "x", "y", "heading", "speed"])
        writer.writeheader()
        writer.writerows(trajectory_rows)
        tmp_path = f.name
    try:
        result = evaluate_run(
            tmp_path, own_vehicle=own_vehicle, start_xy=start_xy, goal_xy=goal_xy,
            nominal_speed=nominal_speed, collision_radius_m=collision_radius_m,
            safe_distance_m=safe_distance_m, reached_radius_m=reached_radius_m,
            checkpoint_codes=checkpoint_codes, run_level_codes=run_level_codes, verbose=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    result["compliance"]["findings"] = findings
    return result


LLM_COMPLIANCE_SYSTEM = """You write plain-language explanations for COLREG compliance findings \
that a deterministic checker has ALREADY detected -- you never judge, score, or re-derive \
whether a rule applied, whether a manoeuvre was correct, or whether risk of collision existed; \
all of that is given to you as an already-decided fact (each finding's CODE). Your only job is \
to turn each finding into one clear, human-readable sentence explaining WHY that code fired and \
what should have happened instead. \
You are given a list of findings, each with: a situation-report fragment (every contact's \
range/bearing/CPA/TCPA/band/expected rule for that instant), own-ship's own self-reported \
decision (action + cited encounter_rule/conduct_rule), the finding's band (safe/early/acute/ \
passed), and its code -- one of: \
A_fabricated_risk (cited a rule with no real risk), \
B_wrong_direction (turned toward the give-way-mandated wrong side), \
C_degrees_over_limit (requested a physically-impossible turn), \
D_no_action_when_required (held course despite an acute, imminent risk), \
E_encounter_mismatch (cited a rule that doesn't match the actual encounter geometry), \
E_role_fabrication (cited a rule when no real encounter existed at all), \
E_unclassified_encounter (cited no rule despite a real encounter existing), \
B_17c (a stand-on vessel acted before it was actually acute -- Rule 17(a)(i)), \
E_8c (cited Rule 8 without a genuine emergency stop or stationary-object encounter), \
P_port_toward_contact (turned to port toward a contact on own-ship's own port side). \
For each finding, write ONE explanation string covering, in this order, as one or two \
sentences: (1) WHEN it happened (the finding's step, in seconds), (2) WHAT own-ship actually \
did, (3) WHAT the code means here in plain language, and (4) WHAT should have happened instead \
(use the finding's own expected_encounter_rule/expected_conduct_rule/expected_direction -- \
never invent a different one). \
Reply with ONLY a JSON object, no other text -- no preamble, no analysis, no summary before or \
after it:
{"explanations": ["t=<seconds>s: <what own-ship did> -- <what the code means>; the \
COLREG-compliant action would have been <concrete correct manoeuvre>.", ...]}
If given an empty findings list, reply {"explanations": []}."""




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
    """Human-readable rendering of _ground_truth_at_checkpoint()'s dict -- used as each
    finding's "situation_report" fragment (see _compliance_findings()), the ONLY per-
    contact text the STAP-4 LLM explanation step ever sees, never the raw trajectory."""
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


def _format_findings_for_llm(findings: list[dict]) -> str:
    """Renders _compliance_findings()'s list as the user message llm_compliance_check()
    sends Claude -- compliance-rebuild STAP 4 (2026-09-23): no more raw trajectory CSV,
    just the already-decided findings themselves (situation/decision/band/expected
    rule-direction/code), since the LLM only ever explains a given finding now, never
    re-derives whether one applies."""
    lines = []
    for i, f in enumerate(findings, 1):
        d = f["decision"]
        lines.append(
            f"Finding {i} [{f['code']}] t={f['step']:.0f}s\n"
            f"  Situation: {f['situation_report']}\n"
            f"  Own-ship decision: action={d.get('action', '?')}, "
            f"encounter_rule={d.get('encounter_rule', 'none')}, conduct_rule={d.get('conduct_rule', 'none')}\n"
            f"  Band: {f['band']}; expected encounter_rule={f['expected_encounter_rule']}, "
            f"conduct_rule={f['expected_conduct_rule']}, direction={f['expected_direction']}"
        )
    return "\n\n".join(lines)


def llm_compliance_check(findings: list[dict], model: str = "claude-sonnet-4-5") -> dict:
    """Compliance-rebuild STAP 4 (2026-09-23): plain-language EXPLANATION of findings the
    deterministic compliance_axis() has already scored -- the LLM never judges, scores, or
    re-derives anything anymore (see LLM_COMPLIANCE_SYSTEM). `findings` is
    score_trajectory()'s own `result["compliance"]["findings"]` (from
    _compliance_findings()) -- never the raw trajectory.

    Deliberately NOT called during live stepping or by default in a sweep -- it's a single
    network round-trip (real latency + cost) that no longer affects any score, so it's
    opt-in only (run_llm_scenario.py's `--explain` flag, default off). An empty findings
    list needs no explanation at all -- returned immediately, no API call.

    Returns {"explanations": [str, ...]} -- one string per finding, in the same when/what/
    why/what-should-have-happened form the old audit used, now for an already-decided
    finding rather than a judgement call."""
    if not findings:
        return {"explanations": []}

    import anthropic
    from core.io import load_env

    load_env(WORKSPACE_ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env -- cannot run the LLM explanation.")

    client = anthropic.Anthropic(api_key=key)
    user_msg = _format_findings_for_llm(findings)
    resp = client.messages.create(
        model=model, max_tokens=2048, system=LLM_COMPLIANCE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    # Prefer the LAST complete {...} object that actually parses AND has an "explanations"
    # key (rather than requiring the ENTIRE reply to be pure JSON) -- see _extract_json_objects.
    for candidate in reversed(_extract_json_objects(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "explanations" in parsed:
            return {"explanations": [str(e) for e in parsed["explanations"]]}
    return {"explanations": [f"[LLM explanation -- could not parse response] {text[:200]}"]}
