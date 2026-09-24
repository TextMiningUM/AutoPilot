"""audit_runs.py — deterministic, read-only RUN-AUDITOR for precomputed OOW mission runs.

No LLM calls, no corrections, no changes to any run file -- only measures and reports.
Every error class this project has previously found by hand (stale situation_report,
a verdict hiding a safety violation, parse leaks, a fabricated stand-on status, wrong
encounter classification, zigzag, token manoeuvres) is caught here automatically, with
the SAME definitions every time, and the current data conventions (two-field decision
schema, real_risk() with a horizon, per-run variable limits, stationary contacts, GOAL
COURSE CHECK) are enforced at inference too, not just in training data.

REUSED, NEVER DUPLICATED
------------------------
- app.measurement.measure_decision_quality(): Checks A/B/C (called as-is; this module
  adds Check D and everything in sections 2.2-3.x around it, never a second copy).
- pipeline.oow_agent_spec: real_risk(), RISK_HORIZON_S, classify_encounter(),
  goal_course_action()/goal_course_check_line(), ACTIONS, validate_action_json(),
  classify_rules() (the encounter/conduct-rule mapping table).
- app.simulation.VesselConstraints, app.narrate.cpa_tcpa, app.units conversions.
- Evaluation Functions/evaluate_run.py's safety-gate STRUCTURE is mirrored (not
  imported -- that module works off a CSV trajectory format, this one off the run
  JSON's own embedded trajectory) for check 1.5's verdict-consistency recomputation.

USAGE
-----
    python audit_runs.py --tag units_v2 [--configs ...] [--missions ...] \
        --out-dir _analysis/audit/<tag>/ [--baseline-tag units_v1]

Exit code 1 if any BLOCKER fired anywhere in the audited set, 0 otherwise (ERROR/WARN
never affect the exit code -- only the report).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent
_SIM_DIR = _ANALYSIS_DIR.parent            # Basic Simulator/
_REPO_ROOT = _SIM_DIR.parent                # Auto Pilot/
for _p in (_SIM_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core import review_path, safe_write_jsonl
from app.llm_runs import RUNS_DIR, parse_run_filename
from app.measurement import measure_decision_quality
from app.narrate import cpa_tcpa
from app.simulation import VesselConstraints
from app.units import kn_to_mps, m_to_nm, nm_to_m
from pipeline.oow_agent_spec import (
    ACTIONS, RISK_HORIZON_S, STAND_ON_TCPA_S, bearing_and_range, classify_encounter,
    classify_rules, goal_course_action, goal_course_check_line, real_risk,
    relative_bearing, validate_action_json,
)

QUIET_MISSIONS = {"UM01", "UM02"}  # the canary: genuinely no real risk, ever
COLLISION_RADIUS_M = 15.0          # matches simulation.py / evaluate_run.py
GOAL_RADIUS_NM = 0.1               # matches simulation.py's GOAL_RADIUS_M

# Section 1.4: the training principle (never hand the model its own answer) must also
# hold at inference -- these words/phrases must never appear in a situation_report.
_LEAKED_LABEL_PATTERNS = [
    re.compile(p, re.I) for p in (
        r"applicable colreg rules", r"give-way", r"give way vessel", r"stand-on",
        r"stand on vessel", r"head-on", r"head on situation", r"crossing situation",
        r"overtaking situation", r"we are meeting", r"both vessels are give-way",
    )
]

_GIVE_WAY_TURN_CONDUCT_RULES = ("Rule 14", "Rule 16")
_GIVE_WAY_TURN_ENCOUNTER_RULES = ("Rule 14", "Rule 15")
_DELIBERATION_WORDS = re.compile(r"\bwait\b|\bactually\b|\blet me re|\bhmm\b", re.I)


# ─────────────────────────────────────────────────────────────────────────────
# Finding helper
# ─────────────────────────────────────────────────────────────────────────────
def _f(severity: str, code: str, step: int | None, message: str, **details) -> dict:
    return {"severity": severity, "code": code, "step": step, "message": message, "details": details}


# ─────────────────────────────────────────────────────────────────────────────
# Schema detection (first step of every run, per spec's SCHEMA-ROBUUSTHEID)
# ─────────────────────────────────────────────────────────────────────────────
def detect_schema(run: dict, path: Path) -> dict:
    name_parts = path.stem.split("__")
    checkpoints = run.get("checkpoints") or []
    first_decision = (checkpoints[0].get("decision") if checkpoints else {}) or {}
    two_field = "encounter_rule" in first_decision or "conduct_rule" in first_decision
    one_field_only = "rule_applied" in first_decision and not two_field
    return {
        "filename_parts": len(name_parts),
        "decision_schema": "one_field" if one_field_only else "two_field",
        "has_measurement": bool(checkpoints) and "measurement" in checkpoints[0],
        "has_reasoning_raw": bool(checkpoints) and "reasoning_raw" in checkpoints[0],
        "has_debug": bool(checkpoints) and "debug" in checkpoints[0],
        "is_legacy": one_field_only,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory helpers
# ─────────────────────────────────────────────────────────────────────────────
def traj_by_vehicle_time(run: dict) -> dict[str, dict[float, dict]]:
    out: dict[str, dict[float, dict]] = defaultdict(dict)
    for row in run.get("trajectory") or []:
        out[row["vehicle"]][round(row["time"], 2)] = row
    return out


def _nearest(series: dict[float, dict], t: float) -> dict | None:
    if not series:
        return None
    key = min(series, key=lambda k: abs(k - t))
    return series[key] if abs(key - t) < 1e-3 else series.get(round(t, 2)) or series[key]


# ─────────────────────────────────────────────────────────────────────────────
# Constraints extraction -- there is no dedicated JSON field for these (yet), only the
# free-text prompt embeds them (debug.user_msg); a per-run PARAMETER, never assumed to
# be 500m/30deg. Falls back to VesselConstraints() defaults (flagged INFO, not BLOCKER)
# for configs (e.g. bare_qwen) whose prompt never states them.
# ─────────────────────────────────────────────────────────────────────────────
_RE_SAFE_DIST = re.compile(r"safe passing distance is ([\d.]+)\s*NM\b")
_RE_MAX_RUDDER = re.compile(r"may request at most (\d+(?:\.\d+)?) degrees")
_RE_TURN_RATE = re.compile(r"heading changes at most (\d+(?:\.\d+)?) deg/s")
_RE_MAX_SPEED = re.compile(r"Speed is capped at ([\d.]+)\s*kt\b")
_RE_ACCEL = re.compile(r"\(~?([\d.]+)\s*kt/min up / ~?([\d.]+)\s*kt/min down\)")


def _extract_constraints_from_text(user_msg: str) -> dict | None:
    m_safe = _RE_SAFE_DIST.search(user_msg)
    m_rudder = _RE_MAX_RUDDER.search(user_msg)
    m_rate = _RE_TURN_RATE.search(user_msg)
    if not (m_safe and m_rudder and m_rate):
        return None
    m_speed = _RE_MAX_SPEED.search(user_msg)
    m_accel = _RE_ACCEL.search(user_msg)
    return {
        "min_cpa_m": nm_to_m(float(m_safe.group(1))),
        "max_rudder_angle_deg": float(m_rudder.group(1)),
        "turn_rate_deg_s": float(m_rate.group(1)),
        "max_speed_mps": kn_to_mps(float(m_speed.group(1))) if m_speed else None,
        "max_acceleration_mps2": kn_to_mps(float(m_accel.group(1))) / 60.0 if m_accel else None,
        "max_deceleration_mps2": kn_to_mps(float(m_accel.group(2))) / 60.0 if m_accel else None,
    }


def extract_constraints(run: dict) -> tuple[VesselConstraints, list[dict]]:
    findings: list[dict] = []
    per_checkpoint: list[dict | None] = []
    for cp in run.get("checkpoints") or []:
        user_msg = (cp.get("debug") or {}).get("user_msg") or ""
        per_checkpoint.append(_extract_constraints_from_text(user_msg))

    non_null = [c for c in per_checkpoint if c]
    if not non_null:
        findings.append(_f("INFO", "constraints_not_recorded_using_defaults", None,
                          "No checkpoint's prompt stated the safe distance/turn limits "
                          "(e.g. bare_qwen) -- falling back to VesselConstraints() defaults."))
        return VesselConstraints(), findings

    first = non_null[0]
    for i, c in enumerate(per_checkpoint):
        if c is not None and c != first:
            findings.append(_f("BLOCKER", "BLOCKER_1_3_constraint_drift",
                              run["checkpoints"][i]["step"],
                              "Safe distance / turn-rate limits stated in the prompt "
                              "changed mid-run -- these are per-mission constants.",
                              expected=first, found=c))
    kwargs = {}
    if first.get("min_cpa_m") is not None:
        kwargs["min_cpa_m"] = first["min_cpa_m"]
    if first.get("max_rudder_angle_deg") is not None:
        kwargs["max_rudder_angle_deg"] = first["max_rudder_angle_deg"]
    if first.get("turn_rate_deg_s") is not None:
        kwargs["turn_rate_deg_s"] = first["turn_rate_deg_s"]
    if first.get("max_speed_mps") is not None:
        kwargs["max_speed_mps"] = first["max_speed_mps"]
    if first.get("max_acceleration_mps2") is not None:
        kwargs["max_acceleration_mps2"] = first["max_acceleration_mps2"]
    if first.get("max_deceleration_mps2") is not None:
        kwargs["max_deceleration_mps2"] = first["max_deceleration_mps2"]
    dt = (run.get("params") or {}).get("dt")
    if dt:
        kwargs["time_step_s"] = dt
    return VesselConstraints(**kwargs), findings


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- INTEGRITY (BLOCKER-class)
# ─────────────────────────────────────────────────────────────────────────────
_RE_OWN_LINE = re.compile(
    r"Own-ship at \((-?[\d.]+), (-?[\d.]+)\) NM, heading ([\d.]+), speed ([\d.]+) kt")
_RE_CONTACT_LINE = re.compile(
    r'Other ship \d+: named "([^"]+)", range ([\d.]+) NM, relative bearing (-?[\d.]+) deg, '
    r'heading ([\d.]+), speed ([\d.]+) kt, CPA ([\d.]+) NM, TCPA (-?[\d.]+)s')
_RE_GOAL_LINE = re.compile(r"^GOAL COURSE CHECK: .*$", re.M)


def check_1_1_situation_vs_trajectory(run: dict, traj: dict[str, dict[float, dict]]) -> list[dict]:
    findings: list[dict] = []
    for cp in run.get("checkpoints") or []:
        report = cp.get("situation_report") or ""
        own_row = _nearest(traj.get("own_ship", {}), cp["time"])
        if own_row is None:
            continue
        for name, rng_nm, rel_brg, hdg, spd_kt, cpa_nm, tcpa_s in _RE_CONTACT_LINE.findall(report):
            tgt_row = _nearest(traj.get(name, {}), cp["time"])
            if tgt_row is None:
                findings.append(_f("BLOCKER", "BLOCKER_1_1_situation_trajectory_mismatch",
                                  cp["step"], f"Contact {name!r} in situation_report has no "
                                  "matching trajectory entry at this time.", contact=name))
                continue
            true_brg, true_rng = bearing_and_range(own_row["x"], own_row["y"], tgt_row["x"], tgt_row["y"])
            true_rel = relative_bearing(own_row["heading"], true_brg)
            true_cpa, true_tcpa = cpa_tcpa(own_row["x"], own_row["y"], own_row["heading"], own_row["speed"],
                                          tgt_row["x"], tgt_row["y"], tgt_row["heading"], tgt_row["speed"])
            recomputed = {
                "range_m": true_rng, "rel_bearing_deg": true_rel,
                "cpa_m": true_cpa, "tcpa_s": true_tcpa,
            }
            reported = {
                "range_m": nm_to_m(float(rng_nm)), "rel_bearing_deg": float(rel_brg),
                "cpa_m": nm_to_m(float(cpa_nm)), "tcpa_s": float(tcpa_s),
            }
            tol = {
                "range_m": max(2.0, 0.01 * recomputed["range_m"]),
                "rel_bearing_deg": 1.0,
                "cpa_m": max(5.0, 0.02 * recomputed["cpa_m"]),
                "tcpa_s": 2.0,
            }
            for key in recomputed:
                if abs(recomputed[key] - reported[key]) > tol[key]:
                    findings.append(_f(
                        "BLOCKER", "BLOCKER_1_1_situation_trajectory_mismatch", cp["step"],
                        f"Contact {name!r} field {key!r} in situation_report does not match "
                        "the trajectory recomputation at this checkpoint's time.",
                        contact=name, field=key, reported=reported[key], recomputed=recomputed[key],
                    ))
    return findings


def check_1_2_goal_course_check(run: dict, traj: dict[str, dict[float, dict]]) -> list[dict]:
    findings: list[dict] = []
    gx, gy = run["mission"]["goal"]["x"], run["mission"]["goal"]["y"]
    for cp in run.get("checkpoints") or []:
        own_row = _nearest(traj.get("own_ship", {}), cp["time"])
        if own_row is None:
            continue
        m = _RE_GOAL_LINE.search(cp.get("situation_report") or "")
        if not m:
            findings.append(_f("BLOCKER", "BLOCKER_1_2_goal_course_check_mismatch",
                              cp["step"], "situation_report is missing a GOAL COURSE CHECK line."))
            continue
        expected = goal_course_check_line(own_row["x"], own_row["y"], own_row["heading"], gx, gy)
        if m.group(0).strip() != expected.strip():
            findings.append(_f("BLOCKER", "BLOCKER_1_2_goal_course_check_mismatch",
                              cp["step"], "GOAL COURSE CHECK line does not byte-match the "
                              "recomputation from own-ship's trajectory position/heading.",
                              expected=expected, found=m.group(0)))
    return findings


def check_1_4_leaked_labels(run: dict) -> list[dict]:
    findings: list[dict] = []
    for cp in run.get("checkpoints") or []:
        report = cp.get("situation_report") or ""
        for pat in _LEAKED_LABEL_PATTERNS:
            if pat.search(report):
                findings.append(_f("BLOCKER", "BLOCKER_1_4_leaked_label", cp["step"],
                                  f"situation_report leaks the training-answer phrase "
                                  f"{pat.pattern!r} into the model's own input.", pattern=pat.pattern))
    return findings


def _interp_xy(series: list[tuple[float, float, float]], t: float) -> tuple[float, float] | None:
    """Mirrors Evaluation Functions/evaluate_run.py's interp_xy() exactly -- same linear
    interpolation between recorded (t, x, y) samples. Screening-set-B audit follow-up
    (2026-09-23): BLOCKER_1_5 compared its own coarse 10s-raw-sample minimum against
    evaluate_run.py's 1s-interpolated minimum and flagged a mismatch (88.98m vs 82.7m) --
    a discrete 10s scan can straddle and miss the TRUE closest point entirely. Using the
    identical interpolation here means the two can never disagree over sampling
    resolution again."""
    if not series or t < series[0][0] or t > series[-1][0]:
        return None
    for i in range(len(series) - 1):
        t0, x0, y0 = series[i]
        t1, x1, y1 = series[i + 1]
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
    return None


def _min_separation_over_run(own_series: dict[float, dict], tgt_series: dict[float, dict],
                             dt: float = 1.0) -> float:
    own_pts = sorted((t, r["x"], r["y"]) for t, r in own_series.items())
    tgt_pts = sorted((t, r["x"], r["y"]) for t, r in tgt_series.items())
    if not own_pts or not tgt_pts:
        return float("inf")
    t_start = max(own_pts[0][0], tgt_pts[0][0])
    t_end = min(own_pts[-1][0], tgt_pts[-1][0])
    if t_end < t_start:
        return float("inf")
    best = float("inf")
    t = t_start
    while t <= t_end:
        po, pt = _interp_xy(own_pts, t), _interp_xy(tgt_pts, t)
        if po and pt:
            best = min(best, math.hypot(po[0] - pt[0], po[1] - pt[1]))
        t += dt
    return best


def check_1_5_verdict_consistency(run: dict, traj: dict[str, dict[float, dict]]) -> list[dict]:
    findings: list[dict] = []
    own_series = traj.get("own_ship", {})
    target_names = [n for n in traj if n != "own_ship"]
    min_seps = {n: _min_separation_over_run(own_series, traj[n]) for n in target_names}
    overall_min = min(min_seps.values(), default=float("inf"))
    collided = overall_min < COLLISION_RADIUS_M

    outcome_verdict = (run.get("outcome") or {}).get("verdict")
    if collided and outcome_verdict != "collision":
        findings.append(_f("BLOCKER", "BLOCKER_1_5_verdict_inconsistent", None,
                          "Recomputed minimum separation is below the collision radius "
                          "but outcome.verdict does not say so.",
                          recomputed_min_sep_m=overall_min, outcome_verdict=outcome_verdict))
    if not collided and outcome_verdict == "collision":
        findings.append(_f("BLOCKER", "BLOCKER_1_5_verdict_inconsistent", None,
                          "outcome.verdict says collision but recomputed minimum separation "
                          "never drops below the collision radius.",
                          recomputed_min_sep_m=overall_min))

    gx, gy = run["mission"]["goal"]["x"], run["mission"]["goal"]["y"]
    reached = any(math.hypot(r["x"] - gx, r["y"] - gy) <= nm_to_m(GOAL_RADIUS_NM)
                 for r in own_series.values())
    if reached and outcome_verdict not in ("reached_goal",) and not collided:
        findings.append(_f("BLOCKER", "BLOCKER_1_5_verdict_inconsistent", None,
                          "Own-ship's trajectory enters the goal radius but "
                          "outcome.verdict does not say reached_goal.", outcome_verdict=outcome_verdict))

    ev = run.get("evaluation") or {}
    safety = ev.get("safety") or {}
    ev_min_cpa = safety.get("min_cpa_m")
    if ev_min_cpa is not None:
        # Screening-set-B audit follow-up (2026-09-23): _min_separation_over_run() now
        # uses the SAME 1s-interpolation as evaluate_run.min_cpa_over_run() (previously a
        # coarse 10s raw-sample scan, which could straddle and miss the true minimum by
        # tens of metres) -- remaining tolerance only covers genuine floating-point/
        # rounding noise, not a systematic sampling-resolution gap.
        tol = max(1.0, 0.005 * overall_min)
        if abs(ev_min_cpa - overall_min) > tol:
            findings.append(_f("BLOCKER", "BLOCKER_1_5_verdict_inconsistent", None,
                              "evaluation.safety.min_cpa_m does not match the recomputed minimum "
                              "separation over the trajectory.",
                              evaluation_min_cpa_m=ev_min_cpa, recomputed_min_sep_m=overall_min))

    constraints, _ = extract_constraints(run)
    cpa_violation = overall_min < constraints.min_cpa_m
    verdict = ev.get("verdict")
    # Exact "PASS" only -- evaluate_run.py's run-writer fix (2026-09-23) now emits
    # "PASS_WITH_CPA_VIOLATION" for exactly this case, which must never re-trigger this
    # BLOCKER; runs generated before that fix still legitimately show it (their embedded
    # verdict really is a bare, now-known-wrong "PASS").
    if cpa_violation and not collided and verdict == "PASS":
        findings.append(_f("BLOCKER", "BLOCKER_1_5_verdict_inconsistent", None,
                          "A CPA-safety-distance violation occurred (min separation below "
                          "this mission's safe distance) but the run is labelled a plain "
                          "PASS/reached_goal without any safety flag.",
                          recomputed_min_sep_m=overall_min, safe_distance_m=constraints.min_cpa_m,
                          verdict=verdict))
    return findings


def check_1_6_kinematics(run: dict, constraints: VesselConstraints) -> list[dict]:
    findings: list[dict] = []
    own_rows = sorted((r for r in run.get("trajectory") or [] if r["vehicle"] == "own_ship"),
                      key=lambda r: r["time"])
    dt = (run.get("params") or {}).get("dt") or constraints.time_step_s
    max_turn = constraints.turn_rate_deg_s * dt + 1e-6
    max_accel = constraints.max_acceleration_mps2 * dt + 1e-6
    max_decel = constraints.max_deceleration_mps2 * dt + 1e-6
    for prev, cur in zip(own_rows, own_rows[1:]):
        step_dt = cur["time"] - prev["time"]
        if step_dt <= 0:
            continue
        scale = step_dt / dt if dt else 1.0
        dh = abs(relative_bearing(prev["heading"], cur["heading"]))
        if dh > max_turn * scale + 1e-6:
            findings.append(_f("BLOCKER", "BLOCKER_1_6_kinematics_violation", None,
                              f"Heading changed {dh:.1f} deg in one step, exceeding the "
                              f"turn-rate limit ({max_turn * scale:.1f} deg allowed).",
                              time=cur["time"], delta_heading=dh, limit=max_turn * scale))
        ds = cur["speed"] - prev["speed"]
        limit = (max_accel if ds > 0 else max_decel) * scale
        if abs(ds) > limit + 1e-6:
            findings.append(_f("BLOCKER", "BLOCKER_1_6_kinematics_violation", None,
                              f"Speed changed {ds:.2f} m/s in one step, exceeding the "
                              f"accel/decel limit ({limit:.2f} m/s allowed).",
                              time=cur["time"], delta_speed=ds, limit=limit))
    return findings


def check_1_7_cadence(run: dict) -> list[dict]:
    """2026-09-24: decision cadence became ADAPTIVE (app.run_llm_scenario, live_decision_
    interval) -- the gap between two checkpoints is decided by the EARLIER checkpoint's own
    "decision_interval_steps" (logged per-checkpoint since that field exists), not one fixed
    params.decision_interval for the whole run. Older archived logs (pre-2026-09-24, or any
    run made with an explicit --decision-interval override, "decision_interval_mode":
    "fixed") have no per-checkpoint field -- falls back to params.decision_interval for
    those, unchanged behaviour."""
    findings: list[dict] = []
    checkpoints = run.get("checkpoints") or []
    run_level_interval = (run.get("params") or {}).get("decision_interval")
    dt = (run.get("params") or {}).get("dt")
    final_step = (run.get("outcome") or {}).get("final_step")
    times = [cp["time"] for cp in checkpoints]
    if times != sorted(times):
        findings.append(_f("ERROR", "ERROR_1_7_cadence_violation", None,
                          "Checkpoint times are not monotonically increasing."))
    if dt:
        for prev, cur in zip(checkpoints, checkpoints[1:]):
            interval = prev.get("decision_interval_steps", run_level_interval)
            if not interval:
                continue
            expected_gap = interval * dt
            gap = cur["time"] - prev["time"]
            if abs(gap - expected_gap) > 1e-6:
                findings.append(_f("ERROR", "ERROR_1_7_cadence_violation", cur["step"],
                                  f"Gap between checkpoints is {gap:.1f}s, expected "
                                  f"{expected_gap:.1f}s (this checkpoint's own decision_interval "
                                  "* dt).", gap=gap))
    if checkpoints and final_step is not None and checkpoints[-1]["step"] > final_step:
        findings.append(_f("ERROR", "ERROR_1_7_cadence_violation", checkpoints[-1]["step"],
                          "A checkpoint exists beyond outcome.final_step.", final_step=final_step))
    return findings


def _extract_json_objects(text: str) -> list[str]:
    """Balanced-brace scan for every top-level {...} object in `text` -- duplicated
    (deliberately, not imported) from app.agents._extract_json_objects, since app.agents
    pulls in torch/transformers/sentence_transformers at module level and this auditor is
    meant to stay dependency-light/importable without a GPU or the models present. Keep
    this in sync by hand if app.agents._extract_json_objects ever changes."""
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


def _reparse_raw_decision(raw_text: str) -> dict | None:
    """Re-derives {decision, parse_ok, schema_errors} from a checkpoint's raw model text,
    duplicating (not importing, same reason as _extract_json_objects above) app.agents.
    _parse_json_action()'s parse/schema split. Returns None if raw_text is falsy.

    Screening-set-B audit follow-up (2026-09-23), point 1: these run logs were generated
    BEFORE the app.agents._parse_json_action() fix landed, so their logged `decision`/
    `_parse_error` reflect the OLD buggy behaviour (discarding a valid-JSON-but-schema-
    invalid decision and replacing it with a hold_course fallback). Re-running the actual
    model is out of scope (explicitly not requested) -- but every checkpoint's raw model
    text is ALREADY saved verbatim as `reasoning_raw`, so the auditor can retroactively
    recompute the CORRECT parse_ok/schema_errors split from that saved text without any
    new model calls. audit_one_run() only applies this correction to checkpoints the OLD
    code flagged as `_parse_error` -- checkpoints that already parsed fine are left as-is."""
    if not raw_text:
        return None
    text = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    schema_invalid_candidate, schema_invalid_errors = None, None
    for candidate in reversed(_extract_json_objects(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not (isinstance(parsed, dict) and "action" in parsed):
            continue
        errors = validate_action_json(parsed)
        if not errors:
            return {"decision": parsed, "parse_ok": True, "schema_errors": []}
        if schema_invalid_candidate is None:
            schema_invalid_candidate, schema_invalid_errors = parsed, errors
    if schema_invalid_candidate is not None:
        schema_invalid_candidate["_schema_errors"] = schema_invalid_errors
        return {"decision": schema_invalid_candidate, "parse_ok": True,
               "schema_errors": schema_invalid_errors}
    return {"decision": {"action": "hold_course", "encounter_rule": "none", "conduct_rule": "none",
                        "reasoning": f"[parse error -- raw model output] {text[:300]}",
                        "_parse_error": True},
           "parse_ok": False, "schema_errors": []}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- DECISION QUALITY (ERROR-class)
# ─────────────────────────────────────────────────────────────────────────────
def check_2_0_parse_format(cp: dict) -> list[dict]:
    findings: list[dict] = []
    decision = cp.get("decision") or {}
    step = cp["step"]
    reasoning = decision.get("reasoning") or ""
    if decision.get("_parse_error") or "[parse error" in reasoning:
        findings.append(_f("ERROR", "ERROR_2_0_parse_error", step, "Decision failed to parse."))
    for field_name in ("action", "degrees", "encounter_rule", "conduct_rule", "reasoning"):
        val = decision.get(field_name)
        if isinstance(val, str) and ("<think>" in val or "</think>" in val):
            findings.append(_f("ERROR", "ERROR_2_0_thinking_leak", step,
                              f"Decision field {field_name!r} leaks <think> tags.", field=field_name))
    errors = validate_action_json({k: v for k, v in decision.items()
                                   if k not in ("_parse_error", "_schema_errors")})
    for err in errors:
        findings.append(_f("ERROR", "ERROR_2_0_invalid_json", step, err))
    return findings


def check_2_1_check_d(cp: dict, situation: list[dict], constraints: VesselConstraints) -> list[dict]:
    findings: list[dict] = []
    decision = cp.get("decision") or {}
    action = decision.get("action")
    conduct_rule = decision.get("conduct_rule") or decision.get("rule_applied") or "none"
    if action not in ("hold_course", "speed_up"):
        return findings
    for c in situation:
        if real_risk(c.get("cpa_m"), c.get("tcpa_s"), constraints.min_cpa_m):
            code = "D_stationary" if c.get("stationary") else "D_no_action_despite_risk"
            if conduct_rule != "Rule 17":
                findings.append(_f("ERROR", code, cp["step"],
                                  f"Contact {c.get('name')!r} has real collision risk but the "
                                  f"action taken was {action!r} with conduct_rule {conduct_rule!r} "
                                  "(not Rule 17 stand-on).", contact=c.get("name"),
                                  cpa_m=c.get("cpa_m"), tcpa_s=c.get("tcpa_s")))
    return findings


def check_2_1b_unclassified_encounter(cp: dict, situation: list[dict],
                                      constraints: VesselConstraints) -> list[dict]:
    """A contact below the safe passing distance (real risk OR early-action, see
    measurement.py's INFO_early_action) that the model failed to recognize as any kind of
    COLREG encounter at all -- encounter_rule left "none", or conduct_rule is a general
    Rule 7/8 lookout/stop citation with no 13/14/15 encounter mentioned anywhere. Found via
    51/84 (61%) of the old screening_standard_cloud A_fabricated_risk hits being exactly
    this pattern instead (Imazu01, a scripted head-on/Rule 14 scenario, only cited Rule 14
    on 3 of its checkpoints)."""
    d = cp.get("decision") or {}
    enc = d.get("encounter_rule") or "none"
    cond = d.get("conduct_rule") or "none"
    below_safe = [c for c in situation if c.get("cpa_m") is not None
                and c["cpa_m"] < constraints.min_cpa_m]
    if not below_safe:
        return []
    unclassified = enc == "none" or (
        cond in ("Rule 7", "Rule 8") and not any(r in (enc, cond) for r in ("Rule 13", "Rule 14", "Rule 15")))
    if not unclassified:
        return []
    worst = min(below_safe, key=lambda c: c["cpa_m"])
    return [_f("ERROR", "E_unclassified_encounter", cp["step"],
              f"Contact {worst.get('name')!r} is below the safe distance "
              f"(cpa={worst['cpa_m']:.0f}m) but encounter_rule={enc!r}/conduct_rule={cond!r} "
              "recognizes no COLREG encounter at all.",
              contact=worst.get("name"), cpa_m=worst["cpa_m"], tcpa_s=worst.get("tcpa_s"),
              encounter_rule=enc, conduct_rule=cond)]


def check_2_2_rule_matrix(cp: dict, situation: list[dict], config: str | None) -> list[dict]:
    findings: list[dict] = []
    d = cp.get("decision") or {}
    step = cp["step"]
    enc, cond, action = d.get("encounter_rule"), d.get("conduct_rule"), d.get("action")
    worst = min(situation, key=lambda c: c.get("cpa_m", float("inf"))) if situation else None
    ctx = {"config": config, "cpa_m": worst.get("cpa_m") if worst else None,
          "tcpa_s": worst.get("tcpa_s") if worst else None,
          "reasoning": d.get("reasoning")}
    if enc == "Rule 14" and cond == "Rule 17":
        findings.append(_f("ERROR", "E_rule_matrix_head_on_stand_on", step,
                          "encounter Rule 14 (head-on) with conduct Rule 17 (stand-on) is "
                          "impossible -- head-on has no stand-on vessel.", **ctx))
    if enc == "none" and cond in ("Rule 13", "Rule 14", "Rule 15", "Rule 16", "Rule 17"):
        # Screening-set-B audit follow-up (2026-09-23): renamed from
        # E_rule_matrix_conduct_without_encounter -- previously this branch could barely
        # ever fire for the "none"/"Rule 17" case specifically, since _parse_json_action()
        # discarded that exact (valid but schema-inconsistent) decision entirely before
        # the auditor ever saw its real fields (see agents.py's parse/schema split fix).
        findings.append(_f("ERROR", "E_rule_matrix_none_with_conduct", step,
                          f"conduct_rule {cond!r} cited with encounter_rule 'none'.",
                          conduct_rule=cond, **ctx))
    if cond == "Rule 16" and action == "hold_course":
        findings.append(_f("ERROR", "E_rule_matrix_give_way_no_action", step,
                          "conduct_rule Rule 16 (give-way duty) cited but action is hold_course.",
                          **ctx))
    if cond == "Rule 8" and action == "speed_up":
        findings.append(_f("ERROR", "E_rule_matrix_rule8_speed_up", step,
                          "conduct_rule Rule 8 cited with a speed_up action.", **ctx))
    return findings


def check_2_3_encounter_mismatch(cp: dict, own_row: dict, target_rows: dict[str, dict]) -> list[dict]:
    findings: list[dict] = []
    d = cp.get("decision") or {}
    cited_enc, cited_cond = d.get("encounter_rule"), d.get("conduct_rule")
    if cited_enc in (None, "none") or own_row is None or not target_rows:
        return findings
    classified = {
        name: classify_encounter(own_row["x"], own_row["y"], own_row["heading"],
                                 row["x"], row["y"], row["heading"])
        for name, row in target_rows.items()
    }
    if any(rules[0] == cited_enc for _enc, rules, _rel in classified.values()):
        name, (enc, rules, _rel) = None, (None, None, None)
    else:
        # None of the contacts geometrically support the cited encounter_rule -- report
        # against the contact with the smallest range as the most likely intended one.
        name = min(target_rows, key=lambda n: math.hypot(
            target_rows[n]["x"] - own_row["x"], target_rows[n]["y"] - own_row["y"]))
        enc, rules, _rel = classified[name]
        findings.append(_f("ERROR", "E_encounter_mismatch", cp["step"],
                          f"Cited encounter_rule {cited_enc!r} does not match the geometric "
                          f"encounter classification ({enc} -> {rules[0]}) for contact {name!r}.",
                          contact=name, geometric_encounter=enc, geometric_rules=rules))
    if name is None:
        return findings
    role_map = {"head_on": "mutual", "crossing_target_on_starboard": "give_way",
               "crossing_target_on_port": "stand_on", "we_are_overtaking_target": "overtaking_give_way",
               "target_is_overtaking_us": "overtaking_stand_on"}
    geo_role = role_map.get(enc)
    if geo_role:
        _, geo_conduct = classify_rules(geo_role, d.get("action") or "hold_course")
        cited_is_stand_on = cited_cond == "Rule 17"
        geo_is_stand_on = geo_conduct == "Rule 17"
        if cited_is_stand_on != geo_is_stand_on:
            findings.append(_f("ERROR", "E_role_fabrication", cp["step"],
                              f"Cited conduct_rule {cited_cond!r} implies a stand-on/give-way "
                              f"role inconsistent with the geometric role for contact {name!r}.",
                              contact=name, geometric_role=geo_role))
    return findings


def check_2_4_direction(cp: dict, situation: list[dict], constraints: VesselConstraints) -> list[dict]:
    """Wraps measurement.py's Checks A/B/C, adds the Rule-13-both-sides-ok exemption and
    the Rule 17(c) port-turn-toward-a-port-side-contact check."""
    findings: list[dict] = []
    result = measure_decision_quality(cp.get("decision") or {}, situation, constraints)
    d = cp.get("decision") or {}
    enc = d.get("encounter_rule")
    for code in result["checks_fired"]:
        if code == "B_wrong_direction" and enc == "Rule 13":
            continue  # Rule 13 accepts either side -- never a violation
        severity = "INFO" if code.startswith("INFO_") else "ERROR"
        detail = result["details"].get(code) or result["details"].get(code.split("_")[0], {})
        findings.append(_f(severity, code, cp["step"], f"{code} fired.", **detail))
    if d.get("conduct_rule") == "Rule 17" and d.get("action") == "turn_left":
        for c in situation:
            if c.get("rel_bearing_deg") is not None and c["rel_bearing_deg"] < 0:
                findings.append(_f("ERROR", "B_17c", cp["step"],
                                  "Stand-on vessel acting under 17(b)/(c) turned to port toward "
                                  "a contact on her own port side.", contact=c.get("name")))
    return findings


def check_2_5_magnitude(cp: dict, situation: list[dict], constraints: VesselConstraints) -> list[dict]:
    findings: list[dict] = []
    d = cp.get("decision") or {}
    action, degrees = d.get("action"), d.get("degrees")
    if action not in ("turn_left", "turn_right") or not isinstance(degrees, (int, float)):
        return findings
    for c in situation:
        cpa_m, tcpa_s = c.get("cpa_m"), c.get("tcpa_s")
        if cpa_m is None or not real_risk(cpa_m, tcpa_s, constraints.min_cpa_m):
            continue
        shortfall = constraints.min_cpa_m - cpa_m
        if shortfall > 0.4 * constraints.min_cpa_m and degrees < 10:
            findings.append(_f("WARN", "W_token_manoeuvre", cp["step"],
                              f"Turn of only {degrees} deg while CPA shortfall is "
                              f"{shortfall:.0f}m ({shortfall / constraints.min_cpa_m:.0%} of "
                              "the safe distance) -- Rule 16 requires a substantial alteration.",
                              contact=c.get("name"), degrees=degrees, shortfall_m=shortfall))
    return findings


def check_2_6_temporal(checkpoints: list[dict], situations: dict[int, list[dict]]) -> list[dict]:
    findings: list[dict] = []
    actions = [(cp["step"], (cp.get("decision") or {}).get("action")) for cp in checkpoints]
    for i in range(len(actions) - 2):
        a, b, c = actions[i][1], actions[i + 1][1], actions[i + 2][1]
        if {a, c} == {"turn_left", "turn_right"} and a != b and b in ("turn_left", "turn_right") and a == c:
            findings.append(_f("WARN", "W_zigzag", actions[i + 2][0],
                              "3 consecutive checkpoints alternate turn_left/turn_right.",
                              window=[a, b, c]))
        speed_actions = {"speed_up", "slow_down"}
        if a in speed_actions and c in speed_actions and a != c:
            findings.append(_f("WARN", "W_speed_oscillation", actions[i + 2][0],
                              "speed_up/slow_down oscillation within 3 consecutive steps.",
                              window=[a, b, c]))

    prior_real_risk_contacts: set[str] = set()
    first_action_step_after_risk: dict[str, int] = {}
    for i, cp in enumerate(checkpoints):
        step = cp["step"]
        situation = situations.get(step, [])
        d = cp.get("decision") or {}
        action, cond = d.get("action"), d.get("conduct_rule")
        if i > 0 and cond in (None, "none") and action in ("hold_course", "turn_left", "turn_right"):
            for name in prior_real_risk_contacts:
                match = next((c for c in situation if c.get("name") == name), None)
                if match and match.get("closing") and (match.get("tcpa_s") or -1) >= 0:
                    findings.append(_f("WARN", "W_premature_resume", step,
                                      f"Resumed a no-rule action while contact {name!r} still "
                                      "has positive closing speed and non-negative TCPA.",
                                      contact=name))
        for c in situation:
            name = c.get("name")
            if real_risk(c.get("cpa_m"), c.get("tcpa_s"), 500.0):
                prior_real_risk_contacts.add(name)
                first_action_step_after_risk.setdefault(name, step)
            elif name in prior_real_risk_contacts:
                prior_real_risk_contacts.discard(name)

    for cp in checkpoints:
        d = cp.get("decision") or {}
        if d.get("conduct_rule") not in (None, "none"):
            situation = situations.get(cp["step"], [])
            for c in situation:
                tcpa = c.get("tcpa_s")
                first_seen = first_action_step_after_risk.get(c.get("name"))
                if tcpa is not None and tcpa < 60 and first_seen == cp["step"]:
                    decisions_earlier = [s for s in first_action_step_after_risk.values() if s < cp["step"]]
                    if len(decisions_earlier) == 0:
                        continue
    return findings


_RULE_CITATION_RE = re.compile(r"\brule\s*\d+\s*(\([a-z]\))?", re.I)
_SUBRULE_RE = re.compile(r"\b\d+\s*\([a-z]\)", re.I)
_NO_RISK_RE = re.compile(
    r"\bno (real |immediate |significant |collision )*risk\b|\bsafe\b|\bclear\b|"
    r"\bnot a (real |immediate |collision )*risk\b|\bno danger\b", re.I)
_RISK_RE = re.compile(
    r"\brisk\b|\bcollision course\b|\bdanger\b|\bmust give way\b|\bgive[- ]way\b|"
    r"\bmust (alter|turn)\b", re.I)


def _mask_rule_citations(text: str) -> str:
    """Blanks "Rule 15", "17(b)", "Rule 17(b)" style citations before extracting numbers
    -- these are rule/sub-rule references, never a measurement the model could fabricate,
    and were ~all of gate_b's false positives (150x '500', plus every bare rule number)."""
    text = _RULE_CITATION_RE.sub(" ", text)
    text = _SUBRULE_RE.sub(" ", text)
    return text


def _round_sig(x: float, sig: int = 2) -> float:
    if x == 0:
        return 0.0
    from math import floor, log10
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


_UNIT_CONVERSIONS = (1852.0, 0.5144, 1 / 60.0)  # NM->m, kn->m/s, s->min


def _known_numbers_from_report(report: str, constraints: VesselConstraints) -> list[float]:
    """Every number a reasoning could legitimately restate: the raw situation_report
    values, their 2-significant-figure rounding (models paraphrase, they don't quote
    verbatim), each of those after a unit conversion (NM->m/kn->m/s/s->min) then rounded
    again, plus this run's own constraint values (safe distance/max turn/horizon) -- never
    a hardcoded 500/30."""
    raw = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", report)]
    known: set[float] = set()
    for n in raw:
        known.add(n)
        r2 = _round_sig(n, 2)
        known.add(r2)
        for factor in _UNIT_CONVERSIONS:
            known.add(_round_sig(n * factor, 2))
            known.add(_round_sig(r2 * factor, 2))
    known.add(constraints.min_cpa_m)
    known.add(constraints.turn_rate_deg_s * constraints.time_step_s)
    known.add(RISK_HORIZON_S)
    return sorted(known)


def _nearest_known(x: float, known: list[float]) -> float | None:
    return min(known, key=lambda k: abs(k - x)) if known else None


def _is_known_number(x: float, known: list[float], rel_tol: float = 0.05, abs_tol: float = 0.5) -> bool:
    nearest = _nearest_known(x, known)
    if nearest is None:
        return False
    return abs(nearest - x) <= max(abs_tol, rel_tol * max(abs(x), abs(nearest)))


def _extract_risk_conclusion(reasoning: str) -> str:
    """{"risk", "no_risk", "unknown"} -- checked in this order because "no risk" contains
    the substring "risk", so a naive `"risk" in text` check (the old version) misread
    every "no risk"/"safe"/"clear" conclusion as an affirmative risk statement, which was
    exactly why gate_c mismatched 100% of UM01/UM02 (the quiet canary missions)."""
    if _NO_RISK_RE.search(reasoning):
        return "no_risk"
    if _RISK_RE.search(reasoning):
        return "risk"
    return "unknown"


def check_2_7_reasoning_vs_decision(cp: dict, situation: list[dict],
                                    constraints: VesselConstraints) -> list[dict]:
    """Reimplementation of the B3 teacher-pipeline's acceptance gates a-e applied to the
    MODEL's own answer (no shared gate functions were found already factored out anywhere
    importable -- if/when they are moved into oow_agent_spec.py, switch this to call them)."""
    findings: list[dict] = []
    d = cp.get("decision") or {}
    reasoning = (d.get("reasoning") or "").lower()
    step = cp["step"]
    below_safe = [c for c in situation if c.get("cpa_m") is not None
                and c["cpa_m"] < constraints.min_cpa_m]
    real_risk_contacts = [c for c in below_safe
                          if real_risk(c.get("cpa_m"), c.get("tcpa_s"), constraints.min_cpa_m)]
    early_action_contacts = [c for c in below_safe if c not in real_risk_contacts
                            and c.get("tcpa_s") is not None and c["tcpa_s"] > RISK_HORIZON_S]

    # gate a -- decisive real-risk contact must be named
    if real_risk_contacts:
        decisive = min(real_risk_contacts, key=lambda c: c.get("cpa_m", float("inf")))
        if decisive.get("name") and decisive["name"].lower() not in reasoning:
            findings.append(_f("ERROR", "G_gate_a_contact_missing", step,
                              f"Reasoning never mentions the decisive contact {decisive['name']!r}."))

    # gate b -- fabricated numbers: mask rule/sub-rule citations first, then compare every
    # remaining number against the FULL prompt text's own values (raw, rounded, unit-
    # converted) and this run's constraint values, with a tolerance (models round/
    # paraphrase). Screening-set-B audit follow-up (2026-09-23): situation_report ALONE
    # (narrate()'s bare contact/GOAL-COURSE-CHECK text) never included the constraint
    # line (safe distance/max turn/risk horizon) or the history-of-previous-decisions
    # text, both of which ARE part of what the model actually read (debug.user_msg) --
    # 567 (this run's real derived risk horizon, correctly read off the constraint line)
    # and 1400 (a contact's own TCPA, correctly read off a history line) were flagged as
    # "fabricated" 87x/29x purely because their source text was never in the known-set at
    # all. debug.user_msg is the actual, complete text sent to the model; situation_report
    # is kept only as a fallback for older logs that never stored debug.user_msg.
    reasoning_masked = _mask_rule_citations(reasoning)
    numbers_in_reasoning = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", reasoning_masked)]
    full_prompt_text = (cp.get("debug") or {}).get("user_msg") or cp.get("situation_report") or ""
    known = _known_numbers_from_report(full_prompt_text, constraints)
    fabricated, nearest_map = [], {}
    for n in numbers_in_reasoning:
        if n <= 1:
            continue  # too common/uninformative ("1 other ship", bare direction counts) to be useful signal
        if not _is_known_number(n, known):
            fabricated.append(n)
            nearest_map[n] = _nearest_known(n, known)
    if fabricated:
        findings.append(_f("WARN", "G_gate_b_number_fabricated", step,
                          "Reasoning cites number(s) not present in (or derivable from) "
                          "situation_report.", numbers=fabricated[:5],
                          nearest_known={str(n): nearest_map[n] for n in fabricated[:5]}))

    # gate c -- risk conclusion vs real_risk(), using the same 3-way split as Check A:
    # "unknown" (no explicit conclusion found) is informational, never a mismatch; in the
    # early-action zone (below safe distance, TCPA beyond horizon) either conclusion is
    # a defensible read, so nothing is flagged there either way.
    extracted = _extract_risk_conclusion(reasoning)
    worst = min(situation, key=lambda c: c.get("cpa_m", float("inf"))) if situation else None
    c_details = {"extracted": extracted, "real_risk": bool(real_risk_contacts),
                "cpa_m": worst.get("cpa_m") if worst else None,
                "tcpa_s": worst.get("tcpa_s") if worst else None, "horizon_s": RISK_HORIZON_S}
    if extracted == "unknown":
        findings.append(_f("INFO", "G_gate_c_risk_unknown", step,
                          "Could not extract an explicit risk/no-risk conclusion from the reasoning.",
                          **c_details))
    elif real_risk_contacts:
        if extracted != "risk":
            findings.append(_f("WARN", "G_gate_c_risk_mismatch", step,
                              "Reasoning's stated risk conclusion does not match real_risk().",
                              **c_details))
    elif not early_action_contacts and extracted != "no_risk":
        findings.append(_f("WARN", "G_gate_c_risk_mismatch", step,
                          "Reasoning's stated risk conclusion does not match real_risk().",
                          **c_details))

    # gate d -- direction word vs action taken
    action = d.get("action")
    if action in ("turn_left", "turn_right"):
        word = "starboard" if action == "turn_right" else "port"
        other = "port" if action == "turn_right" else "starboard"
        if other in reasoning and word not in reasoning:
            findings.append(_f("WARN", "G_gate_d_direction_mismatch", step,
                              f"Reasoning says {other!r} but the action taken is {action!r}.",
                              direction_word=other, action=action))

    # gate e -- cited rule numbers should appear in the reasoning text
    for field_name in ("encounter_rule", "conduct_rule"):
        val = d.get(field_name)
        if val and val != "none" and val.lower() not in reasoning:
            findings.append(_f("INFO", "G_gate_e_rule_mismatch", step,
                              f"{field_name}={val!r} not mentioned in reasoning text.",
                              field=field_name, value=val))
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- REASONING COMPLEXITY (WARN-class)
# ─────────────────────────────────────────────────────────────────────────────
def check_3_truncation(cp: dict, max_new_tokens: int | None) -> list[dict]:
    findings: list[dict] = []
    raw = cp.get("reasoning_raw") or ""
    if not raw or not max_new_tokens:
        return findings
    approx_tokens = len(raw) / 4
    ends_clean = raw.rstrip().endswith("}") or "</think>" in raw
    if not ends_clean and approx_tokens > 0.8 * max_new_tokens:
        findings.append(_f("WARN", "W_truncated", cp["step"],
                          "Reasoning ends without a closing JSON object / mid-sentence, near "
                          "the max_new_tokens budget -- likely truncated.",
                          approx_tokens=approx_tokens, max_new_tokens=max_new_tokens))
    return findings


def check_3_self_correction(cp: dict) -> list[dict]:
    raw = cp.get("reasoning_raw") or ""
    n = len(_DELIBERATION_WORDS.findall(raw))
    if n > 4:
        return [_f("WARN", "W_deliberation_loop", cp["step"],
                   f"{n} self-correction markers ('wait'/'actually'/'let me re'/'hmm') in one "
                   "reasoning block.", count=n)]
    return []


def check_3_repetition(cp: dict) -> list[dict]:
    raw = cp.get("reasoning_raw") or cp.get("decision", {}).get("reasoning") or ""
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", raw) if len(s.strip()) > 15]
    counts = Counter(sentences)
    repeated = [s for s, n in counts.items() if n >= 2]
    if repeated:
        return [_f("WARN", "W_repetition", cp["step"],
                   "Identical sentence repeated >=2x in one reasoning block.",
                   example=repeated[0][:120])]
    return []


def check_3_length(cp: dict, baseline_p95_chars: float | None) -> list[dict]:
    raw = cp.get("reasoning_raw") or cp.get("decision", {}).get("reasoning") or ""
    if baseline_p95_chars and len(raw) > 3 * baseline_p95_chars:
        return [_f("WARN", "W_reasoning_long", cp["step"],
                   f"Reasoning is {len(raw)} chars, over 3x the bare/v0 baseline P95.",
                   chars=len(raw), baseline_p95=baseline_p95_chars)]
    return []


def situation_for_checkpoint(run: dict, traj: dict[str, dict[float, dict]], cp: dict) -> list[dict]:
    own_row = _nearest(traj.get("own_ship", {}), cp["time"])
    out: list[dict] = []
    if own_row is None:
        return out
    for name, series in traj.items():
        if name == "own_ship":
            continue
        row = _nearest(series, cp["time"])
        if row is None:
            continue
        brg, rng = bearing_and_range(own_row["x"], own_row["y"], row["x"], row["y"])
        rel = relative_bearing(own_row["heading"], brg)
        cpa, tcpa = cpa_tcpa(own_row["x"], own_row["y"], own_row["heading"], own_row["speed"],
                             row["x"], row["y"], row["heading"], row["speed"])
        oh, th = math.radians(own_row["heading"]), math.radians(row["heading"])
        vox, voy = own_row["speed"] * math.sin(oh), own_row["speed"] * math.cos(oh)
        vtx, vty = row["speed"] * math.sin(th), row["speed"] * math.cos(th)
        dx, dy = row["x"] - own_row["x"], row["y"] - own_row["y"]
        dvx, dvy = vtx - vox, vty - voy
        closing = (dvx ** 2 + dvy ** 2) >= 1e-6 and -(dx * dvx + dy * dvy) > 0
        out.append({"name": name, "range_m": rng, "rel_bearing_deg": rel, "cpa_m": cpa,
                   "tcpa_s": tcpa, "closing": closing, "stationary": row["speed"] < 0.05})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-run orchestration
# ─────────────────────────────────────────────────────────────────────────────
def audit_one_run(run: dict, path: Path) -> dict:
    schema = detect_schema(run, path)
    findings: list[dict] = []
    traj = traj_by_vehicle_time(run)
    constraints, cfindings = extract_constraints(run)
    findings += cfindings

    if not schema["is_legacy"]:
        findings += check_1_1_situation_vs_trajectory(run, traj)
        findings += check_1_2_goal_course_check(run, traj)
    else:
        findings.append(_f("INFO", "legacy_schema", None,
                          "legacy schema: checks 1.1/1.2/2.1-2.7 (two-field-schema-dependent) "
                          "not applicable to this run."))
    findings += check_1_4_leaked_labels(run)
    findings += check_1_5_verdict_consistency(run, traj)
    findings += check_1_6_kinematics(run, constraints)
    findings += check_1_7_cadence(run)

    checkpoints = run.get("checkpoints") or []
    situations: dict[int, list[dict]] = {}
    reasoning_lens = []
    for cp in checkpoints:
        # Point 1 retroactive fix: only touch checkpoints the (old, buggy) generation-time
        # parser flagged as a parse failure -- re-derive the true parse_ok/decision from the
        # saved raw text instead of trusting that stale flag. Checkpoints that already
        # parsed fine at generation time are left completely untouched.
        if (cp.get("decision") or {}).get("_parse_error"):
            reparsed = _reparse_raw_decision(cp.get("reasoning_raw"))
            if reparsed is not None:
                cp = {**cp, "decision": reparsed["decision"]}
        situation = situation_for_checkpoint(run, traj, cp)
        situations[cp["step"]] = situation
        findings += check_2_0_parse_format(cp)
        if not schema["is_legacy"]:
            findings += check_2_1_check_d(cp, situation, constraints)
            findings += check_2_1b_unclassified_encounter(cp, situation, constraints)
            findings += check_2_2_rule_matrix(cp, situation, run.get("config"))
            own_row = _nearest(traj.get("own_ship", {}), cp["time"])
            target_rows = {n: _nearest(s, cp["time"]) for n, s in traj.items() if n != "own_ship"}
            target_rows = {n: r for n, r in target_rows.items() if r is not None}
            findings += check_2_3_encounter_mismatch(cp, own_row, target_rows)
            findings += check_2_5_magnitude(cp, situation, constraints)
            findings += check_2_7_reasoning_vs_decision(cp, situation, constraints)
        findings += check_2_4_direction(cp, situation, constraints)
        findings += check_3_truncation(cp, (run.get("params") or {}).get("max_new_tokens"))
        findings += check_3_self_correction(cp)
        findings += check_3_repetition(cp)
        raw = cp.get("reasoning_raw") or cp.get("decision", {}).get("reasoning") or ""
        reasoning_lens.append(len(raw))
    if not schema["is_legacy"]:
        findings += check_2_6_temporal(checkpoints, situations)

    severity_counts = Counter(f["severity"] for f in findings)
    return {
        "path": str(path), "mission_id": run.get("mission_id"), "config": run.get("config"),
        "weights": run.get("weights", "W0_base"), "tag": run.get("tag"),
        "schema": schema, "findings": findings,
        "severity_counts": dict(severity_counts),
        "mean_reasoning_chars": statistics.fmean(reasoning_lens) if reasoning_lens else 0.0,
        "n_checkpoints": len(checkpoints),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation / report
# ─────────────────────────────────────────────────────────────────────────────
def build_summary(results: list[dict]) -> dict:
    by_config: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_mission_type: dict[str, list[dict]] = defaultdict(list)
    canary: list[dict] = []
    for r in results:
        by_config[(r["config"], r["weights"])].append(r)
        mtype = _mission_type(r["mission_id"])
        by_mission_type[mtype].append(r)
        if r["mission_id"] in QUIET_MISSIONS:
            canary.append(r)

    def _metrics(rs: list[dict]) -> dict:
        n_checkpoints = sum(r["n_checkpoints"] for r in rs) or 1
        code_counts = Counter()
        for r in rs:
            for f in r["findings"]:
                code_counts[f["code"]] += 1
        rate = lambda prefix: sum(v for k, v in code_counts.items() if k.startswith(prefix)) / n_checkpoints
        n_blockers = sum(r["severity_counts"].get("BLOCKER", 0) for r in rs)
        n_real_findings = sum(1 for r in rs for f in r["findings"] if f["severity"] != "INFO")
        return {
            "n_runs": len(rs), "n_checkpoints": n_checkpoints,
            "A_rate": rate("A_fabricated_risk"), "B_rate": rate("B_wrong_direction"),
            "D_rate": rate("D_"), "E_encounter_mismatch_rate": rate("E_encounter_mismatch"),
            "E_role_fabrication_rate": rate("E_role_fabrication"),
            "E_unclassified_encounter_rate": rate("E_unclassified_encounter"),
            "E_rule_matrix_none_with_conduct_rate": rate("E_rule_matrix_none_with_conduct"),
            "early_action_rate": rate("INFO_early_action"),
            "parse_fail_rate": rate("ERROR_2_0_parse_error"),
            "gate_a_fail_rate": rate("G_gate_a"), "gate_b_fail_rate": rate("G_gate_b"),
            # exact code, not the "G_gate_c" prefix -- that would also swallow the
            # informational G_gate_c_risk_unknown code into a "failure" rate
            "gate_c_fail_rate": rate("G_gate_c_risk_mismatch"),
            "gate_c_unknown_rate": rate("G_gate_c_risk_unknown"),
            "gate_d_fail_rate": rate("G_gate_d"),
            "gate_e_fail_rate": rate("G_gate_e"),
            "n_blockers": n_blockers, "n_real_findings": n_real_findings,
            "mean_composite": statistics.fmean(
                [(r.get("evaluation_composite") or 0) for r in rs]) if rs else 0.0,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(results),
        "schema_versions": Counter(r["schema"]["decision_schema"] for r in results),
        "by_config": {f"{c}::{w}": _metrics(rs) for (c, w), rs in by_config.items()},
        "by_mission_type": {m: _metrics(rs) for m, rs in by_mission_type.items()},
        "canary": _metrics(canary) if canary else None,
        "blockers": [f for r in results for f in r["findings"] if f["severity"] == "BLOCKER"],
        "top_worst": _top_worst_checkpoints(results),
    }


def _mission_type(mission_id: str) -> str:
    if mission_id in QUIET_MISSIONS:
        return "quiet"
    if mission_id.startswith("Imazu") or mission_id.startswith("s"):
        return "imazu_scenario"
    return "um_bearing_sweep"


def _top_worst_checkpoints(results: list[dict], n: int = 10) -> list[dict]:
    scored = []
    for r in results:
        by_step: dict[int, list[dict]] = defaultdict(list)
        for f in r["findings"]:
            if f["step"] is not None:
                by_step[f["step"]].append(f)
        for step, fs in by_step.items():
            weight = sum({"BLOCKER": 4, "ERROR": 2, "WARN": 1, "INFO": 0}.get(f["severity"], 0) for f in fs)
            scored.append((weight, r["path"], step, fs))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"path": p, "step": s, "codes": [f["code"] for f in fs], "weight": w}
            for w, p, s, fs in scored[:n]]


def render_markdown_report(summary: dict, baseline: dict | None = None,
                           baseline_reason: str | None = None) -> str:
    lines = ["# Audit report", "", f"Generated: {summary['generated_at']}", "",
            f"Schema versions seen: {dict(summary['schema_versions'])}", f"Runs audited: {summary['n_runs']}", ""]
    if summary["blockers"]:
        lines.append(f"## \U0001F6D1 {len(summary['blockers'])} BLOCKER(s)")
        for b in summary["blockers"][:50]:
            lines.append(f"- `{b['code']}` step={b['step']}: {b['message']}")
    else:
        lines.append("## \u2705 No blockers")
    lines.append("")
    lines.append("## Primary metrics per config x weights")
    lines.append("| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | "
                 "E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, m in summary["by_config"].items():
        lines.append(f"| {key} | {m['n_runs']} | {m['A_rate']:.2%} | {m['B_rate']:.2%} | "
                     f"{m['D_rate']:.2%} | {m['E_encounter_mismatch_rate']:.2%} | "
                     f"{m['E_role_fabrication_rate']:.2%} | {m['E_unclassified_encounter_rate']:.2%} | "
                     f"{m['E_rule_matrix_none_with_conduct_rate']:.2%} | "
                     f"{m['gate_b_fail_rate']:.2%} | {m['gate_c_fail_rate']:.2%} | "
                     f"{m['parse_fail_rate']:.2%} | {m['n_real_findings']} |")
    lines.append("")
    lines.append("## Per mission-type")
    lines.append("| type | runs | A-rate | B-rate | D-rate | E_unclass |")
    lines.append("|---|---|---|---|---|---|")
    for key, m in summary["by_mission_type"].items():
        lines.append(f"| {key} | {m['n_runs']} | {m['A_rate']:.2%} | {m['B_rate']:.2%} | "
                     f"{m['D_rate']:.2%} | {m['E_unclassified_encounter_rate']:.2%} |")
    lines.append("")
    if summary["canary"]:
        c = summary["canary"]
        lines.append(f"## Canary (UM01/UM02) -- A-rate should be ~0: **{c['A_rate']:.2%}**")
    lines.append("")
    lines.append("## Top-10 worst checkpoints")
    for w in summary["top_worst"]:
        lines.append(f"- weight={w['weight']} `{Path(w['path']).name}` step={w['step']}: {w['codes']}")
    if baseline_reason:
        lines.append("")
        lines.append(f"## Baseline comparison\nniet vergelijkbaar: {baseline_reason}")
    elif baseline:
        lines.append("")
        lines.append("## Baseline comparison")
        lines.append("| config::weights | A-rate now | A-rate baseline | delta |")
        lines.append("|---|---|---|---|")
        for key, m in summary["by_config"].items():
            bm = baseline["by_config"].get(key)
            if bm:
                lines.append(f"| {key} | {m['A_rate']:.2%} | {bm['A_rate']:.2%} | "
                             f"{m['A_rate'] - bm['A_rate']:+.2%} |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _load_runs(tag: str, configs: list[str] | None, missions: list[str] | None,
               runs_dir: Path | None = None) -> list[tuple[dict, Path]]:
    """Scans ONLY `runs_dir` (default RUNS_DIR), non-recursively -- never any sibling
    archive folder (e.g. "MIssions Data V1/", "Mission No Speed Increase/") and never its
    own _review/ subfolder, both of which glob("*.json") on RUNS_DIR itself can't reach.
    Pass an explicit `runs_dir` (e.g. RUNS_DIR / "screening_standard_cloud") to audit an
    archived subfolder ("set A") separately from the live top-level runs ("set B") --
    screening-set-B audit follow-up (2026-09-23)."""
    runs_dir = runs_dir or RUNS_DIR
    out = []
    schemas_seen: set[str] = set()
    prompt_hashes_seen: dict[str, Path] = {}
    for path in sorted(runs_dir.glob("*.json")):
        if path.name.startswith("_sweep_"):
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        parsed = parse_run_filename(path)
        run_tag = run.get("tag") or parsed["tag"]
        if run_tag != tag:
            continue
        if configs and (run.get("config") or parsed["config"]) not in configs:
            continue
        if missions and (run.get("mission_id") or parsed["mission_id"]) not in missions:
            continue
        schema = detect_schema(run, path)
        schemas_seen.add(schema["decision_schema"])
        # Screening-set-B audit follow-up (2026-09-23): a prompt_hash mismatch within one
        # tag means the runs were generated under DIFFERENT system-prompt/constraint-line
        # wording -- mixing them into one aggregate would silently blend two incomparable
        # populations, exactly the failure mode the existing schema-mixing guard already
        # prevents. Older runs with no prompt_hash at all (pre-dating this field) are
        # never compared -- only genuinely DIFFERENT non-null hashes are a hard error.
        prompt_hash = (run.get("params") or {}).get("prompt_hash")
        if prompt_hash:
            for seen_hash, seen_path in prompt_hashes_seen.items():
                if seen_hash != prompt_hash:
                    raise SystemExit(
                        f"Refusing to mix runs with different prompt_hash within tag {tag!r}: "
                        f"{seen_path.name} ({seen_hash[:12]}...) vs {path.name} ({prompt_hash[:12]}...).")
            prompt_hashes_seen[prompt_hash] = path
        out.append((run, path))
    if len(schemas_seen) > 1:
        raise SystemExit(f"Refusing to mix schemas {schemas_seen} within tag {tag!r} in one aggregate.")
    return out


def run_audit(tag: str, configs: list[str] | None, missions: list[str] | None,
             out_dir: Path, baseline_tag: str | None = None,
             runs_dir: Path | None = None, baseline_runs_dir: Path | None = None) -> int:
    runs = _load_runs(tag, configs, missions, runs_dir)
    if not runs:
        print(f"No runs found for tag={tag!r}.")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "audit_summary.json").write_text(json.dumps(
            {"n_runs": 0, "generated_at": datetime.now(timezone.utc).isoformat(),
             "by_config": {}, "by_mission_type": {}, "canary": None, "blockers": [], "top_worst": [],
             "schema_versions": {}, "baseline_tag": baseline_tag, "baseline": None,
             "baseline_reason": "no runs for this tag", "checkpoint_findings": []}, indent=1),
            encoding="utf-8")
        return 0

    # Deliberately NOT one *_audit.json copy per run -- that mirrored every run in _llm_runs/
    # into the output folder 1:1 and looked like (and functioned as) an unwanted duplicate of
    # production run data. Everything the dashboard needs (per-checkpoint findings included)
    # is folded into the single aggregate audit_summary.json instead.
    results = [audit_one_run(run, path) for run, path in runs]

    summary = build_summary(results)
    checkpoint_findings = [
        {"run_path": r["path"], "mission_id": r["mission_id"], "config": r["config"],
         "weights": r.get("weights"), "step": f["step"], "code": f["code"],
         "severity": f["severity"], "message": f["message"], "details": f.get("details")}
        for r in results for f in r["findings"] if f["step"] is not None
    ]
    baseline_summary, baseline_reason = None, None
    if baseline_tag:
        base_runs = _load_runs(baseline_tag, configs, missions, baseline_runs_dir)
        if not base_runs:
            baseline_reason = f"no runs found for baseline tag {baseline_tag!r}"
        else:
            base_results = [audit_one_run(r, p) for r, p in base_runs]
            base_schema = {r["schema"]["decision_schema"] for r in base_results}
            cur_schema = {r["schema"]["decision_schema"] for r in results}
            if base_schema != cur_schema:
                baseline_reason = f"schema mismatch ({cur_schema} vs {base_schema})"
            else:
                baseline_summary = build_summary(base_results)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_summary.json").write_text(
        json.dumps({**summary, "schema_versions": dict(summary["schema_versions"]),
                  "baseline_tag": baseline_tag, "baseline": baseline_summary,
                  "baseline_reason": baseline_reason, "checkpoint_findings": checkpoint_findings},
                  indent=1, default=str), encoding="utf-8")
    (out_dir / "audit_report.md").write_text(
        render_markdown_report(summary, baseline_summary, baseline_reason), encoding="utf-8")

    n_blockers = len(summary["blockers"])
    print(f"Audited {len(results)} run(s): {n_blockers} BLOCKER(s). Report: {out_dir / 'audit_report.md'}")
    return 1 if n_blockers else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--missions", nargs="+", default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--baseline-tag", default=None)
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="scan this directory instead of RUNS_DIR (non-recursive) -- "
                         "e.g. an archived subfolder like _llm_runs/screening_standard_cloud/")
    ap.add_argument("--baseline-runs-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (_ANALYSIS_DIR / "audit" / args.tag)
    sys.exit(run_audit(args.tag, args.configs, args.missions, out_dir, args.baseline_tag,
                       args.runs_dir, args.baseline_runs_dir))


if __name__ == "__main__":
    main()
