#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_run.py — composite, multi-axis scoring for a collision-avoidance
agent's trajectory, extending score_scenario.py's CPA logic.

Design, following (in spirit -- not reproducing exact undisclosed formulas)
the four-axis framework in Woerner, Benjamin, Novitzky & Leonard,
"Quantifying protocol evaluation for autonomous collision avoidance:
toward establishing COLREGS compliance metrics" (Autonomous Robots, 2019):
    - SAFETY        (hard gate on actual collisions, PLUS a weighted continuous
      term for near-misses -- see below)
    - COMPLIANCE    (COLREG rule violations)
    - TEMPORAL efficiency (time to reach the goal vs. a straight-line baseline)
    - SPATIAL efficiency  (distance sailed vs. straight-line distance)
plus two extras you specifically asked for:
    - MANOEUVRE COUNT (fewer, larger, decisive alterations preferred --
      this is also directly grounded in COLREG: Rule 8(b) explicitly says
      "a succession of small alterations of course and/or speed should be
      avoided", so penalizing manoeuvre count isn't just an efficiency
      nicety, it's a literal rule)
    - SMOOTHNESS (penalize large deltas in heading-rate / speed-rate --
      standard quadratic control-effort penalty from optimal control)

INPUT FORMAT
    Same trajectory CSV as score_scenario.py: time,vehicle,x,y,heading,speed.
"""
import argparse
import csv
import math
from collections import defaultdict


# ---------------------------------------------------------------------
# Trajectory loading (shared with score_scenario.py's format)
# ---------------------------------------------------------------------
def load_csv(path):
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["vehicle"]].append((
                float(row["time"]), float(row["x"]), float(row["y"]),
                float(row.get("heading", 0.0)), float(row.get("speed", 0.0)),
            ))
    for v in data:
        data[v].sort(key=lambda r: r[0])
    return data


def interp_xy(series, t):
    if not series or t < series[0][0] or t > series[-1][0]:
        return None
    for i in range(len(series) - 1):
        t0, x0, y0, *_ = series[i]
        t1, x1, y1, *_ = series[i + 1]
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
    return None


# ---------------------------------------------------------------------
# Safety axis (gate)
# ---------------------------------------------------------------------
def min_cpa_over_run(own, target, dt=1.0):
    t_start = max(own[0][0], target[0][0])
    t_end = min(own[-1][0], target[-1][0])
    best = float("inf")
    t = t_start
    while t <= t_end:
        po, pt = interp_xy(own, t), interp_xy(target, t)
        if po and pt:
            best = min(best, math.hypot(po[0] - pt[0], po[1] - pt[1]))
        t += dt
    return best


def safety_axis(own, targets, collision_radius_m, safe_distance_m):
    """Returns (passed: bool, min_cpa_overall: float, safety_score: float in [0,1]).
    passed=False means an actual collision occurred -- caller should gate
    the whole composite score to 0/FAIL on this, not just lower it."""
    if not targets:
        return True, float("inf"), 1.0
    worst = min(min_cpa_over_run(own, t) for t in targets.values())
    collided = worst < collision_radius_m
    # smooth reward for clearing the safety threshold; no extra credit for
    # excessive distance beyond it (avoid rewarding over-conservatism)
    score = 0.0 if collided else min(1.0, worst / safe_distance_m)
    return (not collided), worst, score


# ---------------------------------------------------------------------
# Temporal / spatial efficiency
# ---------------------------------------------------------------------
def path_length(series):
    total = 0.0
    for i in range(1, len(series)):
        _, x0, y0, *_ = series[i - 1]
        _, x1, y1, *_ = series[i]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def efficiency_axes(own, start_xy, goal_xy, nominal_speed, reached_radius_m=25.0):
    straight_dist = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
    baseline_time = straight_dist / nominal_speed if nominal_speed > 0 else float("inf")

    # did it actually arrive? find first time within reached_radius_m of goal
    t_arrival = None
    for t, x, y, hdg, spd in own:
        if math.hypot(x - goal_xy[0], y - goal_xy[1]) <= reached_radius_m:
            t_arrival = t
            break

    if t_arrival is None:
        # never arrived: this is the "stop and wait forever" / "endless
        # detour" case. Score both efficiency axes at 0 rather than letting
        # them be undefined or infinite -- an explicit failure mode.
        return {
            "arrived": False, "time_actual_s": None, "time_ratio": None,
            "temporal_score": 0.0,
            "path_length_m": path_length(own), "path_ratio": None,
            "spatial_score": 0.0,
        }

    t_start = own[0][0]
    time_actual = t_arrival - t_start
    time_ratio = time_actual / baseline_time if baseline_time > 0 else float("inf")
    # score: 1.0 at the baseline, decaying as you take longer; never negative
    temporal_score = max(0.0, min(1.0, 1.0 / time_ratio)) if time_ratio > 0 else 0.0

    own_up_to_arrival = [p for p in own if p[0] <= t_arrival]
    actual_path = path_length(own_up_to_arrival)
    path_ratio = actual_path / straight_dist if straight_dist > 0 else float("inf")
    spatial_score = max(0.0, min(1.0, 1.0 / path_ratio)) if path_ratio > 0 else 0.0

    return {
        "arrived": True, "time_actual_s": time_actual, "time_ratio": time_ratio,
        "temporal_score": temporal_score,
        "path_length_m": actual_path, "path_ratio": path_ratio,
        "spatial_score": spatial_score,
    }


# ---------------------------------------------------------------------
# Manoeuvre count + smoothness
# ---------------------------------------------------------------------
def heading_delta(h1, h0):
    return (h1 - h0 + 540) % 360 - 180


def manoeuvre_and_smoothness_axes(own, heading_rate_deadband_deg_s=0.6,
                                   max_reasonable_manoeuvres=6,
                                   max_reasonable_heading_rate=5.0,
                                   max_reasonable_speed_rate=1.0):
    """Counts discrete manoeuvre events and computes a smoothness penalty
    from heading-rate / speed-rate variance (quadratic control-effort
    penalty). A manoeuvre event starts when heading-rate crosses the
    deadband from below, OR when it reverses sign while still above the
    deadband -- the latter is essential to catch a continuous zigzag
    (alternating small corrections that never drop back to near-zero
    heading-rate would otherwise be counted as a single sustained
    manoeuvre instead of the many discrete corrections they actually are;
    confirmed against a synthetic zigzag trajectory during testing, where
    the sign-reversal check was the difference between counting 1
    manoeuvre and correctly counting dozens).

    The deadband is a heading-RATE threshold (deg/s), not a raw per-step
    degrees threshold -- comparing dt-scaled degrees against a fixed
    constant previously made manoeuvre/smoothness scores dependent on the
    trajectory's sampling interval dt (the Basic Simulator's adjustable
    1-60s time-step slider), so identical maneuvering behaviour scored
    differently just from a different dt. A rate-based threshold is
    dt-invariant by construction."""
    if len(own) < 2:
        return {"manoeuvre_count": 0, "manoeuvre_score": 1.0,
                "mean_abs_heading_rate": 0.0, "mean_abs_speed_rate": 0.0,
                "smoothness_score": 1.0}

    heading_rates, speed_rates = [], []
    manoeuvre_count = 0
    in_manoeuvre = False
    last_sign = 0

    for i in range(1, len(own)):
        t0, x0, y0, h0, s0 = own[i - 1]
        t1, x1, y1, h1, s1 = own[i]
        dt = max(1e-6, t1 - t0)
        dh = heading_delta(h1, h0)
        heading_rate = dh / dt
        heading_rates.append(abs(heading_rate))
        speed_rates.append(abs(s1 - s0) / dt)

        above_deadband = abs(heading_rate) > heading_rate_deadband_deg_s
        sign = (1 if dh > 0 else -1) if above_deadband else 0

        if above_deadband:
            if not in_manoeuvre:
                manoeuvre_count += 1
                in_manoeuvre = True
            elif last_sign != 0 and sign != last_sign:
                # direction reversal mid-manoeuvre: a new, separate event
                manoeuvre_count += 1
            last_sign = sign
        else:
            in_manoeuvre = False
            last_sign = 0

    mean_hr = sum(heading_rates) / len(heading_rates)
    mean_sr = sum(speed_rates) / len(speed_rates)

    manoeuvre_score = max(0.0, 1.0 - manoeuvre_count / max_reasonable_manoeuvres)
    smoothness_hr = max(0.0, 1.0 - mean_hr / max_reasonable_heading_rate)
    smoothness_sr = max(0.0, 1.0 - mean_sr / max_reasonable_speed_rate)
    smoothness_score = 0.5 * smoothness_hr + 0.5 * smoothness_sr

    return {
        "manoeuvre_count": manoeuvre_count, "manoeuvre_score": manoeuvre_score,
        "mean_abs_heading_rate": mean_hr, "mean_abs_speed_rate": mean_sr,
        "smoothness_score": smoothness_score,
    }


# ---------------------------------------------------------------------
# COLREG compliance axis (compliance-rebuild STAP 3, 2026-09-23)
# ---------------------------------------------------------------------
# One weighted deduction PER OCCURRENCE of a code, starting from 1.0, clipped to [0,1] --
# no other numbers live in compliance_axis() itself. checkpoint-level codes: measurement.
# py's A_fabricated_risk/B_wrong_direction/C_degrees_over_limit/D_no_action_when_required,
# plus app.evaluation's auditor codes (E_encounter_mismatch/E_role_fabrication/
# E_unclassified_encounter/B_17c/E_8c/P_port_toward_contact), all computed against
# app.evaluation._ground_truth_at_checkpoint()'s STAP-2 dict, never real_risk() alone.
# run-level codes: P_wrong_side_pass (per offending contact) and cpa_violation (once, if
# min separation anywhere in the run fell below safe_distance_m without an actual
# collision) -- "collision" is a separate hard gate, not a weighted deduction (see below).
COMPLIANCE_WEIGHTS = {
    "B_wrong_direction": 0.15,
    "P_port_toward_contact": 0.15,
    "E_role_fabrication": 0.15,
    "D_no_action_when_required": 0.15,
    "E_encounter_mismatch": 0.05,
    "E_unclassified_encounter": 0.05,
    "A_fabricated_risk": 0.03,
    "C_degrees_over_limit": 0.03,
    "B_17c": 0.15,
    "E_8c": 0.05,
    "P_wrong_side_pass": 0.30,
    "cpa_violation": 0.30,
}

# Short, demo-facing English label per code -- single source of truth (2026-09-24), baked
# straight into every NEW run's compliance.breakdown entries so the raw JSON is
# self-explanatory without a code lookup table. Every COMPLIANCE_WEIGHTS key plus the
# "collision" hard-gate code (not itself weighted, see compliance_axis()) must appear here
# -- see test_compliance_axis.py's test_every_weighted_code_has_a_label. Older, already-
# generated runs are NOT migrated -- their breakdown entries stay the bare
# [code, step_or_detail, deduction] lists they were written with; app/streamlit_app.py and
# app/sweep_dashboard.py fall back to this same dict to label them at DISPLAY time instead.
COMPLIANCE_LABELS = {
    "A_fabricated_risk": "Hallucinated Risk",
    "B_wrong_direction": "Wrong Turn Direction",
    "C_degrees_over_limit": "Implausibly Large Turn Order",
    "D_no_action_when_required": "No Action Despite Risk",
    "E_encounter_mismatch": "Wrong Rule Cited",
    "E_role_fabrication": "Rule Cited Without Real Encounter",
    "E_unclassified_encounter": "Encounter Not Recognized",
    "B_17c": "Stand-On Vessel Acted Too Early",
    "E_8c": "Misapplied Emergency-Stop Rule",
    "P_port_toward_contact": "Turned Toward Contact",
    "P_wrong_side_pass": "Wrong Passing Side",
    "cpa_violation": "Safe-Distance Violation",
    "collision": "Collision",
}

# 2026-09-24 split: every weighted code is either about what the ship physically DID
# ("manoeuvre" -- was the actual turn/hold/stop COLREG-correct and safe, recomputed
# purely from the recorded trajectory) or about what the model SAID about it
# ("explanation" -- does the self-reported encounter_rule/conduct_rule/"real risk" claim
# match the geometric ground truth). These used to be blended into one "compliance"
# number, which let citation/labelling mistakes alone (e.g. a benign, safe manoeuvre
# mislabelled with a fabricated rule) crash the score to 0.0 exactly as hard as an actual
# wrong-direction turn into a contact -- see compliance_axis()/explanation_axis() below.
COMPLIANCE_CATEGORY = {
    "B_wrong_direction": "manoeuvre",
    "C_degrees_over_limit": "manoeuvre",
    "D_no_action_when_required": "manoeuvre",
    "B_17c": "manoeuvre",
    "P_port_toward_contact": "manoeuvre",
    "P_wrong_side_pass": "manoeuvre",
    "cpa_violation": "manoeuvre",
    "A_fabricated_risk": "explanation",
    "E_encounter_mismatch": "explanation",
    "E_role_fabrication": "explanation",
    "E_unclassified_encounter": "explanation",
    "E_8c": "explanation",
}


def _scored_axis(checkpoint_codes, run_level_codes, collided, category):
    """Shared implementation for compliance_axis()/explanation_axis() -- identical
    deduction mechanics (one weighted deduction per occurrence, starting from 1.0,
    clipped to [0,1]), differing only in which codes' category is being scored."""
    if collided:
        return 0.0, [{"code": "collision", "label": COMPLIANCE_LABELS["collision"],
                      "at": None, "deduction": -1.0}]
    breakdown = []
    score = 1.0
    for step, codes in checkpoint_codes:
        for code in codes:
            if COMPLIANCE_CATEGORY.get(code) != category:
                continue
            weight = COMPLIANCE_WEIGHTS.get(code)
            if weight:
                score -= weight
                breakdown.append({"code": code, "label": COMPLIANCE_LABELS.get(code, code),
                                 "at": step, "deduction": -weight})
    for code, detail in run_level_codes:
        if COMPLIANCE_CATEGORY.get(code) != category:
            continue
        weight = COMPLIANCE_WEIGHTS.get(code)
        if weight:
            score -= weight
            breakdown.append({"code": code, "label": COMPLIANCE_LABELS.get(code, code),
                             "at": detail, "deduction": -weight})
    return max(0.0, min(1.0, score)), breakdown


def compliance_axis(checkpoint_codes, run_level_codes, collided=False):
    """Deterministic MANOEUVRE-compliance score -- no LLM call, always computable.
    Answers "was the physical action own-ship took COLREG-correct and safe", using only
    the "manoeuvre"-category codes (B_wrong_direction, C_degrees_over_limit,
    D_no_action_when_required, B_17c, P_port_toward_contact, P_wrong_side_pass,
    cpa_violation) -- see COMPLIANCE_CATEGORY. Citation/labelling accuracy is scored
    separately by explanation_axis(), never blended in here (2026-09-24 split -- see its
    own comment above COMPLIANCE_CATEGORY for why).

    checkpoint_codes: list of (step_label, [code, ...]) -- one entry per audited
    checkpoint, each code in COMPLIANCE_WEIGHTS deducted once per occurrence.
    run_level_codes: list of (code, detail) -- trajectory-level findings (detail is
    typically a contact name or None), same deduction table.
    collided=True is a HARD GATE: returns (0.0, [{"code": "collision", ...}]) regardless
    of every other input -- an actual collision makes the rest of the audit moot, exactly
    like safety_axis()'s own gate on the composite score.

    Returns (score: float in [0,1], breakdown: list of {"code", "label", "at"
    (step_or_detail), "deduction"} dicts) -- breakdown's deductions always sum to
    score - 1.0 (before the final clip), so every score is traceable back to the specific
    findings that produced it."""
    return _scored_axis(checkpoint_codes, run_level_codes, collided, "manoeuvre")


def explanation_axis(checkpoint_codes, run_level_codes, collided=False):
    """Deterministic EXPLANATION-compliance score -- the counterpart to compliance_axis().
    Answers "did the model's own stated encounter_rule/conduct_rule/'real risk' claim
    match the geometric ground truth", using only the "explanation"-category codes
    (A_fabricated_risk, E_encounter_mismatch, E_role_fabrication, E_unclassified_encounter,
    E_8c) -- see COMPLIANCE_CATEGORY. Deliberately independent of whether the physical
    manoeuvre itself was safe: a benign, safe hold_course mislabelled with a fabricated
    rule scores badly HERE, not on compliance_axis(), and vice versa. Same signature/
    mechanics as compliance_axis() (see its docstring) -- same collision hard-gate too,
    since there is no meaningful citation-accuracy story left to tell once a run has
    actually collided."""
    return _scored_axis(checkpoint_codes, run_level_codes, collided, "explanation")


# ---------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------
# 2026-09-26: "safety" is now a weighted term, not just a hard collision gate --
# previously a razor-thin near-miss (e.g. 26.5m CPA, safety_score=0.053) fed the
# composite ONLY through a flat, one-time -0.30 "cpa_violation" compliance
# deduction, identical regardless of how close the near-miss actually was. That
# let PASS_WITH_CPA_VIOLATION runs that were seconds from an actual collision
# still score ~0.9. safety_score is continuous (min_cpa/safe_distance_m) and now
# directly drags the composite down in proportion to how close the call was.
DEFAULT_WEIGHTS = {
    "safety": 0.35,
    "compliance": 0.20,
    "temporal": 0.10,
    "spatial": 0.10,
    "manoeuvre": 0.10,
    "smoothness": 0.15,
}

def evaluate_run(csv_path, own_vehicle, start_xy, goal_xy, nominal_speed,
                  collision_radius_m=15.0, safe_distance_m=50.0, reached_radius_m=25.0,
                  checkpoint_codes=(), run_level_codes=(), weights=None, verbose=True):
    weights = weights or DEFAULT_WEIGHTS
    data = load_csv(csv_path)
    own = data[own_vehicle]
    targets = {k: v for k, v in data.items() if k != own_vehicle}

    passed, min_cpa, safety_score = safety_axis(own, targets, collision_radius_m, safe_distance_m)
    eff = efficiency_axes(own, start_xy, goal_xy, nominal_speed, reached_radius_m=reached_radius_m)
    man = manoeuvre_and_smoothness_axes(own)
    # cpa_violation is derived HERE (min_cpa is already computed above for safety_axis)
    # rather than asked of the caller -- a genuine near-miss that never reached an actual
    # collision is still always a run-level compliance fact, not something callers should
    # have to remember to compute themselves.
    all_run_level_codes = list(run_level_codes)
    if passed and min_cpa < safe_distance_m:
        all_run_level_codes.append(("cpa_violation", None))
    compliance_score, compliance_breakdown = compliance_axis(
        checkpoint_codes, all_run_level_codes, collided=not passed)
    # 2026-09-24 split (see COMPLIANCE_CATEGORY's comment): explanation_score never feeds
    # the composite/verdict below -- it's reported alongside compliance purely as a
    # SEPARATE signal (was the model's stated reasoning/rule-citation accurate), so a run
    # with a perfectly safe, COLREG-correct manoeuvre but sloppy self-reported labelling
    # no longer gets its PASS/FAIL and composite score dragged down for that alone.
    explanation_score, explanation_breakdown = explanation_axis(
        checkpoint_codes, all_run_level_codes, collided=not passed)

    if not passed:
        composite = 0.0
        verdict = "FAIL -- collision occurred"
    elif not eff["arrived"]:
        # Mission-incomplete gate: reaching the goal is the actual point of
        # the mission, not just one axis among several. Without this cap, a
        # vessel that stops and never moves (or wanders forever without
        # arriving) still earns full marks on manoeuvre-count and
        # smoothness -- trivially, since standing still involves no
        # manoeuvres and no control effort at all -- letting those "free"
        # scores drag the composite up to a misleadingly high number
        # (confirmed in testing: without this cap, a stopped vessel scored
        # 0.70, which does not reflect "never completed the mission" as a
        # serious failure). Capped low regardless of how clean the partial
        # trajectory looked.
        composite = min(0.2, weights["compliance"] * compliance_score +
                             weights["manoeuvre"] * man["manoeuvre_score"] +
                             weights["smoothness"] * man["smoothness_score"])
        verdict = "FAIL -- did not reach the goal"
    else:
        composite = (
            weights["safety"] * safety_score +
            weights["compliance"] * compliance_score +
            weights["temporal"] * eff["temporal_score"] +
            weights["spatial"] * eff["spatial_score"] +
            weights["manoeuvre"] * man["manoeuvre_score"] +
            weights["smoothness"] * man["smoothness_score"]
        )
        # No literal collision and the goal was reached, but a genuine COLREG-relevant
        # near-miss (min separation below this mission's own safe_distance_m, just not
        # close enough to be a hull-to-hull collision) must never be silently absorbed
        # into a bare "PASS" -- found via 2026-09-23 audit: 2 runs at 359-395m min
        # separation (safe_distance 500m) were labelled plain PASS with no safety flag
        # anywhere in the record.
        verdict = "PASS" if min_cpa >= safe_distance_m else "PASS_WITH_CPA_VIOLATION"

    result = {
        "verdict": verdict, "composite_score": round(composite, 3),
        "safety": {"passed": passed, "min_cpa_m": round(min_cpa, 1) if min_cpa != float("inf") else None,
                   "score": round(safety_score, 3)},
        "compliance": {"breakdown": compliance_breakdown, "score": round(compliance_score, 3)},
        "explanation_compliance": {"breakdown": explanation_breakdown, "score": round(explanation_score, 3)},
        "temporal": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in eff.items()
                     if k in ("arrived", "time_actual_s", "time_ratio", "temporal_score")},
        "spatial": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in eff.items()
                    if k in ("path_length_m", "path_ratio", "spatial_score")},
        "manoeuvre": {k: round(v, 3) if isinstance(v, float) else v for k, v in man.items()},
    }
    if verbose:
        print(f"Verdict: {result['verdict']}   Composite score: {result['composite_score']}")
        print(f"  Safety:     min CPA {result['safety']['min_cpa_m']} m -> score {result['safety']['score']}")
        print(f"  Compliance: {len(compliance_breakdown)} finding(s) -> score {result['compliance']['score']}")
        print(f"  Explanation: {len(explanation_breakdown)} finding(s) -> score {result['explanation_compliance']['score']}")
        print(f"  Temporal:   arrived={eff['arrived']}, ratio={eff.get('time_ratio')} -> score {eff['temporal_score']:.3f}")
        print(f"  Spatial:    ratio={eff.get('path_ratio')} -> score {eff['spatial_score']:.3f}")
        print(f"  Manoeuvre:  count={man['manoeuvre_count']} -> score {man['manoeuvre_score']:.3f}")
        print(f"  Smoothness: mean|dHdg/dt|={man['mean_abs_heading_rate']:.2f} deg/s, "
              f"mean|dSpd/dt|={man['mean_abs_speed_rate']:.2f} -> score {man['smoothness_score']:.3f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--own-ship", default="opship")
    ap.add_argument("--start-x", type=float, required=True)
    ap.add_argument("--start-y", type=float, required=True)
    ap.add_argument("--goal-x", type=float, required=True)
    ap.add_argument("--goal-y", type=float, required=True)
    ap.add_argument("--nominal-speed", type=float, required=True)
    ap.add_argument("--collision-radius", type=float, default=15.0)
    ap.add_argument("--safe-distance", type=float, default=50.0)
    args = ap.parse_args()
    evaluate_run(args.csv_path, args.own_ship, (args.start_x, args.start_y),
                 (args.goal_x, args.goal_y), args.nominal_speed,
                 args.collision_radius, args.safe_distance)


if __name__ == "__main__":
    main()
