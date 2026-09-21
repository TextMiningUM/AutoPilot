#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_run.py — composite, multi-axis scoring for a collision-avoidance
agent's trajectory, extending score_scenario.py's CPA logic.

Design, following (in spirit -- not reproducing exact undisclosed formulas)
the four-axis framework in Woerner, Benjamin, Novitzky & Leonard,
"Quantifying protocol evaluation for autonomous collision avoidance:
toward establishing COLREGS compliance metrics" (Autonomous Robots, 2019):
    - SAFETY        (hard gate, not a weighted term -- see below)
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

WHY SAFETY IS A GATE, NOT A WEIGHTED TERM
    If every axis is blended into one weighted sum, "stop and never move"
    can look attractive to an optimizer: it trivially maximizes the safety
    term, and a large-enough safety weight can outweigh bad efficiency
    scores. The actual fix isn't just "weight efficiency heavily" -- it's
    structural: a run that produces an actual collision is scored 0
    (or excluded / marked FAIL) regardless of every other axis. Within the
    set of runs that did NOT collide, efficiency/compliance/smoothness then
    genuinely differentiate a good run from a merely-safe one. This also
    makes "stop and wait forever" score badly on its own terms: it never
    reaches the goal, so temporal efficiency is unbounded-bad (or scored 0
    if you cap it), not something a safety bonus can buy back.

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
# COLREG compliance axis (pluggable — this needs scenario-specific rule
# checks; a couple of concrete, generically-useful ones are included)
# ---------------------------------------------------------------------
def compliance_axis(own, violation_checks, llm_compliance_score=None):
    """violation_checks: list of callables(own_trajectory) -> list[str] (each returns a
    list of human-readable violation descriptions, empty if none found) -- kept only to
    surface violation text for display; the actual SCORE now comes directly from
    `llm_compliance_score` (Claude's own 0-1 audit judgement, see app.evaluation.
    llm_compliance_check) instead of a local `1 - 0.34*count` decay, because a bare
    violation COUNT can't tell a technical lateness apart from a violation that caused an
    actual near-miss or collision -- Claude's score is asked to weigh severity, not just
    tally rule numbers.

    Score = 0.0 (NOT innocent-until-proven -- unaudited) until `llm_compliance_score` is
    given, i.e. until the on-demand Claude COLREG audit has actually been run once for this
    trajectory and its result passed in here."""
    violations = []
    for check in violation_checks:
        violations.extend(check(own))
    score = 0.0 if llm_compliance_score is None else max(0.0, min(1.0, llm_compliance_score))
    return violations, score


def check_gave_way_to_port_when_should_be_starboard(own_role_by_time):
    """Example concrete check: flags any interval where own-ship was the
    give-way vessel in a crossing/head-on situation and her heading moved
    to PORT (negative delta) rather than starboard. own_role_by_time:
    list of (t, role, heading) tuples you supply from your own encounter
    classification, not reconstructed here from x/y alone."""
    violations = []
    for i in range(1, len(own_role_by_time)):
        t0, role0, h0 = own_role_by_time[i - 1]
        t1, role1, h1 = own_role_by_time[i]
        if role1 == "give_way" and heading_delta(h1, h0) < -2.0:
            violations.append(f"t={t1:.0f}s: altered to port while give-way "
                              f"(heading {h0:.1f} -> {h1:.1f})")
    return violations


# ---------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "compliance": 0.30,
    "temporal": 0.15,
    "spatial": 0.15,
    "manoeuvre": 0.15,
    "smoothness": 0.25,
}

def evaluate_run(csv_path, own_vehicle, start_xy, goal_xy, nominal_speed,
                  collision_radius_m=15.0, safe_distance_m=50.0, reached_radius_m=25.0,
                  violation_checks=(), weights=None, verbose=True,
                  llm_compliance_score=None):
    weights = weights or DEFAULT_WEIGHTS
    data = load_csv(csv_path)
    own = data[own_vehicle]
    targets = {k: v for k, v in data.items() if k != own_vehicle}

    passed, min_cpa, safety_score = safety_axis(own, targets, collision_radius_m, safe_distance_m)
    eff = efficiency_axes(own, start_xy, goal_xy, nominal_speed, reached_radius_m=reached_radius_m)
    man = manoeuvre_and_smoothness_axes(own)
    violations, compliance_score = compliance_axis(own, violation_checks, llm_compliance_score)

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
            weights["compliance"] * compliance_score +
            weights["temporal"] * eff["temporal_score"] +
            weights["spatial"] * eff["spatial_score"] +
            weights["manoeuvre"] * man["manoeuvre_score"] +
            weights["smoothness"] * man["smoothness_score"]
        )
        verdict = "PASS"

    result = {
        "verdict": verdict, "composite_score": round(composite, 3),
        "safety": {"passed": passed, "min_cpa_m": round(min_cpa, 1) if min_cpa != float("inf") else None,
                   "score": round(safety_score, 3)},
        "compliance": {"violations": violations, "score": round(compliance_score, 3)},
        "temporal": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in eff.items()
                     if k in ("arrived", "time_actual_s", "time_ratio", "temporal_score")},
        "spatial": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in eff.items()
                    if k in ("path_length_m", "path_ratio", "spatial_score")},
        "manoeuvre": {k: round(v, 3) if isinstance(v, float) else v for k, v in man.items()},
    }
    if verbose:
        print(f"Verdict: {result['verdict']}   Composite score: {result['composite_score']}")
        print(f"  Safety:     min CPA {result['safety']['min_cpa_m']} m -> score {result['safety']['score']}")
        print(f"  Compliance: {len(violations)} violation(s) -> score {result['compliance']['score']}")
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
