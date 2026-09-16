#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_scenario.py — post-run scoring for the COLREG MOOS-IvP evaluation missions.

INPUT FORMAT
This script scores a simple CSV trajectory log with columns:
    time,vehicle,x,y,heading,speed
one row per vehicle per report.

Why CSV and not raw .alog directly: MOOS .alog line format and the exact
NODE_REPORT string schema vary a bit by MOOS-IvP version/config, so rather
than guess at a parser that might silently mis-parse your specific version,
this script takes the portable, verifiable CSV format below. Two ways to
produce it from a real run:

  1) Easiest: add a small custom logging hook (e.g. a short uFldNodeBroker/
     pLogger post-process, or a one-off Python script using pymoos to
     subscribe to NODE_REPORT_LOCAL and append rows) that writes exactly
     this CSV as the mission runs.
  2) From an existing .alog: grep for NODE_REPORT lines and extract the
     NAME=, X=, Y=, HDG=/HEADING=, SPD=/SPEED= fields with your MOOS-IvP
     version's `aloggrep`/`alogscan` tools, then reshape into this CSV
     with a short script. Field names in NODE_REPORT strings are stable
     across versions even when the surrounding .alog line format isn't,
     so this is the safest extraction point.

USAGE
    python3 score_scenario.py trajectory.csv --scenario s01_head_on \
        --safe-distance 50 --manifest scenarios_manifest.json
"""
import argparse
import csv
import json
import math
from collections import defaultdict


def load_csv(path):
    data = defaultdict(list)  # vehicle -> list of (t, x, y, heading, speed)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["vehicle"]].append((
                float(row["time"]), float(row["x"]), float(row["y"]),
                float(row.get("heading", 0.0)), float(row.get("speed", 0.0)),
            ))
    for v in data:
        data[v].sort(key=lambda r: r[0])
    return data


def interp(series, t):
    """Linear interpolation of (x,y) at time t from a sorted list of
    (t,x,y,heading,speed) tuples. Returns None if t is out of range."""
    if not series or t < series[0][0] or t > series[-1][0]:
        return None
    for i in range(len(series) - 1):
        t0, x0, y0, h0, s0 = series[i]
        t1, x1, y1, h1, s1 = series[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return (x0, y0)
            f = (t - t0) / (t1 - t0)
            return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
    return None


def compute_cpa(series_a, series_b, dt=1.0):
    """Scan the overlapping time range of two vehicle trajectories and
    return (min_distance, time_of_min_distance)."""
    t_start = max(series_a[0][0], series_b[0][0])
    t_end = min(series_a[-1][0], series_b[-1][0])
    if t_end <= t_start:
        return None, None
    best_d, best_t = float("inf"), None
    t = t_start
    while t <= t_end:
        pa = interp(series_a, t)
        pb = interp(series_b, t)
        if pa and pb:
            d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            if d < best_d:
                best_d, best_t = d, t
        t += dt
    return best_d, best_t


def which_side_passed(series_os, series_ts, t_cpa, window=30.0):
    """Rough heuristic: compare own-ship's heading to the bearing of the
    target just before CPA to classify a starboard or port passing."""
    t_before = t_cpa - window
    pa = interp(series_os, t_before)
    pb = interp(series_ts, t_before)
    if not pa or not pb:
        return "unknown"
    # bearing from OS to TS at t_before
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    # find OS heading at t_before
    os_heading = None
    for (t, x, y, h, s) in series_os:
        if t >= t_before:
            os_heading = h
            break
    if os_heading is None:
        return "unknown"
    rel = (bearing - os_heading + 360) % 360
    return "starboard" if rel <= 180 else "port"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to trajectory CSV (time,vehicle,x,y,heading,speed)")
    ap.add_argument("--scenario", required=True, help="Scenario id, e.g. s01_head_on")
    ap.add_argument("--own-ship", default="opship")
    ap.add_argument("--safe-distance", type=float, default=50.0,
                     help="Minimum acceptable CPA in metres (default 50)")
    ap.add_argument("--manifest", default="scenarios_manifest.json")
    args = ap.parse_args()

    data = load_csv(args.csv_path)
    if args.own_ship not in data:
        raise SystemExit(f"No trajectory data found for own-ship vehicle '{args.own_ship}'")

    manifest = {}
    try:
        with open(args.manifest) as f:
            m = json.load(f)
            manifest = {e["id"]: e for e in m}
    except FileNotFoundError:
        print(f"(manifest file '{args.manifest}' not found — scoring without rule context)")

    target_names = [v for v in data if v != args.own_ship]
    print(f"Scenario: {args.scenario}")
    if args.scenario in manifest:
        info = manifest[args.scenario]
        print(f"  Rule(s): {', '.join(info['rule_refs'])}")
        print(f"  Own-ship role: {info['own_ship_role']}")
    print(f"  Targets found in log: {target_names}")
    print()

    all_pass = True
    for tname in target_names:
        cpa, t_cpa = compute_cpa(data[args.own_ship], data[tname])
        if cpa is None:
            print(f"  [{tname}] No overlapping trajectory data — cannot score.")
            all_pass = False
            continue
        side = which_side_passed(data[args.own_ship], data[tname], t_cpa)
        status = "PASS" if cpa >= args.safe_distance else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{tname}] min CPA = {cpa:.1f} m at t={t_cpa:.1f}s "
              f"(threshold {args.safe_distance:.1f} m) -> {status}")
        print(f"           own-ship appears to pass on the {side} side of {tname}")

    print()
    print(f"Overall: {'PASS' if all_pass else 'FAIL'} "
          f"(all target CPAs >= {args.safe_distance:.1f} m)")
    print()
    print("Note: 'which side passed' is a simple heading/bearing heuristic for a quick "
          "sanity check, not a substitute for manually reviewing the trajectory plot or a "
          "proper rule-compliance checker (e.g. did own-ship alter to starboard rather than "
          "port in a head-on/crossing-give-way case, did she hold course as stand-on, etc.). "
          "Extend this script's checks per-scenario using the pass_criteria in "
          "scenarios_manifest.json for stricter automated grading.")


if __name__ == "__main__":
    main()
