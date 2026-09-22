"""Extensive analysis of Data/missions/_llm_runs/*.json (Imazu01-09 encounter missions +
UM01-10 quiet missions, tag=units_v1, configs bare_qwen/v0_base..v6_pg_scenario).

Read-only analysis for the "top 10 problems per model v0-v6" report requested 2026-09-22.
Prints per-config aggregates + concrete failure-pattern counts/examples. Run with the
project venv from repo root: .venv\\Scripts\\python.exe "Basic Simulator/_tmp_analyze_mission_runs_v2.py"
"""
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict, Counter

RUNS_DIR = Path(__file__).parent / "Data" / "missions"  # search the whole missions tree, not just
# _llm_runs/ -- run logs get archived into dated/named subfolders (e.g. "Missions data v2
# 20260922/") that move around; rglob + the tag filter below finds them wherever they are.
TAG = "units_v1"

CONFIGS_ORDER = ["bare_qwen", "v0_base", "v1_rag", "v2_cot", "v3_rag_cot",
                 "v4_pg", "v5_pg_incident", "v6_pg_scenario"]

# Recursive: run logs now live under a dated archive subfolder (e.g. "Missions data v2
# 20260922/"), not directly in _llm_runs/ -- rglob finds them regardless of which
# subfolder (or none) a given batch was archived into.
files = sorted(p for p in RUNS_DIR.rglob(f"*__*__{TAG}.json"))
print(f"Found {len(files)} run files (tag={TAG})\n")

# ---- per-config aggregate structures ----
agg = defaultdict(lambda: {
    "n": 0, "n_imazu": 0, "n_um": 0,
    "composite": [], "composite_imazu": [], "composite_um": [],
    "verdicts": Counter(),
    "min_cpa": [], "n_unsafe": 0,
    "compliance_score": [], "n_violations": 0, "violation_texts": [],
    "n_fabricated_risk": 0, "n_wrong_rule_number": 0, "n_wrong_side": 0,
    "arrived": 0, "not_arrived": 0, "time_ratio": [],
    "manoeuvre_count": [], "n_zigzag": 0, "n_over30": 0,
    "n_never_resume_after_slow": 0,
    "n_quiet_false_positive": 0,  # UM01/UM02 (genuinely no-risk) where agent manoeuvred anyway
    "latencies": [], "n_slow_checkpoint": 0,  # >20s per-decision latency
    "total_latency_s": [],
    "n_wrong_direction": 0,  # cited Rule 14/15/16 (give-way, must turn starboard) but turned left
    # Deterministic measurement layer (app/measurement.py Checks A/B/C) -- only present on
    # runs generated AFTER that layer was wired into run_llm_scenario.py; older archived
    # runs simply have no "measurement" field per checkpoint and are skipped for these
    # counts (n_measured tracks how many checkpoints actually carried the field).
    "n_checkpoints": 0, "n_measured": 0,
    "n_check_a": 0, "n_check_b": 0, "n_check_c": 0, "n_check_any": 0,
})

QUIET_MISSIONS = {"UM01", "UM02"}  # only these 2 are genuine no-action/degeneracy tests;
# UM03-UM13 are a real bearing-sweep of give-way/stand-on encounters (Rule 13-17), not quiet.

zigzag_examples = []
overturn_examples = []
stall_examples = []
fabricated_examples = []
quiet_fp_examples = []
slow_examples = []
wrong_direction_examples = []
worst_missions = []  # (composite, mission, config)

RULE_NUM_RE = re.compile(r"Rule (\d+)")

for p in files:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        continue
    mission_id = d.get("mission_id", "?")
    config = d.get("config", "?")
    is_um = mission_id.startswith("UM")
    a = agg[config]
    a["n"] += 1
    a["n_imazu"] += 0 if is_um else 1
    a["n_um"] += 1 if is_um else 0

    ev = d.get("evaluation", {})
    verdict = ev.get("verdict", "?")
    a["verdicts"][verdict] += 1
    composite = ev.get("composite_score")
    if composite is not None:
        a["composite"].append(composite)
        (a["composite_um"] if is_um else a["composite_imazu"]).append(composite)
        worst_missions.append((composite, mission_id, config, verdict))

    safety = ev.get("safety", {})
    if safety.get("min_cpa_m") is not None:
        a["min_cpa"].append(safety["min_cpa_m"])
    if safety.get("passed") is False:
        a["n_unsafe"] += 1

    compliance = ev.get("compliance", {})
    if compliance.get("score") is not None:
        a["compliance_score"].append(compliance["score"])
    violations = compliance.get("violations", [])
    a["n_violations"] += len(violations)
    for v in violations:
        a["violation_texts"].append(v)
        if "ground truth says no rule applies" in v or "fabricated" in v.lower():
            a["n_fabricated_risk"] += 1
            if len(fabricated_examples) < 8:
                fabricated_examples.append((mission_id, config, v[:220]))
        if "wrong rule" in v.lower() or "citing the wrong rule number" in v.lower():
            a["n_wrong_rule_number"] += 1
        if "port" in v.lower() and "starboard" in v.lower() and ("should" in v.lower() or "instead" in v.lower()):
            a["n_wrong_side"] += 1

    temporal = ev.get("temporal", {})
    if temporal.get("arrived"):
        a["arrived"] += 1
    else:
        a["not_arrived"] += 1
    if temporal.get("time_ratio") is not None:
        a["time_ratio"].append(temporal["time_ratio"])

    manoeuvre = ev.get("manoeuvre", {})
    if manoeuvre.get("manoeuvre_count") is not None:
        a["manoeuvre_count"].append(manoeuvre["manoeuvre_count"])

    if isinstance(d.get("latency_s"), (int, float)):
        a["total_latency_s"].append(d["latency_s"])

    checkpoints = d.get("checkpoints", [])
    # zigzag: 3 consecutive checkpoints alternating turn_left/turn_right
    actions = []
    for cp in checkpoints:
        dec = cp.get("decision", {})
        action = dec.get("action", "")
        degrees = dec.get("degrees")
        lat = cp.get("latency_s")
        if isinstance(lat, (int, float)):
            a["latencies"].append(lat)
            if lat > 20:
                a["n_slow_checkpoint"] += 1
                if len(slow_examples) < 8:
                    slow_examples.append((mission_id, config, cp.get("step"), round(lat, 1)))
        if action in ("turn_left", "turn_right") and isinstance(degrees, (int, float)) and degrees > 30:
            a["n_over30"] += 1
            if len(overturn_examples) < 8:
                overturn_examples.append((mission_id, config, cp.get("step"), action, degrees))
        actions.append(action)

        a["n_checkpoints"] += 1
        meas = cp.get("measurement")
        if meas is not None:
            a["n_measured"] += 1
            fired = meas.get("checks_fired", [])
            if "A_fabricated_risk" in fired:
                a["n_check_a"] += 1
            if "B_wrong_direction" in fired:
                a["n_check_b"] += 1
            if "C_degrees_over_limit" in fired:
                a["n_check_c"] += 1
            if fired:
                a["n_check_any"] += 1

    zz = 0
    for i in range(2, len(actions)):
        if actions[i-2] == "turn_left" and actions[i-1] == "turn_right" and actions[i] == "turn_left":
            zz += 1
        if actions[i-2] == "turn_right" and actions[i-1] == "turn_left" and actions[i] == "turn_right":
            zz += 1
    if zz > 0:
        a["n_zigzag"] += 1
        if len(zigzag_examples) < 8:
            zigzag_examples.append((mission_id, config, actions))

    # never-resume-after-slow
    speed_actions = [(cp.get("step"), cp.get("decision", {}).get("action", "")) for cp in checkpoints
                     if cp.get("decision", {}).get("action") in ("slow_down", "speed_up", "resume_cruising_speed", "stop")]
    last_slow_idx, resumed_after = None, False
    for i, (step, action) in enumerate(speed_actions):
        if action in ("slow_down", "stop"):
            last_slow_idx, resumed_after = i, False
        elif action in ("speed_up", "resume_cruising_speed") and last_slow_idx is not None:
            resumed_after = True
    if last_slow_idx is not None and not resumed_after:
        a["n_never_resume_after_slow"] += 1
        if len(stall_examples) < 8:
            stall_examples.append((mission_id, config, speed_actions))

    # quiet-mission false positive: UM01/UM02 ONLY (genuinely no real collision risk)
    if mission_id in QUIET_MISSIONS:
        did_manoeuvre = any(cp.get("decision", {}).get("action") not in ("hold_course", None, "")
                             for cp in checkpoints)
        if did_manoeuvre:
            a["n_quiet_false_positive"] += 1
            if len(quiet_fp_examples) < 8:
                first_bad = next(cp for cp in checkpoints if cp.get("decision", {}).get("action") not in ("hold_course", None, ""))
                quiet_fp_examples.append((mission_id, config, first_bad.get("decision", {})))

    # wrong-direction: Rule 14 (head-on)/15/16 (crossing give-way) mandate turning to STARBOARD
    # (turn_right) -- turning left instead is a genuine, dangerous COLREG-direction violation.
    for cp in checkpoints:
        dec = cp.get("decision", {})
        rule = str(dec.get("rule_applied", ""))
        rule_num_m = RULE_NUM_RE.search(rule)
        if rule_num_m and rule_num_m.group(1) in ("14", "15", "16") and dec.get("action") == "turn_left":
            a["n_wrong_direction"] += 1
            if len(wrong_direction_examples) < 10:
                wrong_direction_examples.append((mission_id, config, cp.get("step"), rule, dec.get("reasoning", "")[:200]))

# ---------------- report ----------------
def mean(xs):
    return round(statistics.mean(xs), 3) if xs else None

print(f"{'config':16s} {'n':>3s} {'compo':>6s} {'c_imazu':>7s} {'c_um':>6s} {'unsafe':>6s} "
      f"{'compl':>6s} {'arr%':>5s} {'zzag':>5s} {'>30':>4s} {'no-res':>6s} {'q-FP':>5s} {'fab':>4s} {'wrongdir':>8s}")
for config in CONFIGS_ORDER:
    if config not in agg:
        continue
    a = agg[config]
    arr_pct = round(100 * a["arrived"] / a["n"], 0) if a["n"] else 0
    print(f"{config:16s} {a['n']:>3d} {str(mean(a['composite'])):>6s} {str(mean(a['composite_imazu'])):>7s} "
          f"{str(mean(a['composite_um'])):>6s} {a['n_unsafe']:>6d} {str(mean(a['compliance_score'])):>6s} "
          f"{arr_pct:>5.0f} {a['n_zigzag']:>5d} {a['n_over30']:>4d} {a['n_never_resume_after_slow']:>6d} "
          f"{a['n_quiet_false_positive']:>5d} {a['n_fabricated_risk']:>4d} {a['n_wrong_direction']:>8d}")

print("\n--- deterministic measurement layer (app/measurement.py Checks A/B/C) ---")
print("Primary research metrics: direction-correct-rate = 1 - B_rate, fabricated-risk-rate = A_rate.")
print("(runs with no \"measurement\" field -- archived before this layer existed -- show n_measured=0)")
print(f"{'config':16s} {'n_cp':>6s} {'measured':>8s} {'A_n':>5s} {'A_%':>6s} {'B_n':>5s} {'B_%':>6s} "
      f"{'C_n':>5s} {'C_%':>6s} {'any_%':>6s}")
for config in CONFIGS_ORDER:
    if config not in agg:
        continue
    a = agg[config]
    m = a["n_measured"]
    pct = lambda n: round(100 * n / m, 1) if m else None
    print(f"{config:16s} {a['n_checkpoints']:>6d} {m:>8d} "
          f"{a['n_check_a']:>5d} {str(pct(a['n_check_a'])):>6s} "
          f"{a['n_check_b']:>5d} {str(pct(a['n_check_b'])):>6s} "
          f"{a['n_check_c']:>5d} {str(pct(a['n_check_c'])):>6s} "
          f"{str(pct(a['n_check_any'])):>6s}")

print("\n--- verdict distribution per config ---")
for config in CONFIGS_ORDER:
    if config not in agg:
        continue
    print(f"{config}: {dict(agg[config]['verdicts'])}")

print("\n--- latency (per-decision, seconds) ---")
for config in CONFIGS_ORDER:
    if config not in agg:
        continue
    lats = agg[config]["latencies"]
    if lats:
        print(f"{config:16s} n={len(lats):4d} mean={mean(lats):>6} max={max(lats):>6.1f} "
              f"slow(>20s)={agg[config]['n_slow_checkpoint']}")

print("\n--- worst 15 missions by composite score ---")
for composite, mission_id, config, verdict in sorted(worst_missions)[:15]:
    print(f"{composite:.3f}  {mission_id:10s} {config:16s} {verdict}")

print("\n--- zigzag examples ---")
for mission_id, config, actions in zigzag_examples:
    print(f"[{mission_id}/{config}] {actions}")

print("\n--- >30deg over-turn request examples ---")
for mission_id, config, step, action, degrees in overturn_examples:
    print(f"[{mission_id}/{config}] step={step} {action} {degrees}deg")

print("\n--- never-resume-after-slow examples ---")
for mission_id, config, speed_actions in stall_examples:
    print(f"[{mission_id}/{config}] {speed_actions}")

print("\n--- fabricated-risk / wrong-rule violation examples ---")
for mission_id, config, v in fabricated_examples:
    print(f"[{mission_id}/{config}] {v}")

print("\n--- quiet-mission (UM) false-positive manoeuvre examples ---")
for mission_id, config, dec in quiet_fp_examples:
    print(f"[{mission_id}/{config}] {dec}")

print("\n--- slow-decision (>20s) examples ---")
for mission_id, config, step, lat in slow_examples:
    print(f"[{mission_id}/{config}] step={step} latency={lat}s")

print("\n--- wrong-direction (Rule 14/15/16 cited but turned LEFT instead of starboard) examples ---")
for mission_id, config, step, rule, reasoning in wrong_direction_examples:
    print(f"[{mission_id}/{config}] step={step} rule={rule!r} :: {reasoning}")
