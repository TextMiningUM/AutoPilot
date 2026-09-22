"""One-off scan of _llm_runs/*.json for recurring LLM helm-decision error patterns."""
import json, re
from pathlib import Path
from collections import defaultdict, Counter

RUNS_DIR = Path(__file__).parent / "Data" / "missions" / "_llm_runs"

files = sorted(p for p in RUNS_DIR.glob("*.json") if "__" in p.stem)

by_config = defaultdict(lambda: {
    "n": 0, "n_violations": 0, "n_over_30": 0, "n_missed_goal": 0,
    "n_never_resume_after_slow": 0, "n_fabricated_risk": 0,
})

fabricated_examples = []
overturn_examples = []
stall_examples = []

for p in files:
    d = json.loads(p.read_text(encoding="utf-8"))
    config = d.get("config", "?")
    stats = by_config[config]
    stats["n"] += 1

    ev = d.get("evaluation", {})
    violations = ev.get("compliance", {}).get("violations", [])
    if violations:
        stats["n_violations"] += 1

    # fabricated-risk pattern: violation text says ground truth had no risk but a rule was cited
    for v in violations:
        if "ground truth says no rule applies" in v or "no real risk" in v.lower():
            stats["n_fabricated_risk"] += 1
            if len(fabricated_examples) < 6:
                fabricated_examples.append((p.stem, v))

    checkpoints = d.get("checkpoints", [])
    over_30_here = False
    speed_actions = []  # (step, action)
    for cp in checkpoints:
        dec = cp.get("decision", {})
        action = dec.get("action", "")
        degrees = dec.get("degrees")
        if action in ("turn_left", "turn_right") and isinstance(degrees, (int, float)) and degrees > 30:
            over_30_here = True
            if len(overturn_examples) < 6:
                overturn_examples.append((p.stem, cp.get("step"), action, degrees, dec.get("reasoning", "")[:200]))
        if action in ("slow_down", "speed_up", "resume_cruising_speed", "stop"):
            speed_actions.append((cp.get("step"), action))
    if over_30_here:
        stats["n_over_30"] += 1

    # never-resume-after-slow: a slow_down/stop with no later speed_up/resume_cruising_speed
    last_slow_idx = None
    resumed_after = False
    for i, (step, action) in enumerate(speed_actions):
        if action in ("slow_down", "stop"):
            last_slow_idx = i
            resumed_after = False
        elif action in ("speed_up", "resume_cruising_speed") and last_slow_idx is not None:
            resumed_after = True
    if last_slow_idx is not None and not resumed_after:
        stats["n_never_resume_after_slow"] += 1
        if len(stall_examples) < 6:
            stall_examples.append((p.stem, speed_actions))

    temporal = ev.get("temporal", {})
    if not temporal.get("arrived", True) or temporal.get("time_ratio", 1.0) > 1.5:
        stats["n_missed_goal"] += 1

print(f"{'config':18s} {'n':>4s} {'viol':>5s} {'>30deg':>6s} {'missed':>6s} {'no-resume':>9s} {'fab-risk':>8s}")
for config, s in sorted(by_config.items()):
    print(f"{config:18s} {s['n']:>4d} {s['n_violations']:>5d} {s['n_over_30']:>6d} "
          f"{s['n_missed_goal']:>6d} {s['n_never_resume_after_slow']:>9d} {s['n_fabricated_risk']:>8d}")

print("\n--- fabricated-risk examples ---")
for stem, v in fabricated_examples:
    print(f"[{stem}] {v}")

print("\n--- >30deg turn request examples ---")
for stem, step, action, degrees, reasoning in overturn_examples:
    print(f"[{stem}] step={step} {action} {degrees}deg :: {reasoning}")

print("\n--- never-resume-after-slow examples ---")
for stem, speed_actions in stall_examples:
    print(f"[{stem}] {speed_actions}")
