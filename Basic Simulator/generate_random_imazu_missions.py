"""Generator: Data/missions/RND01.json .. RND{N:02d}.json -- a NEW, procedurally
randomized Imazu-scale mission pool (Stap 2 plan, 2026-09-29 -- see repo memory
"Stap 2 plan" note for the full context/rationale).

Unlike generate_imazu_missions.py (reproduces one fixed published table) or
generate_impossible_missions.py (10 hand-designed adversarial scenarios), this
generator draws RANDOM geometry per mission -- bearing, target speed, range -- while
reusing the SAME guaranteed-collision-course construction (app.geometry.compute_target,
already feasibility-aware: it grows range until own-ship physically has time to turn)
and the SAME encounter classification (generate_imazu_missions.classify(), never a
second copy) so results stay directly comparable to the existing Imazu/IMP/UM corpus.

Each mission carries an additive `failure_category` tag (ignored by app.missions'
loader -- extra JSON keys are simply not read -- so this is fully backward compatible)
biased toward the 3 concretely MEASURED worst spots from the 2026-09-29 nomoto_v2 audit
(see repo memory): RAG-fabricated-risk under multi-target crossing, stand-on-vessel
timing (the IMP13 B_17c pattern), and CoT-under-time-pressure truncation. A
"baseline_mixed" category (unbiased random geometry) is included too, for balance --
not every mined example should come from a deliberately-stressed case.

A fixed fraction of missions (HELD_OUT_FRACTION) is marked `"held_out": true` --
deterministic by mission index, NOT randomly re-rolled per run, so the same missions are
always held out across regenerations. This replaces "Imazu22" as this pool's held-out
marker (Imazu22 itself is a permanently REMOVED exact duplicate, never a real held-out
mission -- see generate_imazu_missions.py's REMOVED_CASES).

Run: python generate_random_imazu_missions.py [--n 60] [--seed 20260929]
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.geometry import compute_target
from app.missions import mission_from_dict
from app.units import mps_to_kn
from generate_imazu_missions import CRITERIA_MAP, NAME_MAP, ROLE_MAP, RULE_MAP, classify, fineness

OUT_DIR = ROOT / "Data" / "missions"
V_OS = 6.173333333333333  # 12 kt, matches the project's established Imazu convention
OWN_START = (0.0, 0.0)
GOAL = (0.0, 18520.0)  # 10 NM straight ahead -- an "Imazu-scale" transit distance
HELD_OUT_FRACTION = 0.15  # ~1 in 7 missions, deterministic by index -- see module docstring

# Per-category (bearing_deg range, range_m band, target-count override) -- biases geometry
# toward the 3 measured-worst spots from the 2026-09-29 audit, plus one unbiased category.
# bearing convention matches classify(): 0=ahead, +=starboard, -=port.
FAILURE_CATEGORIES: dict[str, dict] = {
    # IMP13's own B_17c pattern ("Stand-On Vessel Acted Too Early") broke baseline_vo/
    # v10/v11 identically -- own-ship stand-on (crossing_port) is the geometry that
    # triggers it.
    "stand_on_timing_pressure": {
        "weight": 0.30, "bearing_range": (-140.0, -35.0), "range_band": (6000.0, 11000.0),
        "target_count_weights": {1: 0.6, 2: 0.4, 3: 0.0},
    },
    # v10_super_colreg_rag's measured 18.80% fabricated-risk rate (vs ~3-4% for v0_base/
    # v11) was worst on busier, multi-contact crossing geometry -- force 2-3 targets.
    "multi_target_crossing_rag_risk": {
        "weight": 0.30, "bearing_range": (-150.0, 150.0), "range_band": (6000.0, 12000.0),
        "target_count_weights": {1: 0.0, 2: 0.5, 3: 0.5},
    },
    # v11_super_colreg_rag_cot's worst checkpoints all co-occurred with W_truncated/
    # W_deliberation_loop/ERROR_2_0_thinking_leak -- a materially SHORTER unavoided
    # window (tighter range_m) gives its <think> block less real time to ramble before
    # the decisive moment arrives.
    "cot_time_pressure_tight_window": {
        "weight": 0.20, "bearing_range": (-160.0, 160.0), "range_band": (3000.0, 6000.0),
        "target_count_weights": {1: 0.5, 2: 0.4, 3: 0.1},
    },
    # Unbiased -- not every mined example should come from a deliberately-stressed case.
    "baseline_mixed": {
        "weight": 0.20, "bearing_range": (-170.0, 170.0), "range_band": (5000.0, 13000.0),
        "target_count_weights": {1: 0.35, 2: 0.40, 3: 0.25},
    },
}


def _weighted_choice(rnd: random.Random, weights: dict) -> object:
    keys = list(weights.keys())
    probs = [weights[k] for k in keys]
    return rnd.choices(keys, weights=probs, k=1)[0]


def _sample_target(rnd: random.Random, index: int, n_targets: int, bearing_lo: float,
                   bearing_hi: float, range_lo: float, range_hi: float) -> dict:
    """One target's random geometry -- bearing/range/speed drawn independently per
    target within the mission's category band (evenly spread across the band by index
    first, then jittered, so 2-3 targets in one mission don't cluster on top of each
    other by pure chance). Some (bearing, range, speed) combinations have no real
    solve_intercept() solution (the underlying quadratic's discriminant goes negative --
    same edge case generate_impossible_missions.py's own bisection helper guards
    against) -- retried with a fresh draw from the same band rather than crashing."""
    span = bearing_hi - bearing_lo
    slot = span / n_targets
    for _attempt in range(30):
        jitter = rnd.uniform(-slot * 0.35, slot * 0.35)
        bearing = bearing_lo + slot * (index + 0.5) + jitter
        range_m = rnd.uniform(range_lo, range_hi)
        v_ts = V_OS * rnd.uniform(0.5, 1.3)
        try:
            return compute_target(bearing, range_m, v_ts, V_OS)
        except ValueError:
            continue
    raise RuntimeError(f"no solvable intercept geometry found after 30 attempts "
                      f"(bearing band [{bearing_lo}, {bearing_hi}], range band [{range_lo}, {range_hi}])")


def build_mission(mission_id: str, index: int, rnd: random.Random) -> dict:
    category = _weighted_choice(rnd, {k: v["weight"] for k, v in FAILURE_CATEGORIES.items()})
    spec = FAILURE_CATEGORIES[category]
    n_targets = _weighted_choice(rnd, spec["target_count_weights"])
    bearing_lo, bearing_hi = spec["bearing_range"]
    range_lo, range_hi = spec["range_band"]

    targets_json, types, summaries = [], [], []
    for i in range(n_targets):
        t = _sample_target(rnd, i, n_targets, bearing_lo, bearing_hi, range_lo, range_hi)
        name = f"ts{i + 1}"
        ttype = classify(t["bearing_from_os_deg"], t["heading"], t["speed"], V_OS)
        types.append(ttype)
        summaries.append(f"{name} " + {
            "head_on": "approaches head-on",
            "crossing_stbd": f"crosses from starboard ({fineness(t['bearing_from_os_deg'])})",
            "crossing_port": f"crosses from port ({fineness(t['bearing_from_os_deg'])})",
            "overtaking_give_way": "is a slower target ahead that own-ship must overtake",
            "overtaking_stand_on": "is a faster target overtaking own-ship from astern",
            "parallel_no_risk": "shares own-ship's course/speed with no closing risk",
        }[ttype])
        targets_json.append({
            "name": name, "x_nm": round(t["start_x"] / 1852.0, 4), "y_nm": round(t["start_y"] / 1852.0, 4),
            "heading_deg": t["heading"], "speed_kn": round(mps_to_kn(t["speed"]), 2),
        })

    unique_types = list(dict.fromkeys(types))
    rule_refs: list[str] = []
    for ttype in unique_types:
        for r in RULE_MAP[ttype]:
            if r not in rule_refs:
                rule_refs.append(r)
    if n_targets > 1 and "Rule 8(d)" not in rule_refs:
        rule_refs.append("Rule 8(d)")
    own_ship_role = (ROLE_MAP[types[0]] if n_targets == 1 else
                    " / ".join(f"{ROLE_MAP[t]} re: {targets_json[i]['name']}" for i, t in enumerate(types)))
    pass_criteria = [CRITERIA_MAP[t] for t in unique_types]
    if n_targets > 1:
        pass_criteria.append("Own-ship's manoeuvre for one target does not create a new "
                             "close-quarters situation with another (Rule 8(d)).")
    pass_criteria.append("Minimum CPA to ALL targets stays above the safe-distance threshold.")
    name_parts = " + ".join(NAME_MAP[t] for t in unique_types)

    return {
        "id": mission_id,
        "name": f"Random {index:02d} -- {name_parts}" + (f" ({n_targets} targets)" if n_targets > 1 else ""),
        "rule_refs": rule_refs, "own_ship_role": own_ship_role,
        "description": (f"Procedurally generated (category={category!r}, seed-derived, "
                        f"Stap 2 pool 2026-09-29): " + "; ".join(summaries) + "."),
        "pass_criteria": pass_criteria,
        "own_ship": {"x_nm": round(OWN_START[0] / 1852.0, 4), "y_nm": round(OWN_START[1] / 1852.0, 4),
                    "heading_deg": 0.0, "speed_kn": round(mps_to_kn(V_OS), 2)},
        "goal": {"x_nm": round(GOAL[0] / 1852.0, 4), "y_nm": round(GOAL[1] / 1852.0, 4)},
        "targets": targets_json,
        "failure_category": category,
        "held_out": (index % round(1 / HELD_OUT_FRACTION) == 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_held_out = 0
    for i in range(1, args.n + 1):
        mission_id = f"RND{i:02d}"
        rnd = random.Random(f"{args.seed}::{mission_id}")  # deterministic per mission, reproducible regen
        mission = build_mission(mission_id, i, rnd)
        mission_from_dict(mission)  # round-trip sanity check -- raises if the schema is malformed
        n_held_out += mission["held_out"]
        path = OUT_DIR / f"{mission_id}.json"
        path.write_text(json.dumps(mission, indent=2), encoding="utf-8")
        print(f"  -> wrote {path.relative_to(ROOT)}  "
             f"(category={mission['failure_category']}, {len(mission['targets'])} target(s), "
             f"held_out={mission['held_out']})")
    print(f"\n{args.n} missions written, {n_held_out} held out ({n_held_out / args.n:.0%}).")


if __name__ == "__main__":
    main()
