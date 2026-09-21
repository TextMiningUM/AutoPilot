"""Sweep model configs across missions, scoring each resulting run with the same
Woerner-style composite used elsewhere (app/evaluation.py's score_trajectory), to find the
best-performing agent parameter set per mission.

Reuses app/run_llm_scenario.py's run_one() to generate (or reuse existing) precomputed run
logs -- this is "run_llm_scenario for every config, then rank by score", and inherits its
resume-safety (skips a (mission,config,tag) combo whose log already exists, unless --force).

COARSE-FIRST DESIGN: by default this sweeps only the 8 model CONFIGS per mission (thinking
off, k=2 fixed, same defaults as run_llm_scenario.py) -- a full config x thinking x k grid
would be far too many model calls for this hardware. Once you know which config wins per
mission, run a second, narrower pass to refine just the winning config's other parameters,
e.g.:
    python -m app.sweep_llm_params --missions s01_head_on --configs v3_rag_cot \\
        --tag thinking_on --enable-thinking

Run (one mission, all 8 configs):
    python -m app.sweep_llm_params --missions s01_head_on

Run everything (14 missions x 8 configs = 112 runs -- long, sequential, one GPU):
    python -m app.sweep_llm_params
"""
from __future__ import annotations
import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
REPO_ROOT = ROOT.parent
for p in (ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.missions import list_mission_ids, load_mission
from app.agents import MODEL_CONFIGS
from app.run_llm_scenario import run_one
from app.llm_runs import RUNS_DIR, load_run
from app.evaluation import score_trajectory
from app.simulation import VesselConstraints

SUMMARY_FILE = RUNS_DIR / "_sweep_summary.json"
# Live "what's actually running right now" marker -- sweep_dashboard.py reads this directly
# instead of inferring a current job from the first (mission, config) gap in strict
# q01-first job order, which silently pointed at a long-finished mission (or one nobody is
# even running on THIS host) as soon as jobs complete out of order -- e.g. this same sweep
# script running locally for just s11/s12 while a separate cloud process works through
# q01, q02, ... in parallel, each writing to its own copy of SUMMARY_FILE.
STATUS_FILE = RUNS_DIR / "_sweep_status.json"
_DEFAULT_MIN_CPA_M = VesselConstraints().min_cpa_m


def _write_status(mission_id: str, config: str, tag: str) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps({
        "mission": mission_id, "config": config, "tag": tag,
        "host": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")


def _clear_status() -> None:
    STATUS_FILE.unlink(missing_ok=True)


def score_one(mission, log: dict) -> dict:
    """Scores one precomputed run's trajectory with the same composite evaluate_run.py
    uses elsewhere (Evaluation panel, LLM compliance check) -- safety gate, then weighted
    compliance/temporal/spatial/manoeuvre/smoothness. Keeps the FULL per-axis breakdown
    (violations list, min CPA, path ratios, manoeuvre counts, ...), not just the top-line
    scores, so the on-disk summary is self-sufficient for later inspection.

    Prefers the log's OWN embedded "evaluation" (written by run_llm_scenario.py's run_one()
    at save time) over recomputing -- only recomputes for older logs from before that field
    existed."""
    result = log.get("evaluation") or score_trajectory(
        log["trajectory"], start_xy=(mission.own_ship.x, mission.own_ship.y),
        goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
        safe_distance_m=_DEFAULT_MIN_CPA_M,
        llm_violations=(log.get("colreg_llm_check") or {}).get("violations"),
        llm_compliance_score=(log.get("colreg_llm_check") or {}).get("compliance_score"),
    )
    return {
        "config": log["config"], "tag": log["tag"],
        "composite_score": result["composite_score"], "verdict": result["verdict"],
        "safety": result["safety"], "compliance": result["compliance"],
        "temporal": result["temporal"], "spatial": result["spatial"],
        "manoeuvre": result["manoeuvre"], "latency_s": log.get("latency_s"),
        "colreg_llm_check": log.get("colreg_llm_check"),
    }



def _load_summary_file() -> dict:
    if SUMMARY_FILE.exists():
        return json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    return {}


def _save_row(mission_id: str, row: dict) -> dict:
    """Merges one (mission, config, tag) result into the on-disk summary IMMEDIATELY, so a
    crash/OOM partway through a long sweep still leaves every already-completed job's full
    score on disk instead of only whatever was still in memory when it died."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    merged = _load_summary_file()
    rows = {(r["config"], r["tag"]): r for r in merged.get(mission_id, [])}
    rows[(row["config"], row["tag"])] = row
    merged[mission_id] = sorted(rows.values(), key=lambda r: r["composite_score"], reverse=True)
    SUMMARY_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    return merged


def sweep(missions: list[str], configs: list[str], tag: str = "default", **run_kwargs) -> dict:
    merged = _load_summary_file()
    jobs = [(m, c) for m in missions for c in configs]
    for i, (mission_id, config) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {mission_id} / {config}")
        _write_status(mission_id, config, tag)
        t0 = time.time()
        try:
            out_path = run_one(mission_id, config, tag=tag, **run_kwargs)
            log = load_run(out_path)
            mission = load_mission(mission_id)
            row = score_one(mission, log)
            print(f"  composite={row['composite_score']:.3f} verdict={row['verdict']} "
                 f"({time.time() - t0:.1f}s)")
        except Exception as exc:
            # One bad job (model hiccup, malformed decision JSON, OOM, ...) must not take
            # the rest of the sweep down with it -- record the failure and keep going.
            row = {"config": config, "tag": tag, "composite_score": 0.0,
                  "verdict": f"ERROR: {exc}", "safety": None, "compliance": None,
                  "temporal": None, "spatial": None, "manoeuvre": None,
                  "latency_s": None, "colreg_llm_check": None}
            print(f"  ERROR ({time.time() - t0:.1f}s): {exc}")
        merged = _save_row(mission_id, row)
    _clear_status()
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--missions", nargs="+", default=None, help="default: all missions")
    ap.add_argument("--configs", nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS),
                    help="default: all 8 model configs")
    ap.add_argument("--tag", default="default",
                    help="reused from run_llm_scenario.py -- lets a refinement pass (e.g. "
                         "--enable-thinking) coexist with the coarse sweep's logs instead of "
                         "overwriting them")
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="total step budget -- default: a per-mission recommendation sized "
                         "from the mission's own straight-line transit distance/speed (see "
                         "app.narrate.recommended_max_steps); set explicitly to force the same "
                         "budget across every mission in the sweep")
    ap.add_argument("--decision-interval", type=int, default=None,
                    help="steps between LLM decisions -- default: per-mission recommendation "
                         "(see app.narrate.recommended_decision_interval); set explicitly to "
                         "force the same cadence across every mission in the sweep")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if a log already exists")
    ap.add_argument("--no-colreg-check", action="store_true",
                    help="skip the end-of-mission Anthropic Claude COLREG compliance check for "
                         "every job in this sweep (saves one network call + latency per job; "
                         "needs ANTHROPIC_API_KEY in .env otherwise)")
    args = ap.parse_args()

    missions = args.missions or list_mission_ids()
    # sweep() saves each job's full result to SUMMARY_FILE as it completes -- `summary` here
    # is just the final merged on-disk state (all missions/configs ever swept), returned so
    # the leaderboard below doesn't need to re-read the file.
    summary = sweep(
        missions, args.configs, tag=args.tag, dt=args.dt, max_steps=args.max_steps,
        enable_thinking=args.enable_thinking, max_new_tokens=args.max_new_tokens,
        k=args.k, use_rag=not args.no_rag, force=args.force,
        decision_interval=args.decision_interval,
        check_colreg_compliance=not args.no_colreg_check,
    )
    print(f"\nSaved incrementally to: {SUMMARY_FILE}")

    print("\n=== Best config per mission (all data on disk) ===")
    for mission_id in missions:
        rows = summary.get(mission_id) or []
        if not rows:
            continue
        best = rows[0]
        print(f"  {mission_id:35s} best={best['config']:16s} tag={best['tag']:10s} "
             f"composite={best['composite_score']:.3f} verdict={best['verdict']}")


if __name__ == "__main__":
    main()
