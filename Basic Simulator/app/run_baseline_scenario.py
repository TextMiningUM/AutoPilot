"""Precompute a deterministic-baseline mission run and save it in the EXACT same run-log
schema app/run_llm_scenario.py writes (mission_id, config, weights, tag, generated_at,
latency_s, mission, evaluation, params, outcome, trajectory, checkpoints) -- so every
existing tool that reads that schema (streamlit_app.py's "Play Agent Mission" replay,
app/sweep_dashboard.py, _analysis/audit_runs.py, _sweep_summary.json merging) works on
baseline runs unmodified, just filtered/grouped by `config` (e.g. "baseline_ruletree").

Unlike the LLM agent, every baseline in app/baselines/ is pure Python/CPU with no model
load and effectively zero per-decision latency -- so this defaults to a decision EVERY
simulation step (decision_interval=1), not the LLM path's expensive-call-driven adaptive
cadence. Latency is still recorded per checkpoint (`latency_s`) and for the whole run
(top-level `latency_s`), same fields as an LLM run log, even though the numbers here are
expected to be tiny.

Same end-of-run checks as the LLM path: `find_collision()` after every step (interpolated,
not just same-instant, so a fast pass-through between two samples is never missed) and the
SAME `score_trajectory()` (-> Evaluation Functions/evaluate_run.py) for the final verdict.

Run one:
    python -m app.run_baseline_scenario --missions Imazu01 --configs baseline_ruletree

Run every mission for every registered baseline:
    python -m app.run_baseline_scenario
"""
from __future__ import annotations
import argparse
import json
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

from app.baselines import BASELINE_CONFIGS, DECISION_FUNCS
from app.evaluation import score_trajectory
from app.llm_runs import RUNS_DIR, run_log_path
from app.measurement import measure_decision_quality
from app.missions import list_mission_ids, load_mission, mission_to_dict
from app.narrate import contact_line, recommended_max_steps
from app.simulation import Simulation, VesselConstraints, find_collision

WEIGHTS = "deterministic"  # baselines have no model checkpoint -- fixed, never a chain/merge


def run_one(mission_id: str, config: str, tag: str = "baseline",
           dt: float = 10.0, max_steps: int | None = None,
           decision_interval: int = 1, force: bool = False,
           kinematics_model: str = "kinematics") -> Path:
    out_path = run_log_path(mission_id, config, WEIGHTS, tag)
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to overwrite)")
        return out_path

    mission = load_mission(mission_id)
    decide = DECISION_FUNCS[config]
    max_steps = max_steps if max_steps is not None else recommended_max_steps(mission, dt)
    constraints = VesselConstraints(time_step_s=dt, cruise_speed_mps=mission.own_ship.speed,
                                    kinematics_model=kinematics_model)
    sim = Simulation(mission, constraints)
    checkpoints: list[dict] = []
    outcome = "max_steps_reached"
    step = 0
    next_decision_step = 0

    _t_run_start = time.time()
    for step in range(max_steps):
        if sim.reached_goal():
            outcome = "reached_goal"
            break
        if step >= next_decision_step:
            _t_cp = time.time()
            decision, debug = decide(mission, sim.own, sim.targets, constraints)
            cp_latency_s = time.time() - _t_cp
            contacts_now = [contact_line(sim.own, t, constraints.min_cpa_m, constraints.max_rudder_angle_deg)
                           for t in sim.targets]
            checkpoints.append({
                "step": step, "time": sim.t,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_s": cp_latency_s,
                "situation_report": debug.get("situation"),
                "decision": decision,
                "reasoning_raw": debug.get("raw_response"),
                "measurement": measure_decision_quality(decision, contacts_now, constraints),
                "debug": {kk: vv for kk, vv in debug.items() if kk not in ("situation", "raw_response")},
                "decision_interval_steps": decision_interval,
                "decision_interval_s": decision_interval * dt,
            })
            sim.apply_action(decision)
            next_decision_step = step + max(1, decision_interval)
        sim.step(dt)
        # Same interpolated, ground-truth collision finder used for scoring/plotting
        # (see run_llm_scenario.py's own docstring for why a same-instant-only check can
        # miss two fast-closing vessels passing through each other within one dt step).
        if find_collision(sim.trajectory) is not None:
            outcome = "collision"
            break

    latency_s = time.time() - _t_run_start

    evaluation = score_trajectory(
        sim.trajectory, start_xy=(mission.own_ship.x, mission.own_ship.y),
        goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
        safe_distance_m=constraints.min_cpa_m, max_turn_deg=constraints.max_rudder_angle_deg,
        checkpoints=checkpoints,
    )

    log = {
        "mission_id": mission_id, "config": config, "weights": WEIGHTS,
        "model_variant": config, "tag": tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latency_s": latency_s,
        "mission": mission_to_dict(mission),
        "evaluation": evaluation,
        "colreg_llm_check": {"checked": False, "explanations": None, "error": None},
        "params": {
            "decision_interval": decision_interval, "decision_interval_mode": "fixed",
            "dt": dt, "max_steps": max_steps, "baseline": True,
            "kinematics_model": kinematics_model,
        },
        "outcome": {"verdict": outcome, "final_step": step, "final_time_s": sim.t},
        "trajectory": sim.trajectory,
        "checkpoints": checkpoints,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  [done] {out_path.name}  outcome={outcome}  steps={step}  "
          f"checkpoints={len(checkpoints)}  latency={latency_s:.3f}s  "
          f"composite={log['evaluation']['composite_score']:.3f}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--missions", nargs="+", default=None,
                    help="mission ids to run (default: all missions)")
    ap.add_argument("--configs", nargs="+", default=list(BASELINE_CONFIGS),
                    choices=list(BASELINE_CONFIGS),
                    help="one or more baseline configs (see app.baselines.BASELINE_CONFIGS)")
    ap.add_argument("--tag", default="baseline",
                    help="distinguishes variations of the same mission+config")
    ap.add_argument("--dt", type=float, default=10.0, help="simulation time step (s)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="total step budget -- default: per-mission recommendation, see "
                         "app.narrate.recommended_max_steps")
    ap.add_argument("--decision-interval", type=int, default=1,
                    help="simulation steps between decisions -- default 1 (every step), "
                         "since deterministic baselines are cheap unlike LLM calls")
    ap.add_argument("--kinematics-model", default="kinematics", choices=["kinematics", "nomoto"],
                    help="own-ship heading dynamics: legacy turn-rate slew (default, "
                         "unchanged) or Sawada et al. (2021)'s Nomoto model (opt-in, see "
                         "app.simulation.VesselConstraints)")
    ap.add_argument("--force", action="store_true", help="overwrite existing logs")
    args = ap.parse_args()

    missions = args.missions or list_mission_ids()
    jobs = [(m, c) for m in missions for c in args.configs]
    print(f"Queued {len(jobs)} run(s): {len(missions)} mission(s) x {len(args.configs)} config(s), "
         f"tag={args.tag!r}, kinematics_model={args.kinematics_model!r}")
    for i, (mission_id, config) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {mission_id} / {config}")
        t0 = time.time()
        run_one(mission_id, config, tag=args.tag, dt=args.dt, max_steps=args.max_steps,
                decision_interval=args.decision_interval, force=args.force,
                kinematics_model=args.kinematics_model)
        print(f"  took {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
