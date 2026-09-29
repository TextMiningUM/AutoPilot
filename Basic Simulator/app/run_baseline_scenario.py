"""Precompute a deterministic-baseline mission run and save it in the EXACT same run-log
schema app/run_llm_scenario.py writes (mission_id, config, weights, tag, generated_at,
latency_s, mission, evaluation, params, outcome, trajectory, checkpoints) -- so every
existing tool that reads that schema (streamlit_app.py's "Play Agent Mission" replay,
app/sweep_dashboard.py, _analysis/audit_runs.py, _sweep_summary.json merging) works on
baseline runs unmodified, just filtered/grouped by `config` (e.g. "baseline_ruletree").

Unlike the LLM agent, every baseline in app/baselines/ is pure Python/CPU with no model
load and effectively zero per-decision latency. 2026-09-28: decision cadence is now
ADAPTIVE by default (app.narrate.live_decision_interval, the SAME cadence
run_llm_scenario.py's run_one() uses) instead of a fixed decision EVERY simulation step --
baselines used to get far more decision opportunities than an LLM's sparse cadence, an
unfair comparison once compared side by side (app.sweep_dashboard.py's Baselines-vs-LLM
tab). Pass --decision-interval to opt back into a fixed cadence for the whole run.
Latency is still recorded per checkpoint (`latency_s`) and for the whole run (top-level
`latency_s`), same fields as an LLM run log, even though the numbers here are expected to
be tiny.

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
from app.llm_runs import CANONICAL_BASELINE_TAG_BY_KINEMATICS, RUNS_DIR, run_log_path
from app.measurement import measure_decision_quality
from app.missions import list_mission_ids, load_mission, mission_to_dict
from app.narrate import (
    contact_line, live_decision_interval, recommended_decision_interval,
    recommended_max_steps, transit_step_cap,
)
from app.simulation import Simulation, VesselConstraints, find_collision
from pipeline.oow_agent_spec import derive_risk_horizon_s

WEIGHTS = "deterministic"  # baselines have no model checkpoint -- fixed, never a chain/merge


def run_one(mission_id: str, config: str, tag: str = "baseline",
           dt: float = 10.0, max_steps: int | None = None,
           decision_interval: int | None = None, force: bool = False,
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
    # Adaptive decision cadence (2026-09-28), same default as run_llm_scenario.py's own
    # run_one() -- baselines used to decide EVERY step regardless (decision_interval=1,
    # "cheap, unlike LLM calls"), which gave them far more chances to react/re-plan than an
    # LLM's sparse adaptive cadence, an unfair comparison once results are compared side by
    # side (app.sweep_dashboard.py's Baselines-vs-LLM tab). `--decision-interval` remains
    # available as a FIXED-cadence override for anyone deliberately wanting the old
    # dense-every-step behaviour (decision_interval_mode logged either way).
    fixed_interval = decision_interval
    cap = transit_step_cap(mission, dt)
    next_interval_steps = fixed_interval if fixed_interval is not None else recommended_decision_interval(mission, dt)
    next_decision_step = 0

    _t_run_start = time.time()
    for step in range(max_steps):
        if sim.reached_goal():
            outcome = "reached_goal"
            break
        if step >= next_decision_step:
            if fixed_interval is None:
                risk_horizon_s = derive_risk_horizon_s(
                    constraints.min_cpa_m, constraints.max_rudder_angle_deg, sim.own.speed)
                next_interval_steps = live_decision_interval(
                    sim.own, sim.targets, cap, constraints.min_cpa_m, risk_horizon_s)
            _t_cp = time.time()
            decision, debug = decide(mission, sim.own, sim.targets, constraints)
            cp_latency_s = time.time() - _t_cp
            contacts_now = [contact_line(sim.own, t, constraints.min_cpa_m, constraints.max_rudder_angle_deg)
                           for t in sim.targets]
            sim.apply_action(decision)
            checkpoints.append({
                "step": step, "time": sim.t,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_s": cp_latency_s,
                "situation_report": debug.get("situation"),
                "decision": decision,
                # The resulting ABSOLUTE commanded heading AFTER this decision was applied
                # (2026-09-28) -- sim.target_heading, not just the decision's own possibly-
                # relative action/degrees, so a zigzag/stacking pattern is directly visible
                # in the report regardless of which decision function produced it.
                "commanded_heading_deg": round(sim.target_heading, 1),
                "reasoning_raw": debug.get("raw_response"),
                "measurement": measure_decision_quality(decision, contacts_now, constraints),
                "debug": {kk: vv for kk, vv in debug.items() if kk not in ("situation", "raw_response")},
                "decision_interval_steps": next_interval_steps,
                "decision_interval_s": next_interval_steps * dt,
            })
            next_decision_step = step + max(1, next_interval_steps)
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
            "decision_interval": next_interval_steps if fixed_interval is None else fixed_interval,
            "decision_interval_mode": "fixed" if fixed_interval is not None else "adaptive",
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
    ap.add_argument("--tag", default=None,
                    help="override the canonical tag derived from --kinematics-model "
                         "(app.llm_runs.CANONICAL_BASELINE_TAG_BY_KINEMATICS: kinematics-> "
                         "'baseline', nomoto->'baseline_Nomoto_all', nomoto_v2->'nomoto_v2') "
                         "-- baselines have no other axis to vary (no model/prompt variant), "
                         "so this should almost never be passed; every mission's baseline "
                         "runs must share these SAME 3 tags for app.sweep_dashboard.py's "
                         "Baseline-tag dropdown to group them correctly instead of growing "
                         "a new per-mission entry")
    ap.add_argument("--dt", type=float, default=10.0, help="simulation time step (s)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="total step budget -- default: per-mission recommendation, see "
                         "app.narrate.recommended_max_steps")
    ap.add_argument("--decision-interval", type=int, default=None,
                    help="FIXED simulation steps between decisions -- default: adaptive "
                         "(app.narrate.live_decision_interval, same cadence the LLM agent "
                         "path uses by default) so baselines and the LLM agent are compared "
                         "at a fair, matching decision cadence. Pass an int to opt back "
                         "into the old fixed-cadence-every-N-steps behaviour.")
    ap.add_argument("--kinematics-model", default="kinematics", choices=["kinematics", "nomoto", "nomoto_v2"],
                    help="own-ship heading dynamics: legacy turn-rate slew (default, "
                         "unchanged), Sawada et al. (2021)'s Nomoto model (opt-in, see "
                         "app.simulation.VesselConstraints), or 'nomoto_v2' -- IDENTICAL "
                         "Nomoto physics to 'nomoto', but a distinct label for runs "
                         "generated with this session's absolute steering/adaptive-cadence "
                         "changes, so they're never confused with the frozen 'nomoto' "
                         "reference numbers already on disk")
    ap.add_argument("--force", action="store_true", help="overwrite existing logs")
    args = ap.parse_args()

    canonical_tag = CANONICAL_BASELINE_TAG_BY_KINEMATICS[args.kinematics_model]
    tag = args.tag if args.tag is not None else canonical_tag
    if args.tag is not None and args.tag != canonical_tag:
        print(f"WARNING: --tag {args.tag!r} overrides the canonical tag {canonical_tag!r} "
             f"for kinematics_model={args.kinematics_model!r} -- these runs will NOT be "
             "picked up as part of the standard 3-tag Baseline-tag comparison unless you "
             "know exactly what you're doing.")

    missions = args.missions or list_mission_ids()
    jobs = [(m, c) for m in missions for c in args.configs]
    print(f"Queued {len(jobs)} run(s): {len(missions)} mission(s) x {len(args.configs)} config(s), "
         f"tag={tag!r}, kinematics_model={args.kinematics_model!r}")
    for i, (mission_id, config) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {mission_id} / {config}")
        t0 = time.time()
        run_one(mission_id, config, tag=tag, dt=args.dt, max_steps=args.max_steps,
                decision_interval=args.decision_interval, force=args.force,
                kinematics_model=args.kinematics_model)
        print(f"  took {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
