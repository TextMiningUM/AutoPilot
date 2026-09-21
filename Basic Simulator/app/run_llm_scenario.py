"""Precompute an LLM-driven mission run and save it as a replayable log.

Calls the OOW agent only every `decision_interval` steps, dead-reckoning at the last decision
in between; the streamlit UI's "LLM driven" mode then just scrubs through the saved
trajectory instead of calling the (slow) model live -- that per-step latency is why Full Run
felt unusable interactively (see repo memory, Basic Simulator section).

`decision_interval` defaults to a PER-MISSION recommendation (app.narrate.
recommended_decision_interval -- short for a fast-closing encounter, long for a quiet/slow
one) rather than one fixed value for every mission, since a slow mission with nothing
urgent happening doesn't need the same (expensive) call cadence as a tight crossing --
pass --decision-interval to override it. Comparability across configs is preserved because
ALL configs run against the SAME mission still get the SAME interval (just not necessarily
the same interval as some OTHER mission).

Every agent parameter used (config, thinking, max_new_tokens, k, system prompt, dt,
decision_interval) is stored alongside the trajectory, so multiple variations of the SAME
mission can be run under different --tag values and compared side by side later.

Run one:
    python -m app.run_llm_scenario --missions s01_head_on --configs v3_rag_cot

Run a batch overnight (sequential -- only one GPU):
    python -m app.run_llm_scenario --configs bare_qwen v3_rag_cot v4_pg --tag baseline
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

from app.missions import list_mission_ids, load_mission, mission_to_dict
from app.simulation import Simulation, VesselConstraints, COLLISION_RADIUS_M
from app.agents import ask_oow, MODEL_CONFIGS, SYSTEM_OOW_AGENT, effective_generation_params
from app.evaluation import llm_compliance_check, score_trajectory
from app.llm_runs import RUNS_DIR, run_log_path
from app.narrate import recommended_decision_interval


def run_one(mission_id: str, config: str, tag: str = "default",
           dt: float = 10.0, max_steps: int = 200, enable_thinking: bool = False,
           max_new_tokens: int = 256, k: int = 2, use_rag: bool = True,
           system_prompt: str | None = None, force: bool = False,
           decision_interval: int | None = None, check_colreg_compliance: bool = True) -> Path:
    out_path = run_log_path(mission_id, config, tag)
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to overwrite)")
        return out_path

    mission = load_mission(mission_id)
    effective_interval = (decision_interval if decision_interval is not None
                          else recommended_decision_interval(mission, dt))
    # time_step_s matches --dt (not VesselConstraints' own default) so the agent's per-step
    # turn-degrees estimate in the prompt (see agents.build_oow_prompt) stays accurate
    # regardless of what --dt this run actually uses. cruise_speed_mps matches the mission's
    # OWN designed speed (not VesselConstraints' generic 10.0 m/s default) -- this CLI has no
    # live sidebar for a user to deliberately set a different resume speed, so overriding
    # narrate()'s nominal-speed reference with a mismatched constant would just be wrong
    # (reproduced: told the model s01_head_on's nominal speed was 10 m/s when the mission
    # actually runs at 2.5 m/s).
    constraints = VesselConstraints(time_step_s=dt, cruise_speed_mps=mission.own_ship.speed)
    sim = Simulation(mission, constraints)
    checkpoints: list[dict] = []
    effective_k = k if use_rag else 0
    outcome = "max_steps_reached"
    step = 0
    # What ask_oow() will ACTUALLY use once inside (it silently forces thinking+budget up
    # for CoT configs regardless of what's passed) -- log this instead of the raw args so
    # the dashboard doesn't show "Thinking: off" for a run that in fact had it forced on.
    effective_thinking, effective_max_new_tokens = effective_generation_params(
        config, enable_thinking, max_new_tokens)

    _t_run_start = time.time()
    for step in range(max_steps):
        # Collision check FIRST, and with a `dt`-second look-ahead horizon: two fast-
        # closing vessels (e.g. a 20 m/s head-on Imazu encounter) can pass by/through each
        # other entirely within the UPCOMING step, so a same-instant-only check (or one
        # that runs only after reached_goal() already said no) can silently miss a real
        # collision -- confirmed on Imazu01/v1_rag, which reached its goal well after
        # passing straight through a target with the outcome never once recording it. See
        # Simulation.min_cpa_now()/find_collision()'s docstrings for the confirmed numbers.
        if sim.min_cpa_now(horizon_s=dt) < COLLISION_RADIUS_M:
            outcome = "collision"
            break
        if sim.reached_goal():
            outcome = "reached_goal"
            break
        if step % effective_interval == 0:
            _t_cp = time.time()
            decision, debug = ask_oow(
                mission, sim.own, sim.targets, config=config, system_prompt=system_prompt,
                max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, k=effective_k,
                constraints=constraints,
            )
            cp_latency_s = time.time() - _t_cp
            # Per-checkpoint progress -- without this, a slow config (e.g. RAG+CoT combined,
            # which can take 10x longer per call than CoT alone) looked indistinguishable
            # from a hung process for the whole run's duration.
            print(f"    checkpoint {len(checkpoints) + 1} (step {step}/{max_steps}): "
                  f"{decision.get('action')} -- {cp_latency_s:.1f}s", flush=True)
            checkpoints.append({
                "step": step, "time": sim.t,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_s": cp_latency_s,
                "situation_report": debug.get("situation"),
                "decision": decision,
                "debug": {kk: vv for kk, vv in debug.items() if kk != "situation"},
            })
            sim.apply_action(decision)
        sim.step(dt)

    latency_s = time.time() - _t_run_start

    # One-shot Anthropic Claude judge of the WHOLE completed trajectory's COLREG compliance,
    # run automatically at the end of every mission's calculation (not per-checkpoint -- a
    # single network round-trip after the fact, same call app.evaluation.llm_compliance_check
    # already offered as an on-demand button in the live UI). A failure here (missing API
    # key, network error, ...) must not take the rest of a long sweep down -- recorded as
    # "checked: false" with the error message instead of raising.
    colreg_llm_check = {"checked": False, "violations": None, "compliant_actions": None,
                        "compliance_score": None, "error": None}
    if check_colreg_compliance:
        try:
            audit = llm_compliance_check(sim.trajectory, checkpoints=checkpoints)
            colreg_llm_check["violations"] = audit["violations"]
            colreg_llm_check["compliant_actions"] = audit["compliant_actions"]
            colreg_llm_check["compliance_score"] = audit["compliance_score"]
            colreg_llm_check["checked"] = True
        except Exception as exc:
            colreg_llm_check["error"] = str(exc)

    log = {
        "mission_id": mission_id, "config": config, "tag": tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latency_s": latency_s,
        # Both embedded in full (not just mission_id, a lookup key into a SEPARATE file)
        # so this one log file is self-sufficient even if Data/missions/{id}.json is later
        # moved, renamed, or this log itself gets archived into its own subfolder without
        # that file alongside it -- see app.missions.mission_from_dict/mission_to_dict and
        # app.evaluation.score_trajectory.
        "mission": mission_to_dict(mission),
        "evaluation": score_trajectory(
            sim.trajectory, start_xy=(mission.own_ship.x, mission.own_ship.y),
            goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
            safe_distance_m=constraints.min_cpa_m,
            llm_violations=colreg_llm_check["violations"],
            llm_compliance_score=colreg_llm_check["compliance_score"],
        ),
        "colreg_llm_check": colreg_llm_check,
        "params": {
            "decision_interval": effective_interval, "dt": dt, "max_steps": max_steps,
            "enable_thinking": effective_thinking, "max_new_tokens": effective_max_new_tokens,
            "k": k, "use_rag": use_rag,
            "system_prompt": system_prompt or SYSTEM_OOW_AGENT,
            "system_prompt_is_custom": system_prompt is not None,
        },
        "outcome": {"verdict": outcome, "final_step": step, "final_time_s": sim.t},
        "trajectory": sim.trajectory,
        "checkpoints": checkpoints,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    _cc = (f"score={colreg_llm_check['compliance_score']:.2f} "
          f"({len(colreg_llm_check['violations'])} violation(s), "
          f"{len(colreg_llm_check['compliant_actions'])} compliant action(s))"
          if colreg_llm_check["checked"] else f"not checked ({colreg_llm_check['error']})")
    print(f"  [done] {out_path.name}  outcome={outcome}  steps={step}  "
          f"checkpoints={len(checkpoints)}  latency={latency_s:.1f}s  colreg_check={_cc} "
          f"composite={log['evaluation']['composite_score']:.3f}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--missions", nargs="+", default=None,
                    help="mission ids to run (default: all missions)")
    ap.add_argument("--configs", nargs="+", default=["v3_rag_cot"], choices=list(MODEL_CONFIGS),
                    help="one or more of the 8 model configs (see app.agents.MODEL_CONFIGS)")
    ap.add_argument("--tag", default="default",
                    help="distinguishes variations of the same mission+config -- e.g. run the "
                         "same config twice with --tag thinking_on --enable-thinking vs the "
                         "default, then compare both logs for the same mission")
    ap.add_argument("--dt", type=float, default=10.0, help="simulation time step (s)")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--decision-interval", type=int, default=None,
                    help="simulation steps between LLM decision calls -- default: a per-mission "
                         "recommendation (short for a fast-closing encounter, long for a quiet/slow "
                         "one, see app.narrate.recommended_decision_interval); set explicitly to "
                         "force the same cadence across every mission in this run")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="enable Qwen3's native hidden reasoning channel (slower)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=2, help="RAG chunks for v1_rag/v3_rag_cot")
    ap.add_argument("--no-rag", action="store_true", help="force k=0 regardless of --k")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                    help="path to a text file with a custom system prompt override "
                         "(default: agents.SYSTEM_OOW_AGENT)")
    ap.add_argument("--force", action="store_true", help="overwrite existing logs")
    ap.add_argument("--no-colreg-check", action="store_true",
                    help="skip the end-of-mission Anthropic Claude COLREG compliance check "
                         "(saves one network call + latency per run; needs ANTHROPIC_API_KEY "
                         "in .env otherwise)")
    args = ap.parse_args()

    missions = args.missions or list_mission_ids()
    system_prompt = (args.system_prompt_file.read_text(encoding="utf-8")
                     if args.system_prompt_file else None)

    jobs = [(m, c) for m in missions for c in args.configs]
    print(f"Queued {len(jobs)} run(s): {len(missions)} mission(s) x {len(args.configs)} config(s), "
         f"tag={args.tag!r}")
    for i, (mission_id, config) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {mission_id} / {config}")
        t0 = time.time()
        run_one(
            mission_id, config, tag=args.tag,
            dt=args.dt, max_steps=args.max_steps, enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens, k=args.k, use_rag=not args.no_rag,
            system_prompt=system_prompt, force=args.force,
            decision_interval=args.decision_interval,
            check_colreg_compliance=not args.no_colreg_check,
        )
        print(f"  took {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
