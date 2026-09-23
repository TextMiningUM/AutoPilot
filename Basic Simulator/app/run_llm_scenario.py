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
from app.simulation import Simulation, VesselConstraints, find_collision
from app.agents import ask_oow, MODEL_CONFIGS, SYSTEM_OOW_AGENT, effective_generation_params
from app.evaluation import llm_compliance_check, score_trajectory
from app.llm_runs import RUNS_DIR, run_log_path
from app.measurement import measure_decision_quality
from app.narrate import contact_line, recommended_decision_interval, recommended_max_steps


def run_one(mission_id: str, config: str, weights: str = "W0_base", tag: str = "default",
           dt: float = 10.0, max_steps: int | None = None, enable_thinking: bool = False,
           max_new_tokens: int = 256, k: int = 2, use_rag: bool = True,
           system_prompt: str | None = None, force: bool = False,
           decision_interval: int | None = None, check_colreg_compliance: bool = True) -> Path:
    out_path = run_log_path(mission_id, config, weights, tag)
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to overwrite)")
        return out_path

    mission = load_mission(mission_id)
    effective_interval = (decision_interval if decision_interval is not None
                          else recommended_decision_interval(mission, dt))
    max_steps = max_steps if max_steps is not None else recommended_max_steps(mission, dt)
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
            # Deterministic, read-only measurement (Checks A/B/C -- see app/measurement.py)
            # against the SAME live contacts the agent was actually shown this step. Never
            # changes `decision`/`sim.apply_action()` below -- purely counted and logged.
            contacts_now = [contact_line(sim.own, t, constraints.min_cpa_m, constraints.max_rudder_angle_deg)
                           for t in sim.targets]
            checkpoints.append({
                "step": step, "time": sim.t,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_s": cp_latency_s,
                "situation_report": debug.get("situation"),
                "decision": decision,
                # The FULL raw model output -- every "Wait, ..." / self-correction inside
                # Qwen3's <think> block, not just the final action -- as its OWN top-level
                # field (a sibling of "decision", never buried inside "debug"'s grab-bag of
                # retrieval internals) so it can never silently disappear if debug's shape
                # changes later (e.g. once a different/fine-tuned model gets wired in).
                "reasoning_raw": debug.get("raw_response"),
                "measurement": measure_decision_quality(decision, contacts_now, constraints),
                "debug": {kk: vv for kk, vv in debug.items() if kk not in ("situation", "raw_response")},
            })
            sim.apply_action(decision)
        sim.step(dt)
        # Check the step JUST recorded (not a forward prediction -- see
        # Simulation.min_cpa_now()'s docstring for why a predictive check is unsafe here)
        # against the interpolated, ground-truth collision finder ALSO used for scoring
        # (evaluate_run.py) and plotting (find_collision()) -- two fast-closing vessels
        # (e.g. a 20 m/s head-on Imazu encounter) can pass by/through each other entirely
        # within one dt=10s step, so a same-instant-only check right after stepping could
        # still miss it; this reuses the exact interpolation-aware logic that already
        # catches that case, so outcome="collision" can never be reported without the
        # saved trajectory actually showing it.
        if find_collision(sim.trajectory) is not None:
            outcome = "collision"
            break

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
            audit = llm_compliance_check(sim.trajectory, checkpoints=checkpoints,
                                         safe_distance_m=constraints.min_cpa_m,
                                         max_turn_deg=constraints.max_rudder_angle_deg)
            colreg_llm_check["violations"] = audit["violations"]
            colreg_llm_check["compliant_actions"] = audit["compliant_actions"]
            colreg_llm_check["compliance_score"] = audit["compliance_score"]
            colreg_llm_check["checked"] = True
        except Exception as exc:
            colreg_llm_check["error"] = str(exc)

    log = {
        "mission_id": mission_id, "config": config, "weights": weights, "tag": tag,
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
    ap.add_argument("--configs", nargs="+",
                    default=["bare_qwen", "v0_base", "v7_super_rag", "v8_super_cot_pg", "v9_super_all"],
                    choices=list(MODEL_CONFIGS),
                    help="one or more model configs (see app.agents.MODEL_CONFIGS); default: "
                         "the 5 standard-sweep configs. v1_rag..v6_pg_scenario are an archived "
                         "ablation arm, pass them explicitly if you need to reproduce/extend "
                         "that old sweep")
    ap.add_argument("--tag", default="default",
                    help="distinguishes variations of the same mission+config -- e.g. run the "
                         "same config twice with --tag thinking_on --enable-thinking vs the "
                         "default, then compare both logs for the same mission")
    ap.add_argument("--weights", default="W0_base",
                    help="which checkpoint answered (a SEPARATE axis from --configs, which "
                         "selects the PROMPT) -- always W0_base (base Qwen3-8B) for now; "
                         "no fine-tuned-checkpoint loading path exists yet, see main plan phase F4")
    ap.add_argument("--dt", type=float, default=10.0, help="simulation time step (s)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="total step budget -- default: a per-mission recommendation sized "
                         "from the mission's own straight-line transit distance/speed (see "
                         "app.narrate.recommended_max_steps); set explicitly to force the same "
                         "budget across every mission in this run")
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
            mission_id, config, weights=args.weights, tag=args.tag,
            dt=args.dt, max_steps=args.max_steps, enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens, k=args.k, use_rag=not args.no_rag,
            system_prompt=system_prompt, force=args.force,
            decision_interval=args.decision_interval,
            check_colreg_compliance=not args.no_colreg_check,
        )
        print(f"  took {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
