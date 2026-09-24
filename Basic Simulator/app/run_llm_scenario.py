"""Precompute an LLM-driven mission run and save it as a replayable log.

Calls the OOW agent only every `decision_interval` steps, dead-reckoning at the last decision
in between; the streamlit UI's "LLM driven" mode then just scrubs through the saved
trajectory instead of calling the (slow) model live -- that per-step latency is why Full Run
felt unusable interactively (see repo memory, Basic Simulator section).

2026-09-24: `decision_interval` is now ADAPTIVE by default -- recomputed at every
checkpoint from the CURRENT encounter (app.narrate.live_decision_interval), not frozen
from the mission's t=0 geometry for the whole run (the old approach was found to be
"inversely adaptive": a mission whose encounter opened far off kept the SAME sparse
cadence throughout, including once TCPA had fallen into the acute range). Pass
--decision-interval to opt back into the old fixed-cadence behaviour for the whole run
(e.g. for an apples-to-apples comparison against an older sweep). Comparability across
configs is preserved either way because ALL configs run against the SAME mission see the
SAME sequence of decisions (adaptive recompute is deterministic given the same mission/
constraints/model decisions).

Every agent parameter used (config, thinking, max_new_tokens, k, system prompt, dt,
decision_interval) is stored alongside the trajectory, so multiple variations of the SAME
mission can be run under different --tag values and compared side by side later. Each
checkpoint additionally logs its OWN decision_interval_steps/_s -- the adaptive value
actually used for the gap to the NEXT checkpoint -- since that now varies within a run.

Run one:
    python -m app.run_llm_scenario --missions s01_head_on --configs v3_rag_cot

Run a batch overnight (sequential -- only one GPU):
    python -m app.run_llm_scenario --configs bare_qwen v3_rag_cot v4_pg --tag baseline
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
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
from app.narrate import (
    contact_line, recommended_decision_interval, recommended_max_steps,
    live_decision_interval, transit_step_cap,
)
from pipeline.oow_agent_spec import constraint_line, derive_risk_horizon_s

# Screening-set-B audit follow-up (2026-09-23): identifies which PROMPT VERSION a run was
# generated under (SYSTEM_OOW_AGENT text + constraint_line()'s own source, which renders
# the per-mission constraint sentence) -- audit_runs.py refuses to aggregate runs whose
# prompt_hash differs within one tag, the same guard it already applies to schema
# mismatches. Hashing constraint_line()'s SOURCE (not one rendered instance) captures
# wording changes independent of whatever numeric safe_distance/horizon a given mission
# happens to sample.
_PROMPT_HASH = hashlib.sha256(
    (SYSTEM_OOW_AGENT + inspect.getsource(constraint_line)).encode("utf-8")).hexdigest()
# Human-readable companion to _PROMPT_HASH -- bump whenever SYSTEM_OOW_AGENT/
# constraint_line() wording changes, so an audit report or a human skimming params can
# tell runs apart without diffing hashes. 2026-09-24: uncapped turn orders (change 1) +
# adaptive decision cadence's "next decision point" fact (change 3).
PROMPT_VERSION = "2026-09-24-uncapped-turn-adaptive-cadence"


def run_one(mission_id: str, config: str, weights: str = "W0_base", tag: str = "default",
           dt: float = 10.0, max_steps: int | None = None, enable_thinking: bool = False,
           max_new_tokens: int = 256, k: int = 2, use_rag: bool = True,
           system_prompt: str | None = None, force: bool = False,
           decision_interval: int | None = None, explain: bool = False) -> Path:
    out_path = run_log_path(mission_id, config, weights, tag)
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to overwrite)")
        return out_path

    mission = load_mission(mission_id)
    # Fixed-cadence override (opt-in via --decision-interval) vs. the 2026-09-24 default:
    # ADAPTIVE recompute of the gap to the NEXT checkpoint from the live encounter, every
    # checkpoint (see app.narrate.live_decision_interval's docstring). `cap` is computed
    # ONCE from the mission's t=0 transit distance -- it never goes stale the way a live
    # per-checkpoint recompute of the ENCOUNTER thresholds would if it used frozen mission
    # positions instead of sim.own/sim.targets.
    fixed_interval = decision_interval
    cap = transit_step_cap(mission, dt)
    next_interval_steps = fixed_interval if fixed_interval is not None else recommended_decision_interval(mission, dt)
    initial_interval_steps = next_interval_steps  # logged in params -- see checkpoints for the per-step adaptive value
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
    next_decision_step = 0
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
        if step >= next_decision_step:
            # Adaptive recompute (2026-09-24): decided BEFORE calling ask_oow (from the
            # SAME live state about to be shown to the agent) so THIS call's own prompt
            # can state the resulting gap as a fact (constraint_line()'s "next decision
            # point"). A fixed --decision-interval override keeps next_interval_steps
            # constant instead, and next_decision_in_s along with it.
            if fixed_interval is None:
                risk_horizon_s = derive_risk_horizon_s(
                    constraints.min_cpa_m, constraints.max_rudder_angle_deg, sim.own.speed)
                next_interval_steps = live_decision_interval(
                    sim.own, sim.targets, cap, constraints.min_cpa_m, risk_horizon_s)
            next_decision_in_s = next_interval_steps * dt

            _t_cp = time.time()
            decision, debug = ask_oow(
                mission, sim.own, sim.targets, config=config, system_prompt=system_prompt,
                max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, k=effective_k,
                constraints=constraints, next_decision_in_s=next_decision_in_s,
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
                # The gap to the NEXT checkpoint -- SAME value the prompt above was just
                # told as a fact -- varies within a run now, so it's per-checkpoint, not
                # just one params-level value (see params["decision_interval"] below for
                # the run's INITIAL value).
                "decision_interval_steps": next_interval_steps,
                "decision_interval_s": next_decision_in_s,
            })
            sim.apply_action(decision)
            next_decision_step = step + max(1, next_interval_steps)
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

    # Compliance-rebuild STAP 3/4 (2026-09-23): the deterministic compliance score never
    # needs an API call, so it's always computed. The Claude explanation is a SEPARATE,
    # opt-in step (--explain, default off in a sweep) that only turns score_trajectory()'s
    # own findings into plain-language prose -- it never sees the raw trajectory and never
    # affects the score. A failure here must not take the rest of a long sweep down --
    # recorded as "checked: false" with the error message instead of raising.
    evaluation = score_trajectory(
        sim.trajectory, start_xy=(mission.own_ship.x, mission.own_ship.y),
        goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
        safe_distance_m=constraints.min_cpa_m, max_turn_deg=constraints.max_rudder_angle_deg,
        checkpoints=checkpoints,
    )
    colreg_llm_check = {"checked": False, "explanations": None, "error": None}
    if explain:
        try:
            audit = llm_compliance_check(evaluation["compliance"]["findings"])
            colreg_llm_check["explanations"] = audit["explanations"]
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
        "evaluation": evaluation,
        "colreg_llm_check": colreg_llm_check,
        "params": {
            "decision_interval": initial_interval_steps,
            "decision_interval_mode": "fixed" if fixed_interval is not None else "adaptive",
            "dt": dt, "max_steps": max_steps,
            "enable_thinking": effective_thinking, "max_new_tokens": effective_max_new_tokens,
            "k": k, "use_rag": use_rag,
            "system_prompt": system_prompt or SYSTEM_OOW_AGENT,
            "system_prompt_is_custom": system_prompt is not None,
            "prompt_hash": _PROMPT_HASH,
            "prompt_version": PROMPT_VERSION,
        },
        "outcome": {"verdict": outcome, "final_step": step, "final_time_s": sim.t},
        "trajectory": sim.trajectory,
        "checkpoints": checkpoints,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    _cc = (f"{len(colreg_llm_check['explanations'])} explanation(s)" if colreg_llm_check["checked"]
          else ("not requested" if not explain else f"failed ({colreg_llm_check['error']})"))
    print(f"  [done] {out_path.name}  outcome={outcome}  steps={step}  "
          f"checkpoints={len(checkpoints)}  latency={latency_s:.1f}s  "
          f"compliance={evaluation['compliance']['score']:.2f} explain={_cc} "
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
                    help="simulation steps between LLM decision calls -- default: ADAPTIVE, "
                         "recomputed every checkpoint from the live encounter (see "
                         "app.narrate.live_decision_interval); set explicitly to force one FIXED "
                         "cadence for the whole run instead (e.g. to reproduce an older sweep)")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="enable Qwen3's native hidden reasoning channel (slower)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=2, help="RAG chunks for v1_rag/v3_rag_cot")
    ap.add_argument("--no-rag", action="store_true", help="force k=0 regardless of --k")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                    help="path to a text file with a custom system prompt override "
                         "(default: agents.SYSTEM_OOW_AGENT)")
    ap.add_argument("--force", action="store_true", help="overwrite existing logs")
    ap.add_argument("--explain", action="store_true",
                    help="also ask Anthropic Claude for a plain-language explanation of the "
                         "deterministic compliance findings (one network call + latency per run, "
                         "needs ANTHROPIC_API_KEY in .env) -- never affects the score itself, off "
                         "by default in a sweep")
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
            explain=args.explain,
        )
        print(f"  took {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
