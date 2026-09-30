"""pipeline/train/build_grpo_dataset.py -- Stap 2 Step 9 prep: builds the GRPO training
dataset (prompt + raw simulator state) for pipeline/train/train_grpo.py.

Unlike pipeline/track2/build_oow_scenarios_rnd.py (which produces a FORMATTED teacher
reasoning trace for SFT/DPO/reflection), this script keeps the RAW own-ship/targets/
constraints/mission state alongside each prompt -- pipeline/train/reward_grpo.py needs
that state to recompute oracle_planner.required_direction()/_rollout_cost() against
whatever action the POLICY ITSELF samples at training time (never known in advance,
unlike the oracle-labeled Step 3 data). This is a small, deliberate duplication of
rollout_mission()'s decision-cadence loop (same convention build_oow_scenarios_rnd.py's
own docstring already documents for build_oow_scenarios_leo.py) -- justified here because
the two scripts need fundamentally different OUTPUT SHAPES (formatted text vs. raw
state), not just different content.

Only the "acute_action" bucket (a real turn/stop) is kept -- same rationale as
build_rft_filter.py's own choice: this is where the reward signal is actually
informative; quiet-cruise checkpoints would trivially reward "follow GOAL COURSE CHECK"
with near-zero learning signal either way.

USAGE
-----
    python -m pipeline.train.build_grpo_dataset --out-file Data/OOW/OOW_Agents_Training/oow_grpo_dataset.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from core import AgentPaths

paths = AgentPaths.oow()
CACHE = paths.cache_dir

APP_ROOT = paths.ensure_basic_simulator_importable()

from app import oracle_planner  # noqa: E402
from app.baselines import DECISION_FUNCS  # noqa: E402
from app.missions import list_mission_ids, load_mission  # noqa: E402
from app.narrate import live_decision_interval, narrate, recommended_decision_interval  # noqa: E402
from app.simulation import Simulation, VesselConstraints  # noqa: E402
from pipeline.oow_agent_spec import render_previous_decisions  # noqa: E402
from pipeline.track2.build_oow_scenarios_rnd import FIXED_QUESTION, MAX_CHECKPOINTS_PER_MISSION  # noqa: E402

HISTORY_N = 2  # matches build_oow_scenarios_rnd.py's own convention.


def _bucket_is_acute(action: str) -> bool:
    return action in ("turn_left", "turn_right", "stop")


def rollout_mission_for_grpo(mission, low_level_controller: str | None = None) -> list[dict]:
    """Same decision-cadence/apply_action pattern as build_oow_scenarios_rnd.py's
    rollout_mission(), but the oracle here is used ONLY to decide when/whether a
    checkpoint is acute (bucket filter) -- what gets WRITTEN OUT is the raw own/targets/
    constraints/mission state, not the oracle's own chosen action, since train_grpo.py's
    reward functions need to score whatever the POLICY samples, not replay the oracle's
    single fixed choice. `low_level_controller` -- see build_oow_scenarios_rnd.py's own
    parameter of the same name for the full rationale (opt-in, default None=unchanged)."""
    constraints = VesselConstraints(kinematics_model="nomoto_v2")
    sim = Simulation(mission, constraints)
    dt = constraints.time_step_s
    next_interval_steps = recommended_decision_interval(mission, dt)
    next_decision_step = 0
    transit_cap = next_interval_steps
    history: list[dict] = []
    rows: list[dict] = []
    max_steps = min(2000, MAX_CHECKPOINTS_PER_MISSION * 20)

    for step in range(max_steps):
        if sim.reached_goal():
            break
        if len(rows) >= MAX_CHECKPOINTS_PER_MISSION:
            break
        if step >= next_decision_step:
            next_interval_steps = live_decision_interval(
                sim.own, sim.targets, transit_cap, constraints.min_cpa_m,
                oracle_planner.derive_risk_horizon_s(constraints.min_cpa_m, constraints.max_rudder_angle_deg,
                                                    sim.own.speed))
            situation_report = narrate(mission, sim.own, sim.targets, cruise_speed_mps=constraints.cruise_speed_mps,
                                       safe_distance_m=constraints.min_cpa_m, max_turn_deg=constraints.max_rudder_angle_deg)
            result = oracle_planner.plan(mission, sim.own, sim.targets, constraints)
            if _bucket_is_acute(result["action"]):
                history_prefix = render_previous_decisions(history[-HISTORY_N:])
                user_msg = f"Situation:\n{history_prefix}{situation_report}\n\n{FIXED_QUESTION}"
                rows.append({
                    "rnd_id": mission.id,
                    "prompt": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": user_msg},
                    ],
                    "mission": dataclasses.asdict(mission),
                    "own": dataclasses.asdict(sim.own),
                    "targets": [dataclasses.asdict(t) for t in sim.targets],
                    "constraints": dataclasses.asdict(constraints),
                })
            history.append({"action": result["action"], "degrees": result["degrees"],
                           "conduct_rule": result["conduct_rule"]})
            history = history[-HISTORY_N:]
            sim.apply_action(result)
            next_decision_step = step + next_interval_steps
        elif low_level_controller is not None:
            # Same rationale as build_oow_scenarios_rnd.py's own low_level_controller
            # branch -- never a training-prompt candidate, only keeps the trajectory the
            # next oracle decision sees consistent with a low-level-controller-enabled run.
            ll_decision, _ll_debug = DECISION_FUNCS[low_level_controller](mission, sim.own, sim.targets, constraints)
            sim.apply_action(ll_decision)
        sim.step(dt)
    return rows


def _system_prompt() -> str:
    from pipeline.oow_agent_spec import SYSTEM_OOW_AGENT
    return SYSTEM_OOW_AGENT


def build_dataset(mission_ids: list[str] | None = None, low_level_controller: str | None = None) -> list[dict]:
    ids = mission_ids or [m for m in list_mission_ids() if m.startswith("RND")]
    missions = [load_mission(i) for i in ids]
    trainable = [m for m in missions if not json.loads(
        (APP_ROOT / "Data" / "missions" / f"{m.id}.json").read_text(encoding="utf-8")).get("held_out")]
    rows: list[dict] = []
    for m in trainable:
        rows.extend(rollout_mission_for_grpo(m, low_level_controller=low_level_controller))
    print(f"{len(rows)} acute_action GRPO rows across {len(trainable)} trainable missions")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-file", type=str, default=str(CACHE / "oow_grpo_dataset.jsonl"))
    ap.add_argument("--missions", nargs="+", default=None)
    ap.add_argument("--low-level-controller", default=None, choices=list(DECISION_FUNCS),
                    help="Stap 2 Step 11 (opt-in): fill the gaps between oracle decision "
                         "points with this deterministic baseline (e.g. baseline_ruletree) "
                         "instead of passively coasting -- keeps this dataset's trajectories "
                         "consistent with a low-level-controller-enabled live run")
    args = ap.parse_args()

    rows = build_dataset(args.missions, low_level_controller=args.low_level_controller)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
