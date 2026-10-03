"""
================================================================================
build_oow_decision_reversal_dpo.py -- Stap 2: decision-reversal DPO pairs
================================================================================

Targets a specific failure mode found in the 2026-10-01/02 GRPO A/B evaluation
(IMP12/IMP14 + 20 Imazu missions, see repo memory): the live agent's prompt already
injects a "stay consistent with your last decision unless the situation has genuinely
changed enough" instruction (pipeline/oow_agent_spec.py's render_previous_decisions()).
When that balance tips too far, the model gets stuck repeating `hold_course` long after
risk has escalated past the point it should have committed to a turn -- confirmed
empirically: 3 Imazu missions that previously reached the goal regressed to
max_steps_reached under GRPO, each spending 53-62% of all checkpoints on hold_course.

This script mines the exact corrective signal for that failure, DETERMINISTICALLY (no
LLM, no GPU) from run logs that ALREADY exist on disk (any `app/run_llm_scenario.py`
output, LLM-driven or deterministic baseline alike -- unlike build_dagger_dpo.py, the
trajectory's own driver is irrelevant here since only the ORACLE's own temporal
consistency is examined, never the model's actual action):

  For every pair of CONSECUTIVE checkpoints (t-1, t) within one run's trajectory, the
  oracle's own required action is recomputed independently at each (reusing
  build_dagger_dpo.py's exact state-reconstruction + oracle_plan() call -- never trusting
  the model's self-reported situation text). Whenever the oracle's action at t genuinely
  differs from its own action at t-1, that transition is the moment "genuinely changed
  enough" should fire:

    chosen   = the oracle's correct NEW decision at t (real oracle output, not invented).
    rejected = the oracle's OWN decision from t-1 (real oracle output from an earlier,
               now-stale timestep of the SAME trajectory) -- i.e. literally repeating the
               previous step's answer, the exact stalling behaviour observed.

  The prompt for each pair includes `render_previous_decisions()`'s real history preamble
  (the same text the live agent actually sees), reconstructed from the t-1 decision, so
  training exactly mirrors the live inference-time context this failure occurs in.

Only mines from runs under the CURRENT prompt architecture (same prompt_version gate as
build_dagger_dpo.py/build_outcome_dpo.py) -- an older prompt's trajectory doesn't reflect
what the CURRENT live agent needs to learn.

Mandatory contamination filter (per copilot-instructions.md): reuses
build_measurement_dpo.py's load_gold_texts()/filter_contamination() unchanged.

USAGE
-----
    python -m pipeline.track2.build_oow_decision_reversal_dpo                 # every run log
    python -m pipeline.track2.build_oow_decision_reversal_dpo --tags TAG1 TAG2 # restrict to tags
"""
from __future__ import annotations

import argparse
import json

from core import AgentPaths
from pipeline.oow_agent_spec import PROMPT_VERSION, render_previous_decisions
from pipeline.track2.build_measurement_dpo import FIXED_QUESTION, filter_contamination, load_gold_texts
from pipeline.track2.build_outcome_dpo import _build_constraints, iter_run_paths

paths = AgentPaths.oow()
CACHE = paths.cache_dir
OUT_PATH = CACHE / "oow_decision_reversal_dpo_pairs.jsonl"

APP_ROOT = paths.ensure_basic_simulator_importable()

from app.evaluation import _rows_at_time  # noqa: E402
from app.llm_runs import parse_run_filename  # noqa: E402
from app.missions import Vessel, mission_from_dict  # noqa: E402
from app.oracle_planner import plan as oracle_plan  # noqa: E402

CURRENT_PROMPT_VERSION = PROMPT_VERSION


def _own_and_targets_at(trajectory: list[dict], t: float,
                        own_vehicle: str = "own_ship") -> tuple[Vessel, list[Vessel]] | None:
    """Reconstruct the exact own-ship/target Vessel state at time `t` from a run's own
    saved trajectory -- never the model's self-reported situation text. Identical to
    build_dagger_dpo.py's own helper (small deliberate duplication, same rationale as
    that file's REASONING_SYSTEM_PROMPT note: these two miners are independent passes
    over the same run-log universe and shouldn't share a fragile import chain)."""
    rows = _rows_at_time(trajectory, t)
    own_row = rows.get(own_vehicle)
    if not own_row:
        return None
    own = Vessel(own_vehicle, own_row["x"], own_row["y"], own_row["heading"], own_row["speed"])
    targets = [Vessel(name, row["x"], row["y"], row["heading"], row["speed"])
              for name, row in sorted(rows.items()) if name != own_vehicle]
    return own, targets


def _phrase_action(action: str, degrees: float | None) -> str:
    """Natural-language phrasing of an action/degrees pair -- avoids the awkward
    'hold_course corrects this'-style sentences a raw string-substitution approach would
    produce when the action itself is hold_course/speed_up/etc."""
    if action == "turn_right":
        return f"turn right by {degrees:.0f} deg" if degrees is not None else "turn right"
    if action == "turn_left":
        return f"turn left by {degrees:.0f} deg" if degrees is not None else "turn left"
    return {"hold_course": "hold course", "speed_up": "speed up",
           "slow_down": "slow down", "stop": "stop"}.get(action, action)


def build_reversal_chosen(oracle_decision: dict) -> dict:
    """The oracle's own fresh action/rule at time t, re-expressed as a fluent sentence --
    NEVER oracle_plan()'s own `reasoning` field verbatim, which is an internal debug
    string (e.g. "oracle rollout best offset 15 deg (cost=0.382)") rather than natural
    prose; copilot-instructions.md's "fluent prose, never telegraphic label:value dumps"
    rule applies here same as every other training-data builder (same style as
    build_outcome_dpo.py's build_acute_chosen()/build_goal_chosen())."""
    action, degrees = oracle_decision["action"], oracle_decision.get("degrees")
    conduct_rule = oracle_decision.get("conduct_rule") or "none"
    phrase = _phrase_action(action, degrees)
    if conduct_rule != "none":
        reasoning = (f"The situation has evolved enough that {conduct_rule} now applies -- the ship "
                    f"should {phrase} to maintain safe separation, superseding the earlier assessment.")
    else:
        reasoning = (f"No contact poses a real risk of collision, but the earlier course is no longer "
                    f"the best way toward the goal -- the ship should now {phrase}.")
    return {"action": action, "degrees": degrees,
            "encounter_rule": oracle_decision.get("encounter_rule") or "none",
            "conduct_rule": conduct_rule, "reasoning": reasoning}


def build_reversal_rejected(prev_decision: dict) -> dict:
    """The now-stale decision from t-1, re-expressed as a fluent sentence that reads like
    a model mechanically repeating its last call without noticing the situation changed --
    the exact failure mode this script targets (see module docstring)."""
    action, degrees = prev_decision["action"], prev_decision.get("degrees")
    phrase = _phrase_action(action, degrees)
    reasoning = f"Staying consistent with the previous decision -- continuing to {phrase} still looks appropriate, no need to change course."
    return {"action": action, "degrees": degrees,
            "encounter_rule": prev_decision.get("encounter_rule") or "none",
            "conduct_rule": prev_decision.get("conduct_rule") or "none", "reasoning": reasoning}


def _actions_differ(prev_decision: dict, new_decision: dict) -> bool:
    return prev_decision.get("action") != new_decision.get("action")


def mine_run(run: dict, path_name: str) -> list[dict]:
    trajectory = run.get("trajectory") or []
    checkpoints = run.get("checkpoints") or []
    if not trajectory or "mission" not in run or len(checkpoints) < 2:
        return []
    mission = mission_from_dict(run["mission"])
    constraints = _build_constraints(run)
    system_prompt = run.get("params", {}).get("system_prompt", "")
    seen: set[tuple] = set()
    pairs = []
    prev_state = _own_and_targets_at(trajectory, checkpoints[0]["time"])
    prev_oracle = oracle_plan(mission, *prev_state, constraints) if prev_state else None
    for cp in checkpoints[1:]:
        state = _own_and_targets_at(trajectory, cp["time"])
        if state is None or prev_oracle is None:
            prev_state, prev_oracle = state, (oracle_plan(mission, *state, constraints) if state else None)
            continue
        own, targets = state
        oracle_decision = oracle_plan(mission, own, targets, constraints)
        if not _actions_differ(prev_oracle, oracle_decision):
            prev_oracle = oracle_decision
            continue
        rejected = build_reversal_rejected(prev_oracle)
        # De-dup near-identical repeats within the SAME run -- keep diversity across runs.
        dedup_key = (rejected["action"], rejected.get("degrees"), oracle_decision["action"])
        if dedup_key in seen:
            prev_oracle = oracle_decision
            continue
        seen.add(dedup_key)
        chosen = build_reversal_chosen(oracle_decision)
        history_preamble = render_previous_decisions([
            {"action": prev_oracle.get("action"), "degrees": prev_oracle.get("degrees"),
             "conduct_rule": prev_oracle.get("conduct_rule")},
        ])
        user_msg = (f"{history_preamble}Situation:\n{cp.get('situation_report', '')}\n\n{FIXED_QUESTION}")
        pairs.append({
            "source_file": path_name, "step": cp.get("step"), "outcome": "decision_reversal",
            "prompt": [{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_msg}],
            "chosen": [{"role": "assistant", "content": json.dumps(chosen, ensure_ascii=False)}],
            "rejected": [{"role": "assistant", "content": json.dumps(rejected, ensure_ascii=False)}],
        })
        prev_oracle = oracle_decision
    return pairs


def mine_all(tags: list[str] | None = None) -> list[dict]:
    all_pairs: list[dict] = []
    n_scanned = n_stale = n_skipped_tag = 0
    for p in iter_run_paths():
        if tags is not None and parse_run_filename(p)["tag"] not in tags:
            n_skipped_tag += 1
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "checkpoints" not in data or "mission" not in data:
            continue
        n_scanned += 1
        if data.get("params", {}).get("prompt_version") != CURRENT_PROMPT_VERSION:
            n_stale += 1
            continue
        all_pairs.extend(mine_run(data, p.name))
    print(f"Scanned {n_scanned} run logs ({n_stale} predate the current prompt or have no "
         f"prompt_version [e.g. deterministic baselines], skipped; {n_skipped_tag} skipped "
         f"by --tags filter)")
    return all_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", default=None,
                    help="restrict to run logs with one of these tags (default: every tag)")
    args = ap.parse_args()
    pairs = mine_all(args.tags)
    print(f"Mined {len(pairs)} decision-reversal DPO pairs (deduped, pre-contamination-filter)")
    gold_texts = load_gold_texts()
    pairs = filter_contamination(pairs, gold_texts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_PATH} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
