"""
================================================================================
build_dagger_dpo.py -- Stap 2 Step 7: DAgger DPO pairs from real model-driven rollouts
================================================================================

The "C" method from the Stap 2 plan (see repo memory): unlike build_oow_scenarios_rnd.py
(Step 3, the ORACLE itself sails and its own rollout becomes the chosen answer -- method
"B"), this script never re-simulates anything. It reads ALREADY-RECORDED closed-loop LLM
run logs (app/run_llm_scenario.py output, any tag/mission) -- i.e. the CURRENT model's own
real trajectory, including its own real mistakes -- and at EVERY checkpoint independently
recomputes what app.oracle_planner.plan() would have done from the exact same own-ship/
target state (reconstructed from the run's own saved `trajectory`, never trusting the
model's self-reported situation text). Wherever the model's actual action DIFFERS from the
oracle's, that's a genuine DAgger-style correction: the model visited this exact state by
its own doing (not a state the oracle or a scripted generator would have chosen), and now
gets told what the oracle would have done there instead.

  chosen   = the oracle's own plan() output (action/degrees/encounter_rule/conduct_rule/
             reasoning), verbatim.
  rejected = the model's OWN actual (disagreeing) decision, taken verbatim from the
             checkpoint (see build_measurement_dpo.build_rejected()).

Deliberately only mines DISAGREEMENTS (oracle action != model action), not every visited
state -- classic DAgger relabels every state, but here that would mostly be redundant
reinforcement of already-correct behaviour; a differing action class is the informative
signal. `degrees` is NOT compared (a turn_right 30deg model action vs. the oracle's own
turn_right 45deg pick are both a real, correctly-directed turn -- not a mistake).

Only runs generated under the CURRENT prompt architecture are mined (same PROMPT_VERSION
gate as build_outcome_dpo.py) -- an older/superseded prompt's own trajectory doesn't
reflect what the CURRENT live agent needs correcting. Baseline (deterministic, non-LLM)
run logs have no `params.prompt_version` at all, so they are excluded automatically by
this same check, not specially filtered.

Mandatory contamination filter (per copilot-instructions.md): embeds each pair's user
prompt and drops any row scoring >=CONTAM_THRESH cosine similarity against EITHER
held-out file, reusing build_measurement_dpo.py's load_gold_texts()/filter_contamination()
unchanged.

USAGE
-----
    python -m pipeline.track2.build_dagger_dpo                      # every run log on disk
    python -m pipeline.track2.build_dagger_dpo --tags TAG1 TAG2      # restrict to these tags
"""
from __future__ import annotations

import argparse
import json

from core import AgentPaths
from pipeline.oow_agent_spec import PROMPT_VERSION
from pipeline.track2.build_measurement_dpo import FIXED_QUESTION, build_rejected, filter_contamination, load_gold_texts
from pipeline.track2.build_outcome_dpo import _build_constraints, iter_run_paths

paths = AgentPaths.oow()
CACHE = paths.cache_dir
OUT_PATH = CACHE / "oow_dagger_dpo_pairs.jsonl"

APP_ROOT = paths.ensure_basic_simulator_importable()

from app.evaluation import _rows_at_time  # noqa: E402
from app.llm_runs import parse_run_filename  # noqa: E402
from app.missions import Vessel, mission_from_dict  # noqa: E402
from app.oracle_planner import plan as oracle_plan  # noqa: E402

CURRENT_PROMPT_VERSION = PROMPT_VERSION


def _own_and_targets_at(trajectory: list[dict], t: float,
                        own_vehicle: str = "own_ship") -> tuple[Vessel, list[Vessel]] | None:
    """Reconstruct the exact own-ship/target Vessel state at time `t` from a run's own
    saved trajectory -- never the model's self-reported situation text."""
    rows = _rows_at_time(trajectory, t)
    own_row = rows.get(own_vehicle)
    if not own_row:
        return None
    own = Vessel(own_vehicle, own_row["x"], own_row["y"], own_row["heading"], own_row["speed"])
    targets = [Vessel(name, row["x"], row["y"], row["heading"], row["speed"])
              for name, row in sorted(rows.items()) if name != own_vehicle]
    return own, targets


def build_dagger_chosen(oracle_decision: dict) -> dict:
    return {"action": oracle_decision["action"], "degrees": oracle_decision["degrees"],
            "encounter_rule": oracle_decision["encounter_rule"],
            "conduct_rule": oracle_decision["conduct_rule"], "reasoning": oracle_decision["reasoning"]}


def _actions_disagree(model_decision: dict, oracle_decision: dict) -> bool:
    return model_decision.get("action") != oracle_decision["action"]


def mine_run(run: dict, path_name: str) -> list[dict]:
    trajectory = run.get("trajectory") or []
    if not trajectory or "mission" not in run:
        return []
    mission = mission_from_dict(run["mission"])
    constraints = _build_constraints(run)
    system_prompt = run.get("params", {}).get("system_prompt", "")
    seen: set[str] = set()
    pairs = []
    for cp in run.get("checkpoints", []):
        decision = cp.get("decision") or {}
        state = _own_and_targets_at(trajectory, cp["time"])
        if state is None:
            continue
        own, targets = state
        oracle_decision = oracle_plan(mission, own, targets, constraints)
        if not _actions_disagree(decision, oracle_decision):
            continue
        rejected = build_rejected(decision)
        # De-dup near-identical repeats within the SAME run (a static geometry can repeat
        # the exact same mistake for many consecutive checkpoints) -- keep diversity
        # across runs, drop redundant reinforcement.
        dedup_key = rejected["reasoning"]
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        chosen = build_dagger_chosen(oracle_decision)
        user_msg = (cp.get("debug", {}).get("user_msg")
                   or f"Situation:\n{cp.get('situation_report', '')}\n\n{FIXED_QUESTION}")
        pairs.append({
            "source_file": path_name, "step": cp.get("step"), "outcome": "dagger_oracle_disagreement",
            "prompt": [{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_msg}],
            "chosen": [{"role": "assistant", "content": json.dumps(chosen, ensure_ascii=False)}],
            "rejected": [{"role": "assistant", "content": json.dumps(rejected, ensure_ascii=False)}],
        })
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
    print(f"Mined {len(pairs)} DAgger DPO pairs (deduped, pre-contamination-filter)")
    gold_texts = load_gold_texts()
    pairs = filter_contamination(pairs, gold_texts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_PATH} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
