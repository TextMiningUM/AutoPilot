"""
================================================================================
build_outcome_dpo.py -- Track 2 outcome-tied DPO pairs (RUN-VERDICT-DRIVEN)
================================================================================

Mines DPO pairs DETERMINISTICALLY (no LLM) from precomputed Basic Simulator mission
runs (Basic Simulator/Data/missions/_llm_runs/**/*.json), keyed off the RUN-LEVEL
OUTCOME (app/evaluation.py's score_trajectory() verdict, embedded as
log["evaluation"]) rather than per-decision reasoning quality alone -- contrast with
build_measurement_dpo.py's Check A/B/C mining, which flags a suspicious decision
independent of whether the mission actually ended badly. Three outcome classes are
mined here, per explicit user request:

  1. "FAIL -- collision occurred"     -- every ACUTE-risk checkpoint (a real,
     mandatory-action collision risk existed per app.evaluation._ground_truth_at_
     checkpoint's own band classification) where the model's actual action failed to
     follow the deterministic give-way/stand-on-escalation convention is mined:
     rejected = the model's real decision, chosen = the correct manoeuvre for that
     geometry (turn to starboard for a head-on/give-way-crossing role, "stop" for an
     overdue stand-on emergency escalation [Rule 17(a)(ii)/(b)] or a stationary/non-
     vessel hazard [Rule 8]).
  2. "PASS_WITH_CPA_VIOLATION"        -- the SAME acute-risk mining as (1) -- a near-
     miss that never quite collided is still real evidence of an under-reacting
     decision, and is far more common than an outright collision (see manifest counts).
  3. "FAIL -- did not reach the goal" -- every checkpoint with NO real collision risk
     (ground truth's decisive_contact is None) where the model's action diverged from
     GOAL COURSE CHECK's own recommendation (oow_agent_spec.goal_course_action()) --
     the deterministic "sails past / never corrects" mistake class documented in repo
     memory.

Deliberately conservative: a checkpoint is mined ONLY when a single, unambiguous
correct action can be derived (Rule 13 overtaking, where either side is acceptable, is
skipped rather than guessed) -- fewer, trustworthy pairs are preferred over inventing a
possibly-wrong "chosen" answer.

Ground truth is recomputed independently from the run's own saved trajectory (never
trusts the model's self-reported encounter_rule/conduct_rule/cpa/tcpa), reusing
app.evaluation._ground_truth_at_checkpoint -- the SAME machinery the live compliance
auditor already relies on -- plus pipeline.oow_agent_spec's classify_rules/
goal_course_action/STAND_ON_TCPA_S.

Only runs generated under the CURRENT prompt architecture (the same SYSTEM_OOW_AGENT +
constraint_line() source hash app/run_llm_scenario.py itself stamps into every log's
params.prompt_hash) are mined -- an older/superseded prompt's mistakes don't reflect
what the CURRENT live agent needs to learn not to do.

Mandatory contamination filter (per copilot-instructions.md): embeds each pair's user
prompt and drops any row scoring >=CONTAM_THRESH cosine similarity against EITHER
held-out file (colreg_qa_500_normalised.json / oow_colreg_scenarios_v1.json) -- reuses
build_measurement_dpo.py's load_gold_texts()/filter_contamination() unchanged.

USAGE
-----
    python -m pipeline.track2.build_outcome_dpo
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys

from core import AgentPaths
from pipeline.oow_agent_spec import (
    SYSTEM_OOW_AGENT, STAND_ON_TCPA_S, bearing_and_range, constraint_line,
    goal_course_action, relative_bearing,
)
from pipeline.track2.build_measurement_dpo import (
    FIXED_QUESTION, build_rejected, filter_contamination, load_gold_texts,
)

paths = AgentPaths.oow()
CACHE = paths.cache_dir
OUT_PATH = CACHE / "oow_outcome_dpo_pairs.jsonl"

# "Basic Simulator" has a space in its name, so it isn't a normal importable package --
# same sys.path trick pipeline/eval/measure_archived_checkpoints.py already uses.
APP_ROOT = paths.workspace / "Basic Simulator"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.evaluation import _ground_truth_at_checkpoint, _rows_at_time  # noqa: E402
from app.simulation import VesselConstraints  # noqa: E402

RUNS_DIR = APP_ROOT / "Data" / "missions" / "_llm_runs"

FAIL_COLLISION = "FAIL -- collision occurred"
CPA_VIOLATION = "PASS_WITH_CPA_VIOLATION"
FAIL_GOAL = "FAIL -- did not reach the goal"

# The SAME hash app/run_llm_scenario.py stamps into every log's params.prompt_hash --
# recomputed here (rather than imported from run_llm_scenario.py/app.agents, which pull
# in torch/transformers) since oow_agent_spec.py is deliberately dependency-free.
CURRENT_PROMPT_HASH = hashlib.sha256(
    (SYSTEM_OOW_AGENT + inspect.getsource(constraint_line)).encode("utf-8")).hexdigest()


def iter_run_paths():
    if not RUNS_DIR.exists():
        return
    for p in sorted(RUNS_DIR.rglob("*.json")):
        if "_review" in p.parts:
            continue
        yield p


def expected_acute_response(decisive: dict, constraints: VesselConstraints) -> tuple[str, float | None] | None:
    """The single deterministic correct manoeuvre for an ACUTE-band decisive contact --
    None when the geometry is genuinely ambiguous (an overtaking vessel may legitimately
    pass either side) or the role couldn't be classified, since inventing a "correct"
    answer for those risks teaching a wrong lesson."""
    if decisive["own_role"] == "none" and decisive["expected_conduct_rule"] == "Rule 8":
        return "stop", None  # stationary/non-vessel hazard
    if decisive["own_role"] == "stand_on":
        if decisive["tcpa_s"] < STAND_ON_TCPA_S:
            return "stop", None  # Rule 17(a)(ii)/(b): overdue emergency escalation
        return None  # still correct to hold course this early -- not a mistake to mine
    if decisive["expected_direction"] == "starboard":
        return "turn_right", constraints.max_rudder_angle_deg
    return None  # "either" (overtaking) or unclassified -- too ambiguous to mine safely


def is_acute_mistake(decision: dict, expected_action: str) -> bool:
    actual = decision.get("action")
    if expected_action == "stop":
        return actual != "stop"
    # A turn was expected -- taking it, OR an even-more-conservative "stop", both count
    # as adequate; anything else (hold_course, wrong direction, a speed change) doesn't.
    return actual not in (expected_action, "stop")


def is_goal_mistake(decision: dict, expected_action: str) -> bool:
    return decision.get("action") != expected_action


def build_acute_chosen(decisive: dict, action: str, degrees: float | None) -> dict:
    tail = f" (only {degrees:.0f} deg turn available per command)" if degrees is not None else ""
    reasoning = (
        f"The closest contact's CPA is {decisive['cpa_m']:.0f} m with TCPA "
        f"{decisive['tcpa_s']:.0f} s -- a real risk of collision. "
        f"{decisive['expected_encounter_rule']}/{decisive['expected_conduct_rule']} applies and "
        f"{action}{tail} is required now."
    )
    return {"action": action, "degrees": degrees,
            "encounter_rule": decisive["expected_encounter_rule"],
            "conduct_rule": decisive["expected_conduct_rule"], "reasoning": reasoning}


def build_goal_chosen(action: str, degrees: float | None, off_course_deg: float) -> dict:
    tail = f" by {degrees:.0f} deg" if degrees is not None else ""
    reasoning = (
        f"No contact poses a real risk of collision, but heading is {off_course_deg:.0f} deg off "
        f"the goal bearing -- no COLREG rule applies; correct with {action}{tail} to make progress "
        "toward the goal."
    )
    return {"action": action, "degrees": degrees, "encounter_rule": "none", "conduct_rule": "none",
            "reasoning": reasoning}


def mined_events(run: dict) -> list[dict]:
    """Core mining logic shared by this script (DPO pairs) and
    build_outcome_reflection.py (reflection triples): one entry per checkpoint whose
    actual decision is a confirmed, deterministically-detectable mistake given the run's
    real outcome. Each entry: {"cp", "outcome" ("collision"/"cpa_violation"/
    "goal_not_reached"), "chosen", "decisive" (contact ground truth, or None for the
    goal-not-reached class)}."""
    ev = run.get("evaluation")
    if not ev or ev.get("verdict") not in (FAIL_COLLISION, CPA_VIOLATION, FAIL_GOAL):
        return []
    trajectory = run.get("trajectory") or []
    if not trajectory:
        return []
    params = run.get("params", {})
    mission = run.get("mission", {})
    own_ship = mission.get("own_ship", {})
    goal = mission.get("goal", {})
    constraints = VesselConstraints(time_step_s=params.get("dt", 10.0),
                                    cruise_speed_mps=own_ship.get("speed", 10.0))
    verdict = ev["verdict"]
    events = []
    for cp in run.get("checkpoints", []):
        decision = cp.get("decision") or {}
        gt = _ground_truth_at_checkpoint(trajectory, cp["time"], "own_ship",
                                         constraints.min_cpa_m, constraints.max_rudder_angle_deg)
        contacts_by_name = {c["contact"]: c for c in gt["contacts"]}
        decisive = contacts_by_name.get(gt["decisive_contact"]) if gt["decisive_contact"] else None

        if verdict in (FAIL_COLLISION, CPA_VIOLATION):
            if decisive is None or decisive["band"] != "acute":
                continue
            expected = expected_acute_response(decisive, constraints)
            if expected is None or not is_acute_mistake(decision, expected[0]):
                continue
            chosen = build_acute_chosen(decisive, *expected)
            outcome_tag = "collision" if verdict == FAIL_COLLISION else "cpa_violation"
        else:
            if decisive is not None:
                continue  # a real encounter was in progress here -- not a "sailed past" case
            own_row = _rows_at_time(trajectory, cp["time"]).get("own_ship")
            if not own_row:
                continue
            goal_brg, _ = bearing_and_range(own_row["x"], own_row["y"], goal.get("x", 0.0), goal.get("y", 0.0))
            off_course = relative_bearing(own_row["heading"], goal_brg)
            action, degrees = goal_course_action(own_row["x"], own_row["y"], own_row["heading"],
                                                 goal.get("x", 0.0), goal.get("y", 0.0))
            if not is_goal_mistake(decision, action):
                continue
            chosen = build_goal_chosen(action, degrees, abs(off_course))
            outcome_tag = "goal_not_reached"

        events.append({"cp": cp, "outcome": outcome_tag, "chosen": chosen, "decisive": decisive})
    return events


def mine_run(run: dict, path_name: str) -> list[dict]:
    system_prompt = run.get("params", {}).get("system_prompt", "")
    seen: set[tuple[str, str]] = set()
    pairs = []
    for event in mined_events(run):
        cp = event["cp"]
        decision = cp.get("decision") or {}
        rejected = build_rejected(decision)
        # De-dup near-identical repeats within the SAME run (a static geometry can repeat
        # the exact same mistake for many consecutive checkpoints) -- keep diversity
        # across runs, drop redundant reinforcement.
        dedup_key = (event["outcome"], rejected["reasoning"])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        user_msg = (cp.get("debug", {}).get("user_msg")
                   or f"Situation:\n{cp.get('situation_report', '')}\n\n{FIXED_QUESTION}")
        pairs.append({
            "source_file": path_name, "step": cp.get("step"), "outcome": event["outcome"],
            "prompt": [{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_msg}],
            "chosen": [{"role": "assistant", "content": json.dumps(event["chosen"], ensure_ascii=False)}],
            "rejected": [{"role": "assistant", "content": json.dumps(rejected, ensure_ascii=False)}],
        })
    return pairs


def mine_all() -> list[dict]:
    all_pairs: list[dict] = []
    n_scanned = n_stale = 0
    for p in iter_run_paths():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "checkpoints" not in data or "evaluation" not in data:
            continue
        n_scanned += 1
        if data.get("params", {}).get("prompt_hash") != CURRENT_PROMPT_HASH:
            n_stale += 1
            continue
        all_pairs.extend(mine_run(data, p.name))
    print(f"Scanned {n_scanned} run logs ({n_stale} predate the current prompt, skipped)")
    return all_pairs


def main() -> None:
    pairs = mine_all()
    by_outcome: dict[str, int] = {}
    for p in pairs:
        by_outcome[p["outcome"]] = by_outcome.get(p["outcome"], 0) + 1
    print(f"Mined {len(pairs)} outcome-tied DPO pairs (deduped): {by_outcome}")
    gold_texts = load_gold_texts()
    pairs = filter_contamination(pairs, gold_texts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_PATH} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
