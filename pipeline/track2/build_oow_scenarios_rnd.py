"""
================================================================================
build_oow_scenarios_rnd.py -- Stap 2 Step 3: oracle-labeled closed-loop rollouts
================================================================================

Runs the RND01-60 mission pool (generate_random_imazu_missions.py, repo root of
Basic Simulator/Data/missions/) in a CLOSED LOOP using app.oracle_planner.plan() as the
decision-maker (never the LLM) under REAL Nomoto physics (kinematics_model="nomoto_v2"),
producing oracle-labeled training rows with genuine multi-step history -- the "Rollouts
-> SFT rows with real history and plan field" leg of the Stap 2 plan (see repo memory).

Unlike build_oow_scenarios_leo.py's Leo-state POOL (each row an independent, unconnected
snapshot from a large pre-recorded trajectory corpus), this file's rows come from
mission trajectories THIS SCRIPT ITSELF SIMULATES END TO END -- so "previous decisions"
history is always genuinely real and causally connected within one mission, not merely
plausible-looking. Reuses pipeline.oow_agent_spec.render_previous_decisions() (shared,
same function build_oow_scenarios_leo.py's own Fase B4 history uses) unchanged.

Each row additionally carries an ADDITIVE "oracle_plan" field (the oracle's own
`required_direction`/count-of-rejected-illegal-candidates rationale) -- the "plan field"
the Stap 2 plan asked for -- never shown to Qwen (build_user_message() only ever embeds
`situation_report`), meant for a future reflection-critique consumer.

`held_out` missions (generate_random_imazu_missions.py's own flag) are EXCLUDED from
EVERY output entirely -- reserved for a future eval pass, never mined into training data.

Fase B3 reasoning-text generation mirrors build_oow_scenarios_leo.py's own gated-Claude-
call pattern (pipeline.track2.b3_reasoning_gates, REASONING_SYSTEM_PROMPT duplicated here
per that file's own documented "small deliberate duplication between the two parallel
generators" convention) -- checkpointed/resumable, gated behind the SAME 25-first human-
review sample before a full-population run. UNLIKE that file's own history (see repo
memory's 2026-09-29 fix note), the mandatory contamination filter is wired in from this
script's very first version, not bolted on afterward.

USAGE
-----
    python -m pipeline.track2.build_oow_scenarios_rnd --b3-review-sample 15   # review gate
    python -m pipeline.track2.build_oow_scenarios_rnd --overwrite             # full population
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from core import AgentPaths, CONTAM_THRESH, EMBEDDER_MODEL, load_env, review_path, safe_write_jsonl
from pipeline.oow_agent_spec import SYSTEM_OOW_AGENT, render_previous_decisions

FIXED_QUESTION = "Recommend exactly ONE manoeuvre as the specified JSON object."

paths = AgentPaths.oow()
CACHE = paths.cache_dir
CHECKPOINT_FILE = CACHE / "oow_scenario_RND_b3_checkpoint.jsonl"

# "Basic Simulator" has a space in its name, so it isn't a normal importable package --
# same sys.path trick pipeline/track2/build_outcome_dpo.py already uses.
APP_ROOT = paths.workspace / "Basic Simulator"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app import oracle_planner  # noqa: E402
from app.baselines import DECISION_FUNCS  # noqa: E402
from app.missions import list_mission_ids, load_mission  # noqa: E402
from app.narrate import live_decision_interval, narrate, recommended_decision_interval  # noqa: E402
from app.simulation import Simulation, VesselConstraints  # noqa: E402

HISTORY_N = 2  # last N decisions shown -- matches build_oow_scenarios_leo.py's own convention.
MAX_CHECKPOINTS_PER_MISSION = 60  # safety cap -- no RND mission should need this many.
# Stratification cap (added after the first --skip-llm dry run showed 2152 checkpoints
# across 52 missions, ~41/mission -- mostly near-duplicate "still early band, nothing to
# do yet" rows that differ only by slightly decreasing range). Same lesson Leo's own pool
# already taught this project (LEO_CAP_FRACTION in train_sft.py): uncapped repetitive
# "hold_course, monitoring" rows over-represent one behavior in the SFT mix and drown out
# the rarer, more valuable acute-decision rows -- capped here at GENERATION time instead
# (cheaper: no wasted Claude calls on rows that would be dropped later anyway).
QUIET_CAP_PER_MISSION = 3


def _bucket(row: dict) -> str:
    """acute_action (a real turn/stop -- rare, high-value, ALWAYS kept in full) vs
    risk_monitoring (a real contact is decisive but band isn't acute yet, e.g. a
    stand-on vessel correctly holding before its Rule 17 deadline -- a genuinely
    different lesson from quiet_cruise, capped separately) vs quiet_cruise (no contact
    poses any real risk at all -- the most repetitive bucket, capped hardest)."""
    if row["action"] in ("turn_left", "turn_right", "stop"):
        return "acute_action"
    if row["decisive_contact_name"] is not None:
        return "risk_monitoring"
    return "quiet_cruise"


def stratify_mission_rows(rows: list[dict], quiet_cap: int = QUIET_CAP_PER_MISSION) -> list[dict]:
    """Keeps every acute_action row; evenly subsamples risk_monitoring/quiet_cruise rows
    down to `quiet_cap` each (even INDEX spacing across the bucket, not just the first N,
    so range/TCPA diversity within the capped rows is preserved rather than collapsing to
    only the mission's opening moments)."""
    by_bucket: dict[str, list[dict]] = {"acute_action": [], "risk_monitoring": [], "quiet_cruise": []}
    for r in rows:
        by_bucket[_bucket(r)].append(r)
    kept = list(by_bucket["acute_action"])
    for bucket_name in ("risk_monitoring", "quiet_cruise"):
        bucket_rows = by_bucket[bucket_name]
        if len(bucket_rows) <= quiet_cap:
            kept.extend(bucket_rows)
        else:
            step = len(bucket_rows) / quiet_cap
            kept.extend(bucket_rows[int(i * step)] for i in range(quiet_cap))
    kept.sort(key=lambda r: r["_id"])
    return kept


def rollout_mission(mission, low_level_controller: str | None = None) -> list[dict]:
    """Closed-loop rollout of ONE mission, oracle_planner.plan() deciding every step --
    same decision-cadence/apply_action pattern as Basic Simulator/app/run_llm_scenario.py
    (adaptive live_decision_interval, initial value from recommended_decision_interval),
    but with the oracle instead of an LLM call, and no LLM latency to log.

    `low_level_controller` (opt-in, default None -- see run_llm_scenario.py's own Step 11
    docstring for the full rationale): between oracle decision points, actively apply this
    deterministic baseline's own action instead of passively coasting -- keeps this
    generator's trajectories consistent with what a live low-level-controller-enabled run
    would actually produce. Omitting it is BYTE-IDENTICAL to the pre-Step-11 behaviour --
    the already-committed production oow_scenario_RND_*.jsonl files were generated without
    it and remain valid; regenerating WITH it is a separate, explicit follow-up."""
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
            risk_horizon_s = None  # computed inside oracle_planner/narrate as needed
            next_interval_steps = live_decision_interval(
                sim.own, sim.targets, transit_cap, constraints.min_cpa_m,
                oracle_planner.derive_risk_horizon_s(constraints.min_cpa_m, constraints.max_rudder_angle_deg,
                                                    sim.own.speed))
            situation_report = narrate(mission, sim.own, sim.targets, cruise_speed_mps=constraints.cruise_speed_mps,
                                       safe_distance_m=constraints.min_cpa_m, max_turn_deg=constraints.max_rudder_angle_deg)
            result = oracle_planner.plan(mission, sim.own, sim.targets, constraints)
            decisive = result["decisive_contact"]
            rows.append({
                "_id": f"{mission.id}_{len(rows):03d}", "rnd_id": mission.id,
                "situation_report": situation_report,
                "action": result["action"], "degrees": result["degrees"],
                "encounter_rule": result["encounter_rule"], "conduct_rule": result["conduct_rule"],
                "decisive_contact_name": decisive["name"] if decisive else None,
                "cpa_m": decisive["cpa_m"] if decisive else None,
                "tcpa_s": decisive["tcpa_s"] if decisive else None,
                "prev_decisions": list(history) if history else None,
                "oracle_plan": {
                    "required_direction": oracle_planner.required_direction(sim.own, sim.targets, constraints),
                    "n_rejected_illegal": sum(1 for c in result["rejected_candidates"] if c["illegal"]),
                    "n_candidates": len(result["rejected_candidates"]) + 1,
                },
                "reasoning": None,
            })
            history.append({
                "action": result["action"], "degrees": result["degrees"], "conduct_rule": result["conduct_rule"],
                "real_risk_contact_names": [decisive["name"]] if decisive else [],
            })
            history = history[-HISTORY_N:]
            sim.apply_action(result)
            next_decision_step = step + next_interval_steps
        elif low_level_controller is not None:
            # Deliberately NOT appended to `rows` -- unlike run_llm_scenario.py's live
            # checkpoints (logged in full for audit/dashboard traceability), a low-level
            # infill tick is never a training-data CANDIDATE here (it would explode the row
            # count with low-value physics-infill ticks, exactly what QUIET_CAP_PER_MISSION/
            # stratify_mission_rows() elsewhere in this file exists to avoid) -- it only
            # needs to happen so the SIMULATED TRAJECTORY the next real oracle decision sees
            # matches what a low-level-controller-enabled live run would actually produce.
            ll_decision, _ll_debug = DECISION_FUNCS[low_level_controller](mission, sim.own, sim.targets, constraints)
            sim.apply_action(ll_decision)
        sim.step(dt)
    return rows


REASONING_SYSTEM_PROMPT = """\
You are an expert deck officer analysing a COLREG close-quarters situation. You are \
given a FUSED SITUATION REPORT (own-ship state, contact bearing/range/CPA/TCPA/speed/ \
heading -- no rule numbers or role labels), the ACTION already decided (never change, \
second-guess, or invent a different action/degrees than given), and -- only when a real \
collision risk exists -- which contact (`decisive_contact`, by name) actually drove that \
decision. Never mention any OTHER contact as if it were the reason for the decision.

Write ONE fluent reasoning paragraph, in the officer's own voice, that:
  1. Describes what the geometry of the decisive contact shows (closing or opening, \
which side it is on, how much risk it poses) in your own words -- never a template or \
label dump. If other contacts are present, you may mention them, but the decisive \
contact must be the one your reasoning is actually built on.
  2. From THAT geometry alone, determines which vessel is give-way/stand-on (if \
either) and names BOTH: the encounter rule (Rule 13 overtaking / Rule 14 head-on / \
Rule 15 crossing / 'none' if no real risk), and the conduct rule that governs the \
SPECIFIC action being taken (Rule 16 give-way turn/speed change, Rule 14 head-on turn, \
Rule 17 stand-on, Rule 8 a give-way vessel's own emergency stop, or 'none' if no real \
risk). Whole rule numbers only, no sub-paragraphs.
  3. States the given action (and degrees, if any) and justifies it under the conduct \
rule. Cite ONLY numbers that appear verbatim in the situation text -- never invent, \
convert, round to a different unit, or compute an intermediate value not already given.

Each input record has:
  - situation: the full fused situation report text (may begin with a "Your last N helm
    decision(s)..." history preamble -- that describes PAST decisions only; derive the
    CURRENT encounter_rule/conduct_rule fresh from the CURRENT geometry/contacts alone)
  - action: the action name already decided
  - degrees: turn amount in degrees (only for turn_left/turn_right, else null)
  - decisive_contact: {"name", "cpa_m"} of the contact that drove the decision (omitted
    when there is no real risk)

OUTPUT FORMAT: return ONLY one JSON object, no markdown fences, no commentary:
{"id": "<copied verbatim from input>", "action": "<copied verbatim from input>",
 "degrees": <copied verbatim from input, null if not given>,
 "encounter_rule": "Rule N or 'none'", "conduct_rule": "Rule N or 'none'", "reasoning": "..."}
"""


def build_teacher_payload(rec: dict) -> tuple[dict, dict]:
    situation = rec["situation_report"]
    if rec.get("prev_decisions"):
        situation = render_previous_decisions(rec["prev_decisions"]) + situation
    payload = {"_id": rec["_id"], "situation": situation, "action": rec["action"], "degrees": rec["degrees"]}
    if rec["decisive_contact_name"] is not None:
        payload["decisive_contact"] = {"name": rec["decisive_contact_name"],
                                       "cpa_m": round(rec["cpa_m"], 0) if rec["cpa_m"] is not None else None}
    return payload, {
        "situation": situation, "expected_action": rec["action"], "expected_degrees": rec["degrees"],
        "expected_encounter_rule": rec["encounter_rule"], "expected_conduct_rule": rec["conduct_rule"],
        "real_risk": rec["encounter_rule"] != "none", "cpa_m": rec["cpa_m"], "safe_distance_m": 500.0,
        "tcpa_s": rec["tcpa_s"], "risk_horizon_s": 400.0, "decisive_contact_name": rec["decisive_contact_name"],
    }


def wrong_action_variant(decision: dict) -> dict:
    """Same deterministic principle as build_oow_scenarios_leo.py's own
    wrong_action_variant() -- a plausible but COLREG-incorrect alternative, for DPO
    'rejected' answers."""
    action, degrees = decision["action"], decision["degrees"]
    if action in ("turn_left", "turn_right"):
        wrong_action = "turn_left" if action == "turn_right" else "turn_right"
        return {"action": wrong_action, "degrees": degrees,
               "encounter_rule": decision["encounter_rule"], "conduct_rule": decision["conduct_rule"]}
    if action == "hold_course":
        return {"action": "turn_right", "degrees": 30.0, "encounter_rule": "none", "conduct_rule": "none"}
    return {"action": "hold_course", "degrees": None, "encounter_rule": "none", "conduct_rule": "none"}


def build_user_message(situation_report: str) -> str:
    """The USER turn -- situation text only, matching Basic Simulator/app/agents.py's
    build_oow_prompt() convention exactly, same as build_oow_scenarios_leo.py's own
    build_user_message() (duplicated, not imported -- established convention between the
    parallel Track-2 generators, see REASONING_SYSTEM_PROMPT's own docstring note)."""
    return f"Situation:\n{situation_report}\n\n{FIXED_QUESTION}"


def build_assistant_json(decision: dict, reasoning: str) -> dict:
    return {"action": decision["action"], "degrees": decision["degrees"],
            "encounter_rule": decision["encounter_rule"], "conduct_rule": decision["conduct_rule"],
            "reasoning": reasoning}


def load_checkpoint(path: Path = CHECKPOINT_FILE) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["_id"]] = rec
    return out


def append_checkpoint(recs: list[dict], path: Path = CHECKPOINT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


def filter_contamination(recs: list[dict], gold_questions: list[str]) -> list[dict]:
    """Mandatory (copilot-instructions.md) -- wired in from this script's FIRST version,
    unlike build_oow_scenarios_leo.py's own filter_contamination_batch() which existed
    since Fase B2 but was never actually called until the 2026-09-29 fix (see repo
    memory) -- same CONTAM_THRESH/EMBEDDER_MODEL, embeds `reasoning` text."""
    if not recs or not gold_questions:
        return recs
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER_MODEL)
    gold_embs = model.encode(gold_questions, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    texts = [r.get("reasoning") or "" for r in recs]
    q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    kept, dropped = [], 0
    for rec, emb in zip(recs, q_embs):
        c_sim = float(np.max(gold_embs @ emb)) if len(gold_embs) else 0.0
        if c_sim >= CONTAM_THRESH:
            dropped += 1
            continue
        kept.append(rec)
    print(f"contamination filter: kept={len(kept)}  dropped={dropped} (thresh={CONTAM_THRESH})")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", type=str, default="claude-sonnet-4-5")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gold-file", type=str, default=str(paths.eval_dir / "colreg_qa_500_normalised.json"))
    ap.add_argument("--skip-llm", action="store_true",
                    help="rollout + oracle decisions only -- no API calls, reasoning left None")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting existing production oow_scenario_RND_*.jsonl files")
    ap.add_argument("--b3-review-sample", type=int, default=None,
                    help="generate this many REAL Anthropic reasoning calls from the first "
                        "non-held-out rows and write them to _review/ for human review, then "
                        "exit -- never touches the full population or the checkpoint file")
    ap.add_argument("--low-level-controller", default=None, choices=list(DECISION_FUNCS),
                    help="Stap 2 Step 11 (opt-in, default None=unchanged legacy behaviour): "
                        "fill the gaps between oracle decision points with this deterministic "
                        "baseline instead of passively coasting -- see rollout_mission()'s own "
                        "docstring for the full rationale")
    args = ap.parse_args()

    # A low-level-controller-enabled rollout produces GENUINELY DIFFERENT trajectories
    # (different row _ids/positions, see rollout_mission()'s own docstring) -- reusing the
    # existing checkpoint/production filenames would risk silently attaching stale
    # reasoning (generated against the OLD, no-controller trajectory) to a same-named but
    # actually-different situation. Own checkpoint + output filenames instead, coexisting
    # with the existing _nomoto files rather than overwriting them.
    suffix = "_llc" if args.low_level_controller else ""
    checkpoint_file = CACHE / f"oow_scenario_RND{suffix}_b3_checkpoint.jsonl"

    all_ids = [m for m in list_mission_ids() if m.startswith("RND")]
    missions = [load_mission(m) for m in all_ids]
    trainable = [m for m in missions if not json.loads((paths.workspace / "Basic Simulator" / "Data" / "missions"
                                                       / f"{m.id}.json").read_text(encoding="utf-8")).get("held_out")]
    print(f"{len(missions)} RND missions found, {len(trainable)} trainable (not held out)")

    recs: list[dict] = []
    for m in trainable:
        recs.extend(rollout_mission(m, low_level_controller=args.low_level_controller))
    n_before_strat = len(recs)
    recs = [r for m in trainable for r in stratify_mission_rows([x for x in recs if x["rnd_id"] == m.id])]
    print(f"Rolled out {len(trainable)} missions -> {n_before_strat} decision checkpoints, "
         f"{len(recs)} after per-mission stratification (quiet_cap={QUIET_CAP_PER_MISSION})")

    if args.b3_review_sample is not None:
        load_env(paths.env_file)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set in .env")
        import anthropic
        from pipeline.track2.b3_reasoning_gates import GATE_NAMES, generate_gated_row
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        client = anthropic.Anthropic(api_key=key, default_headers=headers)
        sample = recs[:args.b3_review_sample]
        accepted, rejected = [], []
        gate_rejection_counts = {name: 0 for name in GATE_NAMES}
        for r in sample:
            payload, expected = build_teacher_payload(r)
            obj, attempt_log = generate_gated_row(client, args.model, REASONING_SYSTEM_PROMPT, payload,
                                                  max_attempts=3, max_tokens=args.max_tokens, **expected)
            for attempt in attempt_log:
                for gate_name in attempt["failures"]:
                    gate_rejection_counts[gate_name] += 1
            row = {"_id": r["_id"], "situation": expected["situation"], "ground_truth": {
                "action": expected["expected_action"], "degrees": expected["expected_degrees"],
                "encounter_rule": expected["expected_encounter_rule"], "conduct_rule": expected["expected_conduct_rule"]},
                "n_attempts": len(attempt_log), "attempt_log": attempt_log}
            (accepted if obj is not None else rejected).append({**row, "model_response": obj} if obj else row)
        out_path = review_path(CACHE / "oow_scenario_RND_b3_review_sample.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in accepted + rejected:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path} ({len(accepted) + len(rejected)} review rows: {len(accepted)} accepted, "
             f"{len(rejected)} dropped)")
        print(f"Per-gate rejection counts: {gate_rejection_counts}")
        print("This is a REVIEW-ONLY sample -- no production/checkpoint file was written. "
             "STOP here pending human review before running the full population.")
        return

    if not args.skip_llm:
        load_env(paths.env_file)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set in .env")
        import anthropic
        from pipeline.track2.b3_reasoning_gates import GATE_NAMES, generate_gated_row
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        client = anthropic.Anthropic(api_key=key, default_headers=headers)
        checkpoint = load_checkpoint(checkpoint_file)
        gate_rejection_counts = {name: 0 for name in GATE_NAMES}
        n_accepted = n_rejected = 0
        for i, r in enumerate(recs):
            cp = checkpoint.get(r["_id"])
            if cp is not None:
                if cp.get("reasoning"):
                    r["reasoning"] = cp["reasoning"]
                    n_accepted += 1
                else:
                    n_rejected += 1
                continue
            payload, expected = build_teacher_payload(r)
            obj, attempt_log = generate_gated_row(client, args.model, REASONING_SYSTEM_PROMPT, payload,
                                                  max_attempts=3, max_tokens=args.max_tokens, **expected)
            for attempt in attempt_log:
                for gate_name in attempt["failures"]:
                    gate_rejection_counts[gate_name] += 1
            if obj is not None:
                r["reasoning"] = obj["reasoning"]
                n_accepted += 1
            else:
                n_rejected += 1
            append_checkpoint([{"_id": r["_id"], "reasoning": r.get("reasoning")}], checkpoint_file)
            if (i + 1) % 20 == 0 or i + 1 == len(recs):
                print(f"  [B3] {i + 1}/{len(recs)} processed ({n_accepted} accepted, {n_rejected} dropped)", flush=True)
        print(f"Fase B3 done: {n_accepted} accepted, {n_rejected} dropped")
        print(f"Per-gate rejection counts: {gate_rejection_counts}")
    else:
        print("--skip-llm: reasoning left None pending Fase B3.")

    final_recs = [r for r in recs if r.get("reasoning")]

    if not args.skip_llm:
        gold_questions = [g["question"] for g in json.loads(Path(args.gold_file).read_text(encoding="utf-8"))]
        final_recs = filter_contamination(final_recs, gold_questions)

    sft_rows, dpo_rows, reflect_rows, trace_rows = [], [], [], []
    for r in final_recs:
        history_prefix = render_previous_decisions(r.get("prev_decisions") or [])
        user_msg = build_user_message(history_prefix + r["situation_report"])
        decision = {"action": r["action"], "degrees": r["degrees"],
                   "encounter_rule": r["encounter_rule"], "conduct_rule": r["conduct_rule"]}
        assistant_json = build_assistant_json(decision, r["reasoning"])
        sft_rows.append({"rnd_id": r["rnd_id"], "action": r["action"], "messages": [
            {"role": "system", "content": SYSTEM_OOW_AGENT}, {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
        ]})
        wrong_decision = wrong_action_variant(decision)
        rejected_json = build_assistant_json(wrong_decision, r["reasoning"])
        dpo_rows.append({
            "rnd_id": r["rnd_id"], "action": r["action"],
            "prompt": [{"role": "system", "content": SYSTEM_OOW_AGENT}, {"role": "user", "content": user_msg}],
            "chosen": [{"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}],
            "rejected": [{"role": "assistant", "content": json.dumps(rejected_json, ensure_ascii=False)}],
        })
        draft = f"I will {r['action'].replace('_', ' ')}." if r["action"] else "I will hold course."
        critique = ("This response is too vague -- it must state the exact action parameters "
                   "and cite the specific COLREG rule(s) that justify the decision.")
        reflect_rows.append({"rnd_id": r["rnd_id"], "messages": [
            {"role": "system", "content": SYSTEM_OOW_AGENT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"<think>\nDraft: {draft}\n\n{critique}\n</think>\n\n"
                                              f"{json.dumps(assistant_json, ensure_ascii=False)}"},
        ]})
        trace_rows.append({
            "document_id": f"oow_scenario_rnd_{r['_id']}", "chunk_id": f"oow_scenario_rnd_{r['_id']}",
            "source_file": f"oow_scenario_generator_rnd::{r['rnd_id']}",
            "chapter_title": f"OOW Track 2 (RND oracle) scenario: {r['rnd_id']}",
            "chunk_concepts": [r["rnd_id"], r["action"]],
            "trace": {
                "situation": r["situation_report"], "trigger": None,
                "procedures": [{"step": 1, "action": r["action"], "why": r["reasoning"]}],
                "constraints": [], "prowords_used": [], "channels": [r["encounter_rule"]],
                "regulations": ["COLREG 1972"], "warnings": [],
                "outcomes": [f"Action taken: {r['action']}."], "key_facts": [f"Chosen action: {r['action']}."],
                "question_seeds": [], "oracle_plan": r.get("oracle_plan"),
            },
        })

    outputs = {
        f"oow_scenario_RND_sft_direct_nomoto{suffix}.jsonl": sft_rows,
        f"oow_scenario_RND_sft_cot_nomoto{suffix}.jsonl": sft_rows,
        f"oow_scenario_RND_dpo_pairs_nomoto{suffix}.jsonl": dpo_rows,
        f"oow_scenario_RND_reflection_nomoto{suffix}.jsonl": reflect_rows,
        f"oow_scenario_RND_reasoning_traces_nomoto{suffix}.jsonl": trace_rows,
    }
    for name, rows in outputs.items():
        prod_path = CACHE / name
        out_path = prod_path if args.overwrite else review_path(prod_path)
        safe_write_jsonl(rows, out_path, overwrite=args.overwrite)
        print(f"Wrote {out_path} ({len(rows)} rows)"
             + ("" if args.overwrite else "  [dry-run/review path -- pass --overwrite for production]"))


if __name__ == "__main__":
    main()
