"""
================================================================================
build_oow_scenarios_leo.py — Track 2 augmentation from the Leo MOOS trajectories
================================================================================

Data/OOW/OOW_Scenarios_Leo/moos_temporal_narratives_final.jsonl holds 7928 REAL
MOOS-simulated bridge states across 402 multi-step trajectories (up to 50 cycles
each) -- richer than build_oow_scenarios.py's synthetic single-contact geometry:
0-6 simultaneous contacts, sensor-realism fields (track_quality/age_s/
track_reacquired), own-ship movement_status (underway/paused/stalled), and
encounter/risk vocabularies (parallel, diverging, stationary_contact, risk
uncertain/critical) build_oow_scenarios.py doesn't generate.

It has NO action/gold_answer labels -- it is pure situation state, exactly like
build_oow_scenarios.py's geometry stage before its BUILD pass. This script is
that BUILD pass for the Leo data: own-ship/contact FACTS are re-rendered in our
OWN house prose style (never the source file's ALL-CAPS "CURRENT BRIDGE STATE /
AUTOMATED TRAFFIC ASSESSMENT" label-dump narrative field, which is the exact
telegraphic-dump anti-pattern already identified as harmful for this project --
see copilot-instructions.md) via a deterministic action choice + a Claude-authored
justification, same BUILD-vs-VALIDATE split as build_oow_scenarios.py.

Kept in SEPARATE oow_scenario_Leo_*.jsonl files (never merged into the plain
oow_scenario_*.jsonl files) so this source's contribution stays traceable and
reviewable independently, per explicit user direction.

Action choice reuses the Leo record's OWN already-computed own_role/risk labels
(more informed than re-deriving from raw CPA/TCPA, since Leo's role/risk
computation already accounts for track quality etc. we don't model) rather than
build_oow_scenarios.py's classify_encounter()/choose_action() geometry, which
assumes a single synthetic contact this file's real, variable-count contacts
don't fit -- but the OUTPUT task format (system prompt, action vocabulary, JSON
response schema, GOAL COURSE CHECK line) is shared byte-for-byte with Basic
Simulator/app/agents.py and build_oow_scenarios.py via pipeline/oow_agent_spec.py
(Fase B2, RAG-rebuild-v2 plan, 2026-09-22) -- training on one task format and
evaluating on another is exactly what this project's second core principle forbids.

This is a SAMPLE/REVIEW pass, not the full integration: no eval-file merge, no
PG regeneration yet -- do that only after the sample's reviewed. Every dry-run
(--skip-llm) and review-gate pass writes to a `_review/` subfolder (core.review_path),
NEVER to the tracked production .jsonl files, and a real run refuses to overwrite an
existing production file without --overwrite (core.safe_write_jsonl) -- see this
repo's pipeline-notes.md for the 2026-09-22 incident that made this mandatory.

USAGE
-----
    python -m pipeline.track2.build_oow_scenarios_leo --n 25 --skip-llm   # free dry-run, writes to _review/
    python -m pipeline.track2.build_oow_scenarios_leo --n 25              # real LLM calls, writes to _review/
    python -m pipeline.track2.build_oow_scenarios_leo --n 7928 --overwrite  # full production run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from core import AgentPaths, load_env, review_path, safe_write_jsonl, CONTAM_THRESH, EMBEDDER_MODEL
from pipeline.oow_agent_spec import SYSTEM_OOW_AGENT, ACTIONS, validate_action_json, goal_course_check_line, goal_course_action

paths = AgentPaths.oow()
CACHE = paths.cache_dir
LEO_FILE = paths.workspace / "Data" / "OOW" / "OOW_Scenarios_Leo" / "moos_temporal_narratives_final.jsonl"
# One JSON line per fully-rendered record (gold_answer + wrong_answer already set),
# appended after EACH batch completes -- a crash/interrupt partway through a long run
# (e.g. --n 7928 is ~530 sequential batches, hours unattended) only loses the single
# in-flight batch, not everything already done. Re-running the same --n/--seed skips
# every leo_id already in here instead of re-spending API calls on it.
CHECKPOINT_FILE = CACHE / "oow_scenario_Leo_checkpoint.jsonl"

PASS_CRITERIA = {
    "stop": ["Own-ship stops given a genuinely close-range, still-closing, critical-risk "
             "contact that manoeuvre alone can no longer clear (Rule 17(b)).",
             "Own-ship does not simply hold course when a give-way contact is at critical risk."],
    "alter_course": ["Own-ship takes early, substantial action to keep clear as the give-way vessel, "
                     "scaled to how far CPA falls under the safe passing distance.",
                     "Own-ship does not simply hold course while she holds a give-way obligation.",
                     "Own-ship does not manoeuvre against a contact that poses no real CPA-based risk."],
    "stand_on": ["Own-ship holds course and speed while she is the stand-on vessel.",
                "Own-ship does not needlessly alter away from a stand-on obligation."],
    "stand_on_17b": ["Own-ship (stand-on) takes her own action once it is apparent the give-way "
                     "vessel isn't -- CPA under the safe distance AND the encounter is imminent."],
    "resume": ["Own-ship resumes cruise speed now that no contact requires give-way action.",
              "Own-ship does not needlessly remain at reduced speed with no active encounter."],
    "clear": ["Own-ship maintains course and speed; no contact currently requires action.",
             "Own-ship does not manoeuvre against a contact whose CPA is not below the safe distance."],
}

SAFE_CPA_M = 500.0        # matches Basic Simulator VesselConstraints' default min_cpa_m
CRITICAL_RANGE_M = 200.0  # "so close that collision cannot be avoided by the give-way
                          # vessel's action alone" (Rule 17(b)) -- a genuine GEOMETRY/range
                          # criterion for stop, never a TCPA cutoff.
STAND_ON_TCPA_S = 180.0   # Rule 17(a)(ii)/(b): stand-on may/must act once it becomes
                          # apparent the give-way vessel isn't -- "apparent" needs BOTH a
                          # real CPA shortfall AND the encounter being imminent (short
                          # TCPA), never TCPA alone.
MIN_TURN_DEG = 15.0       # Rule 16's "early and substantial" rules out a token gesture.
MAX_TURN_DEG = 30.0       # matches Basic Simulator VesselConstraints' max_rudder_angle_deg
                          # (a single turn command beyond this is silently capped there).
RESUME_SPEED_MARGIN = 0.5  # own speed must be at least this far under target before
                           # "speed_up" is worth issuing (avoids churn on noise-level gaps)


def render_leo_narrative(state: dict) -> str:
    """Deterministic house-style narrative from a Leo `state` dict -- no LLM, and
    deliberately NOT the source file's own ALL-CAPS-headers 'narrative' field. Matches
    Basic Simulator's narrate.py convention byte-for-byte where the two overlap (own-ship/
    contact phrasing, the shared goal_course_check_line()) -- see Fase B2 (RAG-rebuild-v2
    plan). Deliberately leaves out own_role/encounter_type/active_encounter_rules per
    contact, exactly like narrate.py leaves out encounter classification and applicable
    rules: those are precisely what the model is being asked to derive from the numbers
    (bearing/range/CPA/TCPA) itself, per this project's core "rule knowledge comes from
    data, never handed to the model in the prompt" principle -- handing over "We are
    meeting head-on" or "Rules engaged: Rule 15, Rule 16" would leak the answer instead of
    testing whether the model can derive it."""
    own, mission, contacts, cond = state["own_ship"], state["mission"], state["contacts"], state["conditions"]
    ox, oy = own["x"], own["y"]
    mx, my = mission["x"], mission["y"]
    dist = math.hypot(mx - ox, my - oy)
    bearing = math.degrees(math.atan2(mx - ox, my - oy)) % 360.0
    n = len(contacts)
    lines = [
        # own["name"] is always "LLM_SHIP" in the raw data, but the rendered text always
        # says the generic "Own-ship" (matching narrate.py) rather than that raw identifier.
        f"Own-ship is {own['movement_status']} at ({ox:.1f}, {oy:.1f}), heading {own['heading']:.1f} "
        f"degrees, speed {own['speed']:.2f}. Target cruise speed is {own['target_speed']:.1f}.",
        f"Own-ship vessel type is {own['colregs_vessel_type'].replace('_', ' ')}.",
        f"Mission waypoint is at ({mx:.1f}, {my:.1f}), {dist:.0f} m away, bearing {bearing:.1f} deg.",
        goal_course_check_line(ox, oy, own["heading"], mx, my),
        f"This mission's safe passing distance is {SAFE_CPA_M:.0f}m: CPA below that is a real "
        "collision risk, CPA well above it is safe regardless of how small it looks.",
        f"{n} other ship{'s' if n != 1 else ''}:" if n else "No other ships tracked.",
    ]
    for c in contacts:
        if c.get("cpa_distance_m") is not None and c.get("tcpa_s") is not None:
            cpa_txt = f", CPA {c['cpa_distance_m']:.0f} m, TCPA {c['tcpa_s']:.0f}s"
            if (c.get("closing_speed") or 0) <= 0:
                cpa_txt += " (already past closest point, ranges now increasing)"
        else:
            cpa_txt = " (ranges opening -- no future CPA/TCPA projected)"
        # Sensor-realism fact (data quality), NOT a rule/risk hint -- kept, unlike the
        # removed "Risk assessment: X"/own_role/encounter_type/rule-number fields above.
        track_txt = ""
        if c["risk"] == "uncertain":
            age_txt = f" (age {c['age_s']:.0f}s)" if c.get("age_s") is not None else ""
            track_txt = (f" Track on this contact is {c.get('track_quality', 'unreliable')}{age_txt} "
                        "-- treat its numbers with caution.")
        lines.append(
            f'  - Ship named "{c["name"]}" ({c["colregs_vessel_type"].replace("_", " ")}): range '
            f"{c['range_m']:.0f} m, rel.bearing {c['relative_bearing_deg']:.1f} deg, heading "
            f"{c['heading']:.1f}, speed {c['speed']:.2f}, closing speed {c['closing_speed']:.2f}"
            f"{cpa_txt}.{track_txt}"
        )
    vis = cond.get("visibility_condition", "clear")
    vis_range = cond.get("visibility_range_m")
    cond_line = f"Conditions: visibility is {vis}"
    if vis_range is not None:
        cond_line += f", assessed visibility range is {vis_range:.0f} m"
    cond_line += (", narrow-channel context is indicated" if cond.get("narrow_channel")
                 else ", narrow-channel context is not indicated")
    cond_line += (", traffic-separation context is indicated." if cond.get("traffic_separation_scheme")
                 else ", traffic-separation context is not indicated.")
    lines.append(cond_line)
    return "\n".join(lines)

# Primary rule cited per real-risk encounter type -- the ONE rule that actually drives
# the required action, not every standing/background rule (those stay in "rules" for
# metadata/PG tagging).
_PRIMARY_RULE_BY_ENCOUNTER = {"head_on": "Rule 14", "crossing": "Rule 15",
                             "we_are_overtaking_contact": "Rule 13"}


def _real_risk(c: dict) -> bool:
    """Rule 7 gate: real collision risk is determined EXCLUSIVELY from CPA against the
    safe passing distance (paired with Leo's own risk label as a corroborating check),
    NEVER from TCPA alone -- TCPA=0 can just as easily mean the closest point has already
    passed as mean an imminent collision."""
    cpa = c.get("cpa_distance_m")
    return cpa is not None and cpa < SAFE_CPA_M and c.get("risk") in ("medium", "high", "critical")


def _turn_degrees(cpa_m: float) -> float:
    """Substantial, geometry-scaled turn -- bigger the further CPA falls below the safe
    distance, always at least MIN_TURN_DEG, capped at the physical per-command limit
    (MAX_TURN_DEG). A fixed +30 regardless of shortfall was the original bug's fixed-
    degree half; this scales with how much clearance is actually missing."""
    shortfall = min(1.0, max(0.0, (SAFE_CPA_M - cpa_m) / SAFE_CPA_M))
    return round(MIN_TURN_DEG + shortfall * (MAX_TURN_DEG - MIN_TURN_DEG), 1)


def _diverging_turn(c: dict) -> str:
    """Turn direction when EITHER side is COLREG-permitted (Rule 13 overtaking): turn
    AWAY from whichever side the contact currently sits on, so the manoeuvre increases
    separation instead of cutting across the contact's bow."""
    return "turn_left" if c["relative_bearing_deg"] >= 0 else "turn_right"


def _should_stop(c: dict) -> bool:
    """Rule 17(b)'s own criterion for when manoeuvre alone no longer suffices: genuinely
    close range AND still closing AND risk already critical -- a geometry/range trigger,
    never a TCPA cutoff. This is deliberately rare (the original bug defaulted to "stop"
    on 875/7928 frames from a TCPA<1.5min OR risk=="critical" check alone)."""
    return (c.get("risk") == "critical" and c.get("range_m") is not None
            and c["range_m"] < CRITICAL_RANGE_M and (c.get("closing_speed") or 0) > 0)


def _rule_list(c: dict) -> list[str]:
    nums = sorted(set(c.get("active_encounter_rules") or []) | set(c.get("standing_rules") or []))
    return [f"Rule {n}" for n in nums]


def leo_choose_action(state: dict) -> dict:
    """Deterministic action from the record's OWN own_role/encounter/CPA/TCPA/risk labels
    (trusted, since Leo's own encounter/risk computation already accounts for track
    quality etc. this script does not re-derive). Rule 7 gates everything: a contact only
    drives the action if it poses REAL risk (_real_risk -- CPA vs the safe distance, never
    TCPA alone). No stop-default, no fixed +30-degree turn, no manoeuvre label for a
    stopped/paused own-ship -- see this repo's RAG-rebuild-v2 plan Fase B1 for the full
    diagnosis of what the previous role-only version got wrong (857 manoeuvres on
    low-risk-only contacts, 526 on opening-range contacts, 875 stop-labels, 883 paused
    frames mislabeled as manoeuvres).

    Returns simulator-format fields (action/degrees/rule_applied) directly, matching
    Basic Simulator/app/agents.py's SYSTEM_OOW_AGENT JSON contract -- see Fase B2."""
    own = state["own_ship"]
    contacts = state["contacts"]

    # Never a manoeuvre label for a ship that isn't moving -- there is no manoeuvre to take.
    if own.get("paused") or own.get("stopped"):
        return {"action": None, "degrees": None, "rule_applied": None, "role": "paused",
                "rules": [], "category": "leo_paused", "bucket": "paused"}

    give_way = [c for c in contacts if c["own_role"] in ("give_way", "both_give_way") and _real_risk(c)]
    stand_on = [c for c in contacts if c["own_role"] == "stand_on" and _real_risk(c)]

    if give_way:
        worst = min(give_way, key=lambda c: c["cpa_distance_m"])
        if _should_stop(worst):
            return {"action": "stop", "degrees": None, "rule_applied": "Rule 17",
                    "role": worst["own_role"], "rules": _rule_list(worst),
                    "category": f"leo_{worst['encounter_type']}", "bucket": "stop"}
        degrees = _turn_degrees(worst["cpa_distance_m"])
        if worst["encounter_type"] in ("head_on", "crossing"):
            action = "turn_right"  # Rule 14/15+16: give-way alters to starboard
        else:
            action = _diverging_turn(worst)  # Rule 13: either side permitted
        rule_applied = _PRIMARY_RULE_BY_ENCOUNTER.get(worst["encounter_type"], "none")
        return {"action": action, "degrees": degrees, "rule_applied": rule_applied,
                "role": worst["own_role"], "rules": _rule_list(worst),
                "category": f"leo_{worst['encounter_type']}", "bucket": "alter_course"}

    if stand_on:
        # Rule 17(a)(ii)/(b): own-ship (stand-on) may/must act once it's apparent the
        # give-way vessel isn't -- requires BOTH a real CPA shortfall AND an imminent
        # encounter (short TCPA), never TCPA alone.
        triggered = [c for c in stand_on if (c.get("tcpa_s") or 1e9) < STAND_ON_TCPA_S]
        if triggered:
            worst = min(triggered, key=lambda c: c["cpa_distance_m"])
            degrees = _turn_degrees(worst["cpa_distance_m"])
            return {"action": _diverging_turn(worst), "degrees": degrees, "rule_applied": "Rule 17",
                    "role": "stand_on", "rules": _rule_list(worst),
                    "category": f"leo_stand_on_{worst['encounter_type']}_17b", "bucket": "stand_on_17b"}
        ref = stand_on[0]
        return {"action": "hold_course", "degrees": None, "rule_applied": "Rule 17",
                "role": "stand_on", "rules": _rule_list(ref),
                "category": f"leo_stand_on_{ref['encounter_type']}", "bucket": "stand_on"}

    # No contact poses real risk -- follow GOAL COURSE CHECK exactly (SYSTEM_OOW_AGENT's
    # decision procedure step 3/5), never a fixed hold_course default: only standing rules
    # (2/5/6/7/11) apply, never a specific action-driving rule.
    rule_nums = sorted({n for c in contacts for n in (c.get("standing_rules") or [])}) or [2, 5, 6, 7]
    rules = [f"Rule {n}" for n in rule_nums]
    mission = state["mission"]
    goal_action, goal_degrees = goal_course_action(own["x"], own["y"], own["heading"], mission["x"], mission["y"])
    if goal_action != "hold_course":
        return {"action": goal_action, "degrees": goal_degrees, "rule_applied": "none",
                "role": "cleared", "rules": rules, "category": "leo_clear_goal_turn", "bucket": "clear"}
    if own["speed"] < own["target_speed"] - RESUME_SPEED_MARGIN:
        return {"action": "speed_up", "degrees": None, "rule_applied": "none",
                "role": "cleared", "rules": rules, "category": "leo_clear_resume", "bucket": "resume"}
    return {"action": "hold_course", "degrees": None, "rule_applied": "none",
            "role": "cleared", "rules": rules, "category": "leo_clear_maintain", "bucket": "clear"}



FIXED_QUESTION = "Recommend exactly ONE manoeuvre as the specified JSON object."


def build_user_message(situation_report: str) -> str:
    """The USER turn -- situation text only, matching Basic Simulator/app/agents.py's
    build_oow_prompt() convention exactly (Situation: ... + the same fixed closing
    instruction) so train and eval share one task format."""
    return f"Situation:\n{situation_report}\n\n{FIXED_QUESTION}"


def build_assistant_json(decision: dict, reasoning: str) -> dict:
    """The ASSISTANT turn -- the schema-validated JSON object itself (never free prose),
    reasoning supplied separately (Fase B3's [LLM] step derives it from bearing/CPA/rule,
    never hands the rule to the model to just restate)."""
    return {"action": decision["action"], "degrees": decision["degrees"],
            "rule_applied": decision["rule_applied"], "reasoning": reasoning}


def wrong_action_variant(decision: dict) -> dict:
    """A plausible but COLREG-INCORRECT alternative decision, for DPO 'rejected' answers
    -- deterministic, same principle as build_oow_scenarios.py's own wrong_action_variant()."""
    action, degrees = decision["action"], decision["degrees"]
    if action in ("turn_left", "turn_right"):
        wrong_action = "turn_left" if action == "turn_right" else "turn_right"
        return {"action": wrong_action, "degrees": degrees, "rule_applied": decision["rule_applied"]}
    if action == "hold_course":
        # Wrong: manoeuvring when no real risk exists.
        return {"action": "turn_right", "degrees": MIN_TURN_DEG, "rule_applied": "none"}
    # Wrong response to a stop/speed-up-worthy situation: holding course instead.
    return {"action": "hold_course", "degrees": None, "rule_applied": "none"}


def write_outputs(outputs: dict[str, list[dict]], cache_dir: Path, overwrite: bool) -> None:
    """Writes every {filename: rows} pair in `outputs` under `cache_dir` -- WITHOUT
    --overwrite, every file goes to core.review_path(cache_dir/filename) instead of the
    real production path, and safe_write_jsonl() additionally refuses to clobber an
    existing file even at the production path. `cache_dir` is an explicit parameter (not
    the module-level CACHE constant) so this is unit-testable against a tmp_path without
    ever touching real Data/ files."""
    for name, rows in outputs.items():
        prod_path = cache_dir / name
        out_path = prod_path if overwrite else review_path(prod_path)
        safe_write_jsonl(rows, out_path, overwrite=overwrite)
        print(f"Wrote {out_path} ({len(rows)} rows)"
             + ("" if overwrite else "  [dry-run/review path -- pass --overwrite for production]"))


def stratified_sample(recs: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Round-robins across the action-outcome buckets so even a small sample covers
    stop/alter_course/stand_on/stand_on_17b/resume/clear -- pure random risks missing rare
    buckets (e.g. 'stop' is a small minority of records) in a first-look sample. "paused"
    frames (own-ship not underway) are excluded entirely -- there is no manoeuvre decision
    to make for a stopped ship, so they never become a training instance."""
    rnd = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for r in recs:
        decision = leo_choose_action(r["state"])
        if decision["bucket"] == "paused":
            continue
        buckets.setdefault(decision["bucket"], []).append(r)
    for b in buckets.values():
        rnd.shuffle(b)
    out: list[dict] = []
    bucket_names = list(buckets)
    i = 0
    while len(out) < n and any(buckets.values()):
        b = bucket_names[i % len(bucket_names)]
        if buckets[b]:
            out.append(buckets[b].pop())
        i += 1
    return out


def load_checkpoint() -> dict[str, dict]:
    """leo_id -> fully-rendered record, from every batch a prior run already completed."""
    if not CHECKPOINT_FILE.exists():
        return {}
    out: dict[str, dict] = {}
    for line in CHECKPOINT_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["leo_id"]] = rec
    return out


def append_checkpoint(recs: list[dict]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


def filter_contamination_batch(batch: list[dict], gold_embs, model) -> list[dict]:
    """Per-batch version of build_oow_scenarios.filter_contamination() -- takes a
    PRE-LOADED embedder + pre-computed gold_embs instead of reloading the
    SentenceTransformer on every call, since this now runs once per ~15-record batch
    (~530 times for a full --n 7928 run) instead of once for the whole sample."""
    if not batch:
        return batch
    texts = [r.get("reasoning", "") for r in batch]
    q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    kept, dropped = [], 0
    for rec, emb in zip(batch, q_embs):
        c_sim = float(np.max(gold_embs @ emb)) if len(gold_embs) else 0.0
        if c_sim >= CONTAM_THRESH:
            dropped += 1
            continue
        kept.append(rec)
    if dropped:
        print(f"    contamination filter: kept={len(kept)}  dropped={dropped} (thresh={CONTAM_THRESH})")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n", type=int, default=25, help="sample size (stratified across action buckets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default="claude-sonnet-4-5")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--gold-file", type=str, default=str(paths.eval_dir / "colreg_qa_500_normalised.json"))
    ap.add_argument("--skip-llm", action="store_true",
                    help="geometry+narrative+action only -- no API calls, no reasoning text, for free dry-run testing")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting existing production oow_scenario_Leo_*.jsonl files -- "
                         "without this, a run whose output already exists on disk refuses to write "
                         "(see core.safe_write_jsonl); dry-run/review passes never need this, they "
                         "always write to _review/ instead")
    args = ap.parse_args()

    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(all_recs)} Leo states; sampling {args.n} (stratified by action bucket)...")
    sample = stratified_sample(all_recs, args.n, seed=args.seed)

    recs: list[dict] = []
    for i, r in enumerate(sample):
        decision = leo_choose_action(r["state"])
        recs.append({
            "_id": f"leo{i:05d}", "leo_id": r["id"], "leo_source_file": r["source_file"],
            "category": decision["category"], "action": decision["action"],
            "degrees": decision["degrees"], "rule_applied": decision["rule_applied"],
            "role": decision["role"], "rules": decision["rules"],
            "pass_criteria": PASS_CRITERIA[decision["bucket"]],
            "situation_report": render_leo_narrative(r["state"]),
            "reasoning": None, "wrong_decision": None, "wrong_reasoning": None,
        })

    # Fase B1 (leo_choose_action rewrite) + Fase B2 (unified task format -- system prompt/
    # user framing/JSON response schema, all from pipeline/oow_agent_spec.py) have landed.
    # Fase B3 (the actual [LLM] reasoning-text generation, gated behind the 25-first review
    # per this repo's RAG-rebuild-v2 plan) has NOT started yet -- every rec's "reasoning"
    # stays None below, so the row-building loop naturally emits zero SFT/DPO/reflection
    # rows until that lands. --skip-llm and a real run both currently do the same thing
    # (no API calls at all); the --skip-llm flag is kept for interface compatibility with
    # the eventual B3 implementation, which will only make real Anthropic calls when it
    # is NOT set.
    if not args.skip_llm:
        raise NotImplementedError(
            "Fase B3 (the [LLM] reasoning-text generation step) has not been implemented "
            "yet -- it requires the 25-first review gate per the RAG-rebuild-v2 plan, "
            "which has not been approved. Use --skip-llm for now (geometry/narrative/"
            "action/degrees/rule_applied only, no reasoning text)."
        )
    print("--skip-llm: situation_report/action/degrees/rule_applied are set; "
         "reasoning left None pending Fase B3.")
    final_recs = recs

    sft_rows, dpo_rows, reflect_rows, trace_rows = [], [], [], []
    for r in final_recs:
        if not r.get("reasoning"):
            continue
        user_msg = build_user_message(r["situation_report"])
        decision = {"action": r["action"], "degrees": r["degrees"], "rule_applied": r["rule_applied"]}
        assistant_json = build_assistant_json(decision, r["reasoning"])
        sft_rows.append({"category": r["category"], "action": r["action"], "leo_id": r["leo_id"], "messages": [
            {"role": "system", "content": SYSTEM_OOW_AGENT}, {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
        ]})
        if r.get("wrong_decision") and r.get("wrong_reasoning"):
            rejected_json = build_assistant_json(r["wrong_decision"], r["wrong_reasoning"])
            dpo_rows.append({
                "category": r["category"], "action": r["action"], "leo_id": r["leo_id"],
                "prompt": [{"role": "system", "content": SYSTEM_OOW_AGENT}, {"role": "user", "content": user_msg}],
                "chosen": [{"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}],
                "rejected": [{"role": "assistant", "content": json.dumps(rejected_json, ensure_ascii=False)}],
            })
        trace_rows.append({
            "document_id": f"oow_scenario_leo_{r['leo_id']}", "chunk_id": f"oow_scenario_leo_{r['leo_id']}",
            "source_file": f"oow_scenario_generator_leo::{r['leo_source_file']}",
            "chapter_title": f"OOW Track 2 (Leo) scenario: {r['category']}",
            "chunk_concepts": [r["category"], r["role"]] + r["rules"],
            "trace": {
                "situation": r["situation_report"], "trigger": None,
                "procedures": [
                    {"step": 1, "action": "Assess the fused contact picture (bearing/range/CPA/TCPA) against "
                                          "own-ship's course and speed.",
                     "why": "Rule 7 requires determining if risk of collision exists using all available means."},
                    {"step": 2, "action": f"Determine own-ship's role: {r['role'].replace('_', ' ')}.",
                     "why": f"{', '.join(r['rules']) or 'no encounter rule'} governs this encounter."},
                    {"step": 3, "action": "Execute the action: " + r["action"] +
                                          (f" ({r['degrees']:.0f} deg)" if r["degrees"] is not None else "") + ".",
                     "why": "The chosen action must comply with the applicable COLREG rule(s) above."},
                ],
                "constraints": r["pass_criteria"], "prowords_used": [r["role"]], "channels": r["rules"],
                "regulations": ["COLREG 1972"], "warnings": [],
                "outcomes": [f"Action taken: {r['action']}."],
                "key_facts": [f"Chosen action: {r['action']}."],
                "question_seeds": [{"angle": "what", "text": FIXED_QUESTION}],
            },
        })

    outputs = {
        "oow_scenario_Leo_sft_direct.jsonl": sft_rows,
        "oow_scenario_Leo_sft_cot.jsonl": sft_rows,
        "oow_scenario_Leo_dpo_pairs.jsonl": dpo_rows,
        "oow_scenario_Leo_reflection.jsonl": reflect_rows,
        "oow_scenario_Leo_reasoning_traces.jsonl": trace_rows,
    }
    write_outputs(outputs, CACHE, overwrite=args.overwrite)

    from collections import Counter
    print("category distribution:", Counter(r["category"] for r in final_recs))


if __name__ == "__main__":
    main()
