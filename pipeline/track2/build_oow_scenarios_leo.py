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
don't fit.

This is a SAMPLE/REVIEW pass, not the full integration: no eval-file merge, no
PG regeneration yet -- do that only after the sample's reviewed.

USAGE
-----
    python -m pipeline.track2.build_oow_scenarios_leo --n 25 --skip-llm   # free dry-run
    python -m pipeline.track2.build_oow_scenarios_leo --n 25              # real LLM calls
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

from core import AgentPaths, load_env, CONTAM_THRESH
from pipeline.track2.build_oow_scenarios import (
    SYSTEM_OOW, FIXED_QUESTION, render_all, render_wrong_all, filter_contamination,
)

paths = AgentPaths.oow()
CACHE = paths.cache_dir
LEO_FILE = paths.workspace / "Data" / "OOW" / "OOW_Scenarios_Leo" / "moos_temporal_narratives_final.jsonl"

ROLE_PHRASE = {
    "give_way": "we are the give-way vessel",
    "stand_on": "we are the stand-on vessel",
    "both_give_way": "both vessels are give-way",
    "not_applicable": "no active encounter with this contact",
}

PASS_CRITERIA = {
    "stop": ["Own-ship stops or takes way off given the critical/imminent collision risk.",
             "Own-ship does not simply hold course when a give-way contact is at critical risk."],
    "alter_course": ["Own-ship takes early, substantial action to keep clear as the give-way vessel.",
                     "Own-ship does not simply hold course while she holds a give-way obligation."],
    "stand_on": ["Own-ship holds course and speed while she is the stand-on vessel.",
                "Own-ship does not needlessly alter away from a stand-on obligation."],
    "resume": ["Own-ship resumes cruise speed now that no contact requires give-way action.",
              "Own-ship does not needlessly remain at reduced speed with no active encounter."],
    "clear": ["Own-ship maintains course and speed; no contact currently requires action."],
}

# Full natural-sentence phrasing per (encounter_type, own_role) combination, matching
# build_oow_scenarios.py's role_sentence() house style -- replaces the earlier telegraphic
# "Encounter: X; our role: Y" label-dump. Covers all 9 combinations actually present in
# the Leo dataset; the fallback below only applies to a combination never observed there.
LEO_ROLE_SENTENCES = {
    ("head_on", "both_give_way"): "We are meeting head-on. Both vessels are give-way.",
    ("crossing", "give_way"): "She is crossing our path. We are the give-way vessel.",
    ("crossing", "stand_on"): "She is crossing our path. We are the stand-on vessel.",
    ("crossing", "not_applicable"): "She is on a crossing course, but no active give-way encounter applies yet.",
    ("we_are_overtaking_contact", "give_way"): "We are overtaking her. We are the give-way vessel.",
    ("contact_overtaking_us", "stand_on"): "She is overtaking us. We are the stand-on vessel.",
    ("parallel", "not_applicable"): "She is running a broadly parallel course; no active encounter applies.",
    ("diverging", "not_applicable"): "She is diverging from us and the range is opening; no active encounter applies.",
    ("stationary_contact", "not_applicable"): "She is stationary or barely making way; no active encounter applies.",
}


def leo_role_sentence(encounter_type: str, own_role: str) -> str:
    sentence = LEO_ROLE_SENTENCES.get((encounter_type, own_role))
    if sentence:
        return sentence
    # Fallback for any combination not seen in the dataset today.
    return f"Encounter: {encounter_type.replace('_', ' ')}; {ROLE_PHRASE.get(own_role, own_role)}."


def render_leo_narrative(state: dict) -> str:
    """Deterministic house-style narrative from a Leo `state` dict -- no LLM, and
    deliberately NOT the source file's own ALL-CAPS-headers 'narrative' field. Own-ship/
    contact phrasing matches the Basic Simulator's narrate.py convention (proven to work
    well there) and build_oow_scenarios.py, so all three stay aligned."""
    own, mission, contacts, cond = state["own_ship"], state["mission"], state["contacts"], state["conditions"]
    ox, oy = own["x"], own["y"]
    mx, my = mission["x"], mission["y"]
    dist = math.hypot(mx - ox, my - oy)
    bearing = math.degrees(math.atan2(mx - ox, my - oy)) % 360.0
    n = len(contacts)
    lines = [
        # own["name"] is always "LLM_SHIP" in the raw data, but the rendered text always
        # says the generic "Own-ship" (matching narrate.py/build_oow_scenarios.py) rather
        # than that raw identifier.
        f"Own-ship is {own['movement_status']} at ({ox:.1f}, {oy:.1f}), heading {own['heading']:.1f} "
        f"degrees, speed {own['speed']:.2f}. Target cruise speed is {own['target_speed']:.1f}.",
        f"Own-ship vessel type is {own['colregs_vessel_type'].replace('_', ' ')}.",
        f"Mission waypoint is at ({mx:.1f}, {my:.1f}), {dist:.0f} m away, bearing {bearing:.1f} deg.",
        f"{n} other ship{'s' if n != 1 else ''}:" if n else "No other ships tracked.",
    ]
    for c in contacts:
        rule_nums = sorted(set(c.get("standing_rules") or []) | set(c.get("active_encounter_rules") or []))
        rule_txt = ", ".join(f"Rule {n}" for n in rule_nums) if rule_nums else "none currently engaged"
        cpa_txt = ""
        if c.get("cpa_distance_m") is not None and c.get("tcpa_s") is not None:
            cpa_txt = f" Projected CPA: {c['cpa_distance_m']:.0f} m in {c['tcpa_s'] / 60:.1f} min."
        # Leo's own 'risk' field conflates two different axes: true severity
        # (low/medium/high/critical) and track CONFIDENCE ("uncertain", which turns out to
        # correlate ~100% with track_quality being unreliable/stale, not with CPA/TCPA
        # severity at all). Render them as two separate statements so a model never has to
        # guess which axis "uncertain" belongs to.
        if c["risk"] == "uncertain":
            age_txt = f" (age {c['age_s']:.0f}s)" if c.get("age_s") is not None else ""
            risk_txt = (f"Track on this contact is {c.get('track_quality', 'unreliable')}{age_txt} -- risk "
                       f"cannot be reliably assessed from the current fused picture.")
        else:
            risk_txt = f"Risk assessment: {c['risk']}."
        lines.append(
            f'  - Ship named "{c["name"]}" ({c["colregs_vessel_type"].replace("_", " ")}): range {c["range_m"]:.0f} m, '
            f"{c['sector']} (relative bearing {c['relative_bearing_deg']:.1f} deg). Heading {c['heading']:.1f}, "
            f"speed {c['speed']:.2f}, closing speed {c['closing_speed']:.2f}.{cpa_txt} "
            f"{risk_txt} {leo_role_sentence(c['encounter_type'], c['own_role'])} Rules engaged: {rule_txt}."
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
    lines.append("Allowed actions this cycle: maintain_course, alter_course, set_speed, stop, resume_cruising_speed.")
    # "set_speed (up to 0.0)" reads as a nonsensical range when the ship is deliberately
    # paused/stopped with no current speed target -- state it in words instead of a
    # degenerate 0.0 upper bound in that case.
    speed_clause = (f"set_speed (up to {own['target_speed']:.1f})" if own["target_speed"] > 0
                    else "set_speed (no active speed target while stopped)")
    lines.append("Candidate action parameters: maintain_course; alter_course (degrees between -30 and 30, "
                 f"lookahead distance 200 m); {speed_clause}; stop; resume_cruising_speed.")
    return "\n".join(lines)


def leo_choose_action(state: dict) -> dict:
    """Deterministic action from the record's OWN own_role/risk labels (trusted, since
    Leo's own encounter/risk computation already accounts for track quality etc. this
    script does not re-derive) -- mirrors build_oow_scenarios.py's choose_action() spirit
    but grounded in Leo's richer, variable-count contact picture instead of one synthetic
    contact."""
    own = state["own_ship"]
    contacts = state["contacts"]
    give_way = [c for c in contacts if c["own_role"] in ("give_way", "both_give_way")]
    stand_on = [c for c in contacts if c["own_role"] == "stand_on"]

    if give_way:
        worst = min(give_way, key=lambda c: c.get("tcpa_s") if c.get("tcpa_s") is not None else 1e9)
        tcpa_min = (worst.get("tcpa_s") or 1e9) / 60.0
        critical = any(c["risk"] == "critical" for c in give_way)
        if critical or tcpa_min < 1.5:
            action, params, params_text, bucket = "stop", {}, "stop", "stop"
        else:
            action, params, params_text, bucket = (
                "alter_course", {"degrees": 30, "lookahead_distance_m": 200},
                "alter_course (+30 degrees to starboard, lookahead distance 200 m)", "alter_course")
        rule_nums = sorted(set(worst.get("active_encounter_rules") or []) | set(worst.get("standing_rules") or []))
        return {"action": action, "action_params": params, "action_params_text": params_text,
                "role": worst["own_role"], "rules": [f"Rule {n}" for n in rule_nums],
                "category": f"leo_{worst['encounter_type']}", "bucket": bucket}

    if stand_on:
        ref = stand_on[0]
        rule_nums = sorted(set(ref.get("active_encounter_rules") or []) | set(ref.get("standing_rules") or []))
        return {"action": "maintain_course", "action_params": {}, "action_params_text": "maintain_course",
                "role": "stand_on", "rules": [f"Rule {n}" for n in rule_nums],
                "category": f"leo_stand_on_{ref['encounter_type']}", "bucket": "stand_on"}

    rule_nums = sorted({n for c in contacts for n in (c.get("standing_rules") or [])}) or [2, 5, 6, 7]
    rules = [f"Rule {n}" for n in rule_nums]
    if own["speed"] < own["target_speed"] - 0.5:
        return {"action": "resume_cruising_speed", "action_params": {},
                "action_params_text": "resume_cruising_speed (return to cruising speed)",
                "role": "cleared", "rules": rules, "category": "leo_clear_resume", "bucket": "resume"}
    return {"action": "maintain_course", "action_params": {}, "action_params_text": "maintain_course",
            "role": "cleared", "rules": rules, "category": "leo_clear_maintain", "bucket": "clear"}


def stratified_sample(recs: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Round-robins across the 5 action-outcome buckets so even a small sample covers
    stop/alter_course/stand_on/resume/clear -- pure random risks missing rare buckets
    (e.g. 'stop' is a small minority of records) in a first-look sample."""
    rnd = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for r in recs:
        decision = leo_choose_action(r["state"])
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


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n", type=int, default=25, help="sample size (stratified across action buckets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default="claude-sonnet-4-5")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--gold-file", type=str, default=str(paths.eval_dir / "colreg_qa_500_normalised.json"))
    ap.add_argument("--skip-llm", action="store_true",
                    help="geometry+narrative only -- no API calls, no gold_answer, for free dry-run testing")
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
            "action_params": decision["action_params"], "action_params_text": decision["action_params_text"],
            "role": decision["role"], "rules": decision["rules"],
            "pass_criteria": PASS_CRITERIA[decision["bucket"]],
            "situation_report": render_leo_narrative(r["state"]),
        })

    if args.skip_llm:
        print("--skip-llm: situation_report/action are set; gold_answer left empty.")
    else:
        load_env(paths.env_file)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set in .env")
        import anthropic
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        client = anthropic.Anthropic(api_key=key, default_headers=headers)
        render_all(client, args.model, recs, args.batch_size, args.max_tokens, "leo")

        gold_questions = [g["question"] for g in json.loads(Path(args.gold_file).read_text(encoding="utf-8"))]
        recs = filter_contamination(recs, gold_questions)

        render_wrong_all(client, args.model, recs, args.batch_size, args.max_tokens)

    sft_rows, dpo_rows, reflect_rows, trace_rows = [], [], [], []
    for r in recs:
        if not r.get("gold_answer"):
            continue
        user_msg = f"{r['situation_report']}\n\n{FIXED_QUESTION}"
        sft_rows.append({"category": r["category"], "action": r["action"], "leo_id": r["leo_id"], "messages": [
            {"role": "system", "content": SYSTEM_OOW}, {"role": "user", "content": user_msg},
            {"role": "assistant", "content": r["gold_answer"]},
        ]})
        if r.get("wrong_answer"):
            dpo_rows.append({
                "category": r["category"], "action": r["action"], "leo_id": r["leo_id"],
                "prompt": [{"role": "system", "content": SYSTEM_OOW}, {"role": "user", "content": user_msg}],
                "chosen": [{"role": "assistant", "content": r["gold_answer"]}],
                "rejected": [{"role": "assistant", "content": r["wrong_answer"]}],
            })
        draft = f"I will {r['action'].replace('_', ' ')}."
        critique = ("This response is too vague -- it must state the exact action parameters "
                   "and cite the specific COLREG rule(s) that justify the decision.")
        reflect_rows.append({"category": r["category"], "leo_id": r["leo_id"], "messages": [
            {"role": "system", "content": SYSTEM_OOW}, {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"Draft: {draft}\n\nCritique: {critique}\n\nRefined: {r['gold_answer']}"},
        ]})
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
                     "why": f"{', '.join(r['rules'])} governs this encounter."},
                    {"step": 3, "action": f"Execute the action: {r['action_params_text']}.",
                     "why": "The chosen action must comply with the applicable COLREG rule(s) above."},
                ],
                "constraints": r["pass_criteria"], "prowords_used": [r["role"]], "channels": r["rules"],
                "regulations": ["COLREG 1972"], "warnings": [],
                "outcomes": [f"Action taken: {r['action_params_text']}."],
                "key_facts": [f"Chosen action: {r['action']}."],
                "question_seeds": [{"angle": "what", "text": FIXED_QUESTION}],
            },
        })

    write_jsonl(sft_rows, CACHE / "oow_scenario_Leo_sft_direct.jsonl")
    write_jsonl(sft_rows, CACHE / "oow_scenario_Leo_sft_cot.jsonl")
    write_jsonl(dpo_rows, CACHE / "oow_scenario_Leo_dpo_pairs.jsonl")
    write_jsonl(reflect_rows, CACHE / "oow_scenario_Leo_reflection.jsonl")
    write_jsonl(trace_rows, CACHE / "oow_scenario_Leo_reasoning_traces.jsonl")

    from collections import Counter
    print("category distribution:", Counter(r["category"] for r in recs))


if __name__ == "__main__":
    main()
