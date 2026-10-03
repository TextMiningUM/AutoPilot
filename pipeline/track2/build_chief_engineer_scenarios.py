"""Chief Engineer, Track 2 (design_chief_engineer.md Sec 3.8/4.2) — conversational
malfunction-monitoring dialogues, deterministically derived from the SAME condition-
evaluation functions the live Engine Room dashboard uses
(pipeline/chief_engineer_agent_spec.py), so "what should have been said" is always
consistent with what the dashboard itself would compute -- never an independently
invented storyline (project convention, see that module's own docstring).

Mirrors VHF's Track 2 (vhf_conversations.jsonl): one row per scenario, a ready-to-train
`messages` list (system/user/assistant) that build_sft.py's SFT_DATASETS can include
DIRECTLY -- no extract_conversation_reasoning.py-equivalent mining step needed, since
every turn is already deterministic fluent prose, not a free-text transcript that needs
an LLM to extract structure from.

One scenario = one `generate_synthetic_incident()` call (component x seed x
caught_in_time). The watchkeeping engineer reports the LATEST reading at its own watch
round; the Chief Engineer's reply is `recommend_maintenance()`'s action rendered as
fluent prose (never a "Label: value." dump, same hard rule as build_sft.py's
format_direct_answer()). `caught_in_time=True` rows show a timely, in-good-order
response at first warning/critical severity; `caught_in_time=False` rows show the
emergency response at the final (uncaught) severity the trace reached -- both are
CORRECT Chief Engineer responses to what's actually being reported at that moment, the
difference is only how late the watch round happened to ask.

Contamination filter against the real held-out gold Q&A (chiefengineer_gold_answers.json,
cos >= CONTAM_THRESH dropped) -- mandatory for every new training-data builder, same as
every other Track-1/Track-2 generator in this project.

Output: <cache>/chiefengineer_conversations.jsonl

Safe to run LOCALLY (pure stdlib condition-model + a small CPU sentence-embedder for the
contamination filter, no GPU/API key needed).

Run with: python -m pipeline.track2.build_chief_engineer_scenarios
          python -m pipeline.track2.build_chief_engineer_scenarios --seeds-per-limit 5   # smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths, CONTAM_THRESH, EMBEDDER_MODEL
from pipeline.chief_engineer_agent_spec import (
    all_known_limits, generate_synthetic_incident, recommend_maintenance,
)

paths = AgentPaths.chief_engineer()
CACHE = paths.cache_dir
GOLD_FILE = paths.gold_file
OUT_FILE = CACHE / "chiefengineer_conversations.jsonl"

SEEDS_PER_LIMIT_DEFAULT = 40

SYSTEM_PROMPT = (
    "You are the Chief Engineer, an AI engine-room agent responsible for condition "
    "monitoring and malfunction diagnosis aboard a MAN B&W-class main propulsion engine. "
    "Respond to the watchkeeping engineer's reports accurately, cite the affected "
    "system/component, and give a clear, prioritised instruction grounded in the "
    "component's known operating limits."
)

# Human-readable display names -- component_id/parameter are code-level snake_case slugs
# (chief_engineer_agent_spec.py's own KNOWN_LIMITS keys), never shown verbatim in prose.
_COMPONENT_DISPLAY = {
    "fuel_injection": "the fuel injection system",
    "main_bearing": "the main bearing",
    "cylinder_unit": "the cylinder unit",
    "turbocharger": "the turbocharger",
    "lubricating_oil": "the lubricating oil",
    "cooling_water": "the cooling water system",
}
_PARAMETER_DISPLAY = {
    "viscosity_cst": "viscosity",
    "temperature_c": "temperature",
    "exhaust_temp_spread_c": "exhaust temperature spread",
    "exhaust_gas_temp_c": "exhaust gas temperature",
    "tbn": "TBN (total base number)",
    "iron_content_ppm": "iron content",
}

_SEVERITY_PHRASE = {
    "nominal": "within normal limits",
    "watch": "starting to drift, worth keeping an eye on",
    "warning": "past the warning threshold",
    "critical": "past the critical threshold",
}


def _trend_phrase(readings: list, index: int) -> str:
    """A short clause describing how the value has moved since the previous reading, or
    "" for the very first reading in a trace (nothing to compare against yet)."""
    if index == 0:
        return ""
    prev, cur = readings[index - 1].value, readings[index].value
    if abs(cur - prev) < 1e-6:
        return ", holding steady since the last round"
    return ", up from the last round" if cur > prev else ", down from the last round"


def render_watch_report(incident, index: int) -> str:
    """The watchkeeping engineer's user-turn report of ONE reading, in fluent prose."""
    r = incident.readings[index]
    comp = _COMPONENT_DISPLAY.get(incident.component_id, incident.component_id)
    param = _PARAMETER_DISPLAY.get(incident.parameter, incident.parameter)
    trend = _trend_phrase(incident.readings, index)
    hours = r.timestamp_s / 3600.0
    return (
        f"Chief, {param} on {comp} is now reading {r.value:.1f} {r.unit} at the "
        f"{hours:.1f}h watch round{trend}."
    )


def render_chief_response(incident, index: int) -> str:
    """The Chief Engineer's assistant-turn reply to ONE verdict, in fluent prose built
    from recommend_maintenance() -- never a hand-written second version of that logic."""
    v = incident.verdicts[index]
    comp = _COMPONENT_DISPLAY.get(incident.component_id, incident.component_id)
    rec = recommend_maintenance(v)
    severity_phrase = _SEVERITY_PHRASE[v.severity]
    if rec is None:
        return f"That reading on {comp} is {severity_phrase}. Keep logging it at the usual interval."
    return f"{comp.capitalize()} is {severity_phrase}. {rec.action} ({rec.source_citation})"


def build_scenario(limit, seed: int, caught_in_time: bool) -> dict:
    """One full training row: a seeded, deterministic SyntheticIncident rendered as a
    2-turn (watch-report, Chief Engineer reply) conversation anchored at the incident's
    OWN final reading -- the point of timely intervention (caught_in_time=True) or the
    point the trend reached its worst uncaught severity (caught_in_time=False)."""
    incident = generate_synthetic_incident(limit, seed=seed, caught_in_time=caught_in_time)
    last = len(incident.readings) - 1
    user_msg = render_watch_report(incident, last)
    assistant_msg = render_chief_response(incident, last)
    return {
        "id": f"ceconv_{incident.incident_id}",
        "component_id": incident.component_id,
        "parameter": incident.parameter,
        "severity": incident.verdicts[last].severity,
        "caught_in_time": caught_in_time,
        "narrative": incident.narrative,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
    }


def build_all_scenarios(seeds_per_limit: int) -> list[dict]:
    rows = []
    for limit in all_known_limits():
        for seed in range(seeds_per_limit):
            for caught_in_time in (True, False):
                rows.append(build_scenario(limit, seed, caught_in_time))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds-per-limit", type=int, default=SEEDS_PER_LIMIT_DEFAULT,
                    help="how many distinct seeds to generate per (component, caught_in_time) pair")
    ap.add_argument("--out-file", type=str, default=str(OUT_FILE),
                    help="output jsonl path (default: chiefengineer_conversations.jsonl)")
    args = ap.parse_args()

    rows = build_all_scenarios(args.seeds_per_limit)
    print(f"Raw scenarios: {len(rows)}")

    print("Loading embedder + gold-Q embeddings for contamination filter...")
    model = SentenceTransformer(EMBEDDER_MODEL)
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    gold_embs = model.encode([g["question"] for g in gold], normalize_embeddings=True,
                             batch_size=64, show_progress_bar=False)

    user_texts = [r["messages"][1]["content"] for r in rows]
    q_embs = model.encode(user_texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    kept, dropped = [], 0
    for r, emb in zip(rows, q_embs):
        sim = float(np.max(gold_embs @ emb))
        if sim >= CONTAM_THRESH:
            dropped += 1
            continue
        r["contam_sim"] = round(sim, 3)
        kept.append(r)
    print(f"kept={len(kept)} contam_dropped={dropped}")

    out_file = Path(args.out_file)
    with out_file.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} rows -> {out_file}")


if __name__ == "__main__":
    main()
