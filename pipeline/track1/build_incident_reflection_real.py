"""
================================================================================
build_incident_reflection_real.py -- Fase C3 "C1 pairs" (RAG-rebuild-v2 plan)
================================================================================

Track 1 counterpart to build_measurement_reflection.py: reflection triples grounded in
Fase C1's real incident/CHIRP extractions, never a "too vague, add parameters" critique.

draft    = trace["incident"]["actual_actions_taken"] (what the vessel(s) ACTUALLY did --
           the real mistake, rendered as fluent prose)
critique = trace["incident"]["avoidance_summary"] -- the investigators' OWN counterfactual
           analysis (what, done differently, would have avoided the incident), not a
           synthetic template
refined  = trace["procedures"] (the correct action per the investigators), grounded in
           the trace's own situation summary

Only rows with all three populated are used -- not every report yields one (Rule 5/7/
fatigue/VTS narratives with no clear actual_actions_taken are skipped, never forced).

USAGE
-----
    python -m pipeline.track1.build_incident_reflection_real
"""
from __future__ import annotations

import json

import numpy as np

from core import AgentPaths, load_jsonl, CONTAM_THRESH, EMBEDDER_MODEL
from core.prose import steps_sentence
from pipeline.track1.build_rlhf import SYSTEM

paths = AgentPaths.oow()
CACHE = paths.cache_dir
OUT_FILE = CACHE / "oow_incident_reflection_real.jsonl"

TRACE_FILES = [
    CACHE / "oow_incident_reasoning_traces_fulltext.jsonl",
    CACHE / "oow_chirp_reasoning_traces.jsonl",
]

FIXED_QUESTION = "Given this situation, what should the Officer of the Watch have done, and why?"


def build_row(row: dict) -> dict | None:
    trace = row.get("trace")
    if not trace:
        return None
    procedures = trace.get("procedures") or []
    incident = trace.get("incident", {})
    actual = incident.get("actual_actions_taken") or []
    avoidance_summary = (incident.get("avoidance_summary") or "").strip()
    if not procedures or not actual or not avoidance_summary:
        return None
    situation = trace.get("situation", "")
    draft_text = f"I would have: {steps_sentence(actual)}"
    refined_text = f"{situation} {steps_sentence(procedures)}".strip()
    return {
        "id": row["document_id"], "source_file": row.get("source_file", ""),
        "chapter_title": row.get("chapter_title", ""),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": FIXED_QUESTION},
            {"role": "assistant", "content": f"<think>\nDraft: {draft_text}\n\n{avoidance_summary}\n</think>\n\n"
                                              f"{refined_text}"},
        ],
    }


def load_gold_texts() -> list[str]:
    texts: list[str] = []
    qa_path = paths.eval_dir / "colreg_qa_500_normalised.json"
    if qa_path.exists():
        texts += [g["question"] for g in json.loads(qa_path.read_text(encoding="utf-8"))]
    scenarios_v1_path = paths.eval_dir / "oow_colreg_scenarios_v1.json"
    if scenarios_v1_path.exists():
        texts += [g["situation_report"] for g in json.loads(scenarios_v1_path.read_text(encoding="utf-8"))]
    return texts


def filter_contamination(rows: list[dict], gold_texts: list[str]) -> list[dict]:
    if not rows:
        return rows
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER_MODEL)
    gold_embs = model.encode(gold_texts, normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False)
    texts = [r["messages"][-1]["content"] for r in rows]
    q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    kept, dropped = [], 0
    for r, emb in zip(rows, q_embs):
        c_sim = float(np.max(gold_embs @ emb)) if gold_texts else 0.0
        if c_sim >= CONTAM_THRESH:
            dropped += 1
            continue
        kept.append(r)
    print(f"contamination filter: kept={len(kept)}  dropped={dropped} (thresh={CONTAM_THRESH})")
    return kept


def main() -> None:
    rows = []
    for path in TRACE_FILES:
        if not path.exists():
            print(f"  (skipping {path.name} -- not found)")
            continue
        n_before = len(rows)
        for r in load_jsonl(path):
            row = build_row(r)
            if row is not None:
                rows.append(row)
        print(f"  {path.name}: {len(rows) - n_before} reflection rows")
    print(f"Mined {len(rows)} real incident/CHIRP reflection rows")
    gold_texts = load_gold_texts()
    rows = filter_contamination(rows, gold_texts)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_FILE} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
