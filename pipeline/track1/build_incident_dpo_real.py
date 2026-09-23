"""
================================================================================
build_incident_dpo_real.py -- Fase C2 "C1 pairs" (RAG-rebuild-v2 plan)
================================================================================

The plan's C2 explicitly calls for folding Fase C1's real, investigator-authored
incident/CHIRP extractions into the DPO set "with higher weight (real, expert-authorized
data)" -- DISTINCT from oow_incident_dpo_pairs.jsonl/oow_chirp_dpo_pairs.jsonl (built by
the generic build_rlhf.py, which perturbs the CORRECT answer's own text -- wrong_channel/
missing_step/etc. -- and never touches what the vessel ACTUALLY did wrong).

This script instead pairs, for every trace with BOTH populated:
  chosen   = trace["procedures"]      (the CORRECT action, as the investigators determined it)
  rejected = trace["incident"]["actual_actions_taken"]  (what the vessel(s) ACTUALLY did --
             the REAL mistake, not a synthetic perturbation)

Both rendered as fluent prose (core.prose.steps_sentence -- never a label:value dump, per
this project's established rule) grounded in the trace's own situation summary. Not every
report yields a pair: some are Rule 5/7/fatigue/lookout narratives with no procedures or
no actual_actions_taken populated -- skipped, never forced.

Mandatory contamination filter against BOTH held-out files (copilot-instructions.md rule).

USAGE
-----
    python -m pipeline.track1.build_incident_dpo_real
"""
from __future__ import annotations

import json

import numpy as np

from core import AgentPaths, load_jsonl, CONTAM_THRESH, EMBEDDER_MODEL
from core.prose import steps_sentence
from pipeline.track1.build_rlhf import SYSTEM

paths = AgentPaths.oow()
CACHE = paths.cache_dir
OUT_FILE = CACHE / "oow_incident_dpo_pairs_real.jsonl"

TRACE_FILES = [
    CACHE / "oow_incident_reasoning_traces_fulltext.jsonl",
    CACHE / "oow_chirp_reasoning_traces.jsonl",
]

FIXED_QUESTION = "Given this situation, what should the Officer of the Watch have done, and why?"


def build_pair(row: dict) -> dict | None:
    trace = row.get("trace")
    if not trace:
        return None
    procedures = trace.get("procedures") or []
    actual = trace.get("incident", {}).get("actual_actions_taken") or []
    if not procedures or not actual:
        return None
    situation = trace.get("situation", "")
    chosen_text = f"{situation} {steps_sentence(procedures)}".strip()
    rejected_text = f"{situation} {steps_sentence(actual)}".strip()
    if chosen_text == rejected_text:
        return None
    return {
        "id": row["document_id"], "source_chunk_id": row.get("chunk_id", row["document_id"]),
        "source_file": row.get("source_file", ""), "chapter_title": row.get("chapter_title", ""),
        "pair_type": "real_incident_mistake",
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": FIXED_QUESTION},
        ],
        "chosen": [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
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


def filter_contamination(pairs: list[dict], gold_texts: list[str]) -> list[dict]:
    if not pairs:
        return pairs
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER_MODEL)
    gold_embs = model.encode(gold_texts, normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False)
    texts = [p["chosen"][0]["content"] for p in pairs]
    q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    kept, dropped = [], 0
    for p, emb in zip(pairs, q_embs):
        c_sim = float(np.max(gold_embs @ emb)) if gold_texts else 0.0
        if c_sim >= CONTAM_THRESH:
            dropped += 1
            continue
        kept.append(p)
    print(f"contamination filter: kept={len(kept)}  dropped={dropped} (thresh={CONTAM_THRESH})")
    return kept


def main() -> None:
    pairs = []
    for path in TRACE_FILES:
        if not path.exists():
            print(f"  (skipping {path.name} -- not found)")
            continue
        n_before = len(pairs)
        for row in load_jsonl(path):
            pair = build_pair(row)
            if pair is not None:
                pairs.append(pair)
        print(f"  {path.name}: {len(pairs) - n_before} real chosen/rejected pairs")
    print(f"Mined {len(pairs)} real incident/CHIRP DPO pairs (investigators' own judgment)")
    gold_texts = load_gold_texts()
    pairs = filter_contamination(pairs, gold_texts)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_FILE} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
