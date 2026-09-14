"""Ablation step 1 — precompute prompts for V0..V3 on a stratified sample of gold Q's.

Loads only the CPU embedder (sentence-transformer). Does NOT load Qwen.
Writes _cache/ablation_prompts.json for the inference stage.
"""
from __future__ import annotations
import json, argparse, random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from build_kg import kg_retrieve

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"

GOLD_FILE   = W / "Data" / "VHF" / "VHF_Eval" / "vhf_gold_answers.json"
CHUNKS_FILE = CACHE / "vhf_rag_chunks.json"
EMBS_FILE   = CACHE / "vhf_rag_embeddings.npy"
IDS_FILE    = CACHE / "vhf_rag_chunk_ids.json"
KG_FILE     = CACHE / "vhf_kg.json"
OUT_FILE    = CACHE / "ablation_prompts.json"

SYSTEM_BASE = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Answer accurately, use correct prowords (MAYDAY, PAN PAN, SECURITE, OVER, OUT, THIS IS), "
    "cite VHF channel numbers, and follow ITU/IMO/GMDSS regulations. Be concise."
)
SYSTEM_COT = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Think step by step through the situation, applicable procedure, and constraints "
    "BEFORE giving your final answer. Use correct prowords and channel numbers."
)
SYSTEM_RAG = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Use ONLY the provided context excerpts to answer. If the excerpts don't contain the answer, "
    "say so. Use correct prowords and channel numbers exactly as they appear."
)
SYSTEM_RAG_COT = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Read the excerpts. Think step by step through the situation, procedure, and constraints "
    "using ONLY the excerpts. Then give a precise answer with correct prowords and channel numbers."
)


def stratified_sample(gold: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_sec: dict[str, list[dict]] = defaultdict(list)
    for g in gold:
        by_sec[g["section_id"]].append(g)
    sections = sorted(by_sec)
    picked: list[dict] = []
    i = 0
    while len(picked) < n:
        sec = sections[i % len(sections)]
        pool = [g for g in by_sec[sec] if g not in picked]
        if pool:
            picked.append(rng.choice(pool))
        i += 1
        if i > n * 20:
            break
    return picked[:n]


def format_context(hits: list[dict], chunk_by_id: dict[str, dict]) -> str:
    parts = []
    for j, h in enumerate(hits, 1):
        c = chunk_by_id[h["chunk_id"]]
        parts.append(
            f"[Excerpt {j} — {c['source_file']} / {c['chapter_title']}]\n{c['text']}"
        )
    return "\n\n".join(parts) or "(no relevant excerpts found)"


def build_prompts(q: str, ctx: str) -> dict:
    v0 = [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": q}]
    v1_user = f"Context excerpts from VHF reference documents:\n\n{ctx}\n\nQuestion: {q}"
    v1 = [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": v1_user}]
    v2 = [{"role": "system", "content": SYSTEM_COT},  {"role": "user", "content": q}]
    v3_user = f"Context excerpts from VHF reference documents:\n\n{ctx}\n\nQuestion: {q}"
    v3 = [{"role": "system", "content": SYSTEM_RAG_COT}, {"role": "user", "content": v3_user}]
    return {"v0_base": v0, "v1_rag": v1, "v2_cot": v2, "v3_rag_cot": v3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="pilot sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="retrieval top-k")
    args = ap.parse_args()

    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    print(f"Total gold: {len(gold)}")
    sample = stratified_sample(gold, args.n, seed=args.seed)
    print(f"Stratified sample: n={len(sample)}")
    secs = sorted({s["section_id"] for s in sample})
    print(f"Sections covered: {len(secs)}")

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    embs   = np.load(EMBS_FILE)
    ids    = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    kg     = json.loads(KG_FILE.read_text(encoding="utf-8"))
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    print("Loading embedder (CPU-friendly for retrieval)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    records = []
    for i, g in enumerate(sample, 1):
        hits, q_cons, expanded = kg_retrieve(
            g["question"], model, embs, ids, kg, k=args.k, dense_n=20,
        )
        ctx = format_context(hits, chunk_by_id)
        prompts = build_prompts(g["question"], ctx)
        records.append({
            "q_id":            g["id"],
            "section_id":      g["section_id"],
            "section_title":   g["section_title"],
            "type":            g.get("type", "reference"),
            "question":        g["question"],
            "gold_answer":     g["gold_answer"],
            "expected_points": g.get("expected_points", []),
            "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
            "retrieved_scores":    [round(h["score"], 3) for h in hits],
            "query_concepts":      q_cons,
            "expanded_concepts":   expanded,
            "prompts": prompts,
        })
        if i % 10 == 0:
            print(f"  prepped {i}/{len(sample)}")

    OUT_FILE.write_text(json.dumps({
        "n": len(records),
        "seed": args.seed,
        "k": args.k,
        "configs": ["v0_base", "v1_rag", "v2_cot", "v3_rag_cot"],
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}  ({OUT_FILE.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
