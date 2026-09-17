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

from pipeline.ingest.build_kg import kg_retrieve
from pipeline.ingest.pg_guidance import ProceduralGraph, render_guidance
from core import AgentPaths

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

GOLD_FILE   = paths.gold_file
CHUNKS_FILE = CACHE / f"{_PFX}_rag_chunks.json"
EMBS_FILE   = CACHE / f"{_PFX}_rag_embeddings.npy"
IDS_FILE    = CACHE / f"{_PFX}_rag_chunk_ids.json"
KG_FILE     = CACHE / f"{_PFX}_kg.json"
PG_FILE     = CACHE / f"{_PFX}_pg.json"
OUT_FILE    = CACHE / "ablation_prompts.json"

_ABLATION_PROMPTS = {
    "VHF": dict(
        base=(
            "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
            "Answer accurately, use correct prowords (MAYDAY, PAN PAN, SECURITE, OVER, OUT, THIS IS), "
            "cite VHF channel numbers, and follow ITU/IMO/GMDSS regulations. Be concise."
        ),
        cot=(
            "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
            "Think step by step through the situation, applicable procedure, and constraints "
            "BEFORE giving your final answer. Use correct prowords and channel numbers."
        ),
        rag=(
            "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
            "Use ONLY the provided context excerpts to answer. If the excerpts don't contain the answer, "
            "say so. Use correct prowords and channel numbers exactly as they appear."
        ),
        rag_cot=(
            "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
            "Read the excerpts. Think step by step through the situation, procedure, and constraints "
            "using ONLY the excerpts. Then give a precise answer with correct prowords and channel numbers."
        ),
        doc_label="VHF reference documents",
    ),
    "OOW": dict(
        base=(
            "You are the Officer of the Watch, an AI navigation agent responsible for COLREG-compliant "
            "collision avoidance. Answer accurately, cite the correct COLREG rule number(s), and respect "
            "give-way/stand-on obligations. Be concise."
        ),
        cot=(
            "You are the Officer of the Watch, an AI navigation agent responsible for COLREG-compliant "
            "collision avoidance. Think step by step through the encounter, applicable rule(s), and "
            "give-way/stand-on obligations BEFORE giving your final answer. Cite the rule number(s)."
        ),
        rag=(
            "You are the Officer of the Watch, an AI navigation agent responsible for COLREG-compliant "
            "collision avoidance. Use ONLY the provided context excerpts to answer. If the excerpts don't "
            "contain the answer, say so. Cite rule numbers exactly as they appear in the excerpts."
        ),
        rag_cot=(
            "You are the Officer of the Watch, an AI navigation agent responsible for COLREG-compliant "
            "collision avoidance. Read the excerpts. Think step by step through the encounter, rule(s), "
            "and obligations using ONLY the excerpts. Then give a precise answer citing the rule number(s)."
        ),
        doc_label="COLREG reference documents",
    ),
}[paths.domain]
SYSTEM_BASE    = _ABLATION_PROMPTS["base"]
SYSTEM_COT     = _ABLATION_PROMPTS["cot"]
SYSTEM_RAG     = _ABLATION_PROMPTS["rag"]
SYSTEM_RAG_COT = _ABLATION_PROMPTS["rag_cot"]
_DOC_LABEL     = _ABLATION_PROMPTS["doc_label"]


def stratified_sample(gold: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Pick `n` gold Q&A round-robin across section_ids, so no section dominates the ablation sample."""
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


def build_prompts(q: str, ctx: str, pg_guidance_text: str | None = None) -> dict:
    v0 = [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": q}]
    v1_user = f"Context excerpts from {_DOC_LABEL}:\n\n{ctx}\n\nQuestion: {q}"
    v1 = [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": v1_user}]
    v2 = [{"role": "system", "content": SYSTEM_COT},  {"role": "user", "content": q}]
    v3_user = f"Context excerpts from {_DOC_LABEL}:\n\n{ctx}\n\nQuestion: {q}"
    v3 = [{"role": "system", "content": SYSTEM_RAG_COT}, {"role": "user", "content": v3_user}]
    prompts = {"v0_base": v0, "v1_rag": v1, "v2_cot": v2, "v3_rag_cot": v3}
    if pg_guidance_text:
        v4_user = f"{pg_guidance_text}\n\nQuestion: {q}"
        prompts["v4_pg"] = [{"role": "system", "content": SYSTEM_BASE},
                            {"role": "user", "content": v4_user}]
    return prompts


def main() -> None:
    """CLI entry point: build the v0/v1/v2/v3 ablation prompt sets for a stratified gold sample."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="pilot sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="retrieval top-k")
    ap.add_argument("--gold-file", type=str, default=str(GOLD_FILE),
                    help="held-out gold Q&A file (default: paths.gold_file)")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for ablation_prompts/answers/scored/summary filenames -- pass "
                         "e.g. 'smoke' so a small smoke-test sample never shares (and never gets "
                         "silently combined with) a full run's accumulated answers file")
    args = ap.parse_args()
    global OUT_FILE
    if args.tag:
        OUT_FILE = CACHE / f"ablation_prompts_{args.tag}.json"

    gold = json.loads(Path(args.gold_file).read_text(encoding="utf-8"))
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

    # v4_pg: guidance rendered from the procedural graph, when one exists
    graph = ProceduralGraph(PG_FILE, model) if PG_FILE.exists() else None
    if graph is None:
        print(f"  [v4_pg] {PG_FILE.name} not found -- run pipeline.ingest.build_pg to enable v4_pg")

    records = []
    n_with_guidance = 0
    log_every = 1 if len(sample) <= 20 else 10
    for i, g in enumerate(sample, 1):
        hits, q_cons, expanded = kg_retrieve(
            g["question"], model, embs, ids, kg, k=args.k, dense_n=20,
        )
        ctx = format_context(hits, chunk_by_id)
        pg_text = render_guidance(g["question"], graph) if graph else None
        if pg_text:
            n_with_guidance += 1
        prompts = build_prompts(g["question"], ctx, pg_text or None)
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
        if i % log_every == 0 or i == len(sample):
            print(f"  prepped {i}/{len(sample)}")

    configs = ["v0_base", "v1_rag", "v2_cot", "v3_rag_cot"]
    if graph:
        configs.append("v4_pg")
        print(f"  v4_pg guidance rendered for {n_with_guidance}/{len(records)} questions "
              "(others fall back to the v0 prompt at run time)")
    OUT_FILE.write_text(json.dumps({
        "n": len(records),
        "seed": args.seed,
        "k": args.k,
        "configs": configs,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}  ({OUT_FILE.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
