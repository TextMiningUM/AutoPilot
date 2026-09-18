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
SCENARIOS_FILE = paths.eval_file(f"{_PFX}_colreg_scenarios.json")
CHUNKS_FILE = CACHE / f"{_PFX}_rag_chunks.json"
EMBS_FILE   = CACHE / f"{_PFX}_rag_embeddings.npy"
IDS_FILE    = CACHE / f"{_PFX}_rag_chunk_ids.json"
KG_FILE     = CACHE / f"{_PFX}_kg.json"
PG_FILE     = CACHE / f"{_PFX}_pg.json"
OUT_FILE    = CACHE / "ablation_prompts.json"

# Track 2 (scenario-based) system prompt -- byte-identical to the one used at eval time
# (eval_colreg_scenarios.py / eval_oow_scenarios.py), so the ablation configs only vary
# the USER turn (CoT instruction / RAG context), never the response-format contract.
def _track2_system() -> str:
    if paths.domain == "VHF":
        from pipeline.eval.eval_colreg_scenarios import SYSTEM_COLREG
        return SYSTEM_COLREG
    from pipeline.eval.eval_oow_scenarios import SYSTEM_OOW
    return SYSTEM_OOW


def _track2_situation_text(g: dict) -> str:
    """The scenario-specific text that grounds the question -- VHF's free-text `scenario`
    briefing, or OOW's fixed-format `situation_report` (its `question` is identical across
    every record, so this is what actually varies the retrieval query/prompt)."""
    return g["scenario"] if paths.domain == "VHF" else g["situation_report"]

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


def stratified_sample(gold: list[dict], n: int, seed: int = 0,
                      group_field: str = "section_id") -> list[dict]:
    """Pick `n` gold records round-robin across `group_field` values (section_id for Track 1
    gold Q&A, category for Track 2 scenarios), so no group dominates the ablation sample."""
    rng = random.Random(seed)
    by_sec: dict[str, list[dict]] = defaultdict(list)
    for g in gold:
        by_sec[g[group_field]].append(g)
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


def build_prompts(q: str, ctx: str, pg_guidance: dict[str, str | None] | None = None) -> dict:
    v0 = [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": q}]
    v1_user = f"Context excerpts from {_DOC_LABEL}:\n\n{ctx}\n\nQuestion: {q}"
    v1 = [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": v1_user}]
    v2 = [{"role": "system", "content": SYSTEM_COT},  {"role": "user", "content": q}]
    v3_user = f"Context excerpts from {_DOC_LABEL}:\n\n{ctx}\n\nQuestion: {q}"
    v3 = [{"role": "system", "content": SYSTEM_RAG_COT}, {"role": "user", "content": v3_user}]
    prompts = {"v0_base": v0, "v1_rag": v1, "v2_cot": v2, "v3_rag_cot": v3}
    for cfg_name, text in (pg_guidance or {}).items():
        if not text:
            continue
        user = f"{text}\n\nQuestion: {q}"
        prompts[cfg_name] = [{"role": "system", "content": SYSTEM_BASE},
                             {"role": "user", "content": user}]
    return prompts


def build_prompts_track2(situation: str, question: str, ctx: str,
                         pg_guidance: dict[str, str | None] | None = None) -> dict:
    """Track 2 (scenario) variant of build_prompts(): same V0..V4 shape, but the system
    prompt is ALWAYS the fixed Track 2 response-format contract (SYSTEM_COLREG/SYSTEM_OOW)
    -- only the USER turn gains a CoT instruction and/or RAG context, so CoT/RAG configs
    never risk losing the quoted-transmission / allowed-action-list format rules."""
    system = _track2_system()
    base_user = f"{situation}\n\n{question}"
    cot_instr = ("Think step by step through the situation, the applicable COLREG rule(s), "
                "and any constraints BEFORE giving your final answer.\n\n")
    rag_instr = (f"Context excerpts from {_DOC_LABEL}:\n\n{ctx}\n\n"
                "Use these excerpts if relevant; otherwise rely on your own knowledge.\n\n")
    prompts = {
        "v0_base":    [{"role": "system", "content": system}, {"role": "user", "content": base_user}],
        "v1_rag":     [{"role": "system", "content": system},
                       {"role": "user", "content": rag_instr + base_user}],
        "v2_cot":     [{"role": "system", "content": system},
                       {"role": "user", "content": cot_instr + base_user}],
        "v3_rag_cot": [{"role": "system", "content": system},
                       {"role": "user", "content": rag_instr + cot_instr + base_user}],
    }
    if pg_guidance:
        for cfg_name, text in pg_guidance.items():
            if not text:
                continue
            prompts[cfg_name] = [{"role": "system", "content": system},
                                 {"role": "user", "content": f"{text}\n\n{base_user}"}]
    return prompts


def main() -> None:
    """CLI entry point: build the v0/v1/v2/v3 ablation prompt sets for a stratified gold sample."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="pilot sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="retrieval top-k")
    ap.add_argument("--gold-file", type=str, default=None,
                    help="held-out gold file (default: paths.gold_file, or the Track 2 "
                         "scenarios file when --track2 is set)")
    ap.add_argument("--track2", action="store_true",
                    help="ablate the Track 2 scenario file (conversational compliance / applied "
                         "helm-engine decisions) instead of the Track 1 gold Q&A -- grouped by "
                         "category instead of section_id, fixed Track 2 system prompt")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for ablation_prompts/answers/scored/summary filenames -- pass "
                         "e.g. 'smoke' so a small smoke-test sample never shares (and never gets "
                         "silently combined with) a full run's accumulated answers file")
    args = ap.parse_args()
    global OUT_FILE
    if args.track2:
        OUT_FILE = CACHE / "ablation_prompts_track2.json"
    if args.tag:
        stem = "ablation_prompts_track2" if args.track2 else "ablation_prompts"
        OUT_FILE = CACHE / f"{stem}_{args.tag}.json"
    gold_file = Path(args.gold_file) if args.gold_file else (SCENARIOS_FILE if args.track2 else GOLD_FILE)
    group_field = "category" if args.track2 else "section_id"

    gold = json.loads(gold_file.read_text(encoding="utf-8"))
    print(f"Total gold: {len(gold)}")
    sample = stratified_sample(gold, args.n, seed=args.seed, group_field=group_field)
    print(f"Stratified sample: n={len(sample)}")
    grps = sorted({s[group_field] for s in sample})
    grp_label = "Categories" if group_field == "category" else "Sections"
    print(f"{grp_label} covered: {len(grps)}")

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    embs   = np.load(EMBS_FILE)
    ids    = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    kg     = json.loads(KG_FILE.read_text(encoding="utf-8"))
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    print("Loading embedder (CPU-friendly for retrieval)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # v4_pg (merged graph) plus, when built, source-scoped extra PGs -- each gets its
    # own ablation config so its individual contribution is separately measurable
    # (notebook § 11's KG/PG quality investigation).
    pg_configs: list[tuple[str, Path]] = [("v4_pg", PG_FILE)]
    for cfg_name, suffix in (("v5_pg_incident", "incident"), ("v6_pg_scenario", "scenario")):
        f = CACHE / f"{_PFX}_pg_{suffix}.json"
        if f.exists():
            pg_configs.append((cfg_name, f))
    graphs: dict[str, ProceduralGraph] = {}
    for cfg_name, f in pg_configs:
        if f.exists():
            graphs[cfg_name] = ProceduralGraph(f, model)
        else:
            print(f"  [{cfg_name}] {f.name} not found -- run pipeline.ingest.build_pg to enable it")

    records = []
    n_with_guidance: dict[str, int] = {cfg: 0 for cfg in graphs}
    log_every = 1 if len(sample) <= 20 else 10
    for i, g in enumerate(sample, 1):
        retrieval_query = _track2_situation_text(g) if args.track2 else g["question"]
        hits, q_cons, expanded = kg_retrieve(
            retrieval_query, model, embs, ids, kg, k=args.k, dense_n=20,
        )
        ctx = format_context(hits, chunk_by_id)
        pg_texts: dict[str, str | None] = {}
        for cfg_name, graph in graphs.items():
            text = render_guidance(retrieval_query, graph)
            pg_texts[cfg_name] = text or None
            if text:
                n_with_guidance[cfg_name] += 1
        if args.track2:
            prompts = build_prompts_track2(_track2_situation_text(g), g["question"], ctx, pg_texts)
        else:
            prompts = build_prompts(g["question"], ctx, pg_texts)
        record = {
            "q_id":            g["id"],
            group_field:       g[group_field],
            "question":        g["question"],
            "gold_answer":     g["gold_answer"],
            "expected_points": g.get("expected_points", []),
            "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
            "retrieved_scores":    [round(h["score"], 3) for h in hits],
            "query_concepts":      q_cons,
            "expanded_concepts":   expanded,
            "prompts": prompts,
        }
        if args.track2:
            record["colreg_rules"] = g.get("colreg_rules", [])
            if paths.domain == "VHF":
                record["scenario"] = g["scenario"]
                record["own_vessel"] = g.get("own_vessel", "")
            else:
                record["situation_report"] = g["situation_report"]
                record["correct_action"] = g.get("correct_action")
                record["correct_action_params"] = g.get("correct_action_params", {})
        else:
            record["section_title"] = g.get("section_title", "")
            record["type"] = g.get("type", "reference")
        records.append(record)
        if i % log_every == 0 or i == len(sample):
            print(f"  prepped {i}/{len(sample)}")

    configs = ["v0_base", "v1_rag", "v2_cot", "v3_rag_cot"]
    for cfg_name in graphs:
        configs.append(cfg_name)
        print(f"  {cfg_name} guidance rendered for {n_with_guidance[cfg_name]}/{len(records)} "
              "questions (others fall back to the v0 prompt at run time)")
    OUT_FILE.write_text(json.dumps({
        "n": len(records),
        "seed": args.seed,
        "k": args.k,
        "track2": args.track2,
        "configs": configs,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}  ({OUT_FILE.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
