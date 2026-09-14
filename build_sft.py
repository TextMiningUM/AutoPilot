"""Tutorial 13 § 9 (step 2) — build direct + CoT + RAG SFT datasets from reasoning traces.

Deterministic (no LLM). Reads _cache/vhf_reasoning_traces.jsonl and emits:

  _cache/vhf_sft_direct.jsonl   — Q -> concise answer built from key_facts+procedures
  _cache/vhf_sft_cot.jsonl      — Q -> chain-of-thought reasoning -> answer
  _cache/vhf_sft_rag.jsonl      — (Q + top-3 KG context) -> answer

Contamination filter against 540 gold questions (cos >= 0.85 dropped).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from build_kg import kg_retrieve

W = Path(__file__).resolve().parent
CACHE = W / "_cache"

TRACES_FILE = CACHE / "vhf_reasoning_traces.jsonl"
CHUNKS_FILE = CACHE / "vhf_rag_chunks.json"
EMBS_FILE   = CACHE / "vhf_rag_embeddings.npy"
IDS_FILE    = CACHE / "vhf_rag_chunk_ids.json"
KG_FILE     = CACHE / "vhf_kg.json"
GOLD_FILE   = W / "vhf_gold_answers.json"

DIRECT_OUT = CACHE / "vhf_sft_direct.jsonl"
COT_OUT    = CACHE / "vhf_sft_cot.jsonl"
RAG_OUT    = CACHE / "vhf_sft_rag.jsonl"
STATS_OUT  = CACHE / "vhf_sft_stats.json"

CONTAM_THRESH = 0.85
DEDUP_THRESH  = 0.92

SYSTEM_DIRECT = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Answer accurately, use correct prowords (MAYDAY, PAN PAN, SECURITE, OVER, OUT, THIS IS), "
    "cite VHF channel numbers, and follow ITU/IMO/GMDSS regulations. Be concise."
)
SYSTEM_COT = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Think step by step through the situation, the applicable procedure, and the constraints, "
    "then give a precise answer with correct prowords and channel numbers."
)
SYSTEM_RAG = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Use ONLY the provided context excerpts. If the excerpts don't contain the answer, "
    "say so explicitly. Use correct prowords and channel numbers exactly as they appear."
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def format_direct_answer(trace: dict, angle: str) -> str:
    procs   = trace.get("procedures") or []
    kf      = trace.get("key_facts") or []
    channels = trace.get("channels") or []
    prowords = trace.get("prowords_used") or []
    constraints = trace.get("constraints") or []
    outcomes = trace.get("outcomes") or []
    warnings = trace.get("warnings") or []

    if angle == "how" and procs:
        steps = " ".join(f"{p.get('step', i+1)}. {p.get('action','').rstrip('.')}." for i, p in enumerate(procs))
        tail = ""
        if outcomes:  tail += f" Outcome: {outcomes[0].rstrip('.')}."
        if warnings:  tail += f" Warning: {warnings[0].rstrip('.')}."
        return steps + tail

    if angle == "when":
        parts = []
        trig = trace.get("trigger")
        if trig: parts.append(trig.rstrip("."))
        parts.extend(c.rstrip(".") for c in constraints[:2])
        if not parts and kf: parts.append(kf[0].rstrip("."))
        return ". ".join(parts) + "." if parts else (kf[0] if kf else "")

    if angle == "which" and channels:
        chs = ", ".join(f"Channel {c}" if not str(c).lower().startswith("channel") else c for c in channels)
        base = f"{chs}."
        if kf: base += f" {kf[0].rstrip('.')}."
        return base

    if angle == "why":
        parts = []
        for p in procs[:2]:
            if p.get("why"): parts.append(p["why"].rstrip("."))
        parts.extend(kf[:2])
        return ". ".join(parts) + "." if parts else (kf[0] if kf else "")

    # what / who / default
    parts = kf[:2] if kf else []
    if not parts and procs:
        parts = [procs[0].get("action", "")]
    text = ". ".join(p.rstrip(".") for p in parts if p) + "."
    return text if text.strip(".") else "Information not available in this excerpt."


def format_cot_answer(trace: dict, direct_answer: str) -> str:
    lines = ["Let me reason through this step by step.\n"]
    if trace.get("situation"):
        lines.append(f"Situation: {trace['situation']}")
    if trace.get("trigger"):
        lines.append(f"Trigger: {trace['trigger']}")
    procs = trace.get("procedures") or []
    if procs:
        lines.append("Procedure:")
        for p in procs:
            step = p.get("step", "?")
            act  = p.get("action", "").rstrip(".")
            why  = p.get("why", "").rstrip(".")
            lines.append(f"  {step}. {act}." + (f" (Rationale: {why}.)" if why else ""))
    constraints = trace.get("constraints") or []
    if constraints:
        lines.append("Constraints:")
        for c in constraints[:3]:
            lines.append(f"  - {c}")
    channels = trace.get("channels") or []
    if channels:
        lines.append(f"Channels involved: {', '.join(str(c) for c in channels)}")
    prowords = trace.get("prowords_used") or []
    if prowords:
        lines.append(f"Prowords: {', '.join(prowords)}")
    warnings = trace.get("warnings") or []
    if warnings:
        lines.append("Warnings:")
        for w in warnings[:2]:
            lines.append(f"  - {w}")
    lines.append(f"\nAnswer: {direct_answer}")
    return "\n".join(lines)


def main():
    traces = [t for t in load_jsonl(TRACES_FILE)
              if t.get("trace") and not t.get("skip") and not t.get("error")]
    print(f"Usable traces: {len(traces)}")

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    embs   = np.load(EMBS_FILE)
    ids    = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    kg     = json.loads(KG_FILE.read_text(encoding="utf-8"))
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))

    print("Loading embedder + gold-Q embeddings for contamination filter...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gold_embs = model.encode([g["question"] for g in gold],
                             normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False)

    # ── Flatten all (chunk_id, angle, question) candidates ─────────────────
    flat: list[tuple[dict, str, str, str]] = []  # (trace_rec, angle, question, direct_ans)
    for rec in traces:
        t = rec["trace"]
        seeds = t.get("question_seeds") or []
        for seed in seeds:
            q = (seed.get("text") or "").strip()
            if len(q) < 8:
                continue
            angle = (seed.get("angle") or "what").lower()
            direct = format_direct_answer(t, angle)
            if not direct or direct == "Information not available in this excerpt.":
                continue
            flat.append((rec, angle, q, direct))

    print(f"Raw (question, answer) candidates: {len(flat)}")

    # ── Encode + contamination + dedup ────────────────────────────────────
    print("Encoding candidate questions...")
    q_texts = [q for _, _, q, _ in flat]
    q_embs  = model.encode(q_texts, normalize_embeddings=True, batch_size=128,
                           show_progress_bar=True)

    kept: list[tuple[dict, str, str, str]] = []
    kept_embs: list[np.ndarray] = []
    contam_drop = 0
    dedup_drop  = 0
    for i, cand in enumerate(flat):
        emb = q_embs[i]
        c_sim = float(np.max(gold_embs @ emb))
        if c_sim >= CONTAM_THRESH:
            contam_drop += 1
            continue
        if kept_embs:
            d_sim = float(np.max(np.vstack(kept_embs) @ emb))
            if d_sim >= DEDUP_THRESH:
                dedup_drop += 1
                continue
        kept.append(cand + (round(c_sim, 3),))
        kept_embs.append(emb)

    print(f"kept={len(kept)}  contam_dropped={contam_drop}  dedup_dropped={dedup_drop}")

    # ── Emit direct + cot ─────────────────────────────────────────────────
    print(f"\nWriting {DIRECT_OUT.name} and {COT_OUT.name}...")
    with DIRECT_OUT.open("w", encoding="utf-8") as fd, COT_OUT.open("w", encoding="utf-8") as fc:
        for rec, angle, q, direct_ans, c_sim in kept:
            base = {
                "source_chunk_id": rec["chunk_id"],
                "source_file":     rec["source_file"],
                "chapter_title":   rec["chapter_title"],
                "angle":           angle,
                "expected_points": rec["trace"].get("key_facts", []),
                "contam_sim":      c_sim,
            }
            fd.write(json.dumps({
                **base,
                "messages": [
                    {"role": "system",    "content": SYSTEM_DIRECT},
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": direct_ans},
                ],
            }, ensure_ascii=False) + "\n")
            fc.write(json.dumps({
                **base,
                "messages": [
                    {"role": "system",    "content": SYSTEM_COT},
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": format_cot_answer(rec["trace"], direct_ans)},
                ],
            }, ensure_ascii=False) + "\n")

    # ── Emit RAG ──────────────────────────────────────────────────────────
    print(f"Writing {RAG_OUT.name} (KG top-3 context per Q)...")
    with RAG_OUT.open("w", encoding="utf-8") as fr:
        for i, (rec, angle, q, direct_ans, c_sim) in enumerate(kept, 1):
            hits, q_cons, expanded = kg_retrieve(q, model, embs, ids, kg, k=3, dense_n=20)
            ctx = "\n\n".join(
                f"[Excerpt {j} — {chunk_by_id[h['chunk_id']]['source_file']} / "
                f"{chunk_by_id[h['chunk_id']]['chapter_title']}]\n"
                f"{chunk_by_id[h['chunk_id']]['text']}"
                for j, h in enumerate(hits, 1)
            ) or "(no relevant excerpts found)"
            user_msg = (
                f"Context excerpts from VHF reference documents:\n\n"
                f"{ctx}\n\n"
                f"Question: {q}"
            )
            fr.write(json.dumps({
                "source_chunk_id":     rec["chunk_id"],
                "source_file":         rec["source_file"],
                "chapter_title":       rec["chapter_title"],
                "angle":               angle,
                "expected_points":     rec["trace"].get("key_facts", []),
                "contam_sim":          c_sim,
                "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
                "retrieved_scores":    [round(h["score"], 3) for h in hits],
                "query_concepts":      q_cons,
                "expanded_concepts":   expanded,
                "messages": [
                    {"role": "system",    "content": SYSTEM_RAG},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": direct_ans},
                ],
            }, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"  {i}/{len(kept)}")

    stats = {
        "usable_traces":    len(traces),
        "raw_candidates":   len(flat),
        "kept":             len(kept),
        "contam_dropped":   contam_drop,
        "dedup_dropped":    dedup_drop,
        "direct_bytes":     DIRECT_OUT.stat().st_size,
        "cot_bytes":        COT_OUT.stat().st_size,
        "rag_bytes":        RAG_OUT.stat().st_size,
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2))
    print(f"\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
