"""Tutorial 13 § 10 — multi-hop reasoning dataset.

For every pair of reasoning traces that share a KG concept but come from
DIFFERENT source documents, compose a multi-hop question and a two-part answer:

  Q: "Regarding <concept>, first <angle-A> then <angle-B>?"
  A: "According to <source_A>: <trace_A summary>. According to <source_B>: <trace_B summary>.
     Combined: <both key facts>."

Contamination filter against 540 gold questions.

Output: _cache/vhf_multihop.jsonl
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

W = Path(__file__).resolve().parent
CACHE = W / "_cache"

TRACES_FILE = CACHE / "vhf_reasoning_traces.jsonl"
GOLD_FILE   = W / "vhf_gold_answers.json"
OUT_FILE    = CACHE / "vhf_multihop.jsonl"

CONTAM_THRESH = 0.85
MAX_PAIRS_PER_CONCEPT = 25
RNG = random.Random(42)

SYSTEM = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Some questions require you to synthesize information from multiple reference sources. "
    "Explain each source's contribution, then give a combined precise answer with correct "
    "prowords and channel numbers."
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue


def concept_signature(trace: dict) -> set[str]:
    t = trace.get("trace") or {}
    sig = set(str(x).strip() for x in (trace.get("chunk_concepts") or []) if str(x).strip())
    for c in (t.get("channels") or []):
        sig.add(f"Channel {c}" if str(c).isdigit() else str(c))
    for p in (t.get("prowords_used") or []):
        sig.add(p.upper())
    for r in (t.get("regulations") or []):
        sig.add(r.upper())
    return sig


def summarize_trace(trace: dict) -> str:
    t = trace.get("trace") or {}
    parts: list[str] = []
    if t.get("situation"): parts.append(t["situation"].rstrip("."))
    procs = t.get("procedures") or []
    if procs:
        acts = "; ".join(f"{p.get('action','').rstrip('.')}" for p in procs[:4])
        parts.append(f"steps: {acts}")
    if t.get("channels"):
        parts.append(f"channels: {', '.join(str(c) for c in t['channels'])}")
    if t.get("prowords_used"):
        parts.append(f"prowords: {', '.join(t['prowords_used'])}")
    kf = t.get("key_facts") or []
    if kf: parts.append(kf[0].rstrip("."))
    return ". ".join(parts) + "."


def compose_multihop_question(concept: str, tr_a: dict, tr_b: dict) -> str:
    seeds_a = (tr_a["trace"].get("question_seeds") or [])
    seeds_b = (tr_b["trace"].get("question_seeds") or [])
    fa = seeds_a[0].get("text") if seeds_a else f"how is {concept} handled?"
    fb = seeds_b[0].get("text") if seeds_b else f"what else does the guidance say about {concept}?"
    src_a = tr_a["source_file"].rsplit(".", 1)[0]
    src_b = tr_b["source_file"].rsplit(".", 1)[0]
    return (
        f"Regarding {concept}: {fa.strip().rstrip('?')} "
        f"And in the context of {src_b} — {fb.strip().rstrip('?')}? "
        f"Please explain what each source says and combine into one recommendation."
    )


def compose_multihop_answer(concept: str, tr_a: dict, tr_b: dict) -> str:
    sum_a = summarize_trace(tr_a)
    sum_b = summarize_trace(tr_b)
    kfs = []
    for tr in (tr_a, tr_b):
        for kf in (tr["trace"].get("key_facts") or [])[:2]:
            if kf and kf not in kfs:
                kfs.append(kf.rstrip("."))
    combined = ". ".join(kfs) + "." if kfs else ""
    return (
        f"According to {tr_a['source_file']} ({tr_a['chapter_title']}): {sum_a}\n\n"
        f"According to {tr_b['source_file']} ({tr_b['chapter_title']}): {sum_b}\n\n"
        f"Combined recommendation on {concept}: {combined}"
    )


def main():
    traces = [t for t in load_jsonl(TRACES_FILE)
              if t.get("trace") and not t.get("skip") and not t.get("error")]
    print(f"Usable traces: {len(traces)}")

    # Index by concept
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for tr in traces:
        for con in concept_signature(tr):
            by_concept[con].append(tr)

    # Interesting concepts: >=2 traces from different sources
    interesting: dict[str, list[dict]] = {}
    for con, lst in by_concept.items():
        srcs = {t["source_file"] for t in lst}
        if len(srcs) >= 2 and len(lst) >= 2:
            interesting[con] = lst
    print(f"Concepts with multi-source coverage: {len(interesting)}")

    # Skip generic concepts that would produce trivial pairs
    SKIP = {"VHF", "OVER", "OUT", "THIS IS", "ROGER", "AFFIRMATIVE", "NEGATIVE"}
    interesting = {k: v for k, v in interesting.items() if k not in SKIP}
    print(f"After skipping generic concepts: {len(interesting)}")

    print("Loading embedder + gold-Q embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    gold_embs = model.encode([g["question"] for g in gold],
                             normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False)

    pairs: list[dict] = []
    for con, lst in interesting.items():
        by_src: dict[str, list[dict]] = defaultdict(list)
        for tr in lst:
            by_src[tr["source_file"]].append(tr)
        srcs = list(by_src.keys())
        if len(srcs) < 2: continue

        emitted = 0
        RNG.shuffle(srcs)
        for i, a in enumerate(srcs):
            for b in srcs[i+1:]:
                tr_a = RNG.choice(by_src[a])
                tr_b = RNG.choice(by_src[b])
                q = compose_multihop_question(con, tr_a, tr_b)
                ans = compose_multihop_answer(con, tr_a, tr_b)
                pairs.append({
                    "concept": con,
                    "trace_a": tr_a["chunk_id"],
                    "trace_b": tr_b["chunk_id"],
                    "source_a": tr_a["source_file"],
                    "source_b": tr_b["source_file"],
                    "question": q,
                    "answer": ans,
                    "key_facts": (tr_a["trace"].get("key_facts", []) +
                                  tr_b["trace"].get("key_facts", [])),
                })
                emitted += 1
                if emitted >= MAX_PAIRS_PER_CONCEPT: break
            if emitted >= MAX_PAIRS_PER_CONCEPT: break

    print(f"Raw multi-hop pairs: {len(pairs)}")

    # Contamination filter
    print("Contamination check...")
    q_embs = model.encode([p["question"] for p in pairs],
                          normalize_embeddings=True, batch_size=128,
                          show_progress_bar=True)
    kept = []
    dropped = 0
    for p, emb in zip(pairs, q_embs):
        sim = float(np.max(gold_embs @ emb))
        if sim >= CONTAM_THRESH:
            dropped += 1
            continue
        p["contam_sim"] = round(sim, 3)
        kept.append(p)
    print(f"kept={len(kept)} contam_dropped={dropped}")

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for i, p in enumerate(kept):
            f.write(json.dumps({
                "id": f"mh_{i:04d}",
                "concept":         p["concept"],
                "trace_a":         p["trace_a"],
                "trace_b":         p["trace_b"],
                "source_a":        p["source_a"],
                "source_b":        p["source_b"],
                "expected_points": p["key_facts"],
                "contam_sim":      p["contam_sim"],
                "messages": [
                    {"role": "system",    "content": SYSTEM},
                    {"role": "user",      "content": p["question"]},
                    {"role": "assistant", "content": p["answer"]},
                ],
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_FILE.name}  ({OUT_FILE.stat().st_size/1024:.1f} KB)  rows={len(kept)}")


if __name__ == "__main__":
    main()
