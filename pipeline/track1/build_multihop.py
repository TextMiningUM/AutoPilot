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
import json, random, argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths, load_jsonl, clean as _clean

paths = AgentPaths.vhf()
W = paths.workspace
CACHE = paths.cache_dir

TRACES_FILE = CACHE / "vhf_reasoning_traces.jsonl"
GOLD_FILE   = paths.gold_file
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


def concept_signature(trace: dict) -> set[str]:
    """Concepts/channels/prowords/regulations mentioned in a trace, used to pair related traces."""
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
    """Condense one reasoning trace into a short fluent-prose summary for multi-hop answers."""
    t = trace.get("trace") or {}
    sentences: list[str] = []
    situation = _clean(t.get("situation"))
    if situation:
        sentences.append(situation[:1].upper() + situation[1:] + ".")
    procs = t.get("procedures") or []
    if procs:
        acts = [_clean(p.get("action", "")) for p in procs[:4]]
        acts = [a for a in acts if a]
        if acts:
            joined = acts[0] if len(acts) == 1 else "; then ".join(a[:1].lower() + a[1:] for a in acts)
            sentences.append(f"The steps are: {joined}.")
    channels = t.get("channels") or []
    if channels:
        sentences.append(f"Use {', '.join(str(c) for c in channels)}.")
    prowords = t.get("prowords_used") or []
    if prowords:
        sentences.append(f"Use the prowords {', '.join(prowords)}.")
    kf = t.get("key_facts") or []
    if kf:
        sentences.append(_clean(kf[0]) + ".")
    return " ".join(sentences)


def compose_multihop_question(concept: str, tr_a: dict, tr_b: dict) -> str:
    """Combine one question seed from each of two related traces into one cross-source question."""
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
    """Compose the combined answer citing both sources' summaries and a merged recommendation."""
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


def main() -> None:
    """CLI entry point: pair related traces by shared concepts and write multi-hop Q&A rows."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-file", type=str, nargs="+", default=[str(TRACES_FILE)],
                    help="one or more traces files -- pass both protocol and conversation "
                         "traces to get cross-track multi-hop pairs")
    ap.add_argument("--out-file", type=str, default=str(OUT_FILE))
    ap.add_argument("--extra-gold-file", type=str, default=None,
                    help="optional 2nd held-out file to also filter against (e.g. vhf_colreg_scenarios.json)")
    ap.add_argument("--extra-gold-key", type=str, default="question")
    args = ap.parse_args()

    out_file = Path(args.out_file)
    traces: list[dict] = []
    for tf in args.traces_file:
        traces.extend(t for t in load_jsonl(Path(tf))
                       if t.get("trace") and not t.get("skip") and not t.get("error"))
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
    gold_questions = [g["question"] for g in gold]
    if args.extra_gold_file:
        extra = json.loads(Path(args.extra_gold_file).read_text(encoding="utf-8"))
        gold_questions += [g[args.extra_gold_key] for g in extra if g.get(args.extra_gold_key)]
        print(f"  + {len(extra)} extra held-out questions from {args.extra_gold_file}")
    gold_embs = model.encode(gold_questions,
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

    with out_file.open("w", encoding="utf-8") as f:
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
    print(f"Wrote {out_file.name}  ({out_file.stat().st_size/1024:.1f} KB)  rows={len(kept)}")


if __name__ == "__main__":
    main()
