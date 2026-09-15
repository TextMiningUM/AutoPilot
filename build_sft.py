"""Tutorial 13 § 9 (step 2) — build direct + CoT + RAG SFT datasets from reasoning traces.

Deterministic (no LLM). Reads _cache/vhf_reasoning_traces.jsonl and emits:

  _cache/vhf_sft_direct.jsonl   — Q -> concise answer built from key_facts+procedures
  _cache/vhf_sft_cot.jsonl      — Q -> chain-of-thought reasoning -> answer
  _cache/vhf_sft_rag.jsonl      — (Q + top-3 KG context) -> answer

Contamination filter against 540 gold questions (cos >= 0.85 dropped).
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from build_kg import kg_retrieve

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"

TRACES_FILE = CACHE / "vhf_reasoning_traces.jsonl"
CHUNKS_FILE = CACHE / "vhf_rag_chunks.json"
EMBS_FILE   = CACHE / "vhf_rag_embeddings.npy"
IDS_FILE    = CACHE / "vhf_rag_chunk_ids.json"
KG_FILE     = CACHE / "vhf_kg.json"
GOLD_FILE   = W / "Data" / "VHF" / "VHF_Eval" / "vhf_gold_answers.json"

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


def _clean(s: str | None) -> str:
    return (s or "").strip().rstrip(". ").strip()


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _decap(s: str) -> str:
    if not s:
        return s
    first_word = s.split(" ", 1)[0]
    if len(first_word) > 1 and first_word.isupper():
        return s  # don't mangle acronyms like VHF, GMDSS, ITU
    return s[:1].lower() + s[1:]


def format_direct_answer(trace: dict, angle: str) -> str:
    """Fluent, natural-language answer (no 'Label: value.' dumps) so the
    fine-tuned model learns to speak in plain prose instead of telegraphic
    style -- this previously contaminated every downstream training stage.
    """
    procs       = trace.get("procedures") or []
    kf          = [_clean(k) for k in (trace.get("key_facts") or []) if _clean(k)]
    channels    = trace.get("channels") or []
    constraints = [_clean(c) for c in (trace.get("constraints") or []) if _clean(c)]
    outcomes    = [_clean(o) for o in (trace.get("outcomes") or []) if _clean(o)]
    warnings    = [_clean(w) for w in (trace.get("warnings") or []) if _clean(w)]

    if angle == "how" and procs:
        steps = [_clean(p.get("action", "")) for p in procs]
        steps = [s for s in steps if s]
        if not steps:
            return "Information not available in this excerpt."
        if len(steps) == 1:
            answer = _cap(steps[0]) + "."
        else:
            connectors = ["First", "Then", "Next", "After that", "Finally"]
            pieces = [f"{connectors[i] if i < len(connectors) else 'Then'}, {_decap(s)}"
                      for i, s in enumerate(steps)]
            answer = "; ".join(pieces) + "."
        if outcomes:
            answer += f" Do this correctly and {_decap(outcomes[0])}."
        if warnings:
            answer += f" Just be careful: {_decap(warnings[0])}."
        return answer

    if angle == "when":
        parts = []
        trig = _clean(trace.get("trigger"))
        if trig:
            parts.append(trig)
        parts.extend(constraints[:2])
        if not parts and kf:
            parts.append(kf[0])
        if not parts:
            return ""
        answer = _cap(parts[0])
        if len(parts) > 1:
            answer += ", provided that " + " and ".join(_decap(p) for p in parts[1:])
        return answer + "."

    if angle == "which" and channels:
        chs = [str(c) if str(c).lower().startswith("channel") else f"Channel {c}" for c in channels]
        answer = chs[0] + "." if len(chs) == 1 else ", ".join(chs[:-1]) + f" and {chs[-1]}."
        if kf:
            answer += f" {_cap(kf[0])}."
        return answer

    if angle == "why":
        reasons = [_clean(p.get("why", "")) for p in procs[:2] if p.get("why")]
        reasons.extend(kf[:2])
        reasons = [r for r in reasons if r]
        if not reasons:
            return ""
        if len(reasons) == 1:
            return f"Because {_decap(reasons[0])}."
        return "Because " + ", and because ".join(_decap(r) for r in reasons) + "."

    # what / who / default
    parts = kf[:2] if kf else []
    if not parts and procs:
        act = _clean(procs[0].get("action", ""))
        if act:
            parts = [act]
    if not parts:
        return "Information not available in this excerpt."
    if len(parts) == 1:
        return _cap(parts[0]) + "."
    return _cap(parts[0]) + ", and " + _decap(parts[1]) + "."


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-file", type=str, default=str(TRACES_FILE),
                    help="reasoning traces to build from (default: protocol traces / Track 1)")
    ap.add_argument("--out-prefix", type=str, default="vhf_sft",
                    help="outputs become <prefix>_direct.jsonl / _cot.jsonl / _rag.jsonl / _stats.json")
    ap.add_argument("--extra-gold-file", type=str, default=None,
                    help="optional 2nd held-out file to also filter against (e.g. vhf_colreg_scenarios.json)")
    ap.add_argument("--extra-gold-key", type=str, default="question")
    args = ap.parse_args()

    traces_file = Path(args.traces_file)
    direct_out  = CACHE / f"{args.out_prefix}_direct.jsonl"
    cot_out     = CACHE / f"{args.out_prefix}_cot.jsonl"
    rag_out     = CACHE / f"{args.out_prefix}_rag.jsonl"
    stats_out   = CACHE / f"{args.out_prefix}_stats.json"

    traces = [t for t in load_jsonl(traces_file)
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
    gold_questions = [g["question"] for g in gold]
    if args.extra_gold_file:
        extra = json.loads(Path(args.extra_gold_file).read_text(encoding="utf-8"))
        gold_questions += [g[args.extra_gold_key] for g in extra if g.get(args.extra_gold_key)]
        print(f"  + {len(extra)} extra held-out questions from {args.extra_gold_file}")
    gold_embs = model.encode(gold_questions,
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
    print(f"\nWriting {direct_out.name} and {cot_out.name}...")
    with direct_out.open("w", encoding="utf-8") as fd, cot_out.open("w", encoding="utf-8") as fc:
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
    print(f"Writing {rag_out.name} (KG top-3 context per Q)...")
    with rag_out.open("w", encoding="utf-8") as fr:
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
        "direct_bytes":     direct_out.stat().st_size,
        "cot_bytes":        cot_out.stat().st_size,
        "rag_bytes":        rag_out.stat().st_size,
    }
    stats_out.write_text(json.dumps(stats, indent=2))
    print(f"\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
