"""Tutorial 13 § 12 — reflection / self-critique training data.

For each trace with >=2 procedure steps or >=3 key_facts, build a triple:

  Q:        seed question
  Draft:    incomplete answer (missing step or missing key fact)
  Critique: names what's missing, referencing the expected_points
  Refined:  full answer (all steps + all key facts)

The reflection message format is a single turn where the assistant produces:

  Draft: <partial answer>
  Critique: The draft omits <expected_point>. It should also address <constraint>.
  Refined: <full answer>

This teaches the model to self-check and repair its own output.
Contamination filter against 540 gold questions.

Output: _cache/vhf_reflection.jsonl
"""
from __future__ import annotations
import json, random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"

TRACES_FILE = CACHE / "vhf_reasoning_traces.jsonl"
GOLD_FILE   = W / "Data" / "VHF" / "VHF_Eval" / "vhf_gold_answers.json"
OUT_FILE    = CACHE / "vhf_reflection.jsonl"

CONTAM_THRESH = 0.85
RNG = random.Random(11)

SYSTEM = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "For each question, produce a Draft answer, then a Critique that checks the Draft "
    "against safety-critical requirements (steps, channels, prowords, regulations, warnings), "
    "then a Refined answer that fixes any omissions."
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue


def full_answer(trace: dict) -> str:
    parts: list[str] = []
    if trace.get("situation"): parts.append(trace["situation"].rstrip("."))
    procs = trace.get("procedures") or []
    if procs:
        steps = " ".join(f"{p.get('step','?')}. {p.get('action','').rstrip('.')}." for p in procs)
        parts.append(f"Steps: {steps}")
    if trace.get("channels"):
        parts.append(f"Channels: {', '.join(str(c) for c in trace['channels'])}.")
    if trace.get("prowords_used"):
        parts.append(f"Prowords: {', '.join(trace['prowords_used'])}.")
    if trace.get("warnings"):
        parts.append(f"Warning: {trace['warnings'][0].rstrip('.')}.")
    if trace.get("outcomes"):
        parts.append(f"Outcome: {trace['outcomes'][0].rstrip('.')}.")
    return " ".join(parts).strip()


def draft_answer(trace: dict, mode: str) -> tuple[str, str]:
    """Return (draft, what_was_dropped). mode in {'skip_step','skip_channels','skip_warning','skip_proword'}."""
    parts: list[str] = []
    dropped = ""
    procs = trace.get("procedures") or []
    channels = trace.get("channels") or []
    prowords = trace.get("prowords_used") or []
    warnings = trace.get("warnings") or []

    if trace.get("situation"): parts.append(trace["situation"].rstrip("."))

    if mode == "skip_step" and len(procs) >= 2:
        drop_idx = RNG.randint(1, len(procs) - 1)  # avoid dropping step 1
        kept = [p for i, p in enumerate(procs) if i != drop_idx]
        dropped_p = procs[drop_idx]
        steps = " ".join(f"{i+1}. {p.get('action','').rstrip('.')}." for i, p in enumerate(kept))
        parts.append(f"Steps: {steps}")
        dropped = f"step '{dropped_p.get('action','').rstrip('.')}'"
        if channels: parts.append(f"Channels: {', '.join(str(c) for c in channels)}.")
        if prowords: parts.append(f"Prowords: {', '.join(prowords)}.")
    elif mode == "skip_channels" and channels and procs:
        steps = " ".join(f"{i+1}. {p.get('action','').rstrip('.')}." for i, p in enumerate(procs))
        parts.append(f"Steps: {steps}")
        dropped = f"the channel numbers ({', '.join(str(c) for c in channels)})"
        if prowords: parts.append(f"Prowords: {', '.join(prowords)}.")
    elif mode == "skip_warning" and warnings and procs:
        steps = " ".join(f"{i+1}. {p.get('action','').rstrip('.')}." for i, p in enumerate(procs))
        parts.append(f"Steps: {steps}")
        if channels: parts.append(f"Channels: {', '.join(str(c) for c in channels)}.")
        if prowords: parts.append(f"Prowords: {', '.join(prowords)}.")
        dropped = f"the safety warning ('{warnings[0].rstrip('.')}')"
    elif mode == "skip_proword" and prowords and procs:
        steps = " ".join(f"{i+1}. {p.get('action','').rstrip('.')}." for i, p in enumerate(procs))
        parts.append(f"Steps: {steps}")
        if channels: parts.append(f"Channels: {', '.join(str(c) for c in channels)}.")
        dropped = f"the required prowords ({', '.join(prowords)})"
    else:
        return "", ""
    return " ".join(parts).strip(), dropped


def critique(dropped: str) -> str:
    return f"Reviewing the Draft: it omits {dropped}. That is safety-critical and must be included."


def choose_mode(trace: dict) -> str | None:
    procs = trace.get("procedures") or []
    channels = trace.get("channels") or []
    prowords = trace.get("prowords_used") or []
    warnings = trace.get("warnings") or []
    options: list[str] = []
    if len(procs) >= 2:    options.append("skip_step")
    if channels and procs: options.append("skip_channels")
    if warnings and procs: options.append("skip_warning")
    if prowords and procs: options.append("skip_proword")
    return RNG.choice(options) if options else None


def main():
    traces = [t for t in load_jsonl(TRACES_FILE)
              if t.get("trace") and not t.get("skip") and not t.get("error")]
    print(f"Usable traces: {len(traces)}")

    print("Loading embedder + gold-Q embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    gold_embs = model.encode([g["question"] for g in gold],
                             normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False)

    rows: list[dict] = []
    for rec in traces:
        t = rec["trace"]
        seeds = t.get("question_seeds") or []
        if not seeds: continue
        mode = choose_mode(t)
        if mode is None: continue
        draft, dropped = draft_answer(t, mode)
        if not draft or not dropped: continue
        refined = full_answer(t)
        if len(refined) < 50 or refined == draft: continue
        q = (seeds[0].get("text") or "").strip()
        if len(q) < 8: continue

        assistant_msg = f"Draft: {draft}\n\nCritique: {critique(dropped)}\n\nRefined: {refined}"
        rows.append({
            "source_chunk_id": rec["chunk_id"],
            "source_file":     rec["source_file"],
            "chapter_title":   rec["chapter_title"],
            "reflection_mode": mode,
            "question":        q,
            "assistant":       assistant_msg,
            "expected_points": t.get("key_facts", []),
        })

    print(f"Raw reflection rows: {len(rows)}")

    print("Contamination check...")
    q_embs = model.encode([r["question"] for r in rows],
                          normalize_embeddings=True, batch_size=128,
                          show_progress_bar=True)
    kept = []
    dropped_ct = 0
    for r, emb in zip(rows, q_embs):
        sim = float(np.max(gold_embs @ emb))
        if sim >= CONTAM_THRESH:
            dropped_ct += 1
            continue
        r["contam_sim"] = round(sim, 3)
        kept.append(r)
    print(f"kept={len(kept)} contam_dropped={dropped_ct}")

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for i, r in enumerate(kept):
            f.write(json.dumps({
                "id":              f"refl_{i:05d}",
                "source_chunk_id": r["source_chunk_id"],
                "source_file":     r["source_file"],
                "chapter_title":   r["chapter_title"],
                "reflection_mode": r["reflection_mode"],
                "expected_points": r["expected_points"],
                "contam_sim":      r["contam_sim"],
                "messages": [
                    {"role": "system",    "content": SYSTEM},
                    {"role": "user",      "content": r["question"]},
                    {"role": "assistant", "content": r["assistant"]},
                ],
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_FILE.name}  ({OUT_FILE.stat().st_size/1024:.1f} KB)  rows={len(kept)}")


if __name__ == "__main__":
    main()
