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
import json, random, argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths, load_jsonl, clean as _clean, cap as _cap, decap as _decap, steps_sentence as _steps_sentence

paths = AgentPaths.vhf()
W = paths.workspace
CACHE = paths.cache_dir

TRACES_FILE = CACHE / "vhf_reasoning_traces.jsonl"
GOLD_FILE   = paths.gold_file
OUT_FILE    = CACHE / "vhf_reflection.jsonl"

CONTAM_THRESH = 0.85
RNG = random.Random(11)

SYSTEM = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "For each question, produce a Draft answer, then a Critique that checks the Draft "
    "against safety-critical requirements (steps, channels, prowords, regulations, warnings), "
    "then a Refined answer that fixes any omissions."
)


def full_answer(trace: dict) -> str:
    """Fluent, natural-language answer (no 'Label: value.' dumps)."""
    sentences: list[str] = []
    situation = _clean(trace.get("situation"))
    if situation:
        sentences.append(_cap(situation) + ".")

    steps_sentence = _steps_sentence(trace.get("procedures") or [])
    if steps_sentence:
        sentences.append(steps_sentence)

    channels = [str(c) for c in (trace.get("channels") or [])]
    if channels:
        chs = [c if c.lower().startswith("channel") else f"Channel {c}" for c in channels]
        joined = chs[0] if len(chs) == 1 else ", ".join(chs[:-1]) + f" and {chs[-1]}"
        sentences.append(f"Use {joined}.")

    prowords = trace.get("prowords_used") or []
    if prowords:
        sentences.append(f"Use the prowords {', '.join(prowords)} as appropriate.")

    warnings = trace.get("warnings") or []
    if warnings:
        sentences.append(f"Be careful: {_decap(_clean(warnings[0]))}.")

    outcomes = trace.get("outcomes") or []
    if outcomes:
        sentences.append(f"Done correctly, {_decap(_clean(outcomes[0]))}.")

    return " ".join(sentences).strip()


def draft_answer(trace: dict, mode: str) -> tuple[str, str]:
    """Return (draft, what_was_dropped). mode in {'skip_step','skip_channels','skip_warning','skip_proword'}.

    Deliberately incomplete on purpose (that's the point of the exercise) but
    still fluent prose, not a telegraphic label dump.
    """
    sentences: list[str] = []
    dropped = ""
    procs = trace.get("procedures") or []
    channels = [str(c) for c in (trace.get("channels") or [])]
    prowords = trace.get("prowords_used") or []
    warnings = trace.get("warnings") or []

    situation = _clean(trace.get("situation"))
    if situation:
        sentences.append(_cap(situation) + ".")

    if mode == "skip_step" and len(procs) >= 2:
        drop_idx = RNG.randint(1, len(procs) - 1)  # avoid dropping step 1
        kept = [p for i, p in enumerate(procs) if i != drop_idx]
        dropped_p = procs[drop_idx]
        sentences.append(_steps_sentence(kept))
        dropped = f"the step '{_clean(dropped_p.get('action', ''))}'"
        if channels:
            chs = [c if c.lower().startswith("channel") else f"Channel {c}" for c in channels]
            sentences.append(f"Use {chs[0] if len(chs) == 1 else ', '.join(chs[:-1]) + f' and {chs[-1]}'}.")
        if prowords:
            sentences.append(f"Use the prowords {', '.join(prowords)} as appropriate.")
    elif mode == "skip_channels" and channels and procs:
        sentences.append(_steps_sentence(procs))
        dropped = f"the channel numbers ({', '.join(channels)})"
        if prowords:
            sentences.append(f"Use the prowords {', '.join(prowords)} as appropriate.")
    elif mode == "skip_warning" and warnings and procs:
        sentences.append(_steps_sentence(procs))
        if channels:
            chs = [c if c.lower().startswith("channel") else f"Channel {c}" for c in channels]
            sentences.append(f"Use {chs[0] if len(chs) == 1 else ', '.join(chs[:-1]) + f' and {chs[-1]}'}.")
        if prowords:
            sentences.append(f"Use the prowords {', '.join(prowords)} as appropriate.")
        dropped = f"the safety warning ('{_clean(warnings[0])}')"
    elif mode == "skip_proword" and prowords and procs:
        sentences.append(_steps_sentence(procs))
        if channels:
            chs = [c if c.lower().startswith("channel") else f"Channel {c}" for c in channels]
            sentences.append(f"Use {chs[0] if len(chs) == 1 else ', '.join(chs[:-1]) + f' and {chs[-1]}'}.")
        dropped = f"the required prowords ({', '.join(prowords)})"
    else:
        return "", ""
    return " ".join(s for s in sentences if s).strip(), dropped


def critique(dropped: str) -> str:
    return f"Reviewing the Draft: it omits {dropped}. That is safety-critical and must be included."


def choose_mode(trace: dict) -> str | None:
    """Pick which kind of flawed-draft mode (skip_step/skip_channels/skip_warning/skip_proword)
    this trace has enough material to build, or None if none apply."""
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


def main() -> None:
    """CLI entry point: build draft/critique/full-answer reflection rows from reasoning traces."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-file", type=str, default=str(TRACES_FILE))
    ap.add_argument("--out-file", type=str, default=str(OUT_FILE))
    ap.add_argument("--extra-gold-file", type=str, default=None,
                    help="optional 2nd held-out file to also filter against (e.g. vhf_colreg_scenarios.json)")
    ap.add_argument("--extra-gold-key", type=str, default="question")
    args = ap.parse_args()

    traces_file = Path(args.traces_file)
    out_file    = Path(args.out_file)

    traces = [t for t in load_jsonl(traces_file)
              if t.get("trace") and not t.get("skip") and not t.get("error")]
    print(f"Usable traces: {len(traces)}")

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

    with out_file.open("w", encoding="utf-8") as f:
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
    print(f"Wrote {out_file.name}  ({out_file.stat().st_size/1024:.1f} KB)  rows={len(kept)}")


if __name__ == "__main__":
    main()
