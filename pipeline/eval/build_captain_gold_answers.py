"""Captain Track 1 held-out eval -- captain_gold_answers.json (design_captain_missions.md
Sec 16, mirrors vhf_gold_answers.json / chiefengineer_gold_answers.json's shape).

VHF/OOW's gold Q&A sets were format-converted from an externally-acquired exam/question
bank (SRC exam license content for VHF; a pre-existing 500-question COLREG bank for OOW,
see pipeline/ingest/build_oow_gold_norm.py) -- no equivalent external Captain exam bank
exists. Chief Engineer (the other brand-new domain built this session,
Data/ChiefEngineer/ChiefEngineer_Eval/chiefengineer_gold_answers.json) set the real
precedent instead: a curated set grounded directly in the REAL ingested source documents,
each row carrying expected_points (a factual checklist) and a source_citation.

This script builds that same shape ENTIRELY DETERMINISTICALLY (no LLM call, no
hallucination risk) by reformatting pipeline/track1/extract_captain_reasoning.py's own
already-extracted, source-grounded structured fields -- situation/key_facts/procedures/
question_seeds were themselves extracted from the real chunk text by Claude (the "stronger
teacher model", Sec 16.4); this script performs no further generation, only diverse
selection and fluent-prose reassembly of fields that already exist.

Selection is diverse-by-construction: at most --max-per-source traces per source_file (no
single CHIRP issue or BMP5 section can dominate), richest-content-first within each
source_file (ranked by key_facts count), fully deterministic (no random sampling) so
reruns without a NEW extraction are byte-reproducible.

NEVER used for training -- this IS the held-out file every future Captain training-data
builder must filter against (cosine >= 0.85), per copilot-instructions.md's mandatory
contamination-filtering rule.

Safe to run locally (pure JSON I/O, no GPU/API key needed).

Run with: python -m pipeline.eval.build_captain_gold_answers [--n 200] [--max-per-source 3]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from core import AgentPaths
from core.prose import cap, clean, steps_sentence

paths = AgentPaths.captain()
TRACES_FILE = paths.cache_dir / "captain_reasoning_traces.jsonl"
OUT_FILE = paths.gold_file

DEFAULT_N = 200
DEFAULT_MAX_PER_SOURCE = 3


def load_usable_traces() -> list[dict]:
    """One row per successfully-extracted reasoning trace (skips boilerplate/error rows)."""
    rows = []
    with TRACES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("trace") and not row.get("skip") and not row.get("error"):
                rows.append(row)
    return rows


def select_diverse(rows: list[dict], n: int, max_per_source: int) -> list[dict]:
    """Deterministic diverse selection: group by source_file, richest-content-first within
    each group (more key_facts = more grounded detail to quote), round-robin across
    source_files so no single document dominates the gold set."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source_file"]].append(r)
    for src in by_source:
        by_source[src].sort(key=lambda r: (-len(r["trace"].get("key_facts") or []), r["chunk_id"]))
        by_source[src] = by_source[src][:max_per_source]

    selected: list[dict] = []
    sources = sorted(by_source.keys())
    round_idx = 0
    while len(selected) < n:
        added = False
        for src in sources:
            if round_idx < len(by_source[src]):
                selected.append(by_source[src][round_idx])
                added = True
                if len(selected) >= n:
                    break
        if not added:
            break
        round_idx += 1
    return selected


def build_gold_answer(trace: dict) -> str:
    """Fluent prose, never a telegraphic label:value dump (copilot-instructions' hard
    rule) -- situation + key facts + (if present) the correct procedure, reassembled from
    the trace's own already-extracted, source-grounded sentences."""
    parts = [clean(trace.get("situation", ""))]
    for kf in trace.get("key_facts") or []:
        parts.append(clean(kf))
    sentence = ". ".join(p for p in parts if p) + "."
    procs = trace.get("procedures") or []
    if procs:
        sentence += " " + steps_sentence(procs)
    return sentence


def build_expected_points(trace: dict) -> list[str]:
    """A factual checklist for grading -- the trace's own key_facts plus each correct
    procedure action, exactly the chiefengineer_gold_answers.json convention."""
    points = list(trace.get("key_facts") or [])
    for p in trace.get("procedures") or []:
        action = clean(p.get("action", ""))
        if action:
            points.append(cap(action))
    return points


def build_source_citation(row: dict, trace: dict) -> str:
    cites = trace.get("channels") or trace.get("regulations") or []
    if cites:
        return f"{row['source_file']}, {', '.join(cites[:3])}"
    return row["source_file"]


def build_row(row: dict, idx: int) -> dict | None:
    """One gold_answers.json row, or None if this trace lacks enough material (no
    question_seeds, or no expected_points to grade against)."""
    trace = row["trace"]
    seeds = trace.get("question_seeds") or []
    if not seeds:
        return None
    question = (seeds[0].get("text") or "").strip()
    if not question:
        return None
    expected_points = build_expected_points(trace)
    if not expected_points:
        return None
    return {
        "id": f"captain_{idx:04d}",
        "chunk_id": row["chunk_id"],
        "source_file": row["source_file"],
        "chapter_title": row.get("chapter_title", ""),
        "type": "Theory",
        "question": question,
        "gold_answer": build_gold_answer(trace),
        "expected_points": expected_points,
        "source_citation": build_source_citation(row, trace),
        "mapped_event_type": trace.get("mapped_event_type"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="target number of gold Q&A rows")
    ap.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE,
                    help="cap on rows drawn from any single source_file (keeps the set diverse)")
    args = ap.parse_args()

    rows = load_usable_traces()
    print(f"Usable traces: {len(rows)}")
    selected = select_diverse(rows, args.n, args.max_per_source)
    print(f"Selected: {len(selected)} (target {args.n}, max {args.max_per_source}/source)")

    out, skipped = [], 0
    for i, row in enumerate(selected, 1):
        built = build_row(row, i)
        if built is None:
            skipped += 1
            continue
        out.append(built)

    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written: {len(out)} rows ({skipped} skipped -- no question_seeds/expected_points)")
    print(f"Saved: {OUT_FILE}")

    src_counts = Counter(r["source_file"] for r in out)
    print(f"\nSource files represented: {len(src_counts)}")
    event_counts = Counter(r["mapped_event_type"] for r in out)
    print(f"mapped_event_type distribution: {dict(event_counts)}")


if __name__ == "__main__":
    main()
