"""VHF's existing SFT-multihop + DPO-chosen training data -> case-based RAG chunks.

Mirrors `pipeline/ingest/build_moos_case_rag.py`'s pattern (deterministic, data-derived
content from an existing structured source -> standard per-document JSON so
build_rag.py/build_kg.py pick it up automatically, no changes to those scripts needed) --
per copilot-instructions.md's "reuse before you rebuild".

WHY THIS EXISTS
---------------
VHF's SFT/DPO/compression fine-tuning attempts did not work out and were dropped
(design_vhf_communications.md Sec 9.3) -- the interface runs on base Qwen. But the
TRAINING DATA that was already generated for that abandoned attempt is still real,
useful content: `vhf_multihop.jsonl`/`vhf_colreg_multihop.jsonl` (cross-source
synthesized answers) and the "chosen" (never "rejected" -- that's a deliberately
corrupted negative example, must never leak into RAG as if it were good content) side
of `vhf_dpo_pairs.jsonl`/`vhf_colreg_dpo_pairs.jsonl` (perturbation-robust, complete
answers). Repurposed here as retrievable case-based RAG content instead of wasting it --
per user direction 2026-10-03 ("add the SFT and DPO and other text data we have as
prompt ablation").

Plain SFT files (vhf_sft_direct/cot/rag, vhf_reflection) are deliberately NOT included
here -- their content mostly duplicates either the original rule-text chunks already in
the RAG index or the richer multihop/DPO answers, and reflection's answer additionally
carries a `<think>...</think>` scaffold not meant for direct display. Including them
would just inflate the index with near-duplicates.

DEDUP: DPO files store ~5-6 perturbation-axis ROWS per question (wrong_channel,
wrong_proword, missing_step, ...) that all share the IDENTICAL "chosen" answer --
confirmed by inspection. Deduping by the user question text (across ALL 4 source files
combined) keeps exactly one representative row per distinct question.

Safe to run locally -- deterministic, no LLM calls, no GPU.

Reads:   <cache>/vhf_multihop.jsonl, vhf_colreg_multihop.jsonl,
         vhf_dpo_pairs.jsonl, vhf_colreg_dpo_pairs.jsonl
Writes:  <json_dir>/vhf_training_examples.json

Run with: python -m pipeline.ingest.build_vhf_training_case_rag
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from core import AgentPaths
from core.io import load_jsonl

paths = AgentPaths.vhf()
CACHE = paths.cache_dir
OUT_FILE = paths.json_dir / "vhf_training_examples.json"

# (file, kind) -- kind picks how to extract (question, answer) below.
SOURCES = [
    (CACHE / "vhf_multihop.jsonl", "multihop", "Track 1 (rules/knowledge)"),
    (CACHE / "vhf_colreg_multihop.jsonl", "multihop", "Track 2 (COLREG conversations)"),
    (CACHE / "vhf_dpo_pairs.jsonl", "dpo", "Track 1 (rules/knowledge)"),
    (CACHE / "vhf_colreg_dpo_pairs.jsonl", "dpo", "Track 2 (COLREG conversations)"),
]


def _question_answer(row: dict, kind: str) -> tuple[str, str] | None:
    if kind == "multihop":
        msgs = row.get("messages") or []
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    else:  # dpo -- NEVER use "rejected" (a deliberately corrupted negative example)
        prompt = row.get("prompt") or []
        user = next((m["content"] for m in prompt if m["role"] == "user"), None)
        chosen = row.get("chosen") or []
        assistant = chosen[0]["content"] if chosen else None
    if not user or not assistant:
        return None
    return user.strip(), assistant.strip()


def stable_id(*parts: str) -> str:
    return "case_" + hashlib.md5("|".join(parts).encode()).hexdigest()[:8]


def build() -> dict:
    sections_by_track: dict[str, list[dict]] = {}
    seen_questions: set[str] = set()
    n_total, n_kept, n_dup = 0, 0, 0
    for src_file, kind, track_label in SOURCES:
        if not src_file.exists():
            print(f"  [skip] {src_file.name} not found")
            continue
        for row in load_jsonl(src_file):
            n_total += 1
            qa = _question_answer(row, kind)
            if qa is None:
                continue
            question, answer = qa
            if question in seen_questions:
                n_dup += 1
                continue
            seen_questions.add(question)
            n_kept += 1
            sections_by_track.setdefault(track_label, []).append({
                "section_id": stable_id(src_file.name, question),
                "title": question[:80],
                "type": "qa",
                "text": f"Q: {question}\nA: {answer}",
                "concepts": [],
                "topics": [],
                "steps": [],
                "pages": [],
            })

    doc = {
        "document_id": "vhf_training_examples",
        "source_file": "vhf_training_examples (synthesized from vhf_multihop/vhf_dpo_pairs)",
        "source_type": "synthesized_case_rag",
        "publisher": "Auto Pilot (derived from this project's own VHF training-data builders)",
        "language": "en",
        "chapters": [
            {"chapter_id": f"ch_{i}", "title": track_label, "sections": secs}
            for i, (track_label, secs) in enumerate(sections_by_track.items())
        ],
        "parsing_notes": [
            "Case-based RAG content derived from VHF's own (now-unused for training) "
            "SFT-multihop + DPO-chosen data -- see module docstring.",
        ],
        "metadata": {"n_source_rows_seen": n_total, "n_kept": n_kept, "n_deduped": n_dup},
    }
    return doc


def main() -> None:
    paths.json_dir.mkdir(parents=True, exist_ok=True)
    doc = build()
    OUT_FILE.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    n_sections = sum(len(ch["sections"]) for ch in doc["chapters"])
    print(f"Wrote {OUT_FILE} -- {n_sections} case-based sections "
         f"({doc['metadata']['n_kept']} kept, {doc['metadata']['n_deduped']} deduped "
         f"out of {doc['metadata']['n_source_rows_seen']} source rows)")


if __name__ == "__main__":
    main()
