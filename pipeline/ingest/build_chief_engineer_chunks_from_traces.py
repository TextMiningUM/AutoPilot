"""Build RAG-chunk-shaped records directly from the Chief Engineer known-issues reasoning
traces (pipeline/track1/extract_chief_engineer_known_issues.py), NOT from raw manual text --
there is no build_chief_engineer_json.py yet (Phase 0 of design_chief_engineer.md's roadmap
is still open), so this is a legitimate alternative front door into the SAME shared
build_kg.py / build_sft.py (RAG variant) / build_pg.py pipeline, producing the exact file
shapes (chunk_id/document_id/source_file/source_type/chapter_title/section_titles/types/
token_count/n_sections/concepts/topics/pages/text) those UNCHANGED scripts already expect --
never forking them. Each reasoning trace becomes one retrievable chunk (its own fluent prose
situation/key_facts/procedure text), honestly tagged `source_type="chief_engineer_known_issue"`.

Writes: <cache>/chiefengineer_rag_chunks.json, _rag_embeddings.npy, _rag_chunk_ids.json
(same filenames build_kg.py / build_sft.py / build_pg.py already read via their own
AgentPaths.from_env()-derived `_PFX` prefix -- no changes needed to those scripts).

Safe to run LOCALLY (CPU sentence-embedder only, no GPU/API key).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths, EMBEDDER_MODEL, load_jsonl

paths = AgentPaths.from_env()
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

DEFAULT_TRACES_FILE = CACHE / "chief_engineer_known_issues_traces.jsonl"
CHUNKS_FILE = CACHE / f"{_PFX}_rag_chunks.json"
EMBS_FILE = CACHE / f"{_PFX}_rag_embeddings.npy"
IDS_FILE = CACHE / f"{_PFX}_rag_chunk_ids.json"


def _chunk_text(trace: dict) -> str:
    """Fluent prose built from one trace's fields -- same fields format_direct_answer()
    in build_sft.py already renders into prose, reused here as the embeddable chunk text."""
    parts: list[str] = []
    situation = (trace.get("situation") or "").strip()
    if situation:
        parts.append(situation)
    for p in trace.get("procedures") or []:
        action = (p.get("action") or "").strip()
        if action:
            parts.append(action)
    parts.extend(kf for kf in (trace.get("key_facts") or []) if kf)
    parts.extend(w for w in (trace.get("warnings") or []) if w)
    return " ".join(parts).strip()


def build_chunks(traces_file: Path) -> list[dict]:
    rows = [r for r in load_jsonl(traces_file) if r.get("trace") and not r.get("skip") and not r.get("error")]
    chunks = []
    for row in rows:
        trace = row["trace"]
        text = _chunk_text(trace)
        if len(text) < 30:
            continue
        known_issue = trace.get("known_issue") or {}
        chunks.append({
            "chunk_id": row["chunk_id"],
            "document_id": row.get("source_file", "unknown"),
            "source_file": row.get("source_file", "unknown"),
            "source_type": "chief_engineer_known_issue",
            "chapter_title": row.get("chapter_title", ""),
            "section_titles": [row.get("chapter_title", "")],
            "types": ["known_issue"],
            "token_count": max(1, len(text.split())),
            "n_sections": 1,
            "concepts": row.get("chunk_concepts", []),
            "topics": [known_issue["system"]] if known_issue.get("system") else [],
            "pages": [row["page"]] if row.get("page") else [],
            "text": text,
        })
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-file", type=str, default=str(DEFAULT_TRACES_FILE),
                    help="known-issues reasoning traces jsonl to build chunks from")
    args = ap.parse_args()

    chunks = build_chunks(Path(args.traces_file))
    print(f"Built {len(chunks)} chunks from {args.traces_file}")

    print(f"Embedding with {EMBEDDER_MODEL}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    embs = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                           batch_size=64, show_progress_bar=True)

    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(EMBS_FILE, np.asarray(embs, dtype=np.float32))
    IDS_FILE.write_text(json.dumps([c["chunk_id"] for c in chunks], ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {CHUNKS_FILE.name}, {EMBS_FILE.name}, {IDS_FILE.name}")


if __name__ == "__main__":
    main()
