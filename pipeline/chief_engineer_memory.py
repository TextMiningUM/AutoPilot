"""Chief Engineer memory/retrieval (design_chief_engineer.md Sec 4).

2026-10-03 upgrade: previously did its own from-scratch cosine-similarity retrieval
directly over the raw known-issues traces jsonl (no KG expansion, no procedural-graph
guidance). A concurrent session has since built the proper shared pipeline for this
domain -- `chiefengineer_rag_chunks.json`/`_rag_embeddings.npy`/`_rag_chunk_ids.json`
(pipeline/ingest/build_chief_engineer_chunks_from_traces.py) and `chiefengineer_kg.json`/
`chiefengineer_pg.json` (pipeline/ingest/build_kg.py / build_pg.py, both domain-agnostic
via AgentPaths) -- so this module now reuses that EXACT same hybrid dense+concept-graph
retrieval (`kg_retrieve`, see Basic Simulator/app/agents.py's `_load_retrieval()` for the
established live-app loading pattern this mirrors) plus a procedural-graph guidance block
when the PG file exists, instead of a parallel, strictly weaker duplicate. No reranker
model exists for this domain yet (`_models/ChiefEngineer/` has none) so reranking is
skipped, same graceful-degradation pattern `_load_retrieval()` already uses for OOW.

Falls back to a single honest placeholder hit if the RAG index doesn't exist yet.

Safe to run LOCALLY -- loads the shared CPU sentence-embedding model only
(BAAI/bge-large-en-v1.5), no GPU/API key required. First call is slow (model load);
results are cached in-process via lru_cache.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from core import AgentPaths, EMBEDDER_MODEL
from pipeline.ingest.build_kg import kg_retrieve
from pipeline.ingest.pg_guidance import ProceduralGraph, render_guidance

PATHS = AgentPaths.chief_engineer()
CACHE = PATHS.cache_dir
_PFX = PATHS.domain.lower()

CHUNKS_FILE = CACHE / f"{_PFX}_rag_chunks.json"
EMBS_FILE = CACHE / f"{_PFX}_rag_embeddings.npy"
IDS_FILE = CACHE / f"{_PFX}_rag_chunk_ids.json"
KG_FILE = CACHE / f"{_PFX}_kg.json"
PG_FILE = CACHE / f"{_PFX}_pg.json"

DENSE_N = 20


@dataclass(frozen=True)
class MemoryHit:
    source: str
    text: str
    score: float
    is_placeholder: bool = False


_NO_CORPUS_HIT = MemoryHit(
    source="(no corpus yet)",
    text=(
        "No Chief Engineer RAG index found -- run "
        "`python -m pipeline.ingest.build_chief_engineer_chunks_from_traces` first. "
        "This is an honest placeholder, not retrieved content."
    ),
    score=0.0, is_placeholder=True,
)


@lru_cache(maxsize=1)
def _corpus():
    """Lazily load the RAG chunks/embeddings/KG (+ PG, if built) once per process.
    Returns None if the index hasn't been built yet (caller falls back to
    _NO_CORPUS_HIT)."""
    if not (CHUNKS_FILE.exists() and EMBS_FILE.exists() and IDS_FILE.exists() and KG_FILE.exists()):
        return None
    from sentence_transformers import SentenceTransformer

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    embs = np.load(EMBS_FILE)
    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    kg = json.loads(KG_FILE.read_text(encoding="utf-8"))
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    embedder = SentenceTransformer(EMBEDDER_MODEL, device="cpu")
    pg = ProceduralGraph(PG_FILE, embedder) if PG_FILE.exists() else None
    return embedder, embs, ids, kg, chunk_by_id, pg


def _citation(chunk: dict) -> str:
    bits = [chunk.get("source_file", "?")]
    if chunk.get("chapter_title"):
        bits.append(str(chunk["chapter_title"]))
    if chunk.get("pages"):
        bits.append(f"p.{chunk['pages'][0]}")
    return " \u2014 ".join(bits)


def retrieve(query: str, k: int = 3) -> list[MemoryHit]:
    """Hybrid dense + concept-graph retrieval (kg_retrieve) over the known-issues RAG
    index, plus a trailing procedural-graph guidance hit when PG is available. Returns
    the honest "no corpus yet" placeholder (as a list of 1) if the index hasn't been
    built yet."""
    corpus = _corpus()
    if corpus is None:
        return [_NO_CORPUS_HIT][:k]
    embedder, embs, ids, kg, chunk_by_id, pg = corpus

    hits, _q_cons, _expanded = kg_retrieve(query, embedder, embs, ids, kg, k=k, dense_n=DENSE_N)
    out = [
        MemoryHit(
            source=_citation(chunk_by_id[h["chunk_id"]]),
            text=chunk_by_id[h["chunk_id"]]["text"],
            score=h["score"],
        )
        for h in hits
    ]

    if pg is not None:
        guidance = render_guidance(query, pg)
        if guidance:
            out.append(MemoryHit(source="(procedure guidance)", text=guidance, score=0.0))

    return out
