"""Chief Engineer memory/retrieval (design_chief_engineer.md Sec 4).

Unlike pipeline/captain_memory.py (deliberately a placeholder -- "no real corpus exists
yet"), this module does REAL small-scale embedding retrieval over
Data/ChiefEngineer/ChiefEngineer_Agents_Training/chief_engineer_known_issues_traces.jsonl
using the project's shared embedder convention (core/embedding.py). It scales
automatically as pipeline.track1.extract_chief_engineer_known_issues produces more rows --
no call-site changes needed (same "swap later without changing shape" pattern as
captain_memory.py, except real retrieval exists from day one over however many rows
currently exist). Falls back to a single honest placeholder hit if the traces file is
missing or empty.

Safe to run LOCALLY -- downloads/loads the shared CPU sentence-embedding model only
(BAAI/bge-large-en-v1.5), no GPU/API key required. First call is slow (model load);
results are cached in-process via lru_cache.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core import AgentPaths, EMBEDDER_MODEL, QUERY_PREFIX


@dataclass(frozen=True)
class MemoryHit:
    source: str
    text: str
    score: float
    is_placeholder: bool = False


TRACES_FILE: Path = AgentPaths.chief_engineer().cache_dir / "chief_engineer_known_issues_traces.jsonl"

_NO_CORPUS_HIT = MemoryHit(
    source="(no corpus yet)",
    text=(
        "No Chief Engineer known-issues traces found -- run "
        "`python -m pipeline.track1.extract_chief_engineer_known_issues` first. "
        "This is an honest placeholder, not retrieved content."
    ),
    score=0.0, is_placeholder=True,
)


def _load_traces() -> list[dict]:
    if not TRACES_FILE.exists():
        return []
    rows = []
    with TRACES_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _trace_text(row: dict) -> str:
    """The retrievable text for one trace row: situation + key facts (short, dense --
    good for both embedding and on-screen display)."""
    trace = row.get("trace", {})
    parts = [trace.get("situation") or ""]
    parts += [kf for kf in trace.get("key_facts", []) if kf]
    return " ".join(p for p in parts if p).strip()


def _citation(row: dict) -> str:
    bits = [row.get("source_file", "?")]
    if row.get("chapter_title"):
        bits.append(str(row["chapter_title"]))
    if row.get("page"):
        bits.append(f"p.{row['page']}")
    return " \u2014 ".join(bits)


@lru_cache(maxsize=1)
def _corpus():
    """Lazily load the traces + embed them once per process. Returns None if no traces
    exist yet (caller falls back to _NO_CORPUS_HIT)."""
    rows = [r for r in _load_traces() if _trace_text(r)]
    if not rows:
        return None
    from sentence_transformers import SentenceTransformer
    import numpy as np

    embedder = SentenceTransformer(EMBEDDER_MODEL, device="cpu")
    texts = [_trace_text(r) for r in rows]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embedder, np.asarray(embs), rows


def retrieve(query: str, k: int = 3) -> list[MemoryHit]:
    """Real cosine-similarity retrieval over the known-issues traces corpus. Returns the
    honest "no corpus yet" placeholder (as a list of 1) if no traces exist yet."""
    corpus = _corpus()
    if corpus is None:
        return [_NO_CORPUS_HIT][:k]
    embedder, embs, rows = corpus
    import numpy as np

    q = embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = embs @ q
    order = np.argsort(-scores)[:k]
    return [
        MemoryHit(source=_citation(rows[i]), text=_trace_text(rows[i]), score=float(scores[i]))
        for i in order
    ]
