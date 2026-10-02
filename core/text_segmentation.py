"""Shared text-segmentation helpers used by every text-to-JSON ingest script
(build_vhf_json.py/build_oow_json.py/build_chirp_json.py/screen_incidents.py, and
future Captain parsers) plus the shared chunker (pipeline/ingest/build_rag.py).

Two independent concerns, both now STANDARD for all text (PDF/TXT/MD) processed into
JSON, regardless of domain -- see Docs/rag_chunking_design_and_verification.md for the
full design writeup, literature survey, and verification:

  1. De-hyphenation (join_hyphenated_linebreaks) -- a word a PDF's own line-wrapping
     broke across a line/page boundary ("contin-\\nued") should read as one word
     before any heading/sentence-boundary detection runs on it.
  2. Embedding-based topic-boundary detection (semantic_split_sentence_indices) --
     Hearst (1997) TextTiling's lexical-cohesion depth-score method, adapted to use
     sentence embeddings instead of bag-of-words vectors (the same idea GraphSeg,
     Glavas et al. 2016, and practitioner "semantic chunking" tooling use). Detects a
     topic shift WITHIN what would otherwise be treated as one parsed section, and is
     reused to decide whether two ADJACENT sections are coherent enough to merge into
     one RAG chunk.

Pure algorithmic module -- no sentence-transformers import here, so every function is
fast to unit-test with small synthetic embedding arrays (no model/GPU/network needed).
Callers (build_rag.py) supply real embeddings from the SentenceTransformer they
already load. Local-safe: no training, no GPU, pure Python + numpy.
"""
from __future__ import annotations

import re

import numpy as np

# A hyphen directly before a single line break, immediately followed by a lowercase
# letter -- e.g. "contin-\nued" -> "continued". Deliberately does NOT touch a genuine
# compound hyphen followed by more text on the SAME line ("well-known") or a hyphen
# before an upper-case word start (far more likely a real end-of-sentence hyphen/dash
# followed by a new, separately-capitalized line than a broken word).
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(?=[a-z])")


def join_hyphenated_linebreaks(text: str) -> str:
    """Join a word that a PDF's own line-wrapping broke across a line/page boundary
    ("contin-\\nued" -> "continued"). Call this BEFORE any heading/sentence-boundary
    detection, and before collapsing newlines to spaces -- a hyphen-broken word can
    otherwise defeat both."""
    return _HYPHEN_LINEBREAK_RE.sub(r"\1", text)


# Shared sentence splitter -- previously duplicated independently in build_rag.py's
# _split_oversized_section and pg_guidance.py's split_step_sentences. One copy here now.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences on [.!?] boundaries. Good enough for the prose this
    project ingests (regulation/guide/procedure text) -- not a full sentence-boundary
    disambiguator (e.g. "Rule 1." mid-sentence would still split) -- acceptable since a
    stray extra split only means a slightly-too-fine topic-segmentation window, never a
    correctness problem for the callers below."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def cosine_similarities(embeddings: np.ndarray) -> list[float]:
    """Cosine similarity between every CONSECUTIVE pair of rows in `embeddings`
    (length N-1 for N embeddings, [] if fewer than 2). Assumes rows are already
    L2-normalized (every embedding call in this project uses normalize_embeddings=True,
    see core/embedding.py) -- a plain dot product is then exactly the cosine similarity."""
    if len(embeddings) < 2:
        return []
    return [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]


def depth_scores(similarities: list[float]) -> list[float]:
    """Hearst (1997) TextTiling's 'depth score' at each internal gap: how far
    similarity dips below the highest point on EACH side, summed -- a deep, narrow
    valley (genuine topic shift) scores high even if the surrounding similarities are
    only moderately higher; a shallow dip (ordinary sentence-to-sentence variation)
    scores near zero. Uses the running peak over the whole prefix/suffix on each side
    (a simplified variant of Hearst's original "nearest local peak" -- monotonic and
    robust to picking a spurious nearby ripple as the peak). Using embeddings here
    instead of Hearst's original bag-of-words vectors is the "embedding TextTiling"
    approach practitioner tooling (LangChain's SemanticChunker, LlamaIndex's
    SemanticSplitterNodeParser) and GraphSeg (Glavas et al. 2016) converge on as fast
    + reliable without training data -- see Docs/rag_chunking_design_and_verification.md."""
    n = len(similarities)
    scores = [0.0] * n
    for i in range(n):
        left_peak = max(similarities[: i + 1])
        right_peak = max(similarities[i:])
        scores[i] = max(0.0, (left_peak - similarities[i]) + (right_peak - similarities[i]))
    return scores


def percentile(values: list[float], pct: float) -> float:
    """pct-th percentile of `values` (0.0 for an empty list) -- thin wrapper so callers
    don't need their own numpy import just for this."""
    return float(np.percentile(values, pct)) if values else 0.0


def semantic_split_sentence_indices(
    embeddings: np.ndarray, *, percentile_cutoff: float = 85.0, min_sentences: int = 6,
    min_depth_score: float = 0.05,
) -> list[int]:
    """Sentence indices i (meaning: split BEFORE sentence i, i.e. between sentence i-1
    and sentence i) where embedding similarity dips into a genuine topic-shift valley.

    Adaptive, not a fixed cosine cutoff: only the TOP (100-percentile_cutoff)% deepest
    valleys in THIS text's own depth-score distribution count as CANDIDATE boundaries --
    the same adaptive-threshold idea LangChain/LlamaIndex's semantic chunkers use, since
    a fixed absolute cosine threshold doesn't transfer between a terse procedure card
    and a discursive incident narrative. A percentile rank ALONE isn't enough, though --
    on a short or genuinely single-topic text, the single largest ripple in an otherwise
    flat similarity curve still ranks in the "top 15%" by construction, even though it's
    just noise (confirmed by a real test failure during development: 8 near-identical
    sentence embeddings with only float noise still produced one "boundary" under a pure
    percentile rule). `min_depth_score` is therefore a SECOND, absolute requirement --
    cheap to pick since genuine topic shifts widen the embedding gap by an order of
    magnitude more than embedding noise does (0.05 is far below a real shift's depth
    score, and far above the noise floor measured in testing).

    Returns [] (no split) if there aren't enough sentences for the depth-score's "peak
    on both sides" logic to mean anything (`min_sentences` default 6)."""
    n_sentences = len(embeddings)
    if n_sentences < min_sentences:
        return []
    sims = cosine_similarities(embeddings)
    scores = depth_scores(sims)
    if not scores or max(scores) <= 0.0:
        return []
    threshold = max(percentile(scores, percentile_cutoff), min_depth_score)
    return [i + 1 for i, d in enumerate(scores) if d >= threshold]
