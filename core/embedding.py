"""Single source of truth for the sentence-embedding model and every cosine-similarity
threshold tuned against it -- used for corpus/KG/PG retrieval, contamination filtering,
dedup, and PG canonicalization across both domains.

History: replaced all-MiniLM-L6-v2 (384d) with BAAI/bge-base-en-v1.5 (768d) 2026-09-18
after a calibrated recall@k diagnostic showed bge genuinely retrieves better on this
corpus (see /memories/repo/pipeline-notes.md). Replaced bge-base-en-v1.5 (768d) with
BAAI/bge-large-en-v1.5 (1024d) 2026-09-22 (RAG rebuild Phase A4) for corpus v2 (CHIRP +
Leo MOOS cases + marginal incidents added). Absolute cosine thresholds are NOT portable
between embedding models -- each model has a different random-pair "noise floor" -- so
every threshold below was re-derived by matching the OLD model's own threshold's
percentile-rank within its own random-pair similarity distribution, then reading off the
NEW model's value at that SAME percentile (same relative strictness, new absolute
scale) -- see pipeline/ingest/calibrate_embedder_thresholds.py and
Data/OOW/OOW_Agents_Training/embedder_recalibration_bge_large.json for the measured
values. If the embedder is ever changed again, ALL of these must be recalibrated the
same way, not just copied over. Callers must read the embedding dimension from the
model itself (SentenceTransformer.get_sentence_embedding_dimension()) -- never hardcode
768 or 1024.
"""

EMBEDDER_MODEL = "BAAI/bge-large-en-v1.5"

# bge's asymmetric convention: queries get this instruction prefix, passages/corpus texts
# do not. Only applies to actual retrieval queries (kg_retrieve's `query` arg) -- NOT to
# contamination/dedup/canonicalization comparisons, which compare same-type texts symmetrically.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Question<->question contamination filter (training Q&A vs held-out eval Q&A).
# Was 0.85 under MiniLM, 0.90 under bge-base.
CONTAM_THRESH = 0.883

# Question<->question dedup among newly-generated training rows. Was 0.92 under MiniLM,
# 0.93 under bge-base.
DEDUP_THRESH = 0.944

# Procedure-step-label<->label greedy clustering threshold for PG node canonicalization.
# Was 0.80 under MiniLM, 0.89 under bge-base.
CANON_THRESH = 0.893

# Free text -> PG-node-label matching (online PG retrieval/guidance rendering).
# Was 0.72 under MiniLM, 0.84 under bge-base.
MATCH_THRESH = 0.861

# Loose best-effort anchor-node selection in render_guidance() (deliberately looser than
# MATCH_THRESH -- almost always finds a plausible anchor rather than strictly matching).
# Was 0.35 under MiniLM, 0.65 under bge-base.
ANCHOR_THRESH = 0.660
