"""Tutorial 13 § 8 — Knowledge Graph over VHF chunks.

Builds an inverted-index style KG from _cache/vhf_rag_chunks.json:
  - concept  -> chunk_ids   (inverted index)
  - concept  -> concept     (co-occurrence, symmetric, weighted)
  - topic    -> chunk_ids
  - document -> chunk_ids
  - chunk    -> chunk       (adjacency inside chapter, ordered)
  - alias    -> concept     (query-time normalization)

Writes:  _cache/vhf_kg.json
Also provides hybrid retrieval: dense (embeddings) + graph expansion.
Smoke test at the bottom prints top hits for VHF queries the dense-only run struggled with.
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths, EMBEDDER_MODEL, QUERY_PREFIX

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

CHUNKS_FILE = CACHE / f"{_PFX}_rag_chunks.json"
EMBS_FILE   = CACHE / f"{_PFX}_rag_embeddings.npy"
IDS_FILE    = CACHE / f"{_PFX}_rag_chunk_ids.json"
KG_FILE     = CACHE / f"{_PFX}_kg.json"

# ── Concept aliases: query phrase -> canonical concept ────────────────────
# Lets queries like "how do I DSC alert" resolve to Channel 70 automatically.
CONCEPT_ALIASES: dict[str, list[str]] = {
    "distress channel":       ["Channel 16"],
    "calling channel":        ["Channel 16"],
    "international distress": ["Channel 16", "MAYDAY"],
    "dsc channel":            ["Channel 70", "DSC"],
    "digital selective":      ["DSC", "Channel 70"],
    "dsc alert":              ["DSC", "Channel 70"],
    "urgency":                ["PAN PAN"],
    "urgent":                 ["PAN PAN"],
    "pan pan":                ["PAN PAN"],
    "pan-pan":                ["PAN PAN"],
    "panpan":                 ["PAN PAN"],
    "safety broadcast":       ["SECURITE"],
    "safety announce":        ["SECURITE"],
    "securite":               ["SECURITE"],
    "may day":                ["MAYDAY"],
    "mayday":                 ["MAYDAY"],
    "emergency beacon":       ["EPIRB"],
    "epirb":                  ["EPIRB"],
    "position indicating":    ["EPIRB"],
    "search and rescue":      ["SAR", "MRCC"],
    "coast guard":            ["MRCC", "MCA"],
    "coastguard":             ["MRCC", "MCA"],
    "phonetic":               ["ALFA", "BRAVO", "CHARLIE"],  # any letter triggers phonetic-topic chunks
    "alfa":  ["ALFA"], "alpha": ["ALFA"],
    "bravo": ["BRAVO"], "charlie": ["CHARLIE"],
    "mike":  ["MIKE"], "november": ["NOVEMBER"], "oscar": ["OSCAR"],
    "papa":  ["PAPA"], "quebec": ["QUEBEC"], "romeo": ["ROMEO"],
    "licence":  ["Ofcom", "MCA"],
    "license":  ["Ofcom", "MCA"],
    "mmsi":     ["MMSI"],
    "position": ["MMSI"],
    "handheld": ["VHF"],
    "channel 16": ["Channel 16"],
    "channel 70": ["Channel 70"],
    "channel 13": ["Channel 13"],
    "channel 9":  ["Channel 9"],
    "bridge to bridge": ["Channel 13"],
    "over":     ["OVER"],
    "out":      ["OUT"],
    "this is":  ["THIS IS"],
    "roger":    ["ROGER"],
    "wilco":    ["WILCO"],
    "affirmative": ["AFFIRMATIVE"],
    "negative": ["NEGATIVE"],
    "solas":    ["SOLAS"],
    "gmdss":    ["GMDSS"],
    "smcp":     ["SMCP"],
    "navtex":   ["NAVTEX"],
    "ais":      ["AIS"],
    "inmarsat": ["Inmarsat"],
    "msi":      ["MSI"],
    "squelch":  ["squelch"],
    "ptt":      ["PTT"],
    "push to talk": ["PTT"],
}


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


# ── Build KG ──────────────────────────────────────────────────────────────
def build_kg(chunks: list[dict]) -> dict:
    """Build the concept/topic/adjacency knowledge graph from chunk records."""
    concept_chunks: dict[str, list[str]] = defaultdict(list)
    concept_cooccur: dict[str, Counter] = defaultdict(Counter)
    topic_chunks: dict[str, list[str]]  = defaultdict(list)
    doc_chunks: dict[str, list[str]]    = defaultdict(list)
    chapter_chunks: dict[str, list[str]] = defaultdict(list)
    chunk_meta: dict[str, dict] = {}

    for c in chunks:
        cid = c["chunk_id"]
        chunk_meta[cid] = {
            "document_id":    c["document_id"],
            "source_file":    c["source_file"],
            "source_type":    c["source_type"],
            "chapter_title":  c["chapter_title"],
            "section_titles": c["section_titles"],
            "types":          c["types"],
            "concepts":       c.get("concepts", []),
            "topics":         c.get("topics", []),
            "pages":          c.get("pages", []),
            "token_count":    c["token_count"],
            "n_sections":     c["n_sections"],
        }
        for con in c.get("concepts", []):
            concept_chunks[con].append(cid)
        for t in c.get("topics", []):
            topic_chunks[t].append(cid)
        doc_chunks[c["document_id"]].append(cid)
        ck = f"{c['document_id']}::{c['chapter_title']}"
        chapter_chunks[ck].append(cid)

        cons = c.get("concepts", [])
        for i, a in enumerate(cons):
            for b in cons[i + 1:]:
                concept_cooccur[a][b] += 1
                concept_cooccur[b][a] += 1

    # Chunk adjacency: sequential within chapter (list is already in section order).
    adjacency: dict[str, list[str]] = {}
    for ck, cids in chapter_chunks.items():
        for i, cid in enumerate(cids):
            neigh = []
            if i > 0:            neigh.append(cids[i - 1])
            if i < len(cids)-1:  neigh.append(cids[i + 1])
            adjacency[cid] = neigh

    kg = {
        "chunk_meta":        chunk_meta,
        "concept_chunks":    dict(concept_chunks),
        "concept_cooccur":   {k: dict(v) for k, v in concept_cooccur.items()},
        "topic_chunks":      dict(topic_chunks),
        "document_chunks":   dict(doc_chunks),
        "chapter_chunks":    dict(chapter_chunks),
        "adjacency":         adjacency,
        "aliases":           CONCEPT_ALIASES,
        "stats": {
            "n_chunks":   len(chunk_meta),
            "n_concepts": len(concept_chunks),
            "n_topics":   len(topic_chunks),
            "n_docs":     len(doc_chunks),
            "n_chapters": len(chapter_chunks),
        },
    }
    return kg


# ── Query-time concept extraction ─────────────────────────────────────────
def query_concepts(query: str, kg: dict) -> list[str]:
    """Extract known concepts mentioned in `query` via alias lookup and substring matching."""
    q = " " + query.lower() + " "
    q_norm = re.sub(r"[^a-z0-9 ]+", " ", q)
    q_norm = re.sub(r"\s+", " ", q_norm)
    hits: set[str] = set()

    # 1) Alias lookup
    for phrase, cons in kg["aliases"].items():
        if f" {phrase} " in q_norm:
            hits.update(cons)

    # 2) Direct concept mention (case-insensitive substring, whole-word)
    for con in kg["concept_chunks"]:
        pat = re.escape(con.lower())
        if re.search(rf"(?<![a-z0-9]){pat}(?![a-z0-9])", q_norm):
            hits.add(con)

    return sorted(hits)


# ── Hybrid retrieval ──────────────────────────────────────────────────────
def kg_retrieve(query: str, model, embs: np.ndarray, ids: list[str], kg: dict,
                k: int = 5, dense_n: int = 20,
                concept_boost: float = 0.15,
                cooccur_boost: float = 0.08) -> tuple[list[dict], list[str], list[str]]:
    """Hybrid dense + concept-graph retrieval: dense top-`dense_n` pool, boosted by
    concept and co-occurring-concept matches, then truncated to the top `k`."""
    qe = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    dense_scores = embs @ qe  # cosine (unit-norm)

    # 1) Dense pool
    dense_top = np.argsort(-dense_scores)[:dense_n]
    pool: dict[str, float] = {ids[i]: float(dense_scores[i]) for i in dense_top}

    # 2) Concept expansion
    q_cons = query_concepts(query, kg)
    expanded_cons = set(q_cons)
    for c in q_cons:
        # top-5 by co-occurrence WEIGHT — dict insertion order is first-seen, not strongest
        neighbours = sorted(kg["concept_cooccur"].get(c, {}).items(), key=lambda kv: -kv[1])[:5]
        for c2, _cnt in neighbours:
            expanded_cons.add(c2)

    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    for c in q_cons:
        for cid in kg["concept_chunks"].get(c, []):
            i = id_to_idx.get(cid)
            if i is None: continue
            base = float(dense_scores[i])
            pool[cid] = max(pool.get(cid, 0.0), base) + concept_boost

    for c in expanded_cons - set(q_cons):
        for cid in kg["concept_chunks"].get(c, []):
            i = id_to_idx.get(cid)
            if i is None: continue
            base = float(dense_scores[i])
            pool[cid] = max(pool.get(cid, 0.0), base) + cooccur_boost

    ranked = sorted(pool.items(), key=lambda kv: -kv[1])[:k]
    out = []
    for cid, score in ranked:
        meta = kg["chunk_meta"][cid]
        out.append({
            "chunk_id":     cid,
            "score":        score,
            "dense_score":  float(dense_scores[id_to_idx[cid]]),
            "source_file":  meta["source_file"],
            "chapter":      meta["chapter_title"],
            "types":        meta["types"],
            "concepts":     meta["concepts"],
            "topics":       meta["topics"],
        })
    return out, q_cons, sorted(expanded_cons - set(q_cons))


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point: build the KG from cached chunks/embeddings and print a retrieval smoke test."""
    print("Loading chunks + embeddings...", flush=True)
    chunks = load_chunks()
    embs   = np.load(EMBS_FILE)
    ids    = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    assert len(ids) == embs.shape[0] == len(chunks)

    print(f"  chunks={len(chunks)}  embeddings={embs.shape}", flush=True)

    t0 = time.time()
    kg = build_kg(chunks)
    print(f"KG built in {time.time()-t0:.2f}s", flush=True)
    print(f"  concepts : {kg['stats']['n_concepts']}")
    print(f"  topics   : {kg['stats']['n_topics']}")
    print(f"  chapters : {kg['stats']['n_chapters']}")
    print(f"  aliases  : {len(kg['aliases'])}")

    top_cooc = []
    for c, others in kg["concept_cooccur"].items():
        for c2, w in others.items():
            if c < c2:
                top_cooc.append((w, c, c2))
    top_cooc.sort(reverse=True)
    print("\nTop concept co-occurrences:")
    for w, a, b in top_cooc[:12]:
        print(f"  {w:>3}  {a}  <->  {b}")

    KG_FILE.write_text(json.dumps(kg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {KG_FILE.relative_to(W)}  ({KG_FILE.stat().st_size/1024:.1f} KB)")

    # Smoke test
    print("\nLoading embedder for retrieval smoke test...")
    model = SentenceTransformer(EMBEDDER_MODEL)

    QUERIES = [
        "What is VHF Channel 70 used for?",
        "How do I send a DSC distress alert?",
        "What is the phonetic word for the letter M?",
        "What is bridge to bridge channel?",
        "How do I release the squelch?",
        "What does PTT mean?",
        "What is the difference between MAYDAY and PAN-PAN?",
    ]
    for q in QUERIES:
        print()
        print("=" * 100)
        print("Q:", q)
        results, q_cons, expanded = kg_retrieve(q, model, embs, ids, kg, k=5)
        print(f"   query concepts: {q_cons}  |  expanded: {expanded}")
        print("=" * 100)
        for r, hit in enumerate(results, 1):
            print(f"  {r}. [total={hit['score']:.3f}  dense={hit['dense_score']:.3f}] "
                  f"{hit['source_file']} -> {hit['chapter'][:40]!r}")
            print(f"     types={hit['types']} concepts={hit['concepts']} topics={hit['topics']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
