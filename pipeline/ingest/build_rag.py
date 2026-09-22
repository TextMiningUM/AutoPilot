"""Standalone RAG chunker + embedder — same logic as § 9 in the notebook.

Reads JSONs from _json/, writes:
  _cache/vhf_rag_chunks.json           (chunk records)
  _cache/vhf_rag_embeddings.npy        (numpy float32 array)
  _cache/vhf_rag_chunk_ids.json        (chunk_id list, same order as embeddings)

Run with: python build_rag.py
"""
from __future__ import annotations
import json, re, hashlib, time
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths, EMBEDDER_MODEL
from pipeline.ingest.rag_exclusions import raise_if_excluded_source

paths = AgentPaths.from_env()
WORKSPACE    = paths.workspace
JSON_OUT_DIR = paths.json_dir
CACHE_DIR    = paths.cache_dir
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PFX = paths.domain.lower()

RAG_CHUNKS_FILE = CACHE_DIR / f"{_PFX}_rag_chunks.json"
EMBEDDINGS_FILE = CACHE_DIR / f"{_PFX}_rag_embeddings.npy"
IDS_FILE        = CACHE_DIR / f"{_PFX}_rag_chunk_ids.json"

# ── Chunking parameters ───────────────────────────────────────────────────
CHUNK_TARGET_TOKENS = 400
CHUNK_MAX_TOKENS    = 500
CHUNK_MIN_TOKENS    = 40
TOPIC_JACCARD_MIN   = 0.5

# "rule" added so every individual COLREG rule (Rule 13, Rule 14, ...) is always its
# own standalone chunk, never merged with a neighbouring rule -- found 3 accidental
# multi-rule chunks (Rules 28-30, 32-33, 39-41) that slipped through the token-budget/
# topic-Jaccard merge logic because they're short and share topics (or share no
# topics at all, which also passes the low-token-count 0.2 threshold).
# "chirp_report"/"chirp_comment" added (Phase 2, RAG rebuild 2026-09-22): each CHIRP
# newsletter article is its own independent near-miss case -- merging one article's
# tail into the next unrelated article would corrupt both as retrieval units.
# "moos_case" added (Phase 3, same rebuild): each canonicalized Leo MOOS situation is
# its own independent case, never merged with a neighbouring unrelated case.
STANDALONE_TYPES = {"dialogue", "definition", "procedure", "rule", "chirp_report", "chirp_comment", "moos_case"}

# Set by main() before any chunking/embedding happens.
model: SentenceTransformer | None = None
tokenizer = None


def stable_id(prefix: str, *parts) -> str:
    """Build a short, deterministic id from `prefix` and the string forms of `parts`."""
    h = hashlib.md5(("|".join(str(p) for p in parts)).encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets; 1.0 if both are empty."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def token_count(text: str) -> int:
    """Number of tokenizer tokens in `text` (0 for empty text)."""
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0


# ── Chunker ───────────────────────────────────────────────────────────────
def _can_extend(cur_types, cur_topics, cur_tokens,
                nxt_type, nxt_topics, nxt_tokens) -> bool:
    if nxt_type in STANDALONE_TYPES:
        return False
    if any(t in STANDALONE_TYPES for t in cur_types):
        return False
    if cur_tokens + nxt_tokens > CHUNK_MAX_TOKENS:
        return False
    thresh = TOPIC_JACCARD_MIN
    if cur_tokens < CHUNK_MIN_TOKENS or nxt_tokens < CHUNK_MIN_TOKENS:
        thresh = 0.2
    if jaccard(cur_topics, nxt_topics) < thresh:
        return False
    return True


def _finalise_chunk(sections, chapter, doc, tokens, chapter_index, chunk_index) -> dict:
    doc_id      = doc["document_id"]
    section_ids = [s["section_id"] for s in sections]
    section_texts_with_headers = [
        f"[{s['title']} — {s['type']}]\n{s['text']}" for s in sections
    ]
    text = "\n\n[SECTION BREAK]\n\n".join(s["text"] for s in sections) \
             if len(sections) > 1 else sections[0]["text"]
    text_with_context = (
        f"Source: {doc['source_file']}  ·  Chapter: {chapter['title']}\n\n"
        + "\n\n".join(section_texts_with_headers)
    )
    pages    = sorted({p for s in sections for p in s.get("pages", [])})
    concepts = sorted({c for s in sections for c in s.get("concepts", [])})
    topics   = sorted({t for s in sections for t in s.get("topics", [])})
    return {
        # Structural id (document_id + chapter/chunk POSITION + a text hash), never derived
        # from section_ids/titles -- those repeat across different articles that happen to
        # share a title (e.g. CHIRP's per-issue "Initial Report"/"CHIRP Comment" section
        # titles), which silently collided 64 ids across 196 entries under the old
        # title-based stable_id(doc_id, *section_ids) scheme (RAG rebuild A-nawerk-1,
        # 2026-09-22) -- each entry still carried its own correct text, but chunk_by_id
        # lookups (rerank_hits/format_context) would then return a DIFFERENT entry's text
        # than the one the embedding/retrieval actually matched. Position + text hash makes
        # a collision structurally impossible regardless of upstream title reuse.
        "chunk_id":          stable_id("chunk", doc_id, chapter_index, chunk_index, text),
        "document_id":       doc_id,
        "source_file":       doc["source_file"],
        "source_type":       doc["source_type"],
        "chapter_title":     chapter["title"],
        "section_ids":       section_ids,
        "section_titles":    [s["title"] for s in sections],
        "types":             [s["type"] for s in sections],
        "text":              text,
        "text_with_context": text_with_context,
        "concepts":          concepts,
        "topics":            topics,
        "pages":             pages,
        "token_count":       tokens,
        "n_sections":        len(sections),
    }


def _split_oversized_section(section: dict, max_tokens: int) -> list[dict]:
    """Split a single section whose OWN text already exceeds max_tokens into
    several smaller sections (same metadata, sliced text). Needed for
    incident-report excerpts (build_incident_excerpts.py emits each report's
    whole Analysis/Conclusions/Findings excerpt as ONE section, sometimes
    several pages long) -- chunk_chapter()'s merge loop only ever checked
    the budget when ADDING a section to a chunk, never when a single seed
    section was already over budget on its own, so these came out as
    unsplit 10-30K-character chunks (confirmed: 6 retrieved OOW incident
    chunks combined into an 80KB ablation prompt, ~20K tokens before the
    4096-token truncation, turning a normal ~30s generation into ~18 min)."""
    text = section["text"]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]
    pieces: list[str] = []
    cur, cur_tokens = "", 0
    for p in paras:
        p_tokens = token_count(p)
        if p_tokens > max_tokens:
            # a single paragraph alone is still oversized -- fall back to sentences
            for s in re.split(r"(?<=[.!?])\s+", p):
                s_tokens = token_count(s)
                if cur_tokens + s_tokens > max_tokens and cur:
                    pieces.append(cur.strip())
                    cur, cur_tokens = "", 0
                cur += (" " if cur else "") + s
                cur_tokens += s_tokens
            continue
        if cur_tokens + p_tokens > max_tokens and cur:
            pieces.append(cur.strip())
            cur, cur_tokens = "", 0
        cur += ("\n\n" if cur else "") + p
        cur_tokens += p_tokens
    if cur.strip():
        pieces.append(cur.strip())
    out = []
    for i, piece in enumerate(pieces):
        sub = dict(section)
        sub["text"] = piece
        sub["section_id"] = f"{section['section_id']}_p{i + 1}"
        out.append(sub)
    return out


def chunk_chapter(chapter: dict, doc: dict, chapter_index: int) -> list[dict]:
    """Merge a chapter's sections into token-budgeted, topic-coherent chunks."""
    raw_sections = chapter.get("sections", [])
    if not raw_sections:
        return []
    sections = []
    for s in raw_sections:
        if token_count(s["text"]) > CHUNK_MAX_TOKENS:
            sections.extend(_split_oversized_section(s, CHUNK_MAX_TOKENS))
        else:
            sections.append(s)
    out = []
    i = 0
    chunk_index = 0
    while i < len(sections):
        seed = sections[i]
        seed_tokens = token_count(seed["text"])
        current = [seed]
        cur_tokens = seed_tokens
        cur_types = [seed["type"]]
        cur_topics = set(seed.get("topics", []))
        j = i + 1
        while j < len(sections):
            nxt = sections[j]
            nxt_tokens = token_count(nxt["text"])
            nxt_topics = set(nxt.get("topics", []))
            if not _can_extend(cur_types, cur_topics, cur_tokens,
                               nxt["type"], nxt_topics, nxt_tokens):
                break
            if cur_tokens >= CHUNK_TARGET_TOKENS and nxt_tokens >= CHUNK_MIN_TOKENS:
                break
            current.append(nxt)
            cur_tokens += nxt_tokens
            cur_types.append(nxt["type"])
            cur_topics |= nxt_topics
            j += 1
        out.append(_finalise_chunk(current, chapter, doc, cur_tokens, chapter_index, chunk_index))
        chunk_index += 1
        i = j
    return out


def build_chunks_for_document(doc: dict) -> list[dict]:
    """Chunk every chapter of one parsed document."""
    chunks = []
    for chapter_index, chapter in enumerate(doc.get("chapters", [])):
        chunks.extend(chunk_chapter(chapter, doc, chapter_index))
    return chunks


def _outputs_stale() -> bool:
    """True if any output is missing, or any JSON in JSON_OUT_DIR is newer than
    the oldest output -- same mtime convention as build_vhf_json.py's cleanup
    pass and the notebook's KG staleness check."""
    outputs = (RAG_CHUNKS_FILE, EMBEDDINGS_FILE, IDS_FILE)
    if not all(p.exists() for p in outputs):
        return True
    oldest_output_mtime = min(p.stat().st_mtime for p in outputs)
    return any(p.stat().st_mtime > oldest_output_mtime
               for p in JSON_OUT_DIR.glob("*.json") if not p.name.startswith("_"))


def main() -> None:
    """Load the embedder, chunk every document in _json/, embed the chunks, and
    write chunks/embeddings/ids to _cache/ (plus a retrieval smoke test).
    Skips the whole rebuild if outputs already exist and are newer than every
    source JSON -- pass --force to rebuild unconditionally."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs look up to date")
    args = ap.parse_args()

    if not args.force and not _outputs_stale():
        chunks = json.loads(RAG_CHUNKS_FILE.read_text(encoding="utf-8"))
        print(f"Up to date -- {len(chunks)} chunks, {EMBEDDINGS_FILE.name} unchanged. "
              f"Pass --force to rebuild anyway.")
        return

    global model, tokenizer

    # ── Load embedder + tokenizer ─────────────────────────────────────────
    print("Loading embedder...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(EMBEDDER_MODEL)
    tokenizer = model.tokenizer
    print(f"  ready in {time.time()-t0:.1f}s", flush=True)

    # ── Build corpus ──────────────────────────────────────────────────────
    print("\nBuilding chunks from _json/...", flush=True)
    all_chunks: list[dict] = []
    per_doc: list[tuple[str, int, int]] = []

    for path in sorted(JSON_OUT_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not doc.get("chapters"):
            continue
        # Defense-in-depth: this is the single choke point every source document flows
        # through before becoming a chunk -- see pipeline/ingest/rag_exclusions.py for
        # the full exclusion list/rationale (eval data, model-output artifacts, derived
        # training JSONL, meta-literature never belong in the RAG corpus).
        raise_if_excluded_source(doc["source_file"])
        n_sections = sum(len(ch.get("sections", [])) for ch in doc["chapters"])
        chunks = build_chunks_for_document(doc)
        all_chunks.extend(chunks)
        per_doc.append((doc["source_file"], n_sections, len(chunks)))
        print(f"  {doc['source_file']:<70} sections={n_sections:>4}  chunks={len(chunks):>4}", flush=True)

    RAG_CHUNKS_FILE.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(all_chunks)} chunks from {len(per_doc)} documents", flush=True)
    print(f"Saved: {RAG_CHUNKS_FILE}", flush=True)

    # Stats
    sizes = [c["token_count"] for c in all_chunks]
    sizes_sorted = sorted(sizes)
    n = len(sizes_sorted)
    print(f"\nToken distribution:")
    print(f"  min={min(sizes)}  p25={sizes_sorted[n//4]}  median={sizes_sorted[n//2]}  "
          f"p75={sizes_sorted[3*n//4]}  max={max(sizes)}")
    print(f"  over target ({CHUNK_TARGET_TOKENS}): {sum(1 for s in sizes if s > CHUNK_TARGET_TOKENS)}")
    print(f"  over max ({CHUNK_MAX_TOKENS})   : {sum(1 for s in sizes if s > CHUNK_MAX_TOKENS)}")

    from collections import Counter
    n_sec_distribution = Counter(c["n_sections"] for c in all_chunks)
    print(f"\nSections per chunk:")
    for k in sorted(n_sec_distribution):
        print(f"  {k}: {n_sec_distribution[k]}")

    # ── Embed ─────────────────────────────────────────────────────────────
    print(f"\nEmbedding {len(all_chunks)} chunks on "
          f"{'GPU' if hasattr(model, '_target_device') else 'auto'}...", flush=True)
    t0 = time.time()
    texts = [c["text_with_context"] for c in all_chunks]
    embs = model.encode(
        texts, batch_size=32,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    )
    print(f"  done in {time.time()-t0:.1f}s. Shape: {embs.shape}", flush=True)

    ids = [c["chunk_id"] for c in all_chunks]
    np.save(EMBEDDINGS_FILE, embs)
    IDS_FILE.write_text(json.dumps(ids), encoding="utf-8")
    print(f"Saved: {EMBEDDINGS_FILE.name}  +  {IDS_FILE.name}", flush=True)

    # ── Smoke test ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Retrieval smoke test")
    print("=" * 100)
    chunk_by_id = {c["chunk_id"]: c for c in all_chunks}
    for q in ["What is VHF Channel 70 used for?",
              "How do I send a MAYDAY call?",
              "What is the phonetic word for the letter M?"]:
        print(f"\nQuery: {q}")
        q_emb = model.encode([q], normalize_embeddings=True)[0]
        scores = embs @ q_emb
        top = np.argsort(-scores)[:3]
        for rank, i in enumerate(top, 1):
            c = chunk_by_id[ids[i]]
            preview = c["text"].replace("\n", " ")[:130]
            print(f"  {rank}. [{scores[i]:.3f}] {c['source_file']}  -> {c['chapter_title'][:35]!r}")
            print(f"     types={c['types']} topics={c['topics']}")
            print(f"     text: {preview}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
