"""
================================================================================
extract_chirp_reasoning.py -- Fase C1 (RAG-rebuild-v2 plan): CHIRP article extraction
================================================================================

The incident-report half of Fase C1 already existed from an earlier session
(extract_incident_reasoning.py) but ran against build_incident_excerpts.py's trimmed
excerpts -- fixed there via its new --full-text mode. This script is the OTHER half the
plan calls for and never existed before: extracting the SAME structured
situation/procedures/incident schema from CHIRP newsletter articles that are actually
about collision-avoidance (most CHIRP content is fire/mooring/food-poisoning/manuals --
"vessel-vs-vessel collision-avoidance" is a minority, per the plan's own "CHIRP articles
WITH collision relevance" qualifier).

WHERE THE DATA COMES FROM
--------------------------
oow_rag_chunks.json's source_type=="chirp_newsletter" chunks (build_chirp_json.py's
actual current output type -- NOT "chirp_report"/"chirp_comment", which describe an
internal pre-chunking section split, never the chunk's own top-level source_type).
Each chunk carries (document_id, chapter_title): document_id groups chunks by NEWSLETTER
ISSUE, chapter_title distinguishes individual ARTICLES within that issue (e.g.
"FIRE IN DRYDOCK", "BREAKAWAY FROM MOORINGS") -- so (document_id, chapter_title) is the
grouping key that reconstructs one whole article's full text from its chunks, in their
original (already document-ordered) list order.

An article is "collision relevant" if at least one of its chunks carries a concept from
COLLISION_CONCEPTS (the same crossing/give-way/stand-on/overtaking/close_quarters/risk-
of-collision/cpa_tcpa/near_miss vocabulary already used to identify COLREG-relevant text
elsewhere in this pipeline, e.g. screen_incidents.py's own keyword list).

Reuses extract_incident_reasoning.py's SYSTEM_PROMPT/schema/process_doc/parse_response
UNCHANGED (CHIRP articles are also real vessel incidents/near-misses -- the same
situation/procedures/incident extraction schema fits; genuinely non-collision articles
that slip through the concept filter self-reject via the schema's own {"skip": true}
affordance).

25-first review gate before a full run, per this plan's own "every [LLM] step" rule:
    python -m pipeline.track1.extract_chirp_reasoning --limit 25   # review sample
    python -m pipeline.track1.extract_chirp_reasoning              # full run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from core import AgentPaths, load_env
from pipeline.track1.extract_incident_reasoning import (
    SYSTEM_PROMPT, MODEL, MAX_WORKERS, parse_response, load_done_ids,
)

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

CHUNKS_FILE = CACHE / f"{_PFX}_rag_chunks.json"
OUT_FILE = CACHE / f"{_PFX}_chirp_reasoning_traces.jsonl"

COLLISION_CONCEPTS = {
    "crossing", "give-way", "stand-on", "overtaking", "close_quarters",
    "risk of collision", "action to avoid collision", "cpa_tcpa", "near_miss",
}


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def group_collision_relevant_articles(chunks: list[dict]) -> list[dict]:
    """Groups chirp_newsletter chunks by (document_id, chapter_title) into whole
    articles, keeps only articles with >=1 collision-relevant chunk, joins ALL of that
    article's chunk text (original list order) into one full_text string. Pure function
    (no file I/O) so it's directly unit-testable."""
    chirp = [c for c in chunks if c.get("source_type") == "chirp_newsletter"]
    by_article: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in chirp:
        by_article[(c["document_id"], c["chapter_title"])].append(c)

    docs = []
    for (doc_id, chapter_title), article_chunks in by_article.items():
        concepts = {c for ch in article_chunks for c in ch.get("concepts", [])}
        if not (concepts & COLLISION_CONCEPTS):
            continue
        full_text = "\n\n".join(c["text"] for c in article_chunks)
        source_file = article_chunks[0].get("source_file", doc_id)
        docs.append({
            "document_id": f"chirp_{doc_id}_{_slug(chapter_title)}",
            "source_file": f"{source_file}::{chapter_title}",
            "screening_net_score": None,
            "full_text": full_text,
            "colreg_hits": sorted(concepts & COLLISION_CONCEPTS),
        })
    return docs


def load_collision_relevant_articles(limit: int | None) -> list[dict]:
    """Reads CHUNKS_FILE and applies group_collision_relevant_articles(), optionally
    capped to the first `limit` articles (25-first review gate)."""
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    docs = group_collision_relevant_articles(chunks)
    if limit:
        docs = docs[:limit]
    return docs


def user_prompt(doc: dict) -> str:
    return (
        f"Source: {doc['source_file']} (CHIRP newsletter article)\n"
        f"Matched collision-relevant concepts: {', '.join(doc['colreg_hits'])}\n\n"
        f"Article text:\n\"\"\"\n{doc['full_text']}\n\"\"\""
    )


def process_doc(client: OpenAI, doc: dict) -> tuple[str, dict | None, str | None]:
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.3, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_prompt(doc)}],
            max_tokens=1800,
        )
        raw = resp.choices[0].message.content or ""
        obj = parse_response(raw)
        if obj is None:
            return doc["document_id"], None, "parse-failure"
        return doc["document_id"], obj, None
    except Exception as e:
        return doc["document_id"], None, str(e)[:200]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--limit", type=int, default=None,
                    help="25-first review gate: only process the first N collision-relevant articles.")
    args = ap.parse_args()

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing from .env")
    client = OpenAI()

    docs = load_collision_relevant_articles(args.limit)
    print(f"Collision-relevant CHIRP articles found: {len(docs)}")

    done = load_done_ids(OUT_FILE)
    todo = [d for d in docs if d["document_id"] not in done]
    print(f"Already done: {len(done)}   Todo: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Extracting with {MODEL}, workers={MAX_WORKERS}...")
    t0 = time.time()
    errs = skipped = written = 0
    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_doc, client, d): d for d in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                did, obj, err = fut.result()
                doc = next(d for d in todo if d["document_id"] == did)
                if err or obj is None:
                    errs += 1
                    f_out.write(json.dumps({
                        "document_id": did, "source_file": doc["source_file"],
                        "error": err or "empty", "trace": None,
                    }) + "\n")
                    continue
                if obj.get("skip"):
                    skipped += 1
                    f_out.write(json.dumps({
                        "document_id": did, "source_file": doc["source_file"],
                        "skip": True, "reason": obj.get("reason", ""), "trace": None,
                    }) + "\n")
                    continue
                row = {
                    "document_id": did, "chunk_id": did,
                    "source_file": doc["source_file"],
                    "chapter_title": f"CHIRP article: {doc['source_file']}",
                    "chunk_concepts": doc["colreg_hits"],
                    "screening_net_score": None,
                    "trace": obj,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if i % 10 == 0 or i == len(todo):
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)}  ({rate:.2f}/s)  written={written} skipped={skipped} errs={errs}")

    print(f"\nDone in {time.time()-t0:.0f}s. written={written} skipped={skipped} errs={errs}")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
