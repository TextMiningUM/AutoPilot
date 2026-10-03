"""Chief Engineer, Part B — machinery-relevance screening over already-acquired corpora
(design_chief_engineer.md Sec 7: "needs a machinery-relevance filter (different from the
existing collision-regex filter used for the Processed-Leo MAIB set) -- filter not yet
built").

Two independent corpora, already in this repo for OOW/Captain's own purposes, re-scored
here for a DIFFERENT question (engine-room/machinery relevance, not collision-avoidance
relevance) using the SAME shared scoring mechanism (pipeline/ingest/screen_incidents.py's
score_pages(), now parameterized to accept any keyword vocabulary -- never forked):

1. MAIB/NTSB/TSB/ATSB incident-report PDFs under OOW's own Data/OOW/OOW_Incidents/ (the
   800+ PDF corpus screen_incidents.py already scores for COLREG-relevance) -- re-scored
   here against ENGINE_KEYWORDS/NON_ENGINE_KEYWORDS instead. Deliberately does NOT reuse
   screen_incidents.py's own on-disk page-text cache structurally (different base
   directory assumptions), but a fresh independent cache under Chief Engineer's own
   cache_dir so this script has no coupling to OOW's module-level directory constants.
2. CHIRP newsletter articles already parsed into Data/OOW/OOW_Agents_Training/
   oow_rag_chunks.json (source_type=="chirp_newsletter") -- reconstructed into whole
   articles the same (document_id, chapter_title) grouping way
   extract_chirp_reasoning.py already does for its own COLLISION_CONCEPTS filter, scored
   here against the same machinery keyword vocabulary instead of COLREG concept tags
   (CHIRP chunks are only ever concept-tagged with COLREG vocabulary, so a machinery
   relevance question needs its own text-based score, not the existing `concepts` field).

Output (ranked JSON, same shape convention as screen_incidents.py's own
incident_screening.json so a human only needs to skim a short shortlist):
  Data/ChiefEngineer/ChiefEngineer_Agents_Training/machinery_incident_screening.json
  Data/ChiefEngineer/ChiefEngineer_Agents_Training/machinery_chirp_screening.json

Local-safe: pure Python + pdfplumber, no GPU/API key needed.

Run with: python -m pipeline.ingest.screen_chief_engineer_sources
          python -m pipeline.ingest.screen_chief_engineer_sources --limit 50   # smoke test
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pdfplumber

from core import AgentPaths
from core.text_segmentation import join_hyphenated_linebreaks
from pipeline.ingest.screen_incidents import score_pages

OOW_PATHS = AgentPaths.oow()
CE_PATHS = AgentPaths.chief_engineer()

INCIDENTS_DIR = OOW_PATHS.incidents_dir
TEXT_CACHE_DIR = CE_PATHS.cache_dir / "machinery_incidents_text_cache"
INCIDENT_OUT_FILE = CE_PATHS.cache_dir / "machinery_incident_screening.json"

CHIRP_CHUNKS_FILE = OOW_PATHS.cache_dir / "oow_rag_chunks.json"
CHIRP_OUT_FILE = CE_PATHS.cache_dir / "machinery_chirp_screening.json"

# Positive signal: vocabulary that only shows up when a report actually discusses
# engine-room machinery condition/failure (as opposed to merely mentioning "the engine"
# in passing). Mirrors screen_incidents.py's COLREG_KEYWORDS weighting convention (1-3).
ENGINE_KEYWORDS: dict[str, int] = {
    "main engine": 2, "engine room": 1, "engine failure": 3, "propulsion": 1,
    "loss of propulsion": 3, "blackout": 2, "power failure": 2, "total power loss": 3,
    "generator": 2, "auxiliary engine": 2, "turbocharger": 2, "fuel pump": 2,
    "fuel injection": 2, "lubricating oil": 2, "lube oil": 2, "cooling water": 2,
    "jacket water": 2, "cylinder liner": 2, "crankshaft": 2, "bearing failure": 3,
    "crankcase explosion": 3, "overheating": 1, "governor": 1, "steering gear": 2,
    "steering failure": 3, "boiler": 2, "compressor": 1, "pump failure": 2,
    "unattended machinery space": 2, " ums ": 1, "chief engineer": 2,
    "engine room fire": 2, "condition monitoring": 2, "planned maintenance": 1,
    "oil mist detector": 2, "exhaust gas": 1, "scavenge": 2, "mechanical failure": 2,
    "machinery failure": 3, "machinery space": 1, "fatigue failure": 2,
    "piston": 1, "connecting rod": 2, "turbine": 1, "shaft bearing": 2,
}

# Negative signal: suggests the casualty is unrelated to machinery condition/failure.
# Deliberately does NOT include "grounding"/"ran aground" (unlike COLREG's own negative
# list) -- grounding caused by a propulsion-loss casualty is exactly what this filter
# wants to catch (design_chief_engineer.md Sec 7's own framing), so penalising the word
# "grounding" would suppress real machinery-relevant hits.
NON_ENGINE_KEYWORDS: dict[str, int] = {
    "collision regulations": 2, "give-way vessel": 2, "stand-on vessel": 2,
    "risk of collision": 2, "crossing situation": 1, "head-on situation": 1,
    "overtaking vessel": 1, "close-quarters": 1, "man overboard": 2,
    "person overboard": 2, "mooring line": 1, "cargo shift": 1, "piracy": 2,
    "hijack": 2, "stowaway": 1, "medical evacuation": 1, "food poisoning": 1,
}


def find_pdfs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pdf")) if root.exists() else []


def extract_pages_cached(pdf_path: Path, base_dir: Path) -> list[str]:
    """Same cache-to-disk pattern as screen_incidents.py's own extract_pages_cached, but
    under this script's own independent cache dir (no coupling to OOW's module-level
    INCIDENTS_DIR/CACHE_DIR constants, so any source root can be scanned safely)."""
    rel = pdf_path.relative_to(base_dir)
    cache_file = TEXT_CACHE_DIR / rel.with_suffix(".json")
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [join_hyphenated_linebreaks(page.extract_text() or "") for page in pdf.pages]
    except Exception as e:  # corrupt/scanned/encrypted PDF
        pages = [f"__EXTRACT_ERROR__: {e}"]
    cache_file.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def screen_incident_pdfs(limit: int | None = None) -> list[dict]:
    """Score every OOW_Incidents PDF for machinery/engine-room relevance (independent of,
    and in addition to, screen_incidents.py's own COLREG scoring of the same corpus)."""
    pdfs = find_pdfs(INCIDENTS_DIR)
    if limit:
        pdfs = pdfs[:limit]
    results = []
    for i, pdf_path in enumerate(pdfs, 1):
        pages = extract_pages_cached(pdf_path, INCIDENTS_DIR)
        info = score_pages(pages, pos_keywords=ENGINE_KEYWORDS, neg_keywords=NON_ENGINE_KEYWORDS)
        info["path"] = str(pdf_path.relative_to(INCIDENTS_DIR)).replace("\\", "/")
        results.append(info)
        if i % 100 == 0:
            print(f"  scored {i}/{len(pdfs)}")
    results.sort(key=lambda r: r["net_score"], reverse=True)
    return results


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def group_chirp_articles(chunks: list[dict]) -> list[dict]:
    """Groups chirp_newsletter chunks by (document_id, chapter_title) into whole
    articles, same grouping key as extract_chirp_reasoning.py's own
    group_collision_relevant_articles(). Pure function (no file I/O), directly
    unit-testable."""
    chirp = [c for c in chunks if c.get("source_type") == "chirp_newsletter"]
    by_article: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in chirp:
        by_article[(c["document_id"], c["chapter_title"])].append(c)

    articles = []
    for (doc_id, chapter_title), article_chunks in by_article.items():
        full_text = "\n\n".join(c["text"] for c in article_chunks)
        source_file = article_chunks[0].get("source_file", doc_id)
        articles.append({
            "document_id": f"chirp_{doc_id}_{_slug(chapter_title)}",
            "source_file": f"{source_file}::{chapter_title}",
            "full_text": full_text,
        })
    return articles


def screen_chirp_articles() -> list[dict]:
    """Score every CHIRP newsletter article (already parsed for OOW) for machinery/
    engine-room relevance -- a text-keyword score, NOT the existing `concepts` field
    (CHIRP chunks are only ever concept-tagged with COLREG vocabulary, see
    build_oow_json.py's CONCEPT_KEYWORDS, so it can't answer a machinery-relevance
    question)."""
    if not CHIRP_CHUNKS_FILE.exists():
        print(f"  {CHIRP_CHUNKS_FILE} not found -- skipping CHIRP screening.")
        return []
    chunks = json.loads(CHIRP_CHUNKS_FILE.read_text(encoding="utf-8"))
    articles = group_chirp_articles(chunks)
    results = []
    for a in articles:
        info = score_pages([a["full_text"]], pos_keywords=ENGINE_KEYWORDS, neg_keywords=NON_ENGINE_KEYWORDS)
        info["document_id"] = a["document_id"]
        info["source_file"] = a["source_file"]
        results.append(info)
    results.sort(key=lambda r: r["net_score"], reverse=True)
    return results


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="Only scan the first N incident PDFs (smoke-testing).")
    args = ap.parse_args()

    CE_PATHS.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning incident PDFs under {INCIDENTS_DIR}...")
    incident_results = screen_incident_pdfs(args.limit)
    INCIDENT_OUT_FILE.write_text(json.dumps(incident_results, indent=2), encoding="utf-8")
    n_relevant = sum(1 for r in incident_results if r["net_score"] >= 5)
    print(f"Wrote {INCIDENT_OUT_FILE} -- {n_relevant}/{len(incident_results)} reports score net_score >= 5")
    print("Top 10 incident reports by net_score:")
    for r in incident_results[:10]:
        print(f"  {r['net_score']:>4}  {r['path']}")

    print(f"\nScanning CHIRP articles from {CHIRP_CHUNKS_FILE}...")
    chirp_results = screen_chirp_articles()
    CHIRP_OUT_FILE.write_text(json.dumps(chirp_results, indent=2), encoding="utf-8")
    n_relevant_chirp = sum(1 for r in chirp_results if r["net_score"] >= 5)
    print(f"Wrote {CHIRP_OUT_FILE} -- {n_relevant_chirp}/{len(chirp_results)} articles score net_score >= 5")
    print("Top 10 CHIRP articles by net_score:")
    for r in chirp_results[:10]:
        print(f"  {r['net_score']:>4}  {r['source_file']}")


if __name__ == "__main__":
    main()
