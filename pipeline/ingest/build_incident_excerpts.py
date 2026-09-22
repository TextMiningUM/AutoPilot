"""Turn the shortlisted incident_screening.json entries into small, structured
excerpts -- NOT full reports -- ready for the same § 8 JSON schema the COLREG
rule-text parser (build_oow_json.py) produces.

Rationale: most of an 800-report corpus is vessel-particulars/weather
boilerplate that isn't COLREG-relevant even for a report that IS relevant
overall. For each shortlisted report we keep only:
  1. a short "summary" section (report's own opening paragraph), and
  2. the Analysis / Conclusions / Findings / Causal-factors pages -- located
     via the heading already detected by screen_incidents.py, or (if no
     heading was found) the page window with the highest COLREG keyword
     density -- capped at MAX_EXCERPT_PAGES and cut short if an APPENDIX/ANNEX
     heading is hit (post-conclusions filler).

Output: one JSON file per incident under json_dir (e.g.
Data/OOW/OOW_JSON/incident_<slug>.json), same schema as
colreg_consolidated_2018.json (document_id/source_type/chapters/sections) so
build_rag.py / build_kg.py pick it up automatically alongside the COLREG
rules text -- no changes needed to those scripts.

Run with: python -m pipeline.ingest.build_incident_excerpts [--min-score 15]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core import AgentPaths
from pipeline.ingest.build_oow_json import stable_id, tag_text
from pipeline.ingest.screen_incidents import (
    CACHE_DIR,
    COLREG_KEYWORDS,
    OUT_FILE as SCREENING_FILE,
)

paths = AgentPaths.from_env()
JSON_OUT_DIR = paths.json_dir
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EXCERPT_PAGES = 14
DENSITY_WINDOW = 4
MIN_EXCERPT_CHARS = 200
STOP_HEADING_RE = re.compile(r"^(APPENDIX|ANNEX)\b", re.IGNORECASE)


def slugify(rel_path: str) -> str:
    stem = Path(rel_path).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug[:60]


def load_pages(rel_path: str) -> list[str]:
    cache_file = CACHE_DIR / (rel_path[:-4] + ".json") if rel_path.endswith(".pdf") else CACHE_DIR / (rel_path + ".json")
    return json.loads(cache_file.read_text(encoding="utf-8"))


def excerpt_from_heading(pages: list[str], analysis_page: int) -> tuple[int, int]:
    end = min(analysis_page + MAX_EXCERPT_PAGES, len(pages))
    for i in range(analysis_page + 1, end):
        first_line = next((ln.strip() for ln in (pages[i] or "").split("\n") if ln.strip()), "")
        if STOP_HEADING_RE.match(first_line):
            return analysis_page, i
    return analysis_page, end


def best_density_window(pages: list[str]) -> tuple[int, int] | None:
    """Fallback when no Analysis/Conclusions heading was detected: the
    DENSITY_WINDOW-page window with the highest weighted COLREG keyword count."""
    best_i, best_score = None, 0
    for i in range(len(pages)):
        window_text = "\n".join(pages[i : i + DENSITY_WINDOW]).lower()
        score = sum(window_text.count(kw) * w for kw, w in COLREG_KEYWORDS.items())
        if score > best_score:
            best_i, best_score = i, score
    if best_i is None:
        return None
    return best_i, min(best_i + DENSITY_WINDOW, len(pages))


def filter_collision_relevant_paragraphs(text: str) -> str:
    """Phase 4 (marginal incidents, RAG rebuild 2026-09-22): a 0<=net_score<15 report's
    Analysis/Conclusions excerpt is kept only where it's ACTUALLY collision-relevant, not
    as a whole excerpt -- splits on blank lines and keeps a paragraph only if
    build_oow_json.tag_text() finds at least one concept in it, same tagging already
    used for full incidents (no new keyword list)."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept = [p for p in paras if tag_text(p)[0]]
    return "\n\n".join(kept)


def build_document(entry: dict, marginal: bool = False) -> dict | None:
    pages = load_pages(entry["path"])
    if not pages or pages[0].startswith("__EXTRACT_ERROR__"):
        return None

    span = None
    if entry.get("analysis_page") is not None:
        span = excerpt_from_heading(pages, entry["analysis_page"])
    if span is None:
        span = best_density_window(pages)
    if span is None:
        return None
    start, end = span

    excerpt_text = "\n\n".join(p for p in pages[start:end] if p).strip()
    if marginal:
        excerpt_text = filter_collision_relevant_paragraphs(excerpt_text)
    if len(excerpt_text) < MIN_EXCERPT_CHARS:
        return None

    slug = slugify(entry["path"])
    doc_id = f"incident_marginal_{slug}_{stable_id(entry['path'])}" if marginal else f"incident_{slug}_{stable_id(entry['path'])}"

    sections = []
    summary_text = (pages[0] or "").strip()[:1000]
    if summary_text and not marginal:
        # Marginal incidents skip the opening-paragraph summary entirely -- it's
        # vessel-particulars boilerplate, not itself filtered for relevance, and Phase 4
        # is specifically "collision-relevant sections only", not "a slightly larger excerpt".
        concepts, topics = tag_text(summary_text)
        sections.append({
            "section_id": f"{doc_id}_summary",
            "title": "Incident summary (report opening)",
            "type": "summary",
            "text": summary_text,
            "concepts": concepts,
            "topics": topics,
            "pages": [1],
        })

    concepts, topics = tag_text(excerpt_text)
    sections.append({
        "section_id": f"{doc_id}_analysis",
        "title": "Analysis / conclusions excerpt",
        "type": "incident_analysis",
        "text": excerpt_text,
        "concepts": concepts,
        "topics": topics,
        "pages": list(range(start + 1, end + 1)),
    })

    return {
        "document_id": doc_id,
        "source_file": entry["path"],
        "source_type": "incident_marginal" if marginal else "incident_report",
        "screening_net_score": entry["net_score"],
        "chapters": [{"title": "Incident Report Excerpt", "sections": sections}],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=int, default=15,
                     help="Only build excerpts for reports with screening net_score >= this.")
    ap.add_argument("--marginal", action="store_true",
                     help="Phase 4 (RAG rebuild 2026-09-22): instead of --min-score, build "
                          "collision-relevant-SECTIONS-only excerpts for the 0<=net_score<15 "
                          "'marginal' reports -- a separate, measurable step, never mixed "
                          "into the default --min-score run.")
    args = ap.parse_args()

    if not SCREENING_FILE.exists():
        raise SystemExit(f"Missing {SCREENING_FILE} -- run screen_incidents.py first.")
    entries = json.loads(SCREENING_FILE.read_text(encoding="utf-8"))
    if args.marginal:
        shortlisted = [e for e in entries if 0 <= e["net_score"] < 15]
        print(f"{len(shortlisted)}/{len(entries)} reports are 'marginal' (0 <= net_score < 15)")
    else:
        shortlisted = [e for e in entries if e["net_score"] >= args.min_score]
        print(f"{len(shortlisted)}/{len(entries)} reports score net_score >= {args.min_score}")

    written, skipped = 0, 0
    n_sections_total = 0
    for entry in shortlisted:
        doc = build_document(entry, marginal=args.marginal)
        if doc is None:
            skipped += 1
            continue
        out_path = JSON_OUT_DIR / f"{doc['document_id']}.json"
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
        n_sections_total += sum(len(c["sections"]) for c in doc["chapters"])

    kind = "marginal-incident" if args.marginal else "incident excerpt"
    print(f"Wrote {written} {kind} JSONs ({n_sections_total} sections total) to {JSON_OUT_DIR}")
    if skipped:
        reason = "no collision-relevant section survived filtering" if args.marginal else "no usable excerpt found or extraction error"
        print(f"Skipped {skipped} ({reason})")


if __name__ == "__main__":
    main()
