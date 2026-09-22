"""CHIRP Maritime FEEDBACK newsletters -> structured JSON (Phase 2 of the OOW RAG
rebuild, 2026-09-22).

CHIRP is a confidential near-miss reporting programme; each newsletter bundles several
independent reports (a mariner's own account, "Report Text:"/"CHIRP Narrative:") plus
CHIRP's expert commentary ("CHIRP Comment:") -- the commentary IS the lesson, so it's
kept as its own labelled section, not merged into the report.

SPLIT HEURISTIC (found by manually inspecting MFB-3/2004, MFB-16/2007, MFB-33/2013 --
early/mid/recent issues, per the RAG-rebuild plan's explicit instruction to look before
coding): every ALL-CAPS line of 2-8 words with no digits/colon is treated as an article
(or section-header) boundary -- consistent across all three issues regardless of era.
Section headers like "COMMERCIAL SECTOR REPORTS" match the same pattern but have no body
text before the NEXT such line, so they naturally become empty/near-empty segments and
are dropped by the minimum-body-length filter -- no separate stoplist needed.

Within one article's body, "CHIRP Comment:" (case-insensitive) splits the mariner's own
report from CHIRP's commentary; everything before it is type="chirp_report", everything
from it onward is type="chirp_comment". Articles with no such marker are entirely
type="chirp_report".

Scoring reuses screen_incidents.py's EXACT keyword weights/score_pages() (no new list,
per the RAG-rebuild plan) -- an article is kept only if net_score >= 0.

Output: one JSON per newsletter under OOW_JSON/, same schema build_rag.py expects
(document -> chapters -> sections), so no further RAG/KG wiring is required.

Run with: python -m pipeline.ingest.build_chirp_json
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

import pymupdf

from core import AgentPaths
from pipeline.ingest.rag_exclusions import raise_if_excluded_source
from pipeline.ingest.screen_incidents import score_pages

paths = AgentPaths.from_env()
SRC_DIR = paths.workspace / "Data" / "MarineNewsLetters"
JSON_OUT_DIR = paths.json_dir
TEXT_CACHE_DIR = paths.cache_dir / "chirp_text_cache"

# 2-8 words, letters/spaces/apostrophe/hyphen/ampersand/comma only, no digits/colon --
# matches every article/section-header title observed across all 3 inspected issues
# ("ERRATIC ENCOUNTER", "WAKE WASH", "COLLISION REGULATIONS - NEAR MISS",
# "STEVEDORE'S STOVE", "COMMERCIAL SECTOR REPORTS", ...).
HEADING_RE = re.compile(r"^[A-Z][A-Z' \-&,]{3,58}[A-Z]$")
MIN_BODY_CHARS = 100  # drops section-header-only segments (e.g. "COMMERCIAL SECTOR REPORTS")
CHIRP_COMMENT_RE = re.compile(r"chirp\s+comment\b\s*:?", re.IGNORECASE)
ISSUE_RE = re.compile(r"(?:Issue\s+No|No)\s*:?\s*(\d+)", re.IGNORECASE)
FILENAME_ISSUE_RE = re.compile(r"MFB[-_]?(\d+)", re.IGNORECASE)
# Post-~2018 redesign (CHIRP-MFB-5x/6x): titles are Title Case, not ALL-CAPS, so
# HEADING_RE never matches -- but every article still ends with a literal "Report
# Ends" marker (found by inspecting CHIRP-MFB-52, which HEADING_RE produced zero
# candidates for), used here as a fallback splitter when the primary heuristic finds
# nothing. "OUTLINE:" (present in this era, absent in the ALL-CAPS era) supplies a
# usable synthetic title.
REPORT_ENDS_RE = re.compile(r"report\s+ends", re.IGNORECASE)
OUTLINE_RE = re.compile(r"OUTLINE\s*:\s*(.+)", re.IGNORECASE)


def stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.md5(("|".join(parts)).encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def extract_pages_cached(pdf_path: Path) -> list[str]:
    """Same cache-to-disk pattern as screen_incidents.py's extract_pages_cached --
    independent cache dir since this isn't under OOW_Incidents.

    Uses PyMuPDF, NOT pdfplumber (unlike every other ingest script in this project) --
    these newsletters are 2-column, and pdfplumber's extract_text() sorts words
    primarily by vertical position across the FULL page width, interleaving the left
    and right columns line-by-line (confirmed directly: on CHIRP-MFB-52 page 3,
    pdfplumber's very first line reads "...Co-ordination Centre. RNLI - Yacht sailing
    and motor boats" -- two unrelated columns stitched into one sentence). PyMuPDF's
    default get_text("text") (no sort=True -- that mode was WORSE here, interleaving
    even more aggressively) follows the PDF's own content-stream block order, which for
    every newsletter tested reads the left column fully before the right column."""
    cache_file = TEXT_CACHE_DIR / f"{pdf_path.stem}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pymupdf.open(pdf_path) as doc:
            pages = [page.get_text("text") or "" for page in doc]
    except Exception as e:  # corrupt/scanned/encrypted PDF
        pages = [f"__EXTRACT_ERROR__: {e}"]
    cache_file.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def find_issue_number(pages: list[str], filename: str) -> str:
    # The filename is far more reliable than page text: a newsletter's OWN body often
    # cross-references a DIFFERENT past issue number (e.g. "as shown in Issue no. 14"),
    # which previously hijacked a bare page-text search and caused several distinct
    # newsletters to collide onto the same document_id, silently overwriting each
    # other's output file -- confirmed the hard way (CHIRP-MFB-51/53/55 all resolved to
    # "5" and only the last-processed one survived). Try the filename FIRST.
    m = FILENAME_ISSUE_RE.search(filename)
    if m:
        return m.group(1)
    for ptext in pages[:1]:
        m = ISSUE_RE.search(ptext or "")
        if m:
            return m.group(1)
    return Path(filename).stem


def split_articles(full_text: str) -> list[tuple[str, str]]:
    """[(title, body)] -- body is everything from one HEADING_RE line up to (not
    including) the next one. The text before the first heading (masthead/editorial
    lead-in) is discarded -- it's never an article."""
    lines = full_text.split("\n")
    heading_idxs = [i for i, ln in enumerate(lines) if HEADING_RE.match(ln.strip())]
    articles = []
    for j, idx in enumerate(heading_idxs):
        title = lines[idx].strip()
        end = heading_idxs[j + 1] if j + 1 < len(heading_idxs) else len(lines)
        body = "\n".join(lines[idx + 1:end]).strip()
        articles.append((title, body))
    return articles


def split_articles_modern(full_text: str) -> list[tuple[str, str]]:
    """Fallback for the post-~2018 redesign (see REPORT_ENDS_RE's comment) -- one
    article per "Report Ends"-delimited segment; the text after the FINAL "Report
    Ends" is trailing boilerplate (contact details, disclaimer, ...), never a report,
    so it's dropped."""
    parts = REPORT_ENDS_RE.split(full_text)
    articles = []
    for seg in parts[:-1]:
        seg = seg.strip()
        if not seg:
            continue
        m = OUTLINE_RE.search(seg)
        if m:
            title = m.group(1).strip()[:80]
        else:
            title = next((ln.strip() for ln in seg.splitlines() if ln.strip()), "Untitled report")[:80]
        articles.append((title, seg))
    return articles


def split_report_and_comment(body: str) -> list[tuple[str, str]]:
    """[(section_type, text)] -- "chirp_report" (the mariner's account /
    CHIRP's narrative summary) then, if present, "chirp_comment" (CHIRP's own
    expert commentary -- the lesson)."""
    m = CHIRP_COMMENT_RE.search(body)
    if not m:
        return [("chirp_report", body)]
    report_part = body[:m.start()].strip()
    comment_part = body[m.start():].strip()
    sections = []
    if report_part:
        sections.append(("chirp_report", report_part))
    if comment_part:
        sections.append(("chirp_comment", comment_part))
    return sections


def build_document(pdf_path: Path) -> tuple[dict, dict]:
    """Returns (document, stats) -- stats records how many candidate articles were
    found/kept/dropped for the Phase-2 screening-distribution report."""
    raise_if_excluded_source(pdf_path.name)
    pages = extract_pages_cached(pdf_path)
    issue_no = find_issue_number(pages, pdf_path.name)
    doc_id = f"chirp_mfb_{issue_no.zfill(3)}"
    full_text = "\n".join(pages)

    # Trigger on the MARKER's presence, not on the ALL-CAPS splitter's own output
    # quality -- with PyMuPDF's cleaner column order, the ALL-CAPS heuristic no longer
    # reliably returns EMPTY on modern-format issues, it just matches the wrong things
    # (sidebar/footer captions like "ONLINE" instead of real articles, confirmed on
    # CHIRP-MFB-52), so "zero good candidates" is no longer a reliable modern-format
    # signal. >=2 "Report Ends" occurrences is: this newsletter uses that convention.
    if len(REPORT_ENDS_RE.findall(full_text)) >= 2:
        candidates = split_articles_modern(full_text)
        split_mode = "report_ends"
    else:
        candidates = split_articles(full_text)
        split_mode = "allcaps"
    stats = {"issue": issue_no, "file": pdf_path.name, "n_candidates": len(candidates),
             "split_mode": split_mode, "n_kept": 0, "n_dropped_short": 0, "n_dropped_score": 0}
    chapters = []
    for title, body in candidates:
        if len(body) < MIN_BODY_CHARS:
            stats["n_dropped_short"] += 1
            continue
        net_score = score_pages([body])["net_score"]
        if net_score < 0:
            stats["n_dropped_score"] += 1
            continue
        stats["n_kept"] += 1
        sections = []
        for sec_type, text in split_report_and_comment(body):
            sections.append({
                "section_id": stable_id("oow", doc_id, title, sec_type),
                "title": f"{title} ({'CHIRP comment' if sec_type == 'chirp_comment' else 'report'})",
                "type": sec_type,
                "text": text,
                "concepts": [], "topics": [], "pages": [],
            })
        chapters.append({"title": title, "sections": sections})

    document = {
        "document_id": doc_id,
        "source_file": pdf_path.name,
        "source_type": "chirp_newsletter",
        "chapters": chapters,
    }
    return document, stats


def main() -> None:
    pdfs = sorted(SRC_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} CHIRP newsletters under {SRC_DIR}")
    all_stats = []
    n_written = 0
    seen_doc_ids: dict[str, str] = {}
    for pdf_path in pdfs:
        document, stats = build_document(pdf_path)
        all_stats.append(stats)
        doc_id = document["document_id"]
        if doc_id in seen_doc_ids:
            raise RuntimeError(
                f"document_id collision: {pdf_path.name!r} and {seen_doc_ids[doc_id]!r} "
                f"both resolved to {doc_id!r} -- one would silently overwrite the other's "
                "output file. Fix find_issue_number() before proceeding."
            )
        seen_doc_ids[doc_id] = pdf_path.name
        if not document["chapters"]:
            print(f"  [skip] {pdf_path.name}: 0 articles kept "
                  f"(candidates={stats['n_candidates']}, "
                  f"dropped_short={stats['n_dropped_short']}, dropped_score={stats['n_dropped_score']})")
            continue
        out = JSON_OUT_DIR / f"{document['document_id']}.json"
        out.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        n_written += 1
        n_sections = sum(len(c["sections"]) for c in document["chapters"])
        print(f"  {pdf_path.name:<14} issue={stats['issue']:>3s} mode={stats['split_mode']:<11s} "
              f"candidates={stats['n_candidates']:>3d} kept={stats['n_kept']:>3d} "
              f"sections={n_sections:>3d} -> {out.name}")

    total_candidates = sum(s["n_candidates"] for s in all_stats)
    total_kept = sum(s["n_kept"] for s in all_stats)
    total_dropped_short = sum(s["n_dropped_short"] for s in all_stats)
    total_dropped_score = sum(s["n_dropped_score"] for s in all_stats)
    print(f"\n{n_written}/{len(pdfs)} newsletters produced >=1 kept article")
    print(f"Articles: {total_candidates} candidates -> {total_dropped_short} dropped (too short/section-header), "
          f"{total_dropped_score} dropped (net_score<0), {total_kept} kept")


if __name__ == "__main__":
    main()
