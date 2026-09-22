"""CHIRP Maritime FEEDBACK newsletters -> structured JSON (Phase 2 of the OOW RAG
rebuild, 2026-09-22).

CHIRP is a confidential near-miss reporting programme; each newsletter bundles several
independent reports (a mariner's own account, "Report Text:"/"CHIRP Narrative:") plus
CHIRP's expert commentary ("CHIRP Comment:") -- the commentary IS the lesson, so it's
kept as its own labelled section, not merged into the report.

SPLIT HEURISTIC: CHIRP's format drifted through 5 distinct eras over ~20 years (found
by manually inspecting one issue per era after the initial 3-issue sample, MFB-3/2004,
MFB-16/2007, MFB-33/2013, stopped generalizing to newer issues -- see STORY_START_RE's
comment for the full era breakdown). Two splitters cover all of them:
  - split_articles(): pre-2018 (ALL-CAPS titles, era 1) -- every ALL-CAPS line of 2-8
    words with no digits/colon is an article (or section-header) boundary. Headers like
    "COMMERCIAL SECTOR REPORTS" match the same pattern but have no body text before the
    next such line, so they naturally become empty/near-empty segments and are dropped
    by the minimum-body-length filter -- no separate stoplist needed.
  - split_articles_modern(): 2018-2022 (eras 2-5, Title Case titles) -- anchored on
    whichever "here is the reporter's own account" opening phrase that era uses
    (STORY_START_RE), since none of them use ALL-CAPS titles.

Within one article's body, "CHIRP Comment:" (case-insensitive) splits the mariner's own
report from CHIRP's commentary; everything before it is type="chirp_report", everything
from it onward is type="chirp_comment". Articles with no such marker are entirely
type="chirp_report" -- but comment-less articles need a HIGHER minimum length
(MIN_BODY_CHARS_NO_COMMENT) than articles with one, since PyMuPDF's block-level
extraction otherwise lets page headers/tables-of-contents/copyright notices through as
spurious "articles" (confirmed: 465/778 kept articles had no comment section, and
samples up to ~470 chars were ALL such noise, not genuine comment-less reports).

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
from pipeline.ingest.build_oow_json import tag_text
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
# Chapters with NO chirp_comment section need a HIGHER bar -- confirmed by sampling:
# genuine comment-less reports/follow-ups run 500+ chars, while page headers, tables
# of contents, copyright notices, and a recurring BLANK report-submission-form
# template ("NAME: / ADDRESS: / POST CODE: ...") that gets reprinted in multiple
# issues all cluster under it (samples up to ~470 chars were all noise; residual
# blank-form duplicates above this bar are a known, accepted gap -- a length filter
# alone can't catch a template that happens to be long).
MIN_BODY_CHARS_NO_COMMENT = 500
CHIRP_COMMENT_RE = re.compile(r"chirp\s+comment\b\s*:?", re.IGNORECASE)
ISSUE_RE = re.compile(r"(?:Issue\s+No|No)\s*:?\s*(\d+)", re.IGNORECASE)
FILENAME_ISSUE_RE = re.compile(r"MFB[-_]?(\d+)", re.IGNORECASE)
# CHIRP's newsletter format drifted through (at least) 5 distinct eras over ~20 years,
# each with its own "here is the reporter's own account" phrasing -- found by directly
# inspecting one issue per era (MFB-44, -52, -56, -64) after the ALL-CAPS heuristic
# stopped finding good candidates on them:
#   era 1 (pre-2018, ALL-CAPS titles):      "Report Text:" / "Report text:"
#   era 2 (2018, MFB44-45):                 "What did the reporters tell us?"
#   era 3 (2018 redesign, MFB46-55):        "What the Reporter told us (N):" + a
#                                            closing "Report Ends" marker + "OUTLINE:"
#   era 4 (2019-2021, MFB56-63):            "What the reporter told us:" (no closing
#                                            marker, "OUTLINE:" only on some issues)
#   era 5 (2021-2022, MFB64-67):            "Initial Report" + a reference ID ("M1761")
# Eras 2-5 share no single marker, but each of eras 2/3/4 always uses SOME variant of
# "what (did the reporter(s) tell|the reporter(s) told) us", and era 5 always uses
# "Initial Report" -- combined here as one anchor set. Era 1's "Report Text:" is
# deliberately NOT included: it also appears inside era-1 articles the ALL-CAPS
# splitter already handles correctly, so including it here would double-trigger.
STORY_START_RE = re.compile(
    r"what\s+(?:did\s+the\s+reporters?\s+tell\s+us|the\s+reporters?\s+told\s+us)|initial\s+report",
    re.IGNORECASE,
)
# Kept only as an internal synthetic-title source within split_articles_modern (still
# present on eras 3/4) -- no longer used as the split trigger itself (see build_document).
OUTLINE_RE = re.compile(r"OUTLINE\s*:\s*(.+)", re.IGNORECASE)
# Non-printable control characters PyMuPDF occasionally emits (font-encoding/ligature
# artifacts) that add nothing and can break whitespace-sensitive regexes -- strip
# everything below 0x20 except the newline that carries line structure.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")


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
            pages = [_CONTROL_CHAR_RE.sub("", page.get_text("text") or "") for page in doc]
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
    """Fallback for post-2018 issues (eras 2-5, see STORY_START_RE's comment) -- one
    article per STORY_START_RE match: body runs from that match to the next match (or
    end of document); the text before the FIRST match (masthead/editorial lead-in) is
    discarded, same reasoning as split_articles(). Anchoring on the OPENING marker
    (not "Report Ends", which era 4/5 issues don't even have) is what makes this work
    uniformly across all four modern eras."""
    starts = [m.start() for m in STORY_START_RE.finditer(full_text)]
    articles = []
    for j, start in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(full_text)
        seg = full_text[start:end].strip()
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

    # Trigger on STORY_START_RE's presence (an opening-marker signal every modern era
    # shares in some form), not on the ALL-CAPS splitter's own output quality -- with
    # PyMuPDF's cleaner column order, ALL-CAPS no longer reliably returns EMPTY on
    # modern-format issues, it just matches the wrong things (sidebar/footer captions
    # like "ONLINE" instead of real articles, confirmed on CHIRP-MFB-52).
    if len(STORY_START_RE.findall(full_text)) >= 2:
        candidates = split_articles_modern(full_text)
        split_mode = "story_start"
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
        sections_raw = split_report_and_comment(body)
        has_comment = any(t == "chirp_comment" for t, _ in sections_raw)
        if not has_comment and len(body) < MIN_BODY_CHARS_NO_COMMENT:
            stats["n_dropped_short"] += 1
            continue
        stats["n_kept"] += 1
        sections = []
        for sec_type, text in sections_raw:
            concepts, topics = tag_text(text)
            sections.append({
                "section_id": stable_id("oow", doc_id, title, sec_type),
                "title": f"{title} ({'CHIRP comment' if sec_type == 'chirp_comment' else 'report'})",
                "type": sec_type,
                "text": text,
                "concepts": concepts, "topics": topics, "pages": [],
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
