"""COLREG-Consolidated-2018.pdf -> structured JSON (OOW § 8 equivalent).

VHF's § 8 (source-document -> JSON) lives entirely as in-notebook cells because it
has to cope with ~30 heterogeneous documents (txt/md/pdf, many publishers, no fixed
structure). OOW's Track 1 source is a single, very regularly structured official
text (Convention Articles I-IX, then PART A-E / Rule 1-38, then Annexes I-IV), so a
small standalone parser is simpler and more robust than forcing it through the
generic multi-document classifier.

Output schema matches the convention `pipeline/ingest/build_rag.py` expects:
    {
      "document_id": "colreg_consolidated_2018",
      "source_file": "COLREG-Consolidated-2018.pdf",
      "source_type": "regulation",
      "chapters": [
        {"title": "PART A - GENERAL", "sections": [
            {"section_id": "...", "title": "Rule 1 - Application", "type": "rule",
             "text": "...", "concepts": [...], "topics": [...], "pages": [5]}
        ]}
      ]
    }

Run with: python -m pipeline.ingest.build_oow_json
"""
from __future__ import annotations
import json, re, hashlib
from pathlib import Path

import pdfplumber

from core import AgentPaths

paths = AgentPaths.from_env()
JSON_OUT_DIR = paths.json_dir
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_FILE = paths.source_dir / "COLREG-Consolidated-2018.pdf"
OUT_FILE = JSON_OUT_DIR / "colreg_consolidated_2018.json"

PART_RE   = re.compile(r"^PART\s+([A-E])\s*[-\u2013]\s*(.+)$")
ARTICLE_RE = re.compile(r"^ARTICLE\s+([IVXLC]+)\s*$")
RULE_RE   = re.compile(r"^(?:RULE|Rule)\s+(\d{1,2})\s*$")
ANNEX_RE  = re.compile(r"^ANNEX\s+([IVX]+)\s*[-\u2013]?\s*(.*)$")
SECTION_RE = re.compile(r"^Section\s+([IVX]+)\s*[-\u2013]?\s*(.*)$", re.IGNORECASE)
# Top-level numbered headings inside an Annex (e.g. "1. Definition",
# "2. Vertical positioning and spacing of lights") -- annexes have no
# "Rule N" markers of their own, so without this an entire 10-20 page
# annex would collapse into a single oversized RAG chunk.
ANNEX_HEADING_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z].{0,80})$")

# Rough keyword -> concept/topic tags, scanned over each section's own text
# (no external knowledge -- purely a coarse index for KG retrieval, same role
# CONCEPT_ALIASES plays for VHF).
CONCEPT_KEYWORDS = {
    "give-way":              ["give way", "give-way"],
    "stand-on":              ["stand-on", "stand on vessel", "keep her course"],
    "risk of collision":     ["risk of collision"],
    "overtaking":            ["overtak"],
    "head-on":               ["head-on", "reciprocal or nearly reciprocal"],
    "crossing":              ["crossing situation", "crossing"],
    "restricted visibility": ["restricted visibility"],
    "narrow channel":        ["narrow channel", "fairway"],
    "traffic separation scheme": ["traffic separation scheme"],
    "sound signal":          ["whistle", "sound signal", "bell", "gong"],
    "light":                 ["light", "lights"],
    "shape":                 ["shape", "shapes", "ball", "cone", "cylinder"],
    "not under command":     ["not under command"],
    "restricted in ability to manoeuvre": ["restricted in her ability to manoeuvre"],
    "constrained by draught": ["constrained by her draught"],
    "fishing vessel":        ["fishing vessel", "engaged in fishing"],
    "sailing vessel":        ["sailing vessel", "vessel under sail"],
    "anchored":              ["at anchor", "aground"],
    "safe speed":            ["safe speed"],
    "lookout":               ["proper look-out", "look-out"],
    "action to avoid collision": ["action to avoid collision"],
}


def stable_id(*parts) -> str:
    h = hashlib.md5(("|".join(str(p) for p in parts)).encode()).hexdigest()[:8]
    return f"oow_{h}"


def tag_text(text: str) -> tuple[list[str], list[str]]:
    """Coarse concept/topic tagging by keyword search (topics == concepts here;
    COLREG doesn't need the VHF-style two-tier split)."""
    low = text.lower()
    hits = sorted({tag for tag, kws in CONCEPT_KEYWORDS.items() if any(kw in low for kw in kws)})
    return hits, hits


def extract_pages(pdf_path: Path) -> list[list[str]]:
    """Return a list of (list-of-lines) per page."""
    pages: list[list[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append([ln.strip() for ln in text.split("\n") if ln.strip()])
    return pages


def parse_document(pdf_path: Path) -> dict:
    pages = extract_pages(pdf_path)

    chapters: list[dict] = []
    cur_chapter: dict | None = None
    cur_section: dict | None = None
    cur_lines: list[str] = []
    cur_pages: set[int] = set()

    def flush_section():
        nonlocal cur_section, cur_lines, cur_pages
        if cur_section is not None:
            text = "\n".join(cur_lines).strip()
            concepts, topics = tag_text(text)
            cur_section["text"] = text
            cur_section["concepts"] = concepts
            cur_section["topics"] = topics
            cur_section["pages"] = sorted(cur_pages)
            if cur_chapter is not None and text:
                cur_chapter["sections"].append(cur_section)
        cur_section = None
        cur_lines = []
        cur_pages = set()

    def new_chapter(title: str):
        nonlocal cur_chapter
        flush_section()
        cur_chapter = {"title": title, "sections": []}
        chapters.append(cur_chapter)

    def new_section(section_id: str, title: str, sec_type: str):
        nonlocal cur_section, cur_lines, cur_pages
        flush_section()
        cur_section = {"section_id": section_id, "title": title, "type": sec_type}

    # Preamble chapter for the Convention Articles (I-IX), before PART A starts.
    new_chapter("Convention Articles")
    pending_article_title_next_line = False
    article_num = None

    for page_idx, lines in enumerate(pages, start=1):
        i = 0
        while i < len(lines):
            line = lines[i]

            m_part = PART_RE.match(line)
            m_annex = ANNEX_RE.match(line)
            m_article = ARTICLE_RE.match(line)
            m_rule = RULE_RE.match(line)
            m_section = SECTION_RE.match(line)
            m_annex_heading = (
                ANNEX_HEADING_RE.match(line)
                if cur_chapter is not None and cur_chapter["title"].startswith("ANNEX")
                else None
            )

            if m_part:
                new_chapter(f"PART {m_part.group(1)} - {m_part.group(2).strip()}")
            elif m_annex:
                title = f"ANNEX {m_annex.group(1)}" + (f" - {m_annex.group(2).strip()}" if m_annex.group(2).strip() else "")
                new_chapter(title)
            elif m_annex_heading:
                num, heading = m_annex_heading.group(1), m_annex_heading.group(2).strip()
                new_section(stable_id("annex_heading", cur_chapter["title"], num), heading, "reference")
                cur_pages.add(page_idx)
            elif m_section:
                # Sub-heading within the current PART; fold into the running
                # section text as a marker rather than starting a new chapter.
                if cur_section is not None:
                    cur_lines.append(f"[{line}]")
                else:
                    cur_lines.append(f"[{line}]")
            elif m_article:
                article_num = m_article.group(1)
                pending_article_title_next_line = True
                i += 1
                continue
            elif pending_article_title_next_line:
                new_section(stable_id("article", article_num), f"Article {article_num} - {line}", "article")
                pending_article_title_next_line = False
            elif m_rule:
                rule_num = m_rule.group(1)
                # Title is usually the next non-empty line.
                title_line = lines[i + 1] if i + 1 < len(lines) else f"Rule {rule_num}"
                new_section(stable_id("rule", rule_num), f"Rule {rule_num} - {title_line}", "rule")
                cur_pages.add(page_idx)
                i += 2
                continue
            else:
                if cur_section is None:
                    # Text before the first Article/Rule heading of a chapter
                    # (e.g. annex preambles) -- start a generic section for it.
                    new_section(stable_id("misc", len(chapters), len(cur_lines)), cur_chapter["title"], "reference")
                cur_lines.append(line)
                cur_pages.add(page_idx)
            i += 1

    flush_section()
    # Drop any empty chapters (e.g. if a PART header had no body text captured).
    chapters = [c for c in chapters if c["sections"]]

    return {
        "document_id": "colreg_consolidated_2018",
        "source_file": pdf_path.name,
        "source_type": "regulation",
        "publisher": "IMO",
        "language": "en",
        "chapters": chapters,
    }


def main() -> None:
    if not PDF_FILE.exists():
        raise SystemExit(f"Not found: {PDF_FILE}")
    doc = parse_document(PDF_FILE)
    n_sections = sum(len(c["sections"]) for c in doc["chapters"])
    print(f"Parsed {PDF_FILE.name}: {len(doc['chapters'])} chapters, {n_sections} sections")
    for c in doc["chapters"]:
        print(f"  {c['title']:<60} sections={len(c['sections'])}")
    OUT_FILE.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
