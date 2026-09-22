"""COLREG-Consolidated-2018.pdf + supplementary OOW_Protocols docs -> structured JSON.

OOW's Track 1 core source is a single, very regularly structured official text
(Convention Articles I-IX, then PART A-E / Rule 1-38, then Annexes I-IV), so a small
standalone parser (`parse_document`/`main`) is simpler and more robust than forcing it
through a generic multi-document classifier -- unlike VHF's ~45 heterogeneous documents
(txt/md/pdf, many publishers, no fixed structure), which do need one (see
`build_vhf_json.py`).

`build_extra_docs()` (called from `main()` too) handles everything else dropped into
OOW_Protocols/ that ISN'T the raw COLREG text -- a handful of navigation-maths/
radar-plotting reference docs (bearings, CPA/TCPA, compass conventions) added to fix a
confirmed TCPA-misread gap found via Basic Simulator mission runs (see
`Data/basic_nav_knowledge_gaps.json`), plus `simple_colreg.json` (added later): a
plain-English, example-driven explanation of each COLREG rule, pre-chunked one entry per
rule by whoever authored it, as a companion to (not a replacement for) the verbatim
`COLREG-Consolidated-2018.pdf` text parsed by `parse_document()` above. These are few and
each has its own clear structure, so -- same reasoning as the COLREG parser -- each gets
its own small, specific parser (`parse_markdown_doc` for the plain-heading .md references,
`parse_radar_workbook` for the "Lesson N.N" PDF, `parse_navmath_drills` for the
already-Q&A-shaped drill set, `parse_simple_colreg` for the already-chunked-per-rule
explainer) rather than one generic classifier for just a handful of files.

Output schema matches the convention `pipeline/ingest/build_rag.py` expects (one JSON
per document, glob'd automatically -- no wiring needed elsewhere):
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

PART_RE   = re.compile(r"^PART\s+([A-Z])\s*[-\u2013]\s*(.+)$")
ARTICLE_RE = re.compile(r"^ARTICLE\s+([IVXLC]+)\s*$")
# Trailing content is optional: some rules have an amendment annotation on the
# same line (e.g. "Rule 39 (Added by Res.A.1085(28))") -- without this, those
# rules were silently swallowed into the tail of the PRECEDING rule's section
# instead of getting their own (found via check_consistency_colreg.py's
# invalid_rule_number check flagging Rules 39-41 as missing from the map).
RULE_RE   = re.compile(r"^(?:RULE|Rule)\s+(\d{1,2})\b\s*(?:\(.+\))?\s*$")
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
    # Incident-report vocabulary (build_incident_excerpts.py reuses tag_text() on
    # accident-investigation excerpts -- these terms describe HUMAN-FACTORS causes
    # real incidents attribute failures to, which the COLREG-rule keywords above
    # never mention, so without these every incident chunk went untagged (see
    # notebook's OOW § 4/§ 8 KG/PG quality investigation). Aligned with the fixed
    # fault_type vocabulary in extract_incident_reasoning.py so KG concepts and
    # incident fault categories use the same vocabulary.
    "lookout_failure":       ["failed to maintain a proper look-out", "inadequate lookout",
                              "no lookout was posted", "lookout was not maintained"],
    "distraction":           ["distracted by", "was distracted", "preoccupied with"],
    "equipment_failure":     ["equipment failure", "malfunctioned", "failed to operate",
                              "instrument failure", "radar failure"],
    "procedural_failure":    ["failed to follow", "did not comply with", "procedure was not followed",
                              "deviated from standard procedure"],
    "communication_failure": ["failed to communicate", "no vhf contact", "miscommunication",
                              "communication breakdown", "failed to establish contact"],
    "fatigue":               ["fatigue", "fatigued", "asleep", "excessive hours", "watchkeeper was tired"],
    "environmental":         ["heavy weather", "poor visibility", "adverse weather", "sea state"],
    "radar_arpa":            ["radar", "arpa", "plotting"],
    "ais":                   ["ais", "automatic identification system"],
    "bridge_resource_management": ["bridge resource management", "brm", "bridge team"],
    # Added after inspecting concept-tag coverage on the new CHIRP/MOOS/marginal-incident
    # sections (RAG rebuild Phase A2, 2026-09-22): already validated as COLREG-relevant
    # in screen_incidents.py's own COLREG_KEYWORDS but missing here, the one clear gap
    # found -- everything else untagged was low-value boilerplate (masthead/submission-
    # form text), not missed real content.
    "vessel_traffic_service": ["vessel traffic service", " vts "],
    # Added after inspecting concept-tag coverage on the RAG-rebuild-v2 CHIRP corpus
    # (A-nawerk-3, 2026-09-22): scanned all 367 then-untagged chirp_newsletter chunks
    # for candidate terms -- "near miss"/"near-miss" (13 hits) and "close quarters" (a
    # direct Rule 19(e) term, 1 hit but unambiguous) were the only genuinely COLREG-
    # collision-avoidance-relevant gaps; frequent but off-topic terms in the same scan
    # (pilot 58, tug 22, mooring 11, bare "starboard"/"port side") were deliberately
    # NOT added -- those describe pilotage/berthing seamanship, not rule-of-the-road
    # collision avoidance, and "starboard"/"port" alone are too generic (false-positive
    # on any passing mention) to make a useful concept tag.
    "near_miss":              ["near miss", "near-miss"],
    "close_quarters":         ["close-quarters", "close quarters situation", "close quarters"],
    # Bare "grounded" dropped: false-positives on "grounded in <source>" (a citation
    # phrase, not the nautical sense) once nav_maths_drills.json's explanation fields
    # (which all start "Grounded in the tdgil.com ...") were added -- "grounding"/"ran
    # aground" alone still catch real incident-report groundings just as well.
    "grounding":              ["grounding", "ran aground"],
    "collision":              ["collision occurred", "vessels collided", "struck the"],
    "casualty":               ["casualty", "fatality", "injured", "loss of life"],
    "investigation_finding":  ["investigation found", "contributing factor", "root cause",
                              "recommendation", "lessons learned"],
    # Navigation-maths/radar-plotting vocabulary (compass_directions_reference.md,
    # tdgil_bearings.md, tdgil_cpa.md, free_radar_workbook.pdf, nav_maths_drills.json --
    # see build_extra_docs()) -- added alongside those docs since none of the COLREG-rule
    # or incident-report keywords above ever appear in them, which would otherwise leave
    # every section of these new docs completely untagged (same KG/PG quality problem
    # already found and fixed for incident-report vocabulary, see the notebook's § 4).
    "true_bearing":       ["true bearing", "true north"],
    "magnetic_bearing":   ["magnetic bearing", "variation", "magnetic north", "declination"],
    "relative_bearing":   ["relative bearing"],
    "cpa_tcpa":           ["closest point of approach", " cpa ", "tcpa", "time to closest point"],
    "radar_plotting":     ["relative motion line", " rml ", " drm ", " srm ", "six-minute rule",
                          "6-minute rule", "6 minute rule", "transfer plot"],
    "compass_convention": ["compass rose", "clockwise", "counter-clockwise", "counterclockwise",
                          "compass convention"],
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


# ── Supplementary docs (build_extra_docs) ──────────────────────────────────
# Everything in OOW_Protocols/ that isn't the COLREG text -- see module docstring.

_MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def parse_markdown_doc(md_path: Path, document_id: str, source_type: str, publisher: str) -> dict:
    """Generic '#'/'##'/'###' heading splitter for plain-prose reference docs -- level-1
    headings start a new chapter, level-2/3 start a new section within it. Good enough for
    these few hand-written/fetched reference docs; COLREG's own regularly-structured text
    still gets its dedicated `parse_document()` above, same reasoning as the module
    docstring (a handful of clearly-structured docs don't need a generic classifier)."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    chapters: list[dict] = []
    cur_chapter: dict | None = None
    cur_section: dict | None = None
    cur_lines: list[str] = []

    def flush_section():
        nonlocal cur_section, cur_lines
        if cur_section is not None:
            text = "\n".join(cur_lines).strip()
            if text:
                concepts, topics = tag_text(text)
                cur_section.update(text=text, concepts=concepts, topics=topics, pages=[])
                cur_chapter["sections"].append(cur_section)
        cur_section, cur_lines = None, []

    def ensure_chapter(title: str):
        nonlocal cur_chapter
        flush_section()
        cur_chapter = {"title": title, "sections": []}
        chapters.append(cur_chapter)

    ensure_chapter(md_path.stem.replace("_", " ").title())  # replaced by the first real H1, if any
    for raw in lines:
        line = raw.rstrip()
        if line.strip() == "---":
            continue
        m = _MD_HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 1:
                ensure_chapter(title)
            else:
                flush_section()
                cur_section = {"section_id": stable_id(document_id, title), "title": title, "type": "reference"}
        else:
            if cur_section is None:
                cur_section = {"section_id": stable_id(document_id, cur_chapter["title"], len(cur_chapter["sections"])),
                               "title": cur_chapter["title"], "type": "reference"}
            cur_lines.append(line)
    flush_section()
    chapters = [c for c in chapters if c["sections"]]
    return {"document_id": document_id, "source_file": md_path.name, "source_type": source_type,
           "publisher": publisher, "language": "en", "chapters": chapters}


_LESSON_RE = re.compile(r"^Lesson\s+(\d+\.\d+)\s+(.+)$")
_TOC_DOTLEADER_RE = re.compile(r"\.{4,}")


def parse_radar_workbook(pdf_path: Path) -> dict:
    """free_radar_workbook.pdf-specific: splits on its "Lesson N.N <title>" headings (each
    appears twice -- once in the Table of Contents with a dot-leader, once as the real
    heading -- so ToC lines are dropped by the dot-leader regex before matching)."""
    pages = extract_pages(pdf_path)
    chapters = [{"title": "Radar Plotting Workbook", "sections": []}]
    cur_section: dict | None = None
    cur_lines: list[str] = []
    cur_pages: set[int] = set()

    def flush():
        nonlocal cur_section, cur_lines, cur_pages
        if cur_section is not None:
            text = "\n".join(cur_lines).strip()
            if text:
                concepts, topics = tag_text(text)
                cur_section.update(text=text, concepts=concepts, topics=topics, pages=sorted(cur_pages))
                chapters[0]["sections"].append(cur_section)
        cur_section, cur_lines, cur_pages = None, [], set()

    for page_idx, lines in enumerate(pages, start=1):
        for line in lines:
            if _TOC_DOTLEADER_RE.search(line):
                continue
            m = _LESSON_RE.match(line)
            if m:
                flush()
                num, title = m.group(1), m.group(2).strip()
                cur_section = {"section_id": stable_id("radar_lesson", num), "title": f"Lesson {num} {title}",
                              "type": "procedure"}
                cur_pages.add(page_idx)
            elif cur_section is not None:
                cur_lines.append(line)
                cur_pages.add(page_idx)
            # else: front-matter/title-page text before Lesson 1.1 -- dropped.
    flush()
    return {"document_id": "free_radar_workbook", "source_file": pdf_path.name, "source_type": "workbook",
           "publisher": "Columbia Pacific Maritime", "language": "en", "chapters": chapters}


def parse_navmath_drills(json_path: Path) -> dict:
    """nav_maths_drills.json is already {id, category, topic, question, answer, explanation,
    difficulty} Q&A -- unlike the prose docs above, there's nothing to chunk-detect; each
    drill becomes its own 'qa'-type section (grouped into chapters by `category`) so it still
    flows through § 3-§ 6 exactly like every other document instead of needing a bespoke
    bypass into SFT data."""
    drills = json.loads(json_path.read_text(encoding="utf-8"))
    by_category: dict[str, list[dict]] = {}
    for d in drills:
        by_category.setdefault(d.get("category", "General"), []).append(d)

    chapters = []
    for category, items in by_category.items():
        sections = []
        for d in items:
            text = f"Q: {d['question']}\nA: {d['answer']}"
            if d.get("explanation"):
                text += f"\n(Why: {d['explanation']})"
            concepts, topics = tag_text(text)
            sections.append({"section_id": stable_id("navmath", d["id"]), "title": d.get("topic", d["id"]),
                             "type": "qa", "text": text, "concepts": concepts, "topics": topics, "pages": []})
        chapters.append({"title": category, "sections": sections})
    return {"document_id": "nav_maths_drills", "source_file": json_path.name, "source_type": "drill_qa",
           "publisher": "Auto Pilot project (derived from tdgil.com)", "language": "en", "chapters": chapters}


def parse_simple_colreg(json_path: Path) -> dict:
    """simple_colreg.json is already one pre-chunked record PER RULE (chunk_id, part,
    rule_ref, title, plain_explanation, examples, official_text, chunk_text) -- a
    plain-English/example-driven companion to the verbatim colreg_consolidated_2018.json
    text above, not a replacement for it. Each record becomes its own 'rule'-type section
    (STANDALONE_TYPES in build_rag.py keeps every rule its own chunk, never merged with a
    neighbour), grouped into chapters by `part` exactly like the official text's PART/
    Section grouping. Uses `chunk_text` (title + plain_explanation + examples) as the
    indexed text rather than `official_text`, since the verbatim rule text is already
    covered by colreg_consolidated_2018.json -- indexing it twice would just duplicate the
    same legal text under two document_ids for no retrieval benefit."""
    records = json.loads(json_path.read_text(encoding="utf-8"))
    by_part: dict[str, list[dict]] = {}
    for r in records:
        by_part.setdefault(r.get("part", "General"), []).append(r)

    chapters = []
    for part, items in by_part.items():
        sections = []
        for r in items:
            text = r["chunk_text"]
            concepts, topics = tag_text(text)
            sections.append({
                "section_id": stable_id("simple_colreg", r["chunk_id"]),
                "title": f"{r['rule_ref']} - {r['title']}",
                "type": "rule", "text": text, "concepts": concepts, "topics": topics, "pages": [],
            })
        chapters.append({"title": part, "sections": sections})
    return {"document_id": "simple_colreg", "source_file": json_path.name,
           "source_type": "regulation_plain_explainer",
           "publisher": "Auto Pilot project (derived from COLREG-Consolidated-2018.pdf)",
           "language": "en", "chapters": chapters}


_EXTRA_MD_DOCS = [
    ("compass_directions_reference.md", "compass_directions_reference", "reference", "Auto Pilot project"),
    ("tdgil_bearings.md", "tdgil_bearings", "guide", "tdgil.com"),
    ("tdgil_cpa.md", "tdgil_cpa", "guide", "tdgil.com"),
]


def build_extra_docs() -> None:
    """Parses every non-COLREG doc in OOW_Protocols/ (see module docstring) into its own
    JSON file under OOW_JSON/ -- build_rag.py globs every *.json there, so no further
    wiring is needed for these to flow into § 3 (RAG)/§ 4 (KG)/§ 5 (reasoning traces)."""
    for filename, doc_id, source_type, publisher in _EXTRA_MD_DOCS:
        md_path = paths.source_dir / filename
        if not md_path.exists():
            print(f"  [skip] {filename} not found")
            continue
        doc = parse_markdown_doc(md_path, doc_id, source_type, publisher)
        n_sections = sum(len(c["sections"]) for c in doc["chapters"])
        out = JSON_OUT_DIR / f"{doc_id}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Parsed {filename}: {len(doc['chapters'])} chapters, {n_sections} sections -> {out.name}")

    radar_pdf = paths.source_dir / "free_radar_workbook.pdf"
    if radar_pdf.exists():
        doc = parse_radar_workbook(radar_pdf)
        n_sections = sum(len(c["sections"]) for c in doc["chapters"])
        out = JSON_OUT_DIR / "free_radar_workbook.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Parsed free_radar_workbook.pdf: {n_sections} sections -> {out.name}")
    else:
        print("  [skip] free_radar_workbook.pdf not found")

    drills_file = paths.source_dir / "nav_maths_drills.json"
    if drills_file.exists():
        doc = parse_navmath_drills(drills_file)
        n_sections = sum(len(c["sections"]) for c in doc["chapters"])
        out = JSON_OUT_DIR / "nav_maths_drills.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Parsed nav_maths_drills.json: {len(doc['chapters'])} categories, {n_sections} sections -> {out.name}")
    else:
        print("  [skip] nav_maths_drills.json not found")

    simple_colreg_file = paths.source_dir / "simple_colreg.json"
    if simple_colreg_file.exists():
        doc = parse_simple_colreg(simple_colreg_file)
        n_sections = sum(len(c["sections"]) for c in doc["chapters"])
        out = JSON_OUT_DIR / "simple_colreg.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Parsed simple_colreg.json: {len(doc['chapters'])} chapters, {n_sections} sections -> {out.name}")
    else:
        print("  [skip] simple_colreg.json not found")



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

    print()
    build_extra_docs()


if __name__ == "__main__":
    main()
