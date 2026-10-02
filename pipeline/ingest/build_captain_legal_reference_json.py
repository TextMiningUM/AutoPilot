"""Data/Captain/Legal_Reference/{ecfr,uk_legislation,mars,tsb_canada}/ -> structured
JSON (design_captain_missions.md Sec 16.6/14.1 Tier 1 -- the remainder of the already-
acquired corpus build_captain_json.py explicitly deferred, different formats: XML/HTML).

Four source formats, one parser function each (reused nowhere else, so kept together
in this one file rather than 4 separate scripts -- each format appears in exactly one
subfolder):
  - eCFR XML (US Code of Federal Regulations, 46/33 CFR parts): DIV5/DIV6/DIV8 nested
    structure, no XML namespace. One section per DIV8 ("SECTION").
  - UK legislation XML (legislation.gov.uk's own schema): P1group/P1/P2/... nested
    structure, WITH a default XML namespace (stripped post-parse so plain tag names
    work). One section per P1group (flattens Part/Schedule nesting -- acceptable for
    v1, matches the "let build_rag.py's own semantic splitter handle finer structure"
    approach already used for the Navy yearbooks).
  - MARS HTML (Nautical Institute near-miss reports): full rendered webpage, real
    content lives in `div.entry` (title + body paragraphs + "Lessons learned" list).
  - TSB Canada HTML (marine investigation summary pages): real content lives in the
    `<article>` tag (occurrence summary + "The occurrence" narrative + investigation
    metadata).

All 4 are already in English -- unlike the Navy yearbooks (Dutch), these write
DIRECTLY into Data/Captain/Captain_JSON/ (what build_rag.py globs), no translation step.

Local-safe: pure Python + stdlib xml.etree + beautifulsoup4, no GPU/API key needed.

Run with: python -m pipeline.ingest.build_captain_legal_reference_json
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bs4 import BeautifulSoup

from core import AgentPaths
from pipeline.ingest.build_vhf_json import stable_id

paths = AgentPaths.captain()
LEGAL_REF_DIR = paths.data_root / "Legal_Reference"
JSON_OUT_DIR = paths.json_dir

_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _strip_namespaces(root: ET.Element) -> ET.Element:
    """Removes '{uri}' prefixes from every tag so plain tag names (e.g. 'P1group')
    work regardless of whether the source XML declared a default namespace."""
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


# ── eCFR XML ─────────────────────────────────────────────────────────────
def parse_ecfr_xml(path: Path) -> dict | None:
    root = ET.parse(path).getroot()
    part_title_el = root.find("HEAD")
    part_title = _clean(part_title_el.text or "") if part_title_el is not None else path.stem

    sections = []
    for div8 in root.iter("DIV8"):
        head = div8.find("HEAD")
        title = _clean(head.text or "") if head is not None else div8.get("N", "section")
        text = _clean("".join(div8.itertext()))
        if not text:
            continue
        citation = div8.get("N", "")
        sections.append({
            "section_id": stable_id("ecfr", path.stem, citation or title),
            "title": title, "type": "regulation", "text": text,
            "concepts": [], "topics": [], "pages": [],
        })
    if not sections:
        return None
    return {
        "document_id": path.stem,
        "source_file": path.name,
        "source_type": "regulation",
        "publisher": "US Government (eCFR)",
        "language": "en",
        "chapters": [{"title": part_title, "sections": sections}],
    }


# ── UK legislation XML ───────────────────────────────────────────────────
def parse_uk_legislation_xml(path: Path) -> dict | None:
    root = _strip_namespaces(ET.parse(path).getroot())
    title_el = root.find(".//Title")
    doc_title = _clean(title_el.text or "") if title_el is not None else path.stem

    sections = []
    for p1group in root.iter("P1group"):
        title_el = p1group.find("Title")
        title = _clean(title_el.text or "") if title_el is not None else "Untitled regulation"
        texts = [t.text for t in p1group.iter("Text") if t.text]
        text = _clean(" ".join(texts))
        if not text:
            continue
        sections.append({
            "section_id": stable_id("uk_legislation", path.stem, title, str(len(sections))),
            "title": title, "type": "regulation", "text": text,
            "concepts": [], "topics": [], "pages": [],
        })
    if not sections:
        return None
    return {
        "document_id": f"uk_legislation_{path.stem}",
        "source_file": path.name,
        "source_type": "regulation",
        "publisher": "UK Government (legislation.gov.uk, OGL v3.0)",
        "language": "en",
        "chapters": [{"title": doc_title, "sections": sections}],
    }


# ── MARS HTML (Nautical Institute near-miss reports) ─────────────────────
def parse_mars_html(path: Path) -> dict | None:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    entry = soup.find("div", class_="entry")
    if entry is None:
        return None
    title_el = entry.find("h1")
    title = _clean(title_el.get_text()) if title_el else path.stem

    body_paragraphs = [_clean(p.get_text(separator=" ")) for p in entry.find_all("p")]
    body_paragraphs = [p for p in body_paragraphs if p]
    report_text = " ".join(body_paragraphs)

    lessons = [_clean(li.get_text(separator=" ")) for li in entry.find_all("li")]
    lessons = [l for l in lessons if l]

    sections = []
    if report_text:
        sections.append({
            "section_id": stable_id("mars", path.stem, "report"),
            "title": title, "type": "incident_report", "text": report_text,
            "concepts": [], "topics": [], "pages": [],
        })
    if lessons:
        sections.append({
            "section_id": stable_id("mars", path.stem, "lessons"),
            "title": f"{title} (Lessons learned)", "type": "incident_analysis",
            "text": " ".join(lessons), "concepts": [], "topics": [], "pages": [],
        })
    if not sections:
        return None
    return {
        "document_id": path.stem if path.stem.startswith("mars_") else f"mars_{path.stem}",
        "source_file": path.name,
        "source_type": "incident_report",
        "publisher": "Nautical Institute (MARS)",
        "language": "en",
        "chapters": [{"title": title, "sections": sections}],
    }


# ── TSB Canada HTML (marine investigation summary pages) ─────────────────
def parse_tsb_html(path: Path) -> dict | None:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    article = soup.find("article")
    if article is None:
        return None
    title_el = soup.find("h1")
    title = _clean(title_el.get_text()) if title_el else path.stem

    sections = []
    # Each "field--name-body" div holds one labelled sub-section (its own h2/h3 + text).
    for i, field in enumerate(article.find_all("div", class_="field--name-body")):
        heading_el = field.find(["h2", "h3"])
        sub_title = _clean(heading_el.get_text(separator=" ")) if heading_el else f"{title} (part {i + 1})"
        text = _clean(field.get_text(separator=" "))
        if not text:
            continue
        sections.append({
            "section_id": stable_id("tsb", path.stem, str(i)),
            "title": sub_title, "type": "incident_report", "text": text,
            "concepts": [], "topics": [], "pages": [],
        })
    if not sections:
        return None
    return {
        "document_id": f"tsb_{path.stem}",
        "source_file": path.name,
        "source_type": "incident_report",
        "publisher": "Transportation Safety Board of Canada",
        "language": "en",
        "chapters": [{"title": title, "sections": sections}],
    }


def main() -> None:
    JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (LEGAL_REF_DIR / "ecfr", "*.xml", parse_ecfr_xml),
        (LEGAL_REF_DIR / "uk_legislation", "*.xml", parse_uk_legislation_xml),
        (LEGAL_REF_DIR / "mars", "*.html", parse_mars_html),
        (LEGAL_REF_DIR / "tsb_canada", "*.html", parse_tsb_html),
    ]
    written = 0
    for src_dir, glob_pat, parser_fn in jobs:
        if not src_dir.exists():
            print(f"  [skip] {src_dir} not found")
            continue
        for path in sorted(src_dir.glob(glob_pat)):
            doc = parser_fn(path)
            if not doc:
                print(f"  [skip] {path.relative_to(LEGAL_REF_DIR)}: no usable text extracted")
                continue
            out = JSON_OUT_DIR / f"{doc['document_id']}.json"
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            written += 1
            n_sections = sum(len(c["sections"]) for c in doc["chapters"])
            print(f"  {path.relative_to(LEGAL_REF_DIR)!s:<45} sections={n_sections:>3d} -> {out.name}")

    print(f"\n{written} documents written -> {JSON_OUT_DIR}")


if __name__ == "__main__":
    main()
