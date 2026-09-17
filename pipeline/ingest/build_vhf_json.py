"""VHF source documents (TXT/MD/PDF) -> structured JSON (extracted from notebook § 8-8.7).

VHF's Track 1 corpus is ~45 heterogeneous documents (guides, regulations, procedure
cards, cheatsheets) with no fixed structure, unlike OOW's single COLREG convention
text (see build_oow_json.py) -- hence the generic heading-detection + classification
approach here instead of a hand-rolled per-document parser.

Output schema matches what build_rag.py / build_kg.py expect (same as build_oow_json.py):
    {"document_id", "source_file", "source_type", "publisher", "language",
     "chapters": [{"chapter_id", "title", "sections": [
        {"section_id", "title", "type", "text", "concepts", "topics", "steps", "pages"}
     ]}], "parsing_notes", "metadata": {...}}

Run with: python -m pipeline.ingest.build_vhf_json
"""
from __future__ import annotations
import json, re, hashlib, statistics
from pathlib import Path
from typing import Optional
from collections import Counter

import pdfplumber

from core import AgentPaths

paths = AgentPaths.from_env()
DATA_DIR = paths.source_dir
JSON_OUT_DIR = paths.json_dir
CACHE_DIR = paths.cache_dir
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = JSON_OUT_DIR / "_index.json"
CLEANUP_MARKER_FILE = JSON_OUT_DIR / ".cleanup_done"
CONSISTENCY_EXCLUSIONS_FILE = CACHE_DIR / "consistency_exclusions.json"
CONSISTENCY_NORMALIZATIONS_FILE = CACHE_DIR / "consistency_normalizations.json"

# ── § 8.1a  File -> document classification ───────────────────────────────
# Hand-curated (source_type, publisher, language) per known file. Anything new
# dropped into DATA_DIR gets auto-added as ("reference", "Auto", "en") at import
# time (bottom of this section) so it flows through the pipeline unmodified.
SOURCE_CLASSIFICATION: dict[str, Optional[tuple[str, str, str]]] = {
    "BoatUS.txt":                         ("guide",       "BoatUS",              "en"),
    "CompleteGuideVHF.txt":               ("guide",       "Don Casey",           "en"),
    "MarinePublicVHF.txt":                ("guide",       "Public marine blog",  "en"),
    "NZC VHF.txt":                        ("protocol",    "North Sea Club",      "en"),
    "VHFPro.txt":                         ("guide",       "SMCP compendium",     "en"),
    "VHFProtocolsUS.txt":                 ("guide",       "Mathew F",            "en"),
    "VHF-procedures-Scheldt-area-EN.md":  ("protocol",    "GNA Scheldt",         "en"),

    "Basic MAYDAY Call.pdf":                                                   ("procedure_card", "Reference",    "en"),
    "Calling the coastguard.pdf":                                              ("guide",          "Reference",    "en"),
    "CEPT Regulations.pdf":                                                    ("regulation",     "CEPT",         "en"),
    "Channel Listing.pdf":                                                     ("reference",      "ITU/CEPT",     "en"),
    "DSC alert flow chart.pdf":                                                ("procedure_card", "Reference",    "en"),
    "dg_185583-1.pdf":                                                         ("regulation",     "UK Gov",       "en"),
    "GMDSS VHF DSC procedures for small boat users - GOV.UK.pdf":              ("guide",          "MCA / GOV.UK", "en"),
    "IMO standard marine comms phrases.pdf":                                   ("regulation",     "IMO",          "en"),
    "mgn375.pdf":                                                              ("regulation",     "MCA",          "en"),
    "mgn_324.pdf":                                                             ("regulation",     "MCA",          "en"),
    "MSI.pdf":                                                                 ("protocol",       "Reference",    "en"),
    "msi_leaflet_2010_version.pdf":                                            ("reference",      "Reference",    "en"),
    "PHONETIC ALPHABET.pdf":                                                   ("reference",      "NATO/ICAO",    "en"),
    "R-REC-M.489-2-199510-I!!PDF-E.pdf":                                       ("regulation",     "ITU",          "en"),
    "Ship radio guidance for licencing.pdf":                                   ("regulation",     "Ofcom",        "en"),
    "Ship radio information ofcom.pdf":                                        ("guide",          "Ofcom",        "en"),
    "Ship radio licence application.pdf":                                      ("reference",      "Ofcom",        "en"),
    "VHF-Cheatsheet-v2.pdf":                                                   ("procedure_card", "Reference",    "en"),
    "VHF RADIO REGULATIONS.pdf":                                               ("regulation",     "Reference",    "en"),

    "NL-Marifoon-procedures-7.4.pdf": None,  # Dutch language -- skipped
}

# ── § 8.1b  VHF concept vocabulary (drives extraction of `concepts`) ───────
VHF_CONCEPTS: dict[str, str] = {
    "MAYDAY":            "distress",       "MAYDAY RELAY":     "distress",
    "PAN PAN":           "urgency",        "PAN PAN MEDICO":   "urgency",
    "SECURITE":          "safety",         "SÉCURITÉ":         "safety",
    "SEELONCE MAYDAY":   "distress",       "SEELONCE FEENEE":  "distress",
    "PRUDONCE":          "distress",       "SEELONCE DISTRESS":"distress",

    "OVER":  "prowords", "OUT":   "prowords", "ROGER": "prowords", "WILCO": "prowords",
    "SAY AGAIN": "prowords", "I SPELL": "prowords", "STANDBY": "prowords",
    "CORRECTION": "prowords", "BREAK": "prowords", "ALL STATIONS": "prowords",
    "AFFIRMATIVE": "prowords", "NEGATIVE": "prowords", "THIS IS": "prowords",
    "I SAY AGAIN": "prowords", "WAIT OUT": "prowords",

    "Channel 6":  "channels", "Channel 8":  "channels", "Channel 9":  "channels",
    "Channel 12": "channels", "Channel 13": "channels", "Channel 14": "channels",
    "Channel 16": "channels", "Channel 22A":"channels", "Channel 67": "channels",
    "Channel 70": "channels", "Channel 72": "channels", "Channel 77": "channels",
    "Channel 80": "channels", "Channel 88": "channels",
    "156.800 MHz": "channels", "156.525 MHz": "channels",

    "DSC": "gmdss", "GMDSS": "gmdss", "EPIRB": "gmdss", "SART": "gmdss",
    "AIS-SART": "gmdss", "NAVTEX": "gmdss", "Inmarsat": "gmdss",
    "MMSI": "gmdss", "MRCC": "gmdss", "SAR": "gmdss",
    "Sea Area A1": "gmdss", "Sea Area A2": "gmdss",
    "Sea Area A3": "gmdss", "Sea Area A4": "gmdss",
    "MSI": "gmdss", "AIS": "gmdss",

    "ITU Radio Regulations": "regulation", "SOLAS": "regulation",
    "CEPT": "regulation", "Ofcom": "regulation", "MCA": "regulation",
    "IMO": "regulation", "SMCP": "regulation", "SRC": "regulation",
    "ROC": "regulation", "GOC": "regulation",
    "Ship Station Licence": "regulation", "Cospas-Sarsat": "regulation",

    "ALFA": "phonetic", "BRAVO": "phonetic", "CHARLIE": "phonetic",
    "DELTA": "phonetic", "ECHO": "phonetic", "FOXTROT": "phonetic",
    "PAPA": "phonetic", "TANGO": "phonetic", "YANKEE": "phonetic",
    "ZULU": "phonetic",

    "VHF": "basics", "simplex": "basics", "duplex": "basics",
    "squelch": "basics", "PTT": "basics", "watchkeeping": "basics",
    "line-of-sight": "basics",
}

_CONCEPT_PATTERNS = [
    (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), k, v)
    for k, v in VHF_CONCEPTS.items()
]


def extract_concepts(text: str) -> tuple[list[str], list[str]]:
    """Return (concepts, topics) found in text using the VHF vocabulary."""
    if not text:
        return [], []
    found_concepts, found_topics, seen = [], set(), set()
    for pattern, key, topic in _CONCEPT_PATTERNS:
        if pattern.search(text) and key.lower() not in seen:
            seen.add(key.lower())
            found_concepts.append(key)
            found_topics.add(topic)
    return found_concepts, sorted(found_topics)


# ── § 8.1c  Section type classifier (rule-based) ───────────────────────────
_TYPE_RULES = [
    ("procedure",  re.compile(r"\b(step\s+\d|first,?|second,?|then\s+|finally,?|"
                              r"press\s+PTT|switch\s+to\s+channel|tune\s+to\s+channel|"
                              r"1\.\s|2\.\s|3\.\s|1️⃣|►|•\s+[A-Z])", re.IGNORECASE)),
    ("rule",       re.compile(r"\b(must|shall\s+not|shall\s+|is\s+required|"
                              r"is\s+prohibited|is\s+forbidden|never\s+use|"
                              r"only\s+authorised|only\s+authorized)\b", re.IGNORECASE)),
    ("definition", re.compile(r"\b(is\s+defined\s+as|refers\s+to|means\s+the|"
                              r"is\s+a\s+(term|word|signal|acronym)|"
                              r"stands\s+for|is\s+the\s+abbreviation)\b", re.IGNORECASE)),
    ("warning",    re.compile(r"\b(warning|caution|danger|do\s+not|avoid|"
                              r"never\s+transmit)\b", re.IGNORECASE)),
    ("dialogue",   re.compile(r'\"(?:MAYDAY|PAN PAN|THIS IS|OVER|OUT)|'
                              r'^(Vessel|Coastguard|Marina|Port Control)[\s\:]', re.MULTILINE | re.IGNORECASE)),
    ("example",    re.compile(r"\b(example|for\s+instance|e\.g\.|such\s+as)\b", re.IGNORECASE)),
]


def classify_section(text: str) -> str:
    if not text:
        return "reference"
    for stype, pattern in _TYPE_RULES:
        if pattern.search(text):
            return stype
    return "reference"


# ── § 8.1d  Step extractor ──────────────────────────────────────────────────
_STEP_PATTERNS = [
    re.compile(r"^\s*\d+[\.\)]\s+(.+)$",         re.MULTILINE),
    re.compile(r"^\s*[1-9]️⃣\s*(.+)$",           re.MULTILINE),
    re.compile(r"^Step\s+\d+\s*[:\-]?\s*(.+)$",  re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*[►•\-*]\s+(.+)$",           re.MULTILINE),
]


def extract_steps(text: str) -> list[str]:
    if not text:
        return []
    for pattern in _STEP_PATTERNS:
        found = [m.group(1).strip() for m in pattern.finditer(text) if m.group(1).strip()]
        if len(found) >= 2:
            return found
    return []


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:max_len] or "section"


def stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.md5(("|".join(str(p) for p in parts)).encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def new_document(source_file: str) -> dict:
    info = SOURCE_CLASSIFICATION.get(source_file)
    if info is None:
        return {}
    doc_type, publisher, lang = info
    return {
        "document_id":  slug(Path(source_file).stem),
        "source_file":  source_file,
        "source_type":  doc_type,
        "publisher":    publisher,
        "language":     lang,
        "chapters":     [],
        "parsing_notes": [],
        "metadata":     {"total_words": 0, "total_sections": 0, "primary_topics": []},
    }


# ── § 8.1f  Auto-discover new source files ─────────────────────────────────
def _register_new_sources() -> list[str]:
    _data_exts = {".txt", ".md", ".pdf"}
    present = {p.name for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in _data_exts}
    unknown = sorted(present - set(SOURCE_CLASSIFICATION.keys()))
    for name in unknown:
        SOURCE_CLASSIFICATION[name] = ("reference", "Auto", "en")
    return unknown


# ── § 8.2  TXT / MD -> JSON ─────────────────────────────────────────────────
_STOP_WORDS_LEAD = {
    "the", "a", "an", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "it", "they", "in", "on", "at", "for", "with", "by", "as",
    "if", "when", "where", "how", "why", "what", "however", "moreover",
    "therefore", "thus", "some", "many", "few", "most", "all", "every",
    "before", "after", "during", "under", "over", "between",
}
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z].+)$")


def detect_heading(line: str) -> Optional[tuple[int, str]]:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    m = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
    if m:
        return (min(len(m.group(1)), 3), m.group(2).strip())
    m = _NUMBERED_HEADING_RE.match(stripped)
    if m and not stripped.endswith((".", ":", "?", "!", ",", ";")):
        return (min(stripped.count(".") + 1, 3), stripped)
    words = stripped.split()
    n_words = len(words)
    if n_words < 2 or n_words > 12:
        return None
    if stripped.endswith((".", ":", "?", "!", ",", ";", "…")):
        return None
    if stripped[0].islower():
        return None
    if words[0].lower() in _STOP_WORDS_LEAD:
        return None
    if stripped == stripped.upper() and stripped[0].isalpha():
        return (1, stripped.title())
    n_cap = sum(1 for w in words if w and (w[0].isupper() or w[0].isdigit()))
    if n_cap / n_words >= 0.6:
        return (2, stripped)
    return None


def parse_txt_file(path: Path) -> dict:
    doc = new_document(path.name)
    if not doc:
        return {}
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    lines = raw.split("\n")

    chapters: list[dict] = []
    current_chapter: Optional[dict] = None
    pending_section_title: Optional[str] = None
    buffer: list[str] = []

    def start_chapter(title: str) -> None:
        nonlocal current_chapter
        if current_chapter is not None:
            chapters.append(current_chapter)
        current_chapter = {"chapter_id": stable_id("c", doc["document_id"], len(chapters), title),
                           "title": title, "level": 1, "sections": []}

    def flush_paragraph_to_section() -> None:
        nonlocal buffer, pending_section_title, current_chapter
        text = "\n\n".join(b.strip() for b in "\n".join(buffer).split("\n\n") if b.strip()).strip()
        buffer = []
        if not text:
            pending_section_title = None
            return
        if current_chapter is None:
            start_chapter(doc["document_id"])
        concepts, topics = extract_concepts(text)
        sec_type = classify_section(text)
        steps = extract_steps(text) if sec_type == "procedure" else []
        title = pending_section_title or (text.split("\n", 1)[0][:60].rstrip(":.") or "Body")
        current_chapter["sections"].append({
            "section_id": stable_id("s", doc["document_id"], sum(len(c["sections"]) for c in chapters) + len(current_chapter["sections"]), title),
            "title": title, "level": 2, "type": sec_type, "text": text,
            "concepts": concepts, "topics": topics, "steps": steps,
        })
        pending_section_title = None

    for line in lines:
        heading = detect_heading(line)
        if heading is None:
            buffer.append(line)
            continue
        level, title = heading
        flush_paragraph_to_section()
        if level == 1:
            start_chapter(title)
        else:
            if current_chapter is None:
                start_chapter(doc["document_id"])
            pending_section_title = title

    flush_paragraph_to_section()
    if current_chapter is not None:
        chapters.append(current_chapter)

    if not chapters:
        body = raw.strip()
        concepts, topics = extract_concepts(body)
        chapters = [{"chapter_id": stable_id("c", doc["document_id"], 0), "title": path.stem, "level": 1,
                    "sections": [{"section_id": stable_id("s", doc["document_id"], 0), "title": "Body",
                                 "level": 2, "type": classify_section(body), "text": body,
                                 "concepts": concepts, "topics": topics, "steps": []}]}]

    doc["chapters"] = chapters
    _finalize_metadata(doc)
    return doc


# ── § 8.3  PDF -> JSON ──────────────────────────────────────────────────────
def _extract_lines_with_meta(page) -> list[dict]:
    chars = page.chars
    if not chars:
        return []
    lines: dict[int, list[dict]] = {}
    for ch in chars:
        lines.setdefault(int(round(ch["top"])), []).append(ch)
    out = []
    for y in sorted(lines):
        chs = sorted(lines[y], key=lambda c: c["x0"])
        text = "".join(c["text"] for c in chs).strip()
        if not text:
            continue
        sizes = [c["size"] for c in chs]
        fonts = [c.get("fontname", "") for c in chs]
        is_bold = any(re.search(r"Bold|-B\b|,B\b", f) for f in fonts)
        out.append({"text": text, "size": statistics.median(sizes) if sizes else 0.0,
                    "bold": is_bold, "y": y})
    return out


def _is_pdf_heading(line: dict, page_median_size: float) -> Optional[int]:
    text = line["text"].strip()
    if not text or len(text) > 120:
        return None
    words = text.split()
    if len(words) < 2 or len(words) > 14:
        return None
    if text.endswith((".", ";", ",", "…")):
        return None
    size = line.get("size", 0.0)
    if size >= page_median_size * 1.20:
        return 1 if size >= page_median_size * 1.40 else 2
    if line.get("bold") and 2 <= len(words) <= 10:
        return 2
    heading = detect_heading(text)
    return heading[0] if heading else None


def parse_pdf_file(path: Path, max_pages: int = 200) -> dict:
    doc = new_document(path.name)
    if not doc:
        return {}
    all_lines: list[dict] = []
    empty_pages = 0
    try:
        with pdfplumber.open(str(path)) as pdf:
            n_pages = min(len(pdf.pages), max_pages)
            for i in range(n_pages):
                lines = _extract_lines_with_meta(pdf.pages[i])
                if not lines:
                    empty_pages += 1
                    continue
                for ln in lines:
                    ln["page"] = i + 1
                all_lines.extend(lines)
    except Exception as e:
        doc["parsing_notes"].append(f"pdfplumber_error: {e.__class__.__name__}: {e}")
        return doc

    if not all_lines:
        doc["parsing_notes"].append("scanned_pdf_ocr_needed")
        return doc
    if empty_pages:
        doc["parsing_notes"].append(f"empty_pages: {empty_pages}")

    doc_median_size = statistics.median(l["size"] for l in all_lines) or 10.0

    chapters: list[dict] = []
    current_chapter: Optional[dict] = None
    pending_section_title: Optional[str] = None
    buffer_texts: list[str] = []
    buffer_pages: list[int] = []

    def start_chapter(title: str) -> None:
        nonlocal current_chapter
        if current_chapter is not None:
            chapters.append(current_chapter)
        current_chapter = {"chapter_id": stable_id("c", doc["document_id"], len(chapters), title),
                           "title": title, "level": 1, "sections": []}

    def flush_section() -> None:
        nonlocal buffer_texts, buffer_pages, pending_section_title, current_chapter
        body = " ".join(buffer_texts).strip()
        buffer_texts = []
        pages = sorted(set(buffer_pages))
        buffer_pages = []
        if not body:
            pending_section_title = None
            return
        if current_chapter is None:
            start_chapter(doc["document_id"])
        concepts, topics = extract_concepts(body)
        sec_type = classify_section(body)
        steps = extract_steps(body) if sec_type == "procedure" else []
        title = pending_section_title or body.split(".")[0][:80].strip() or "Body"
        current_chapter["sections"].append({
            "section_id": stable_id("s", doc["document_id"], sum(len(c["sections"]) for c in chapters) + len(current_chapter["sections"]), title),
            "title": title, "level": 2, "type": sec_type, "text": body,
            "concepts": concepts, "topics": topics, "steps": steps, "pages": pages,
        })
        pending_section_title = None

    for ln in all_lines:
        h_level = _is_pdf_heading(ln, doc_median_size)
        text = ln["text"].strip()
        if h_level is None:
            buffer_texts.append(text)
            buffer_pages.append(ln.get("page", 0))
            continue
        flush_section()
        if h_level == 1:
            start_chapter(text)
        else:
            if current_chapter is None:
                start_chapter(doc["document_id"])
            pending_section_title = text

    flush_section()
    if current_chapter is not None:
        chapters.append(current_chapter)

    if not chapters or all(not c["sections"] for c in chapters):
        body = " ".join(l["text"] for l in all_lines).strip()
        if body:
            concepts, topics = extract_concepts(body)
            chapters = [{"chapter_id": stable_id("c", doc["document_id"], 0), "title": path.stem, "level": 1,
                        "sections": [{"section_id": stable_id("s", doc["document_id"], 0), "title": "Body",
                                     "level": 2, "type": classify_section(body), "text": body,
                                     "concepts": concepts, "topics": topics, "steps": [],
                                     "pages": sorted(set(l.get("page", 0) for l in all_lines))}]}]
            doc["parsing_notes"].append("flat_document_no_headings_detected")

    doc["chapters"] = chapters
    _finalize_metadata(doc)
    doc["metadata"]["total_pages"] = max((l.get("page", 0) for l in all_lines), default=0)
    return doc


def _finalize_metadata(doc: dict) -> None:
    all_topics: set[str] = set()
    n_sections = n_words = 0
    for ch in doc["chapters"]:
        for s in ch["sections"]:
            all_topics.update(s.get("topics", []))
            n_sections += 1
            n_words += len(s.get("text", "").split())
    doc["metadata"]["primary_topics"] = sorted(all_topics)
    doc["metadata"]["total_sections"] = n_sections
    doc["metadata"]["total_words"] = n_words


# ── § 8.4  Convert all sources + build index ───────────────────────────────
def convert_all_sources(force: bool = False) -> list[dict]:
    index: list[dict] = []
    skipped: list[str] = []
    todo = [(name, cls) for name, cls in SOURCE_CLASSIFICATION.items() if cls is not None]

    for name in sorted(dict(todo).keys()):
        src = DATA_DIR / name
        if not src.exists():
            skipped.append(f"{name} (file not found)")
            continue
        out_path = JSON_OUT_DIR / f"{Path(name).stem}.json"
        if out_path.exists() and not force:
            doc = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            try:
                if src.suffix.lower() in {".txt", ".md"}:
                    doc = parse_txt_file(src)
                elif src.suffix.lower() == ".pdf":
                    doc = parse_pdf_file(src)
                else:
                    skipped.append(f"{name} (unsupported extension)")
                    continue
            except Exception as e:
                skipped.append(f"{name} ({e.__class__.__name__}: {e})")
                continue
            if not doc:
                skipped.append(f"{name} (empty result)")
                continue
            out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

        index.append({
            "source_file": doc["source_file"], "document_id": doc["document_id"],
            "source_type": doc["source_type"], "publisher": doc["publisher"],
            "language": doc["language"], "chapters": len(doc["chapters"]),
            "sections": doc["metadata"]["total_sections"], "words": doc["metadata"]["total_words"],
            "primary_topics": doc["metadata"]["primary_topics"],
            "parsing_notes": doc.get("parsing_notes", []),
        })

    INDEX_FILE.write_text(json.dumps({"documents": index, "skipped": skipped}, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return index


# ── § 8.7  Cleanup pass ─────────────────────────────────────────────────────
_TITLE_TRIM_RE = re.compile(r"^[\s\*_#`\"'>\-]+|[\s\*_#`\"',.]+$")
_MULTI_WHITESPACE = re.compile(r"\s+")
_SPEAKER_RE = re.compile(
    r"^\s*(Vessel|Coastguard|Marina|Port\s+Control|MV\s|SV\s|MY\s|Harbour|"
    r"Bridge|SAR|MRCC|Skipper|Pilot|Operator|Master|Radio)\s*[:\-]",
    re.MULTILINE | re.IGNORECASE)
_VHF_QUOTE_RE = re.compile(
    r'\"[^\"]*(MAYDAY|PAN PAN|SÉCURITÉ|SECURITE|THIS IS|OVER|OUT|ROGER|WILCO|CHANNEL)[^\"]*\"',
    re.IGNORECASE)
_DEFINITION_HEAD_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 \-]{1,30})\s*(?:is|are|means|refers to|stands for|=|:)\s+")
_STEP_PREFIX_STRIP = re.compile(r"^\s*(?:\d+[\.\)]|[►•\-*])\s*")
BOILERPLATE_TITLES = {"body", "contents", "table of contents", "index", "references",
                     "acknowledgements", "acknowledgments", "foreword", "introduction"}


def clean_title(t: str) -> str:
    if not t:
        return ""
    t = _TITLE_TRIM_RE.sub("", t)
    t = _MULTI_WHITESPACE.sub(" ", t).strip()
    words = t.split()
    if len(words) > 12:
        t = " ".join(words[:8]).rstrip(",;:") + "…"
    return t


def is_dialogue(text: str) -> bool:
    if not text:
        return False
    if _SPEAKER_RE.search(text):
        return True
    return len(_VHF_QUOTE_RE.findall(text)) >= 2


def is_definition(text: str) -> bool:
    if not text:
        return False
    return bool(_DEFINITION_HEAD_RE.match(text.split("\n", 1)[0]))


def classify_section_v2(text: str) -> str:
    if not text:
        return "reference"
    if is_dialogue(text):
        return "dialogue"
    if is_definition(text):
        return "definition"
    return classify_section(text)


def clean_steps(steps: list[str]) -> list[str]:
    return [_STEP_PREFIX_STRIP.sub("", s).strip() for s in steps if s and s.strip()]


def is_boilerplate(sec: dict) -> bool:
    title = sec["title"].strip().lower()
    words = len(sec["text"].split())
    if words < 4:
        return True
    if title in BOILERPLATE_TITLES and words < 30:
        return True
    return False


def _load_excluded_section_ids() -> set[str]:
    if not CONSISTENCY_EXCLUSIONS_FILE.exists():
        return set()
    return {e["section_id"] for e in json.loads(CONSISTENCY_EXCLUSIONS_FILE.read_text(encoding="utf-8"))}


def _load_normalizations() -> dict[str, list[tuple[str, str]]]:
    if not CONSISTENCY_NORMALIZATIONS_FILE.exists():
        return {}
    by_section: dict[str, list[tuple[str, str]]] = {}
    for e in json.loads(CONSISTENCY_NORMALIZATIONS_FILE.read_text(encoding="utf-8")):
        by_section.setdefault(e["section_id"], []).append((e["find"], e["replace"]))
    return by_section


def normalize_section_text(text: str, rules: list[tuple[str, str]]) -> str:
    for find, replace in rules:
        text = re.sub(rf"(?<![A-Za-z]){re.escape(find)}(?![A-Za-z])", replace, text)
    return text


def improve_section(sec: dict) -> dict:
    sec["title"] = clean_title(sec.get("title", ""))
    text = sec.get("text", "")
    sec["type"] = classify_section_v2(text)
    sec["steps"] = clean_steps(extract_steps(text)) if sec["type"] == "procedure" else []
    if not sec["concepts"] or not sec["topics"]:
        c, t = extract_concepts(text)
        sec["concepts"] = c
        sec["topics"] = t
    return sec


def _cleanup_needed() -> bool:
    if not CLEANUP_MARKER_FILE.exists():
        return True
    marker_mtime = CLEANUP_MARKER_FILE.stat().st_mtime
    return any(p.stat().st_mtime > marker_mtime for p in JSON_OUT_DIR.glob("*.json") if not p.name.startswith("_"))


def cleanup_all_jsons(force: bool = False) -> dict:
    if not force and not _cleanup_needed():
        return {"skipped": True, "reason": "no JSON newer than marker"}

    excluded_ids = _load_excluded_section_ids()
    normalizations = _load_normalizations()
    stats = Counter()
    for path in sorted(JSON_OUT_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        stats["documents"] += 1
        before_types = Counter(s["type"] for ch in doc["chapters"] for s in ch["sections"])

        for ch in doc["chapters"]:
            ch["title"] = clean_title(ch["title"])
            kept = []
            for s in ch["sections"]:
                if s.get("section_id") in excluded_ids:
                    stats["dropped_consistency_exclusion"] += 1
                    continue
                rules = normalizations.get(s.get("section_id"))
                if rules:
                    s["text"] = normalize_section_text(s.get("text", ""), rules)
                    stats["normalized_sections"] += 1
                s = improve_section(s)
                if is_boilerplate(s):
                    stats["dropped_boilerplate"] += 1
                    continue
                kept.append(s)
            ch["sections"] = kept

        after_types = Counter(s["type"] for ch in doc["chapters"] for s in ch["sections"])
        for t in ("dialogue", "definition", "rule", "procedure"):
            stats[f"→{t}"] += after_types[t] - before_types[t]

        _finalize_metadata(doc)
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    CLEANUP_MARKER_FILE.write_text("cleanup_all_jsons applied\n", encoding="utf-8")
    return dict(stats)


def main() -> None:
    unknown = _register_new_sources()
    if unknown:
        print(f"New source file(s) auto-classified as ('reference','Auto','en'): {unknown}")
        print("Edit SOURCE_CLASSIFICATION above to give them a better source_type/publisher.")

    index = convert_all_sources()
    print(f"Converted {len(index)} documents -> {JSON_OUT_DIR}")
    print(f"  total sections: {sum(d['sections'] for d in index)}")
    print(f"  total words   : {sum(d['words'] for d in index):,}")

    result = cleanup_all_jsons()
    if result.get("skipped"):
        print(f"Cleanup skipped: {result['reason']}")
    else:
        print("Cleanup stats:", dict(result))


if __name__ == "__main__":
    main()
