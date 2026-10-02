"""Royal Netherlands Navy Annual Reports ("KM Jaarboek" 2016-2023) -> structured JSON,
kept in DUTCH (design_captain_missions.md Sec 5.1 -- found 2026-10-03). Confirmed by
direct inspection: real per-ship/per-year deployment narratives -- commander names,
exercises (Joint Viking/Joint Warrior), maintenance windows, Caribbean counter-
narcotics/SAR support work. Genuinely different in kind from Legal_Reference's
regulatory texts and CHIRP/MAIB's incident reports -- this is "what did a Captain's
ship actually do this year" material, closely matching Captain's own Mission Order /
Mission Progress Report concept (Sec 8). ~450K words total across 7 yearbooks.

Source: "Data/Captain/Annual Reports Royal Netherlands Navy/" -- 6 EPUBs (2017/18/20/
21/22/23) + 1 PDF (2016). EPUBs are flat InDesign exports: one XHTML file per PAGE,
absolute-positioned spans, NO semantic heading tags at all -- so this parser treats
each PAGE as one section (title derived from the chapter + page number only), relying
on build_rag.py's own embedding-based topic-boundary detection (Docs/rag_chunking_
design_and_verification.md) to further split/merge as needed at chunk time, rather
than inventing a bespoke heading-detector for a format that has no real headings to
detect. The 2016 PDF IS 2-column (confirmed directly -- naive pdfplumber interleaves
the columns exactly like CHIRP newsletters did) so it's parsed with PyMuPDF's content-
block order instead, the same fix build_chirp_json.py already established.

DELIBERATELY NOT fed into build_rag.py directly -- output stays in Captain_JSON_NL/
(a SEPARATE directory from Captain_JSON/, which build_rag.py globs), not translated
here. See translate_captain_navy_yearbooks.py (separate script, not yet written) for
the English-language version that actually enters the RAG corpus -- RAG chunks must be
in English, matching every other Captain/VHF/OOW source; translating inline here would
conflate two independently-reviewable steps (extraction correctness vs. translation
quality) into one.

Local-safe: pure Python + pymupdf, no GPU/API key needed.

Run with: python -m pipeline.ingest.build_captain_navy_yearbooks_json
"""
from __future__ import annotations

import html
import json
import re
import zipfile
from pathlib import Path

import pymupdf

from core import AgentPaths
from core.text_segmentation import join_hyphenated_linebreaks
from pipeline.ingest.build_vhf_json import stable_id

paths = AgentPaths.captain()
SRC_DIR = paths.data_root / "Annual Reports Royal Netherlands Navy"
JSON_OUT_DIR = paths.data_root / "Captain_JSON_NL"

_PAGE_FILE_RE = re.compile(r"^(?P<prefix>.+)-(?P<page>\d+)\.xhtml$")
_CHAPTER_NAME_RE = re.compile(r"^\d+_KM_Jaarboek_\d{4}_(.+)$")
_SKIP_CHAPTER_PREFIXES = ("0_omslag_voor", "9_omslag_achter")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _chapter_title(prefix: str) -> str:
    """'5_KM_Jaarboek_2023_DIRECTIE_OPERATIES' -> 'Directie Operaties'."""
    name = Path(prefix).name
    m = _CHAPTER_NAME_RE.match(name)
    body = m.group(1) if m else name
    words = re.split(r"[_\-]+", body)
    return " ".join(w.capitalize() for w in words if w)


def _clean_xhtml_text(raw_html: str) -> str:
    text = _TAG_RE.sub(" ", raw_html)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def parse_yearbook_epub(epub_path: Path) -> dict | None:
    """One EPUB -> one document; each PAGE xhtml file becomes one section."""
    m = re.search(r"(\d{4})", epub_path.stem)
    year = m.group(1) if m else epub_path.stem
    doc_id = f"km_jaarboek_{year}"

    with zipfile.ZipFile(epub_path) as z:
        page_files = sorted(n for n in z.namelist() if n.lower().endswith((".xhtml", ".html")))
        by_chapter: dict[str, list[tuple[int, str]]] = {}
        for name in page_files:
            if name.endswith("toc.xhtml"):
                continue
            pm = _PAGE_FILE_RE.match(Path(name).name)
            if not pm:
                continue
            prefix = pm.group("prefix")
            if prefix.startswith(_SKIP_CHAPTER_PREFIXES):
                continue
            page_no = int(pm.group("page"))
            text = _clean_xhtml_text(z.read(name).decode("utf-8", errors="replace"))
            if text:
                by_chapter.setdefault(prefix, []).append((page_no, text))

    chapters = []
    for prefix in sorted(by_chapter):
        title = _chapter_title(prefix)
        sections = []
        for page_no, text in sorted(by_chapter[prefix]):
            sections.append({
                "section_id": stable_id("navy_yearbook", doc_id, prefix, str(page_no)),
                "title": f"{title} (p.{page_no})",
                "type": "narrative",
                "text": text,
                "concepts": [], "topics": [], "pages": [page_no],
            })
        if sections:
            chapters.append({"title": title, "sections": sections})

    if not chapters:
        return None
    return {
        "document_id": doc_id,
        "source_file": epub_path.name,
        "source_type": "navy_yearbook",
        "publisher": "Koninklijke Marine (Royal Netherlands Navy)",
        "language": "nl",
        "chapters": chapters,
    }


def parse_yearbook_pdf(pdf_path: Path) -> dict | None:
    """The 2016 yearbook -- 2-column PDF, confirmed needing PyMuPDF's content-block
    order (naive pdfplumber interleaves the columns, same issue build_chirp_json.py
    already found and fixed for CHIRP newsletters)."""
    m = re.search(r"(\d{4})", pdf_path.stem)
    year = m.group(1) if m else pdf_path.stem
    doc_id = f"km_jaarboek_{year}"

    chapters = [{"title": "Jaarboek 2016", "sections": []}]
    with pymupdf.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            text = join_hyphenated_linebreaks(page.get_text("text") or "")
            text = _WS_RE.sub(" ", text).strip()
            if not text:
                continue
            chapters[0]["sections"].append({
                "section_id": stable_id("navy_yearbook", doc_id, str(page_idx)),
                "title": f"Jaarboek 2016 (p.{page_idx})",
                "type": "narrative",
                "text": text,
                "concepts": [], "topics": [], "pages": [page_idx],
            })

    if not chapters[0]["sections"]:
        return None
    return {
        "document_id": doc_id,
        "source_file": pdf_path.name,
        "source_type": "navy_yearbook",
        "publisher": "Koninklijke Marine (Royal Netherlands Navy)",
        "language": "nl",
        "chapters": chapters,
    }


def main() -> None:
    JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".epub":
            doc = parse_yearbook_epub(path)
        elif path.suffix.lower() == ".pdf":
            doc = parse_yearbook_pdf(path)
        else:
            continue
        if not doc:
            print(f"  [skip] {path.name}: no usable text extracted")
            continue
        out = JSON_OUT_DIR / f"{doc['document_id']}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
        n_sections = sum(len(c["sections"]) for c in doc["chapters"])
        n_words = sum(len(s["text"].split()) for c in doc["chapters"] for s in c["sections"])
        print(f"  {path.name:<35} chapters={len(doc['chapters']):>2d} sections={n_sections:>4d} "
              f"words={n_words:>7,d} -> {out.name}")

    print(f"\n{written} yearbooks parsed (Dutch) -> {JSON_OUT_DIR}")
    print("NOT yet translated to English -- see translate_captain_navy_yearbooks.py "
          "(not written yet) before this content can enter build_rag.py.")


if __name__ == "__main__":
    main()
