"""Data/ChiefEngineer/ChiefEngineerManuals/{navedtra,wikipedia_background,legal_reference}/
-> structured JSON (design_chief_engineer.md Sec 7/9 -- "No parsing/chunking/RAG-building
has been done on the 21 NAVEDTRA/legal/Wikipedia files acquired... only the 2 MAN Project
Guides feed the known-issues traces so far").

Dispatches each file to an EXISTING parser function that already handles its exact format,
reused directly (never duplicated), per this project's "never fork a pipeline script"
convention -- same pattern build_captain_json.py already established for Captain's own
mixed-format corpus:
  - NAVEDTRA PDFs (10): build_vhf_json.parse_pdf_file()'s generic font-size/bold heading
    detector -- these are plain single-column technical training manuals with no
    COLREG/CHIRP-specific structure of their own, same reasoning build_captain_json.py
    already used for BMP5.pdf.
  - Wikipedia background articles (8, plaintext .txt): build_vhf_json.parse_txt_file()'s
    generic markdown/ALL-CAPS heading detector.
  - MARPOL Annex VI UK SI (1 XML, copied from Data/Captain/Legal_Reference/uk_legislation/):
    build_captain_legal_reference_json.parse_uk_legislation_xml() -- the exact same
    legislation.gov.uk P1group/P1/P2 schema Captain's own copy of this file already uses.

The 2 MAN Project Guide PDFs are DELIBERATELY NOT re-parsed here -- they already feed the
known-issues traces (chief_engineer_known_issues_traces.jsonl) via a different,
LLM-assisted extraction path (pipeline/track1/extract_chief_engineer_known_issues.py),
and that content reaches RAG via pipeline/ingest/build_chief_engineer_chunks_from_traces.py
instead. Re-parsing the same 2 PDFs here would duplicate that content under a different
chunking scheme.

Writes to Data/ChiefEngineer/ChiefEngineer_JSON/ (paths.json_dir) -- build_rag.py's normal
AUTOPILOT_DOMAIN=ChiefEngineer run then chunks/embeds these documents exactly like any
other domain's JSON corpus. See build_chief_engineer_chunks_from_traces.py's new
--append mode for merging the known-issues-trace chunks IN ADDITION to (never instead of)
what build_rag.py produces from this script's output.

Local-safe: pure Python + pdfplumber/xml.etree, no GPU/API key needed.

Run with: python -m pipeline.ingest.build_chief_engineer_json
"""
from __future__ import annotations

import os

# MUST happen before importing build_vhf_json/build_captain_legal_reference_json below --
# each resolves AgentPaths.from_env()/.captain() at ITS OWN import time, same ordering
# requirement build_captain_json.py's own docstring already documents.
os.environ.setdefault("AUTOPILOT_DOMAIN", "ChiefEngineer")

import json

from core import AgentPaths
from pipeline.ingest import build_vhf_json
from pipeline.ingest.build_captain_legal_reference_json import parse_uk_legislation_xml

paths = AgentPaths.chief_engineer()
MANUALS_DIR = paths.source_dir
JSON_OUT_DIR = paths.json_dir
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVEDTRA_DIR = MANUALS_DIR / "navedtra"
WIKIPEDIA_DIR = MANUALS_DIR / "wikipedia_background"
LEGAL_DIR = MANUALS_DIR / "legal_reference"

# Longest NAVEDTRA manual (Principles of Naval Engineering) is 676 pages -- cap comfortably
# above that so no manual gets silently truncated by parse_pdf_file()'s own default cap (200).
NAVEDTRA_MAX_PAGES = 700

# (filename -> (source_type, publisher, language)) registered into build_vhf_json's own
# classification dict, same mechanism build_captain_json.py already uses for its 2 extra
# filenames -- real NAVEDTRA manual titles/years from build_chief_engineer_corpus.py's own
# NAVEDTRA_MANUALS table.
_NAVEDTRA_CLASSIFICATION: dict[str, tuple[str, str, str]] = {
    "engineman1.pdf": ("guide", "NAVEDTRA (US Navy)", "en"),
    "engineman2.pdf": ("guide", "NAVEDTRA (US Navy)", "en"),
    "fireman.pdf": ("guide", "NAVEDTRA (US Navy)", "en"),
    "machinery_repairman.pdf": ("guide", "NAVEDTRA (US Navy)", "en"),
    "fluid_power.pdf": ("guide", "NAVEDTRA (US Navy)", "en"),
    "blueprint_reading_and_sketching.pdf": ("guide", "NAVEDTRA (US Navy)", "en"),
    "basic_machines.pdf": ("guide", "NAVEDTRA (US Navy)", "en"),
    "tools_and_their_uses.pdf": ("guide", "NAVEDTRA (US Navy)", "en"),
    "principles_of_naval_engineering.pdf": ("guide", "NAVPERS (US Navy)", "en"),
    "joint_oil_analysis_program_manual.pdf": ("guide", "NAVAIR (US Navy)", "en"),
}


def _register_classifications() -> None:
    for filename, info in _NAVEDTRA_CLASSIFICATION.items():
        build_vhf_json.SOURCE_CLASSIFICATION.setdefault(filename, info)
    for path in sorted(WIKIPEDIA_DIR.glob("*.txt")) if WIKIPEDIA_DIR.exists() else []:
        build_vhf_json.SOURCE_CLASSIFICATION.setdefault(
            path.name, ("reference", "Wikipedia (CC BY-SA 4.0)", "en"))


def build_navedtra_docs() -> list[dict]:
    if not NAVEDTRA_DIR.exists():
        return []
    docs = []
    for path in sorted(NAVEDTRA_DIR.glob("*.pdf")):
        doc = build_vhf_json.parse_pdf_file(path, max_pages=NAVEDTRA_MAX_PAGES)
        if not doc or not doc.get("chapters"):
            print(f"  [skip] {path.name}: parser returned no chapters")
            continue
        docs.append(doc)
    return docs


def build_wikipedia_docs() -> list[dict]:
    if not WIKIPEDIA_DIR.exists():
        return []
    docs = []
    for path in sorted(WIKIPEDIA_DIR.glob("*.txt")):
        doc = build_vhf_json.parse_txt_file(path)
        if not doc or not doc.get("chapters"):
            print(f"  [skip] {path.name}: parser returned no chapters")
            continue
        docs.append(doc)
    return docs


def build_legal_docs() -> list[dict]:
    if not LEGAL_DIR.exists():
        return []
    docs = []
    for path in sorted(LEGAL_DIR.glob("*.xml")):
        doc = parse_uk_legislation_xml(path)
        if not doc or not doc.get("chapters"):
            print(f"  [skip] {path.name}: parser returned no chapters")
            continue
        docs.append(doc)
    return docs


def main() -> None:
    _register_classifications()
    docs = build_navedtra_docs() + build_wikipedia_docs() + build_legal_docs()

    written = 0
    seen_doc_ids: dict[str, str] = {}
    for doc in docs:
        doc_id = doc["document_id"]
        if doc_id in seen_doc_ids:
            raise RuntimeError(
                f"document_id collision: {doc['source_file']!r} and "
                f"{seen_doc_ids[doc_id]!r} both resolved to {doc_id!r} -- one would "
                "silently overwrite the other's output file."
            )
        seen_doc_ids[doc_id] = doc["source_file"]
        out = JSON_OUT_DIR / f"{doc_id}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
        n_sections = sum(len(c.get("sections", [])) for c in doc["chapters"])
        print(f"  {doc['source_file']:<45} chapters={len(doc['chapters']):>3d} "
              f"sections={n_sections:>4d} -> {out.name}")

    print(f"\n{written}/{len(docs)} documents written -> {JSON_OUT_DIR}")


if __name__ == "__main__":
    main()
