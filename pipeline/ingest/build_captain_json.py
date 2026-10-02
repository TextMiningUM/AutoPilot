"""Data/Captain/CaptainProtocol/ (+ Legal_Reference/bmp5/) -> structured JSON
(design_captain_missions.md Sec 16.6 step 1 -- the first concrete Captain pipeline
script, everything else in Sec 6.3's RAG/KG/PG stack depends on this running first).

Captain's own source folder (Sec 6.1) is deliberately a flat COPY of material from
VHF/OOW plus raw CHIRP newsletters, not a fork of any domain's parser -- this script
DISPATCHES each file to the parser FUNCTION that already handles its exact format,
reused directly from build_vhf_json.py/build_oow_json.py/build_chirp_json.py (never
duplicating their logic), per this project's "never fork a pipeline script" convention.
Each copied file keeps ITS OWN source domain's concept tagging (VHF_CONCEPTS /
CONCEPT_KEYWORDS) -- these files' content really is VHF/OOW vocabulary (MAYDAY/
channels, COLREG rules), not re-tagged with a new Captain vocabulary. A Captain-
specific concept vocabulary (ISM/SOLAS articles, the 5 v1 brown-envelope categories,
Sec 16.1) is a separate, later addition once the rest of Legal_Reference is wired in.

Scope of THIS pass (explicitly not the full Sec 14.1 corpus list yet):
  - Data/Captain/CaptainProtocol/: 5 VHF docs, 2 OOW docs, 66 raw CHIRP newsletter PDFs.
  - Data/Captain/Legal_Reference/bmp5/BMP5.pdf (single-column body text, verified
    readable -- see Docs/rag_chunking_design_and_verification.md's own column-layout
    check -- reuses VHF's generic heading-detection PDF parser, no COLREG/CHIRP-
    specific structure of its own).
  - NOT YET (deferred, different formats needing their own new parsing logic):
    Legal_Reference/{ecfr,uk_legislation}/*.xml, Legal_Reference/{mars,tsb_canada}/*.html.

Local-safe: pure Python + pdfplumber/pymupdf, no GPU/API key needed.

Run with: python -m pipeline.ingest.build_captain_json
"""
from __future__ import annotations

import os

# MUST happen before importing build_vhf_json/build_oow_json/build_chirp_json below --
# each of those modules calls AgentPaths.from_env() at ITS OWN import time for its own
# DATA_DIR/CACHE_DIR/JSON_OUT_DIR globals. Without this, they would resolve against
# whatever AUTOPILOT_DOMAIN happens to already be set to (default "VHF"), risking
# writing a cache file (e.g. build_chirp_json's chirp_text_cache/) into VHF's/OOW's
# actual TRACKED production folders instead of Captain's own.
os.environ.setdefault("AUTOPILOT_DOMAIN", "Captain")

import json

from core import AgentPaths
from pipeline.ingest import build_chirp_json, build_oow_json, build_vhf_json

paths = AgentPaths.captain()
JSON_OUT_DIR = paths.json_dir
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact filenames routed to each existing parser -- everything else in CaptainProtocol/
# is, by construction (see Data/Captain/CaptainProtocol/manifest.json), a raw CHIRP
# newsletter PDF, so no prefix-pattern matching is needed for that branch.
_VHF_PDF_FILES = {
    "Basic MAYDAY Call.pdf",
    "Calling the coastguard.pdf",
    "DSC alert flow chart.pdf",
    "GMDSS VHF DSC procedures for small boat users - GOV.UK.pdf",
}
_VHF_TXT_FILES = {"Securite_Mayday_Repeat_And_Escalation_Sequences.txt"}
_OOW_COLREG_PDF = "COLREG-Consolidated-2018.pdf"
_OOW_SIMPLE_COLREG_JSON = "simple_colreg.json"
_SKIP_FILES = {"manifest.json"}  # provenance metadata, not a source document

# Classification entries for the 2 copied files not already in build_vhf_json's own
# SOURCE_CLASSIFICATION dict (matched by exact filename, domain-agnostic) -- same
# role as that module's own _register_new_sources() auto-registration, just explicit
# here since we want a real category, not the generic ("reference", "Auto", "en") default.
build_vhf_json.SOURCE_CLASSIFICATION.setdefault(
    "Securite_Mayday_Repeat_And_Escalation_Sequences.txt",
    ("procedure_card", "Reference", "en"),
)
build_vhf_json.SOURCE_CLASSIFICATION.setdefault(
    "BMP5.pdf",
    ("guide", "BIMCO/ICS/IGP&I Clubs/INTERTANKO/OCIMF", "en"),
)


def build_captain_protocol_docs() -> list[dict]:
    """Parses every file in CaptainProtocol/, dispatching to the right existing parser."""
    docs: list[dict] = []
    for path in sorted(paths.source_dir.iterdir()):
        if not path.is_file() or path.name in _SKIP_FILES:
            continue
        if path.name in _VHF_PDF_FILES:
            doc = build_vhf_json.parse_pdf_file(path)
        elif path.name in _VHF_TXT_FILES:
            doc = build_vhf_json.parse_txt_file(path)
        elif path.name == _OOW_COLREG_PDF:
            doc = build_oow_json.parse_document(path)
        elif path.name == _OOW_SIMPLE_COLREG_JSON:
            doc = build_oow_json.parse_simple_colreg(path)
        elif path.suffix.lower() == ".pdf":
            doc, _stats = build_chirp_json.build_document(path)
        else:
            print(f"  [skip] {path.name}: no parser routed for this filename yet")
            continue
        if not doc or not doc.get("chapters"):
            print(f"  [skip] {path.name}: parser returned no chapters")
            continue
        docs.append(doc)
    return docs


def build_bmp5_doc() -> dict | None:
    """BMP5.pdf (Legal_Reference/bmp5/) -- reuses VHF's generic heading-detection PDF
    parser since BMP5 has no COLREG/CHIRP-specific structure of its own."""
    bmp5_path = paths.data_root / "Legal_Reference" / "bmp5" / "BMP5.pdf"
    if not bmp5_path.exists():
        print(f"  [skip] {bmp5_path} not found")
        return None
    doc = build_vhf_json.parse_pdf_file(bmp5_path)
    if not doc or not doc.get("chapters"):
        print(f"  [skip] {bmp5_path.name}: parser returned no chapters")
        return None
    return doc


def main() -> None:
    docs = build_captain_protocol_docs()
    bmp5_doc = build_bmp5_doc()
    if bmp5_doc is not None:
        docs.append(bmp5_doc)

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
        print(f"  {doc['source_file']:<70} chapters={len(doc['chapters']):>3d} "
              f"sections={n_sections:>4d} -> {out.name}")

    print(f"\n{written}/{len(docs)} documents written -> {JSON_OUT_DIR}")


if __name__ == "__main__":
    main()
