"""Tests for pipeline/ingest/build_captain_json.py's output -- reads the already-produced
Data/Captain/Captain_JSON/*.json files on disk (same early-return-if-missing convention
as tests/test_rag_exclusions.py), rather than re-invoking the builder in-process (avoids
mutating build_vhf_json.SOURCE_CLASSIFICATION's shared module-level dict during a test
run, and avoids a slow full PDF re-parse on every test run)."""
from __future__ import annotations

import json

from core import AgentPaths

paths = AgentPaths.captain()
JSON_DIR = paths.json_dir


def _load_all_docs() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(JSON_DIR.glob("*.json"))]


def test_captain_json_dir_has_documents():
    if not JSON_DIR.exists():
        return  # builder hasn't been run yet -- nothing to check
    docs = _load_all_docs()
    assert len(docs) >= 70, f"expected ~73 parsed documents, found {len(docs)}"


def test_every_document_has_the_shared_schema_shape():
    if not JSON_DIR.exists():
        return
    for doc in _load_all_docs():
        assert "document_id" in doc
        assert "source_file" in doc
        assert "chapters" in doc and isinstance(doc["chapters"], list)
        assert len(doc["chapters"]) > 0
        for chapter in doc["chapters"]:
            assert "sections" in chapter
            for section in chapter["sections"]:
                assert "section_id" in section
                assert "text" in section and section["text"]


def test_no_document_id_collisions():
    if not JSON_DIR.exists():
        return
    docs = _load_all_docs()
    ids = [d["document_id"] for d in docs]
    assert len(ids) == len(set(ids)), "duplicate document_id across parsed Captain documents"


def test_expected_source_documents_are_present():
    if not JSON_DIR.exists():
        return
    source_files = {d["source_file"] for d in _load_all_docs()}
    assert "COLREG-Consolidated-2018.pdf" in source_files
    assert "simple_colreg.json" in source_files
    assert "BMP5.pdf" in source_files
    assert "Basic MAYDAY Call.pdf" in source_files
    assert "Securite_Mayday_Repeat_And_Escalation_Sequences.txt" in source_files
    chirp_count = sum(1 for d in _load_all_docs() if d.get("source_type") == "chirp_newsletter")
    assert chirp_count >= 60, f"expected ~66 CHIRP newsletters parsed, found {chirp_count}"
