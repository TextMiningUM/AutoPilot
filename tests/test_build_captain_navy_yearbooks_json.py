"""Tests for pipeline/ingest/build_captain_navy_yearbooks_json.py -- reads the already-
produced Data/Captain/Captain_JSON_NL/*.json files (same early-return-if-missing
convention as tests/test_rag_exclusions.py / test_build_captain_json.py)."""
from __future__ import annotations

import json

from core import AgentPaths

paths = AgentPaths.captain()
JSON_NL_DIR = paths.data_root / "Captain_JSON_NL"


def _load_all_docs() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(JSON_NL_DIR.glob("*.json"))]


def test_json_nl_dir_is_separate_from_captain_json():
    """RAG chunks must be in English -- build_rag.py globs paths.json_dir
    (Captain_JSON/) only, so the Dutch-language output must live elsewhere."""
    assert JSON_NL_DIR != paths.json_dir
    assert JSON_NL_DIR.name == "Captain_JSON_NL"


def test_all_seven_yearbooks_parsed():
    if not JSON_NL_DIR.exists():
        return
    docs = _load_all_docs()
    assert len(docs) == 7, f"expected 7 yearbooks (2016-2023, no 2019), found {len(docs)}"
    years = {d["document_id"] for d in docs}
    assert years == {f"km_jaarboek_{y}" for y in
                     ["2016", "2017", "2018", "2020", "2021", "2022", "2023"]}


def test_every_document_is_tagged_dutch_and_has_the_shared_schema_shape():
    if not JSON_NL_DIR.exists():
        return
    for doc in _load_all_docs():
        assert doc["language"] == "nl"
        assert doc["source_type"] == "navy_yearbook"
        assert "document_id" in doc and "source_file" in doc
        assert doc["chapters"], f"{doc['document_id']} has no chapters"
        for chapter in doc["chapters"]:
            for section in chapter["sections"]:
                assert section["text"], f"{doc['document_id']}/{section['section_id']} has empty text"


def test_no_document_id_collisions():
    if not JSON_NL_DIR.exists():
        return
    ids = [d["document_id"] for d in _load_all_docs()]
    assert len(ids) == len(set(ids))
