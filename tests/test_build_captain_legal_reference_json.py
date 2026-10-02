"""Tests for pipeline/ingest/build_captain_legal_reference_json.py -- reads the already-
produced Data/Captain/Captain_JSON/*.json files for the 4 Legal_Reference source types
(same early-return-if-missing convention as tests/test_build_captain_json.py)."""
from __future__ import annotations

import json

from core import AgentPaths

paths = AgentPaths.captain()
JSON_DIR = paths.json_dir


def _load(doc_id: str) -> dict | None:
    p = JSON_DIR / f"{doc_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def test_ecfr_documents_parsed():
    doc = _load("ecfr_ism_title46_part138")
    if doc is None:
        return
    assert doc["source_type"] == "regulation"
    assert doc["language"] == "en"
    assert doc["chapters"][0]["sections"]
    assert all(s["text"] for s in doc["chapters"][0]["sections"])


def test_uk_legislation_documents_parsed():
    doc = _load("uk_legislation_uk_ism")
    if doc is None:
        return
    assert doc["source_type"] == "regulation"
    assert doc["publisher"].startswith("UK Government")
    assert len(doc["chapters"][0]["sections"]) > 5


def test_mars_documents_parsed():
    doc = _load("mars_202627")
    if doc is None:
        return
    assert doc["source_type"] == "incident_report"
    titles = [s["title"] for s in doc["chapters"][0]["sections"]]
    assert any("Lessons learned" in t for t in titles)


def test_tsb_documents_parsed():
    doc = _load("tsb_M23C0032")
    if doc is None:
        return
    assert doc["source_type"] == "incident_report"
    assert doc["publisher"] == "Transportation Safety Board of Canada"
    # heading and body text must not be glued together with no separating space
    for s in doc["chapters"][0]["sections"]:
        assert "occurrenceOn" not in s["text"]


def test_no_document_id_collisions_across_all_four_formats():
    ids = []
    for prefix in ("ecfr_", "uk_legislation_", "mars_", "tsb_"):
        for p in JSON_DIR.glob(f"{prefix}*.json"):
            ids.append(json.loads(p.read_text(encoding="utf-8"))["document_id"])
    assert len(ids) == len(set(ids))
