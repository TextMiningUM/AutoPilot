"""Fase C1 (RAG-rebuild-v2 plan) tests for pipeline/track1/extract_incident_reasoning.py's
--full-text mode (load_incident_docs_full_text) -- pure I/O logic, no API calls."""
import json

import pipeline.track1.extract_incident_reasoning as m


def test_load_incident_docs_full_text_filters_by_net_score_and_joins_pages(tmp_path, monkeypatch) -> None:
    screening = [
        {"path": "uk/relevant.pdf", "net_score": 57, "colreg_hits": {"cpa": 3}},
        {"path": "uk/irrelevant.pdf", "net_score": 3, "colreg_hits": {}},
    ]
    screening_file = tmp_path / "incident_screening.json"
    screening_file.write_text(json.dumps(screening), encoding="utf-8")

    text_cache_dir = tmp_path / "incidents_text_cache"
    (text_cache_dir / "uk").mkdir(parents=True)
    (text_cache_dir / "uk" / "relevant.json").write_text(
        json.dumps(["page one text", "page two text"]), encoding="utf-8")

    monkeypatch.setattr(m, "SCREENING_FILE", screening_file)
    monkeypatch.setattr(m, "TEXT_CACHE_DIR", text_cache_dir)

    docs = m.load_incident_docs_full_text(None)
    assert len(docs) == 1  # irrelevant.pdf's low net_score excludes it, AND it has no cache file
    doc = docs[0]
    assert doc["document_id"] == "incident_fulltext_relevant"
    assert doc["full_text"] == "page one text\n\npage two text"
    assert doc["truncated"] is False
    assert doc["screening_net_score"] == 57


def test_load_incident_docs_full_text_truncates_at_the_char_cap(tmp_path, monkeypatch) -> None:
    screening = [{"path": "uk/long.pdf", "net_score": 20, "colreg_hits": {}}]
    screening_file = tmp_path / "incident_screening.json"
    screening_file.write_text(json.dumps(screening), encoding="utf-8")
    text_cache_dir = tmp_path / "incidents_text_cache"
    (text_cache_dir / "uk").mkdir(parents=True)
    (text_cache_dir / "uk" / "long.json").write_text(json.dumps(["x" * 100]), encoding="utf-8")

    monkeypatch.setattr(m, "SCREENING_FILE", screening_file)
    monkeypatch.setattr(m, "TEXT_CACHE_DIR", text_cache_dir)
    monkeypatch.setattr(m, "FULLTEXT_MAX_CHARS", 10)

    docs = m.load_incident_docs_full_text(None)
    assert len(docs[0]["full_text"]) == 10
    assert docs[0]["truncated"] is True


def test_load_incident_docs_full_text_skips_missing_cache_file(tmp_path, monkeypatch) -> None:
    screening = [{"path": "uk/missing.pdf", "net_score": 20, "colreg_hits": {}}]
    screening_file = tmp_path / "incident_screening.json"
    screening_file.write_text(json.dumps(screening), encoding="utf-8")
    text_cache_dir = tmp_path / "incidents_text_cache"
    text_cache_dir.mkdir()

    monkeypatch.setattr(m, "SCREENING_FILE", screening_file)
    monkeypatch.setattr(m, "TEXT_CACHE_DIR", text_cache_dir)

    assert m.load_incident_docs_full_text(None) == []
