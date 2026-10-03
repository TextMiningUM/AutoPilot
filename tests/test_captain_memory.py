"""Tests for pipeline/captain_memory.py (design_captain_missions.md Sec 16.1).

As of 2026-10-03, captain_reasoning_traces.jsonl is a real, committed corpus (built by
pipeline/track1/extract_captain_reasoning.py), so retrieve() now returns REAL hits for
event types with a real mapped_event_type match -- these tests confirm that behaviour,
plus the placeholder fallback for event types/queries with no real hits. No GPU/API key
needed (CPU-only JSON I/O)."""
from pipeline.captain_memory import MemoryHit, retrieve


def test_retrieve_returns_real_hits_for_event_types_present_in_the_corpus():
    # engine_failure/fog/distress_call/commercial_instruction all have real mapped
    # reasoning traces in the committed corpus (confirmed counts: 63/40/37/17).
    for event_type in ("engine_failure", "fog", "distress_call", "commercial_instruction"):
        hits = retrieve(event_type)
        assert hits
        assert all(isinstance(h, MemoryHit) for h in hits)
        assert all(not h.is_placeholder for h in hits), f"{event_type} should have real hits"


def test_retrieve_falls_back_to_placeholder_for_whale_zone():
    # whale_zone is table-determined (Sec 13.A.4) and has 0 real mapped chunks in the
    # current corpus -- this is expected, not a bug (see extract_captain_reasoning.py).
    hits = retrieve("whale_zone")
    assert hits
    assert all(h.is_placeholder for h in hits)


def test_retrieve_falls_back_to_a_generic_placeholder_for_an_unknown_query():
    hits = retrieve("some_unrelated_query_string")
    assert len(hits) == 1
    assert "no real" in hits[0].text.lower() or "no corpus" in hits[0].source.lower()


def test_retrieve_respects_k():
    hits = retrieve("engine_failure", k=1)
    assert len(hits) == 1


def test_real_hits_are_ranked_by_severity_descending():
    hits = retrieve("engine_failure", k=20)
    severity_scores = [h.score for h in hits]
    assert severity_scores == sorted(severity_scores, reverse=True)


def test_real_hit_source_cites_a_specific_article_or_source_file():
    hits = retrieve("distress_call")
    for h in hits:
        assert h.source.strip()
        assert h.text.strip()
