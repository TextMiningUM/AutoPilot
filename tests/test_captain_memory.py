"""Tests for pipeline/captain_memory.py (design_captain_missions.md Sec 16.1 skeleton) --
confirms the placeholder shape/behaviour, never that any content is "real" retrieval. No
GPU/API key needed."""
from pipeline.captain_memory import MemoryHit, retrieve


def test_retrieve_returns_placeholder_hits_for_known_event_types():
    hits = retrieve("engine_failure")
    assert hits
    assert all(isinstance(h, MemoryHit) for h in hits)
    assert all(h.is_placeholder for h in hits)


def test_retrieve_falls_back_to_a_generic_placeholder_for_an_unknown_query():
    hits = retrieve("some_unrelated_query_string")
    assert len(hits) == 1
    assert "no real" in hits[0].text.lower() or "no corpus" in hits[0].source.lower()


def test_retrieve_respects_k():
    hits = retrieve("engine_failure", k=1)
    assert len(hits) == 1


def test_every_v1_judgement_event_type_has_at_least_one_hit():
    # The 3 genuine judgement cases (Sec 13.A.4) -- fog/whale_zone are table-determined and
    # deliberately have no grounding entries.
    for event_type in ("engine_failure", "distress_call", "commercial_instruction"):
        assert retrieve(event_type), f"expected at least one placeholder hit for {event_type!r}"
