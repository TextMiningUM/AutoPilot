"""Unit tests for the Captain-domain-aware rendering added to build_reflection.py,
build_multihop.py, and build_pg_sft.py (2026-10-03) -- these fields (channels/
prowords_used) hold VHF-specific meanings by default, but OOW/Captain repurpose them
(COLREG rule citations / vessel roles for OOW; regulation-article citations / role-actor
tags for Captain), so rendering must be domain-aware to avoid nonsensical text like "Use
the prowords master, dpa." Monkeypatches each module's own `paths` object (AgentPaths is
frozen, so a whole new instance is swapped in, not a single field) -- no API/GPU calls needed."""
from core import AgentPaths
import pipeline.track1.build_reflection as refl
import pipeline.track1.build_multihop as multihop
import pipeline.track1.build_pg_sft as pg_sft


def test_reflection_channels_sentence_is_captain_aware(monkeypatch):
    monkeypatch.setattr(refl, "paths", AgentPaths.captain())
    assert refl._channels_sentence(["ISM Code Art. 5"]) == "This is governed by ISM Code Art. 5."


def test_reflection_prowords_sentence_is_captain_aware(monkeypatch):
    monkeypatch.setattr(refl, "paths", AgentPaths.captain())
    assert refl._prowords_sentence(["master", "dpa"]) == "This involves the master and dpa."


def test_reflection_channels_dropped_desc_is_captain_aware(monkeypatch):
    monkeypatch.setattr(refl, "paths", AgentPaths.captain())
    desc = refl._channels_dropped_desc(["ISM Code Art. 5"])
    assert "regulation/article citations" in desc


def test_reflection_prowords_dropped_desc_is_captain_aware(monkeypatch):
    monkeypatch.setattr(refl, "paths", AgentPaths.captain())
    desc = refl._prowords_dropped_desc(["master"])
    assert "role/actor context" in desc


def test_reflection_vhf_rendering_unaffected(monkeypatch):
    """Regression guard: adding the Captain branch must not change VHF's own output."""
    monkeypatch.setattr(refl, "paths", AgentPaths.vhf())
    assert refl._channels_sentence(["16"]) == "Use Channel 16."
    assert refl._prowords_sentence(["MAYDAY"]) == "Use the prowords MAYDAY as appropriate."


def test_multihop_summarize_trace_is_captain_aware(monkeypatch):
    monkeypatch.setattr(multihop, "paths", AgentPaths.captain())
    trace = {"trace": {"situation": "an engine failure occurred", "channels": ["ISM Code Art. 5"],
                       "prowords_used": ["master"]}}
    summary = multihop.summarize_trace(trace)
    assert "This involves ISM Code Art. 5." in summary
    assert "This concerns master." in summary
    assert "Use ISM Code Art. 5." not in summary


def test_multihop_summarize_trace_vhf_unaffected(monkeypatch):
    monkeypatch.setattr(multihop, "paths", AgentPaths.vhf())
    trace = {"trace": {"situation": "a distress call", "channels": ["16"], "prowords_used": ["MAYDAY"]}}
    summary = multihop.summarize_trace(trace)
    assert "Use 16." in summary
    assert "Use the prowords MAYDAY." in summary


def test_pg_sft_family_phrase_covers_the_5_v1_event_types():
    for event_type in ("engine_failure", "fog", "distress_call", "whale_zone", "commercial_instruction"):
        assert event_type in pg_sft.FAMILY_PHRASE
        assert pg_sft.FAMILY_PHRASE[event_type] != pg_sft.FAMILY_PHRASE["general"]
