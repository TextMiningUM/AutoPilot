"""Unit tests for pipeline/ingest/build_kg.py's CONCEPT_ALIASES domain selection --
specifically the 2026-10-03 addition of _CAPTAIN_ALIASES (previously an unconditional
VHF-else fallback would have silently given Captain the VHF-radio-procedure alias set,
with no ISM/SOLAS/BMP5/event-type coverage at all). Pure logic only, no API/model calls."""
from pipeline.ingest.build_kg import (
    _CONCEPT_ALIASES_BY_DOMAIN, _CAPTAIN_ALIASES, _VHF_ALIASES, _OOW_ALIASES,
)


def test_captain_gets_its_own_alias_set_not_the_vhf_fallback():
    assert _CONCEPT_ALIASES_BY_DOMAIN["Captain"] is _CAPTAIN_ALIASES
    assert _CONCEPT_ALIASES_BY_DOMAIN["Captain"] is not _VHF_ALIASES


def test_vhf_and_oow_still_resolve_to_their_own_sets():
    assert _CONCEPT_ALIASES_BY_DOMAIN["VHF"] is _VHF_ALIASES
    assert _CONCEPT_ALIASES_BY_DOMAIN["OOW"] is _OOW_ALIASES


def test_captain_aliases_cover_the_5_v1_event_types():
    mapped_targets = {t for targets in _CAPTAIN_ALIASES.values() for t in targets}
    for event_type in ("engine_failure", "fog", "distress_call", "whale_zone", "commercial_instruction"):
        assert event_type in mapped_targets, f"no alias maps to {event_type}"


def test_captain_aliases_cover_the_core_instruments():
    mapped_targets = {t for targets in _CAPTAIN_ALIASES.values() for t in targets}
    for instrument in ("ISM Code", "SOLAS", "MARPOL", "STCW", "BMP5"):
        assert instrument in mapped_targets, f"no alias maps to {instrument}"
