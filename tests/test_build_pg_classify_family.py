"""Unit tests for pipeline/ingest/build_pg.py's classify_family() -- specifically the
2026-10-03 addition that lets Captain traces (which already carry a deterministic
mapped_event_type field from extract_captain_reasoning.py) short-circuit the VHF/OOW-
specific regex-based family classification. Pure logic only, no API/model calls."""
from pipeline.ingest.build_pg import classify_family, FAMILY_RULES


def test_prefers_mapped_event_type_when_present_and_mapped():
    trace = {"mapped_event_type": "engine_failure", "situation": "irrelevant prose"}
    assert classify_family(trace) == "engine_failure"


def test_falls_back_to_regex_when_mapped_event_type_is_unmapped():
    trace = {"mapped_event_type": "unmapped", "situation": "A vessel issued a MAYDAY call."}
    assert classify_family(trace) == "distress"


def test_falls_back_to_regex_when_mapped_event_type_absent():
    """VHF/OOW traces never carry this field at all -- must behave exactly as before."""
    trace = {"situation": "A vessel issued a MAYDAY call."}
    assert classify_family(trace) == "distress"


def test_falls_back_to_general_with_no_signal_at_all():
    trace = {"situation": "Some unrelated prose with no family markers."}
    assert classify_family(trace) == "general"


def test_colreg_encounter_regex_still_matches_rule_numbers():
    trace = {"situation": "Rule 15 crossing situation between two power-driven vessels."}
    assert classify_family(trace) == "colreg_encounter"


def test_family_rules_cover_every_documented_family_name():
    names = {fam for fam, _ in FAMILY_RULES}
    assert names == {"distress", "urgency", "safety", "dsc", "colreg_encounter", "routine_call"}
