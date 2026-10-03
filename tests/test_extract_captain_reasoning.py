"""Unit tests for pipeline/track1/extract_captain_reasoning.py -- pure logic only
(parse_response, cross_validate_event_type), no API calls."""
from pipeline.track1.extract_captain_reasoning import (
    parse_response, cross_validate_event_type, EVENT_TYPES, EVENT_KEYWORD_GUARDS,
)


def test_event_types_match_procedure_library():
    assert set(EVENT_TYPES) == {
        "engine_failure", "fog", "distress_call", "whale_zone", "commercial_instruction",
    }


def test_every_event_type_has_a_keyword_guard():
    assert set(EVENT_KEYWORD_GUARDS.keys()) == set(EVENT_TYPES)


def test_parse_response_plain_json():
    assert parse_response('{"a": 1}') == {"a": 1}


def test_parse_response_tolerates_surrounding_prose():
    raw = 'Sure, here it is:\n{"a": 1, "b": [2, 3]}\nHope that helps.'
    assert parse_response(raw) == {"a": 1, "b": [2, 3]}


def test_parse_response_returns_none_on_unparseable_text():
    assert parse_response("not json at all") is None


def test_cross_validate_keeps_a_genuinely_supported_mapping():
    trace = {"mapped_event_type": "engine_failure", "severity": "moderate", "context_flags": ["deadline_pressure"]}
    cross_validate_event_type(trace, "The main engine suffered a cooling-water pump failure.")
    assert trace["mapped_event_type"] == "engine_failure"
    assert trace["severity"] == "moderate"
    assert "mapped_event_type_llm_guess" not in trace


def test_cross_validate_downgrades_a_hallucinated_mapping():
    trace = {"mapped_event_type": "whale_zone", "severity": "serious", "context_flags": ["foo"]}
    cross_validate_event_type(trace, "The master reviewed the annual safety report with no particular incident.")
    assert trace["mapped_event_type"] == "unmapped"
    assert trace["severity"] is None
    assert trace["context_flags"] == []
    assert trace["mapped_event_type_llm_guess"] == "whale_zone"


def test_cross_validate_is_case_insensitive():
    trace = {"mapped_event_type": "fog", "severity": "minor", "context_flags": []}
    cross_validate_event_type(trace, "RESTRICTED VISIBILITY was reported by the lookout.")
    assert trace["mapped_event_type"] == "fog"


def test_cross_validate_forces_unmapped_fields_even_if_llm_left_them_populated():
    trace = {"mapped_event_type": "unmapped", "severity": "serious", "context_flags": ["x"]}
    cross_validate_event_type(trace, "General statutory text with no event relevance.")
    assert trace["severity"] is None
    assert trace["context_flags"] == []


def test_cross_validate_downgrades_an_out_of_vocabulary_event_type():
    """Real bug found on the first full run: the LLM returned "crossing" (a COLREG concept,
    not one of the 5 v1 event types) -- the keyword-guard check alone never caught it since
    it only validated KNOWN event types, silently passing invalid values through."""
    trace = {"mapped_event_type": "crossing", "severity": "moderate", "context_flags": ["x"]}
    cross_validate_event_type(trace, "A crossing situation developed between two vessels.")
    assert trace["mapped_event_type"] == "unmapped"
    assert trace["mapped_event_type_llm_guess"] == "crossing"
    assert trace["severity"] is None
    assert trace["context_flags"] == []
