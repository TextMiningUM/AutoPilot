"""Fase C3 tests for pipeline/track1/build_incident_reflection_real.py -- real incident/
CHIRP reflection triples grounded in investigators' own avoidance_summary critique."""
from pipeline.track1.build_incident_reflection_real import build_row


def _row(procedures=None, actual_actions=None, avoidance_summary=""):
    return {
        "document_id": "incident_x", "source_file": "x.pdf",
        "chapter_title": "Incident report: x",
        "trace": {
            "situation": "Two vessels collided in a crossing situation.",
            "procedures": procedures,
            "incident": {"actual_actions_taken": actual_actions, "avoidance_summary": avoidance_summary},
        },
    }


def test_builds_a_row_when_all_three_fields_present() -> None:
    row = _row(
        procedures=[{"step": 1, "action": "alter course to starboard", "why": "Rule 15"}],
        actual_actions=[{"actor": "give-way vessel", "action": "held course"}],
        avoidance_summary="Had the give-way vessel altered course early, the collision would have been avoided.",
    )
    result = build_row(row)
    assert result is not None
    content = result["messages"][-1]["content"]
    assert "Draft:" in content and "Held course" in content
    assert "Critique: Had the give-way vessel altered course early" in content
    assert "Refined:" in content and "Alter course to starboard" in content


def test_returns_none_when_avoidance_summary_missing() -> None:
    row = _row(
        procedures=[{"step": 1, "action": "x", "why": "y"}],
        actual_actions=[{"actor": "a", "action": "b"}],
        avoidance_summary="",
    )
    assert build_row(row) is None


def test_returns_none_when_no_trace() -> None:
    assert build_row({"document_id": "x", "trace": None}) is None


def test_returns_none_when_procedures_or_actions_missing() -> None:
    assert build_row(_row(procedures=[], actual_actions=[{"actor": "a", "action": "b"}],
                         avoidance_summary="x")) is None
    assert build_row(_row(procedures=[{"step": 1, "action": "a", "why": "b"}], actual_actions=[],
                         avoidance_summary="x")) is None
