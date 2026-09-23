"""Fase C2 tests for pipeline/track1/build_incident_dpo_real.py -- the real incident/CHIRP
"investigator's own judgment" chosen/rejected pairs (distinct from build_rlhf.py's text-
perturbation pairs)."""
from pipeline.track1.build_incident_dpo_real import build_pair


def _row(procedures=None, actual_actions=None):
    return {
        "document_id": "incident_x", "chunk_id": "incident_x", "source_file": "x.pdf",
        "chapter_title": "Incident report: x",
        "trace": {
            "situation": "Two vessels collided in a crossing situation.",
            "procedures": procedures,
            "incident": {"actual_actions_taken": actual_actions},
        },
    }


def test_builds_a_pair_when_both_procedures_and_actions_are_present() -> None:
    row = _row(
        procedures=[{"step": 1, "action": "alter course to starboard", "why": "Rule 15"}],
        actual_actions=[{"actor": "give-way vessel", "action": "held course"}],
    )
    pair = build_pair(row)
    assert pair is not None
    assert "Alter course to starboard" in pair["chosen"][0]["content"]
    assert "Held course" in pair["rejected"][0]["content"]
    assert pair["chosen"][0]["content"] != pair["rejected"][0]["content"]
    assert pair["prompt"][0]["role"] == "system"
    assert pair["prompt"][1]["content"]


def test_returns_none_when_procedures_missing() -> None:
    row = _row(procedures=[], actual_actions=[{"actor": "x", "action": "y"}])
    assert build_pair(row) is None


def test_returns_none_when_actual_actions_missing() -> None:
    row = _row(procedures=[{"step": 1, "action": "x", "why": "y"}], actual_actions=[])
    assert build_pair(row) is None


def test_returns_none_when_no_trace() -> None:
    assert build_pair({"document_id": "x", "trace": None}) is None


def test_returns_none_when_chosen_and_rejected_would_be_identical() -> None:
    row = _row(
        procedures=[{"step": 1, "action": "kept a proper lookout", "why": "Rule 5"}],
        actual_actions=[{"actor": "vessel", "action": "kept a proper lookout"}],
    )
    assert build_pair(row) is None
