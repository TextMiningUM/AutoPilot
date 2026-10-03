"""Unit tests for pipeline/eval/build_captain_gold_answers.py -- pure logic, synthetic
traces only, no real corpus/API calls needed."""
from pipeline.eval.build_captain_gold_answers import (
    select_diverse, build_gold_answer, build_expected_points, build_source_citation,
    build_row,
)


def _row(chunk_id, source_file, key_facts, question_seeds=None, procedures=None,
        situation="A situation.", channels=None, regulations=None, mapped_event_type=None):
    if question_seeds is None:
        question_seeds = [{"angle": "what", "text": "What is X?"}]
    return {
        "chunk_id": chunk_id, "source_file": source_file, "chapter_title": "Ch",
        "trace": {
            "situation": situation,
            "key_facts": key_facts,
            "question_seeds": question_seeds,
            "procedures": procedures or [],
            "channels": channels or [],
            "regulations": regulations or [],
            "mapped_event_type": mapped_event_type,
        },
    }


def test_select_diverse_caps_rows_per_source_file():
    rows = [_row(f"c{i}", "A.pdf", ["fact"]) for i in range(10)] + \
          [_row(f"d{i}", "B.pdf", ["fact"]) for i in range(10)]
    selected = select_diverse(rows, n=100, max_per_source=2)
    counts = {}
    for r in selected:
        counts[r["source_file"]] = counts.get(r["source_file"], 0) + 1
    assert counts == {"A.pdf": 2, "B.pdf": 2}


def test_select_diverse_prefers_richer_traces_within_a_source():
    rich = _row("rich", "A.pdf", ["f1", "f2", "f3"])
    poor = _row("poor", "A.pdf", ["f1"])
    selected = select_diverse([poor, rich], n=1, max_per_source=1)
    assert selected == [rich]


def test_select_diverse_is_deterministic_across_runs():
    rows = [_row(f"c{i}", f"doc{i % 3}.pdf", ["fact"]) for i in range(9)]
    sel1 = select_diverse(rows, n=5, max_per_source=2)
    sel2 = select_diverse(rows, n=5, max_per_source=2)
    assert [r["chunk_id"] for r in sel1] == [r["chunk_id"] for r in sel2]


def test_select_diverse_respects_n_target():
    rows = [_row(f"c{i}", "A.pdf", ["fact"]) for i in range(20)]
    selected = select_diverse(rows, n=5, max_per_source=100)
    assert len(selected) == 5


def test_build_gold_answer_is_fluent_prose_not_a_label_value_dump():
    trace = {
        "situation": "A vessel suffered engine failure.",
        "key_facts": ["The cooling pump failed.", "Speed was capped at 7 knots."],
        "procedures": [{"step": 1, "action": "notify the DPA", "why": "duty to report"}],
    }
    answer = build_gold_answer(trace)
    assert "label:" not in answer.lower() and "Steps:" not in answer
    assert "notify the dpa" in answer.lower()
    assert answer.startswith("A vessel suffered engine failure.")


def test_build_expected_points_includes_key_facts_and_procedure_actions():
    trace = {"key_facts": ["Fact one."], "procedures": [{"action": "do the thing"}]}
    points = build_expected_points(trace)
    assert "Fact one." in points
    assert "Do the thing" in points


def test_build_expected_points_empty_when_nothing_to_grade():
    assert build_expected_points({"key_facts": [], "procedures": []}) == []


def test_build_source_citation_prefers_channels_over_bare_filename():
    trace = {"channels": ["ISM Code Art. 5"], "regulations": ["ISM Code"]}
    cite = build_source_citation({"source_file": "x.pdf"}, trace)
    assert cite == "x.pdf, ISM Code Art. 5"


def test_build_source_citation_falls_back_to_bare_filename():
    cite = build_source_citation({"source_file": "x.pdf"}, {"channels": [], "regulations": []})
    assert cite == "x.pdf"


def test_build_row_skips_traces_with_no_question_seeds():
    row = _row("c1", "A.pdf", ["fact"], question_seeds=[])
    assert build_row(row, 1) is None


def test_build_row_skips_traces_with_no_expected_points():
    row = _row("c1", "A.pdf", key_facts=[], procedures=[])
    assert build_row(row, 1) is None


def test_build_row_produces_the_full_expected_schema():
    row = _row("c1", "A.pdf", ["fact one"], mapped_event_type="engine_failure")
    built = build_row(row, 7)
    assert built["id"] == "captain_0007"
    assert built["chunk_id"] == "c1"
    assert built["question"] == "What is X?"
    assert built["mapped_event_type"] == "engine_failure"
    assert "fact one" in built["expected_points"][0].lower()
