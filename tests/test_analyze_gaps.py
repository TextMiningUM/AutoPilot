"""Unit tests for pipeline/eval/analyze_gaps.py — the reusable data-coverage
gap-analysis tool.

Run with: python -m tests.test_analyze_gaps
"""
from __future__ import annotations

from pipeline.eval.analyze_gaps import (
    build_report, dominant_failure_metrics, group_rows, rank_groups, worst_samples,
)

ROWS = [
    {"section_id": "A", "question": "q1", "gold_answer": "g1", "answer": "a1",
     "metrics": {"Composite": 0.9, "Cover": 0.9, "Faith": 0.9}},
    {"section_id": "A", "question": "q2", "gold_answer": "g2", "answer": "a2",
     "metrics": {"Composite": 0.8, "Cover": 0.8, "Faith": 0.8}},
    {"section_id": "B", "question": "q3", "gold_answer": "g3", "answer": "a3",
     "metrics": {"Composite": 0.1, "Cover": 0.0, "Faith": 0.5}},
    {"section_id": "B", "question": "q4", "gold_answer": "g4", "answer": "a4",
     "metrics": {"Composite": 0.2, "Cover": 0.0, "Faith": 0.6}},
    {"section_id": "C", "question": "q5", "gold_answer": "g5", "answer": "a5",
     "metrics": {"Composite": 0.5, "Cover": None, "Faith": float("nan")}},
]


def test_group_rows_groups_by_field() -> None:
    groups = group_rows(ROWS, "section_id")
    assert set(groups.keys()) == {"A", "B", "C"}
    assert len(groups["A"]) == 2
    assert len(groups["B"]) == 2


def test_group_rows_missing_field_bucketed_as_question_mark() -> None:
    groups = group_rows([{"question": "q"}], "section_id")
    assert "?" in groups


def test_rank_groups_worst_first() -> None:
    groups = group_rows(ROWS, "section_id")
    ranked = rank_groups(groups)
    # B (mean 0.15) should rank worst, then C (0.5), then A (0.85)
    assert [k for k, _, _ in ranked] == ["B", "C", "A"]


def test_dominant_failure_metrics_excludes_composite() -> None:
    counts = dominant_failure_metrics(group_rows(ROWS, "section_id")["B"])
    assert "Composite" not in counts
    assert counts.get("Cover") == 2  # Cover=0.0 is lower than Faith in both B rows


def test_dominant_failure_metrics_skips_missing_values() -> None:
    # group C has Cover=None and Faith=nan; with no valid candidates it must not crash
    counts = dominant_failure_metrics(group_rows(ROWS, "section_id")["C"])
    assert counts == {}


def test_worst_samples_returns_lowest_composite_first() -> None:
    samples = worst_samples(group_rows(ROWS, "section_id")["B"], n=1)
    assert len(samples) == 1
    assert samples[0]["question"] == "q3"  # Composite 0.1 is lower than 0.2


def test_build_report_shape() -> None:
    report = build_report(ROWS, "section_id", top_n_groups=2, samples_per_group=1)
    assert report["group_by"] == "section_id"
    assert report["n_rows_total"] == 5
    assert report["n_groups_total"] == 3
    assert len(report["worst_groups"]) == 2
    assert report["worst_groups"][0]["group"] == "B"


if __name__ == "__main__":
    test_group_rows_groups_by_field()
    test_group_rows_missing_field_bucketed_as_question_mark()
    test_rank_groups_worst_first()
    test_dominant_failure_metrics_excludes_composite()
    test_dominant_failure_metrics_skips_missing_values()
    test_worst_samples_returns_lowest_composite_first()
    test_build_report_shape()
    print("OK — analyze_gaps.py behaves as expected.")
