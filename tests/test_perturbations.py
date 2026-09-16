"""Unit tests for the DPO perturbation functions in pipeline/track1/build_rlhf.py.

Each perturbation must return a *different*, still-fluent answer (never a
telegraphic label:value dump), or None when the trace lacks the field needed
for that particular perturbation.

Run with: python -m tests.test_perturbations
"""
from __future__ import annotations

from pipeline.track1 import build_rlhf as rlhf

TRACE = {
    "situation": "a vessel is approaching a narrow channel",
    "procedures": [
        {"action": "hail the vessel on the radio"},
        {"action": "agree a passing arrangement"},
    ],
    "channels": ["16"],
    "prowords_used": ["MAYDAY"],
    "regulations": ["COLREG Rule 9"],
    "warnings": ["do not overtake in the channel"],
    "outcomes": ["both vessels pass safely"],
}


def test_perturb_wrong_channel_swaps_to_a_different_channel() -> None:
    rlhf.RNG.seed(1)
    rejected = rlhf.perturb_wrong_channel(TRACE)
    assert rejected is not None
    assert "16" not in rejected  # original channel must be gone
    assert "Use Channel" in rejected


def test_perturb_wrong_channel_none_without_channels() -> None:
    trace = {**TRACE, "channels": []}
    assert rlhf.perturb_wrong_channel(trace) is None


def test_perturb_wrong_proword_uses_swap_table() -> None:
    rejected = rlhf.perturb_wrong_proword(TRACE)
    assert rejected is not None
    assert "PAN PAN" in rejected
    assert "MAYDAY" not in rejected


def test_perturb_wrong_proword_none_for_unswappable_proword() -> None:
    trace = {**TRACE, "prowords_used": ["UNKNOWN_PROWORD"]}
    assert rlhf.perturb_wrong_proword(trace) is None


def test_perturb_missing_step_drops_one_step() -> None:
    rejected = rlhf.perturb_missing_step(TRACE)
    assert rejected is not None
    chosen = rlhf.build_chosen_answer(TRACE)
    assert rejected != chosen


def test_perturb_missing_step_none_with_fewer_than_two_steps() -> None:
    trace = {**TRACE, "procedures": TRACE["procedures"][:1]}
    assert rlhf.perturb_missing_step(trace) is None


def test_perturb_drop_regulation() -> None:
    rejected = rlhf.perturb_drop_regulation(TRACE)
    assert rejected is not None
    assert "COLREG Rule 9" not in rejected

    trace = {**TRACE, "regulations": []}
    assert rlhf.perturb_drop_regulation(trace) is None


def test_perturb_drop_warning() -> None:
    rejected = rlhf.perturb_drop_warning(TRACE)
    assert rejected is not None
    assert "overtake" not in rejected

    trace = {**TRACE, "warnings": []}
    assert rlhf.perturb_drop_warning(trace) is None


if __name__ == "__main__":
    test_perturb_wrong_channel_swaps_to_a_different_channel()
    test_perturb_wrong_channel_none_without_channels()
    test_perturb_wrong_proword_uses_swap_table()
    test_perturb_wrong_proword_none_for_unswappable_proword()
    test_perturb_missing_step_drops_one_step()
    test_perturb_missing_step_none_with_fewer_than_two_steps()
    test_perturb_drop_regulation()
    test_perturb_drop_warning()
    print("OK — build_rlhf.py perturbation functions behave as expected.")
