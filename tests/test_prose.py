"""Unit tests for core/prose.py — the fluent-prose text helpers.

Run with: python -m tests.test_prose
"""
from __future__ import annotations

from core.prose import cap, clean, decap, steps_sentence


def test_clean_strips_whitespace_and_trailing_period() -> None:
    assert clean("  hail the vessel.  ") == "hail the vessel"
    assert clean("no trailing period") == "no trailing period"
    assert clean(None) == ""
    assert clean("") == ""


def test_cap_uppercases_first_letter_only() -> None:
    assert cap("hail the vessel") == "Hail the vessel"
    assert cap("") == ""
    assert cap("Already capitalised") == "Already capitalised"


def test_decap_lowercases_first_letter() -> None:
    assert decap("Hail the vessel") == "hail the vessel"
    assert decap("") == ""


def test_decap_preserves_multiletter_acronyms() -> None:
    # Must NOT mangle "VHF Channel 16" into "vHF Channel 16".
    assert decap("VHF Channel 16 is for hailing") == "VHF Channel 16 is for hailing"
    assert decap("GMDSS requires a watch") == "GMDSS requires a watch"
    # Single-letter first words still get lowercased normally.
    assert decap("A distress call follows") == "a distress call follows"


def test_steps_sentence_empty_input() -> None:
    assert steps_sentence([]) == ""
    assert steps_sentence([{"action": ""}, {"action": "  "}]) == ""


def test_steps_sentence_single_step() -> None:
    assert steps_sentence([{"action": "hail on channel 16"}]) == "Hail on channel 16."


def test_steps_sentence_multiple_steps_use_connectives() -> None:
    procs = [
        {"action": "hail on channel 16"},
        {"action": "switch to a working channel"},
        {"action": "pass traffic"},
    ]
    result = steps_sentence(procs)
    assert result.startswith("First, hail on channel 16")
    assert "Then, switch to a working channel" in result
    assert "Next, pass traffic" in result
    assert result.endswith(".")
    # Never falls back to telegraphic "Steps: ..." labels.
    assert "Steps:" not in result


if __name__ == "__main__":
    test_clean_strips_whitespace_and_trailing_period()
    test_cap_uppercases_first_letter_only()
    test_decap_lowercases_first_letter()
    test_decap_preserves_multiletter_acronyms()
    test_steps_sentence_empty_input()
    test_steps_sentence_single_step()
    test_steps_sentence_multiple_steps_use_connectives()
    print("OK — core/prose.py helpers behave as expected.")
