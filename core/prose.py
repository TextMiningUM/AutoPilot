"""Fluent-prose text helpers shared by the training-data builders.

Previously copy-pasted independently into build_sft.py, build_rlhf.py,
build_reflection.py and (partially) build_multihop.py during the telegraphic-
style fix. One copy here now, so a future edge-case fix (like the acronym
gotcha below) only needs to happen once.
"""
from __future__ import annotations


def clean(s: str | None) -> str:
    return (s or "").strip().rstrip(". ").strip()


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def decap(s: str) -> str:
    """Lowercase the first letter -- but not if it starts with a multi-letter
    all-caps acronym (VHF, GMDSS, ITU, ...), which this would otherwise mangle
    into "vHF"."""
    if not s:
        return s
    first_word = s.split(" ", 1)[0]
    if len(first_word) > 1 and first_word.isupper():
        return s
    return s[:1].lower() + s[1:]


def steps_sentence(procs: list[dict]) -> str:
    """Join a list of {"action": ...} procedure steps into one flowing sentence
    using natural connectives ("First, ...; Then, ...") instead of a numbered
    label:value dump."""
    steps = [clean(p.get("action", "")) for p in procs]
    steps = [s for s in steps if s]
    if not steps:
        return ""
    if len(steps) == 1:
        return cap(steps[0]) + "."
    connectors = ["First", "Then", "Next", "After that", "Finally"]
    pieces = [f"{connectors[i] if i < len(connectors) else 'Then'}, {decap(s)}"
              for i, s in enumerate(steps)]
    return "; ".join(pieces) + "."
