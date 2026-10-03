"""Crypto easter egg (design_vhf_communications.md Sec 12.7) -- NOT a real GMDSS/VHF
procedure, purely for fun. Kept in its own module/UI section, deliberately separate from
the real regulatory content so it's never mistaken for an actual radio procedure.

A simple Caesar (shift) cipher over A-Z (case-preserved, non-letters left untouched) --
classic, deterministic, trivially testable. "Decode" is just "encode" with the shift
negated.
"""
from __future__ import annotations

ALPHABET_SIZE = 26


def caesar_encode(text: str, shift: int) -> str:
    """Shifts every letter by `shift` positions (wrapping A<->Z), case-preserved.
    Non-letters (spaces, digits, punctuation) pass through unchanged."""
    shift %= ALPHABET_SIZE
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - ord("a") + shift) % ALPHABET_SIZE + ord("a")))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - ord("A") + shift) % ALPHABET_SIZE + ord("A")))
        else:
            out.append(ch)
    return "".join(out)


def caesar_decode(text: str, shift: int) -> str:
    """Inverse of caesar_encode -- same shift, opposite direction."""
    return caesar_encode(text, -shift)
