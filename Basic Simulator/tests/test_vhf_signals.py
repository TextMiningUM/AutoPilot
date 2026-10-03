"""Tests for Basic Simulator/app/vhf_signals.py -- all pure data/string functions, no GPU/
model/network calls needed (per copilot-instructions.md)."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app.vhf_signals as vs  # noqa: E402


def test_morse_known_letters() -> None:
    assert vs.MORSE_CODE["A"] == ".-"
    assert vs.MORSE_CODE["S"] == "..."
    assert vs.MORSE_CODE["O"] == "---"


def test_morse_sos_round_trip() -> None:
    encoded = vs.text_to_morse("SOS")
    assert encoded == "... --- ..."
    assert vs.morse_to_text(encoded) == "SOS"


def test_morse_multi_word() -> None:
    encoded = vs.text_to_morse("SOS HELP")
    assert " / " in encoded
    assert vs.morse_to_text(encoded) == "SOS HELP"


def test_morse_decode_ignores_unknown_code() -> None:
    assert vs.morse_to_text("....----") == ""  # not a valid letter/digit code -- dropped, no crash


def test_phonetic_alphabet_has_all_26_letters() -> None:
    assert set(vs.PHONETIC_ALPHABET) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert vs.PHONETIC_ALPHABET["A"] == "Alfa"
    assert vs.PHONETIC_ALPHABET["J"] == "Juliett"  # NOT "Juliet" -- double-T is the real ICAO/NATO spelling


def test_flag_blazons_cover_all_letters_and_match_patterns() -> None:
    assert set(vs.FLAG_BLAZONS) == set(vs.PHONETIC_ALPHABET)
    assert set(vs.FLAG_PATTERNS) == set(vs.PHONETIC_ALPHABET)
    # sourced verbatim from Wikipedia's blazon table -- spot-check a few exact strings
    assert vs.FLAG_BLAZONS["A"] == "Swallowtailed, per pale argent and azure"
    assert vs.FLAG_BLAZONS["Q"] == "Or"
    assert vs.FLAG_BLAZONS["Z"] == "Per saltire or, sable, gules and azure"


def test_flag_patterns_reference_only_known_tinctures() -> None:
    color_keys = ("colors", "field", "band", "border", "inner", "saltire", "cross",
                 "lozenge", "roundel")
    for letter, pattern in vs.FLAG_PATTERNS.items():
        for key in color_keys:
            if key not in pattern:
                continue
            value = pattern[key]
            tinctures = value if isinstance(value, list) else [value]
            for tincture in tinctures:
                assert tincture in vs.TINCTURE_COLORS, f"{letter}.{key}={tincture!r}"


def test_flag_svg_renders_every_letter_as_valid_looking_svg() -> None:
    for letter in vs.PHONETIC_ALPHABET:
        svg = vs.flag_svg(letter)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert "fill=" in svg


def test_flag_svg_swallowtail_only_on_a_and_b() -> None:
    assert "clipPath" in vs.flag_svg("A")
    assert "clipPath" in vs.flag_svg("B")
    assert "clipPath" not in vs.flag_svg("C")


def test_flag_signal_example_has_send_and_receive() -> None:
    example = vs.flag_signal_example("A")
    assert "send" in example and "receive" in example
    assert "Alfa" in example["send"] and "Alfa" in example["receive"]
    assert "diver down" in example["send"].lower()
    assert "diver down" in example["receive"].lower()


def test_load_flag_meanings_has_all_26_letters() -> None:
    meanings = vs.load_flag_meanings()
    assert set(meanings) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert "diver down" in meanings["A"].lower()


def test_pattern_to_symbols() -> None:
    assert vs.pattern_to_symbols(["short"]) == "\u2022"
    assert vs.pattern_to_symbols(["short", "short"]) == "\u2022 \u2022"
    assert vs.pattern_to_symbols(["prolonged", "short", "short"]) == "\u2014 \u2022 \u2022"


def test_manoeuvring_signals_shape() -> None:
    required = {"name", "pattern", "rule"}
    for sig in vs.MANOEUVRING_SIGNALS:
        assert required <= sig.keys()
        assert all(p in ("short", "prolonged") for p in sig["pattern"])


def test_restricted_visibility_signals_shape() -> None:
    required = {"name", "pattern", "rule", "interval"}
    for sig in vs.RESTRICTED_VISIBILITY_SIGNALS:
        assert required <= sig.keys()
        assert all(p in ("short", "prolonged") for p in sig["pattern"])


def test_danger_signal_is_at_least_five_short_blasts() -> None:
    danger = next(s for s in vs.MANOEUVRING_SIGNALS if "Danger" in s["name"])
    assert len(danger["pattern"]) >= 5
    assert all(p == "short" for p in danger["pattern"])
