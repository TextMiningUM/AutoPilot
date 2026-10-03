"""Tests for the crypto easter egg (app/vhf_crypto.py) -- pure functions, no model/GPU."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.vhf_crypto import caesar_encode, caesar_decode  # noqa: E402


def test_caesar_encode_basic_shift() -> None:
    assert caesar_encode("abc", 1) == "bcd"
    assert caesar_encode("xyz", 1) == "yza"  # wraps


def test_caesar_encode_preserves_case_and_non_letters() -> None:
    assert caesar_encode("Hello, World!", 3) == "Khoor, Zruog!"


def test_caesar_decode_is_inverse() -> None:
    for shift in (0, 1, 13, 25, -5, 40):
        original = "The quick Brown Fox, Channel 16!"
        assert caesar_decode(caesar_encode(original, shift), shift) == original


def test_caesar_encode_zero_shift_is_identity() -> None:
    assert caesar_encode("MAYDAY", 0) == "MAYDAY"
