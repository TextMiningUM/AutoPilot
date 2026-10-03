"""Tests for the crypto tab (app/vhf_crypto.py) -- pure functions, no model/GPU."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.vhf_crypto import (  # noqa: E402
    caesar_encode, caesar_decode, encrypt_aes_gcm, decrypt_aes_gcm,
    generate_rsa_keypair, encrypt_rsa_hybrid, decrypt_rsa_hybrid,
)


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


def test_aes_gcm_round_trip() -> None:
    message = "MEET AT BUOY DELTA ONE TWO"
    blob = encrypt_aes_gcm(message, "correct horse battery staple")
    assert decrypt_aes_gcm(blob, "correct horse battery staple") == message


def test_aes_gcm_wrong_passphrase_raises() -> None:
    blob = encrypt_aes_gcm("secret cargo manifest", "right-passphrase")
    try:
        decrypt_aes_gcm(blob, "wrong-passphrase")
    except ValueError as e:
        assert "passphrase" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for a wrong passphrase")


def test_aes_gcm_malformed_blob_raises() -> None:
    try:
        decrypt_aes_gcm("not valid json at all", "any-passphrase")
    except ValueError as e:
        assert "malformed" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for a malformed blob")


def test_generate_rsa_keypair_returns_parseable_pem() -> None:
    private_pem, public_pem = generate_rsa_keypair()
    assert "BEGIN PRIVATE KEY" in private_pem
    assert "BEGIN PUBLIC KEY" in public_pem


def test_rsa_hybrid_round_trip() -> None:
    private_pem, public_pem = generate_rsa_keypair()
    message = "RENDEZVOUS AT 51.9N 004.5E, CHANNEL 16"
    blob = encrypt_rsa_hybrid(message, public_pem)
    assert decrypt_rsa_hybrid(blob, private_pem) == message


def test_rsa_hybrid_wrong_private_key_raises() -> None:
    _, public_pem = generate_rsa_keypair()
    other_private_pem, _ = generate_rsa_keypair()  # a different, unrelated keypair
    blob = encrypt_rsa_hybrid("classified cargo manifest", public_pem)
    try:
        decrypt_rsa_hybrid(blob, other_private_pem)
    except ValueError as e:
        assert "private key" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for the wrong private key")
