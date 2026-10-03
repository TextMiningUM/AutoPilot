"""Crypto tab (design_vhf_communications.md Sec 12.7/12.12) -- NOT a real GMDSS/VHF
procedure. Two parts, kept in the same module but clearly distinct:

1. A Caesar (shift) cipher -- the original easter egg, purely for fun, trivially breakable.
2. REAL, standard cryptography (AES-256-GCM, PBKDF2-HMAC-SHA256, RSA-OAEP via the
   `cryptography` library) -- there is no public "NATO standard" text cipher to
   responsibly reproduce (real military crypto systems are classified, certified
   hardware), so this instead implements the actual NIST/RFC-standard primitives that
   OpenPGP itself is built from (a random AES session key encrypts the message, RSA-OAEP
   encrypts that session key for the recipient -- the same hybrid-encryption pattern as
   PGP). Genuinely secure if used correctly, but still kept out of the real VHF/GMDSS
   content -- never a substitute for an actual distress/safety procedure.
"""
from __future__ import annotations
import base64
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

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


# ── Real cryptography: AES-256-GCM (passphrase) + RSA-OAEP hybrid (PGP-style) ──────────
_PBKDF2_ITERATIONS = 600_000  # OWASP 2023 minimum recommendation for PBKDF2-HMAC-SHA256
_SALT_LEN = 16
_NONCE_LEN = 12
_AES_KEY_LEN = 32  # AES-256


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=_AES_KEY_LEN, salt=salt,
                     iterations=_PBKDF2_ITERATIONS)
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_aes_gcm(message: str, passphrase: str) -> str:
    """Real AES-256-GCM authenticated encryption, key derived from `passphrase` via
    PBKDF2-HMAC-SHA256 (600k iterations). Returns a JSON blob (salt/nonce/ciphertext,
    each base64) safe to copy/paste as plain text."""
    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    key = _derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, message.encode("utf-8"), None)
    blob = {
        "v": 1, "alg": "AES-256-GCM+PBKDF2-SHA256",
        "salt": base64.b64encode(salt).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }
    return json.dumps(blob)


def decrypt_aes_gcm(blob_json: str, passphrase: str) -> str:
    """Inverse of encrypt_aes_gcm(). Raises ValueError if the blob is malformed, or if
    the passphrase is wrong / the message was tampered with (GCM is authenticated, so
    this is a real integrity check, not just garbled output)."""
    try:
        blob = json.loads(blob_json)
        salt = base64.b64decode(blob["salt"])
        nonce = base64.b64decode(blob["nonce"])
        ciphertext = base64.b64decode(blob["ciphertext"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError("Malformed encrypted blob -- paste the exact text produced by Encrypt.") from e
    key = _derive_key(passphrase, salt)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except InvalidTag as e:
        raise ValueError("Wrong passphrase or corrupted/tampered message.") from e
    return plaintext.decode("utf-8")


def generate_rsa_keypair() -> tuple[str, str]:
    """Generates a real RSA-2048 keypair (the same asymmetric primitive OpenPGP itself
    uses). Returns (private_key_pem, public_key_pem) -- both PEM text, safe to display/
    copy. The private key is NOT password-protected here -- never reuse a key generated
    in this demo tool for anything real."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def encrypt_rsa_hybrid(message: str, public_key_pem: str) -> str:
    """"PGP-style" hybrid encryption -- the same scheme OpenPGP actually uses: a random
    AES-256 session key encrypts the message (AES-GCM), then RSA-OAEP(SHA-256) encrypts
    that session key with the recipient's public key. Returns a JSON blob."""
    try:
        public_key = serialization.load_pem_public_key(public_key_pem.encode())
    except ValueError as e:
        raise ValueError("Malformed public key -- paste the exact PEM text from Generate keypair.") from e
    session_key = os.urandom(_AES_KEY_LEN)
    nonce = os.urandom(_NONCE_LEN)
    ciphertext = AESGCM(session_key).encrypt(nonce, message.encode("utf-8"), None)
    enc_session_key = public_key.encrypt(
        session_key,
        rsa_padding.OAEP(mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(), label=None),
    )
    blob = {
        "v": 1, "alg": "RSA-OAEP-SHA256+AES-256-GCM",
        "enc_session_key": base64.b64encode(enc_session_key).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }
    return json.dumps(blob)


def decrypt_rsa_hybrid(blob_json: str, private_key_pem: str) -> str:
    """Inverse of encrypt_rsa_hybrid() -- decrypts the session key with the RSA private
    key, then the message with AES-GCM."""
    try:
        private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    except ValueError as e:
        raise ValueError("Malformed private key -- paste the exact PEM text from Generate keypair.") from e
    try:
        blob = json.loads(blob_json)
        enc_session_key = base64.b64decode(blob["enc_session_key"])
        nonce = base64.b64decode(blob["nonce"])
        ciphertext = base64.b64decode(blob["ciphertext"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError("Malformed encrypted blob -- paste the exact text produced by Encrypt.") from e
    try:
        session_key = private_key.decrypt(
            enc_session_key,
            rsa_padding.OAEP(mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(), label=None),
        )
    except ValueError as e:
        raise ValueError("Wrong private key or corrupted/tampered message.") from e
    try:
        plaintext = AESGCM(session_key).decrypt(nonce, ciphertext, None)
    except InvalidTag as e:
        raise ValueError("Corrupted/tampered message.") from e
    return plaintext.decode("utf-8")
