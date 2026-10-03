"""Tests for Basic Simulator/app/vhf_audio.py -- pure numpy/scipy signal generation, no
GPU/model calls. The piper-tts/pyttsx3-based voice effect and the openai-whisper-based
speech-to-text are each tested only for their graceful-None fallback / fast-failure
paths by default -- a real transcription run downloads/loads a ~139 MB model and is
gated behind the AUTOPILOT_TEST_WHISPER=1 opt-in env var so the normal test suite stays
fast, offline, and network-free (per copilot-instructions.md's testing conventions)."""
import os
import sys
import wave
from io import BytesIO
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app.vhf_audio as vhf_audio  # noqa: E402


def _wav_duration_s(wav_bytes: bytes) -> float:
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()


def test_synthesize_blast_wav_is_valid_wav() -> None:
    wav_bytes = vhf_audio.synthesize_blast_wav(["short"])
    assert wav_bytes.startswith(b"RIFF")
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == vhf_audio.SAMPLE_RATE


def test_synthesize_blast_wav_empty_pattern_does_not_crash() -> None:
    wav_bytes = vhf_audio.synthesize_blast_wav([])
    assert wav_bytes.startswith(b"RIFF")


def test_synthesize_blast_wav_prolonged_longer_than_short() -> None:
    short_dur = _wav_duration_s(vhf_audio.synthesize_blast_wav(["short"]))
    prolonged_dur = _wav_duration_s(vhf_audio.synthesize_blast_wav(["prolonged"]))
    assert prolonged_dur > short_dur


def test_synthesize_blast_wav_five_short_longer_than_one_short() -> None:
    one = _wav_duration_s(vhf_audio.synthesize_blast_wav(["short"]))
    five = _wav_duration_s(vhf_audio.synthesize_blast_wav(["short"] * 5))
    assert five > one * 4  # 5 blasts + 4 gaps vs. 1 blast


def test_synthesize_morse_wav_is_valid_wav() -> None:
    wav_bytes = vhf_audio.synthesize_morse_wav("... --- ...")  # SOS
    assert wav_bytes.startswith(b"RIFF")
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == vhf_audio.SAMPLE_RATE


def test_synthesize_morse_wav_empty_does_not_crash() -> None:
    wav_bytes = vhf_audio.synthesize_morse_wav("")
    assert wav_bytes.startswith(b"RIFF")


def test_synthesize_morse_wav_dah_longer_than_dit() -> None:
    dit_dur = _wav_duration_s(vhf_audio.synthesize_morse_wav("."))
    dah_dur = _wav_duration_s(vhf_audio.synthesize_morse_wav("-"))
    assert dah_dur > dit_dur


def test_synthesize_morse_wav_faster_wpm_is_shorter() -> None:
    slow = _wav_duration_s(vhf_audio.synthesize_morse_wav("...", wpm=5))
    fast = _wav_duration_s(vhf_audio.synthesize_morse_wav("...", wpm=30))
    assert fast < slow


def test_synthesize_vhf_voice_wav_returns_bytes_or_none() -> None:
    """Doesn't assert pyttsx3 IS installed (optional, Windows-only) -- just that the
    function never raises and returns either real WAV bytes or a clean None."""
    result = vhf_audio.synthesize_vhf_voice_wav("TEXAS SPIRIT, OVER")
    assert result is None or result.startswith(b"RIFF")


def test_transcribe_wav_bytes_raises_on_malformed_audio() -> None:
    """Fast, offline failure path -- malformed WAV bytes must raise before any model
    load/download is attempted (openai-whisper IS installed in this environment, so this
    confirms the WAV-decoding step fails fast rather than silently loading a model)."""
    with pytest.raises(Exception):  # noqa: B017 -- wave module's own exception type varies
        vhf_audio.transcribe_wav_bytes(b"not a real wav file")


@pytest.mark.skipif(os.environ.get("AUTOPILOT_TEST_WHISPER") != "1",
                    reason="real transcription downloads/loads a ~139 MB model -- opt in "
                          "with AUTOPILOT_TEST_WHISPER=1")
def test_transcribe_wav_bytes_round_trips_tts_output() -> None:
    """Opt-in integration test: synthesize speech, transcribe it back, and confirm at
    least one expected word survives the TTS->distortion->STT round trip."""
    tts_wav = vhf_audio.synthesize_vhf_voice_wav("TEXAS SPIRIT OVER")
    assert tts_wav is not None, "pyttsx3 not installed -- can't generate audio to transcribe"
    text = vhf_audio.transcribe_wav_bytes(tts_wav)
    assert text is not None
    assert "TEXAS" in text.upper() or "SPIRIT" in text.upper() or "OVER" in text.upper()

