"""Audio synthesis + speech-to-text for the VHF Communications page (design doc Sec
12.13/12.14) -- sound-signal blasts, Morse code tones, a "distorted VHF radio voice"
effect (text-to-speech), and transcribing recorded/played-back speech back to text.

Pure numpy/scipy DSP for blasts/Morse (scipy is already a pinned project dependency,
no new one added for this). The other pieces are each OPTIONAL and gracefully return
None if their package/model isn't available -- all LOCAL Streamlit-Simulator-only
convenience features, never needed for the training/eval pipeline, NOT expected to be
available on the cloud GPU pod:
  - `synthesize_vhf_voice_wav()`'s raw speech comes from, in priority order:
      1. `piper-tts` (offline neural TTS, natural-sounding) -- needs the
         `en_US-lessac-medium` voice model downloaded once into
         `_models/piper_voices/` via `python -m piper.download_voices
         en_US-lessac-medium --download-dir _models/piper_voices` (one-time, needs
         internet; fully offline afterward).
      2. `pyttsx3` (Windows SAPI, offline, always available if installed) -- clearly
         more robotic-sounding, kept only as a fallback if piper's model isn't present.
      3. None, if neither is available.
    Either way, the raw speech then gets a bandpass filter + mild clipping + static
    noise applied to approximate the sound of a real VHF transmission.
  - `transcribe_wav_bytes()` needs `openai-whisper` (runs on CPU here, deliberately, to
    avoid contending with a live Qwen GPU session -- same convention as this project's
    offline embedding/reranker scripts). Decodes the WAV itself and passes the raw array
    straight to Whisper, bypassing Whisper's own ffmpeg-based loader entirely -- no
    system ffmpeg install required. The model checkpoint downloads on first use (one-
    time, needs internet); after that it's fully offline.

Safe to run locally -- no GPU required for blasts/Morse/TTS; STT runs on CPU by design.
"""
from __future__ import annotations
import io
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, resample

SAMPLE_RATE = 22050  # Hz -- matches pyttsx3's SAPI output, one rate used everywhere here

# ── Piper voice model location (see module docstring for the one-time download command) ─
_PIPER_VOICE_DIR = Path(__file__).resolve().parent.parent.parent / "_models" / "piper_voices"
_PIPER_VOICE_NAME = "en_US-lessac-medium"
_piper_voice_cache = None  # lazy singleton, set on first successful load

# ── Sound-signal (ship's whistle) blasts -- COLREG Rule 32(c)/(d) timing ───────────────
_SHORT_BLAST_S = 1.0       # Rule 32(c): "a blast of about one second's duration"
_PROLONGED_BLAST_S = 5.0   # Rule 32(d): "four to six seconds" -- midpoint used here
_GAP_S = 1.0               # silence between blasts -- UX spacing, not a regulatory number
_BLAST_FREQ_HZ = 400       # illustrative ship's-whistle tone (real pitch varies by vessel length, Annex III)

# ── Morse code audio -- standard "PARIS" WPM timing ─────────────────────────────────────
_MORSE_FREQ_HZ = 700  # typical practice-oscillator tone


def _tone(freq: float, duration_s: float, amplitude: float = 0.6) -> np.ndarray:
    """One sine-wave tone with a short fade in/out (avoids audible clicks at the edges)."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * freq * t)
    fade = min(200, n // 4)
    if fade > 0:
        ramp = np.linspace(0, 1, fade)
        samples[:fade] *= ramp
        samples[-fade:] *= ramp[::-1]
    return samples


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_s))


def _to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float [-1, 1] samples -> 16-bit PCM mono WAV bytes, playable directly by st.audio()."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def synthesize_blast_wav(pattern: list[str]) -> bytes:
    """Renders a sound-signal pattern (e.g. ["short","short"]) as a WAV clip -- the same
    "short"/"prolonged" vocabulary used by vhf_signals.pattern_to_symbols()."""
    if not pattern:
        return _to_wav_bytes(np.zeros(1))
    segments = []
    for i, p in enumerate(pattern):
        duration = _SHORT_BLAST_S if p == "short" else _PROLONGED_BLAST_S
        segments.append(_tone(_BLAST_FREQ_HZ, duration))
        if i < len(pattern) - 1:
            segments.append(_silence(_GAP_S))
    return _to_wav_bytes(np.concatenate(segments))


def synthesize_morse_wav(morse: str, wpm: int = 15, freq: int = _MORSE_FREQ_HZ) -> bytes:
    """Renders a dot-dash string (as produced by vhf_signals.text_to_morse()) as a WAV
    clip, using the standard "PARIS" WPM calibration: dit = 1200/wpm ms, dah = 3x dit,
    intra-character gap = 1x dit, inter-character gap = 3x dit, inter-word gap = 7x dit."""
    morse = morse.strip()
    if not morse:
        return _to_wav_bytes(np.zeros(1))
    dit = 1.2 / wpm  # seconds
    segments: list[np.ndarray] = []
    words = morse.split(" / ")
    for w_i, word in enumerate(words):
        chars = word.split(" ")
        for c_i, code in enumerate(chars):
            for s_i, symbol in enumerate(code):
                duration = dit if symbol == "." else dit * 3
                segments.append(_tone(freq, duration))
                if s_i < len(code) - 1:
                    segments.append(_silence(dit))
            if c_i < len(chars) - 1:
                segments.append(_silence(dit * 3))
        if w_i < len(words) - 1:
            segments.append(_silence(dit * 7))
    return _to_wav_bytes(np.concatenate(segments))


def _get_piper_voice():
    """Lazy-loads the Piper neural voice model if `piper-tts` is installed AND the
    model file has been downloaded (see module docstring) -- returns None otherwise."""
    global _piper_voice_cache
    if _piper_voice_cache is not None:
        return _piper_voice_cache
    try:
        from piper import PiperVoice
    except ImportError:
        return None
    model_path = _PIPER_VOICE_DIR / f"{_PIPER_VOICE_NAME}.onnx"
    if not model_path.exists():
        return None
    _piper_voice_cache = PiperVoice.load(str(model_path))
    return _piper_voice_cache


def _synthesize_raw_speech(text: str) -> tuple[bytes, int] | None:
    """Raw (undistorted) speech PCM + its sample rate, from the best available offline
    TTS engine -- Piper (natural neural voice) if its model is downloaded, else pyttsx3
    (SAPI, more robotic but installed by default on Windows) as a fallback. None if
    neither is usable."""
    voice = _get_piper_voice()
    if voice is not None:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice.synthesize_wav(text, w)
        buf.seek(0)
        with wave.open(buf, "rb") as w:
            framerate = w.getframerate()
            raw = w.readframes(w.getnframes())
        return raw, framerate

    try:
        import pyttsx3
    except ImportError:
        return None

    import os
    import tempfile

    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with wave.open(tmp_path, "rb") as w:
            framerate = w.getframerate()
            raw = w.readframes(w.getnframes())
    finally:
        os.remove(tmp_path)
    return raw, framerate


def synthesize_vhf_voice_wav(text: str) -> bytes | None:
    """Text-to-speech (piper-tts if its voice model is downloaded, else pyttsx3/SAPI as
    a fallback -- see module docstring) + a bandpass filter (300-3000 Hz, standard
    narrowband voice bandwidth) + mild clipping + static noise, to approximate the sound
    of a real VHF radio transmission. Returns None if neither TTS engine is usable, so
    callers can show a friendly message instead of crashing."""
    raw_speech = _synthesize_raw_speech(text)
    if raw_speech is None:
        return None
    raw, framerate = raw_speech

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Narrowband bandpass -- approximates real analog VHF/AM voice-channel bandlimiting,
    # not a certified radio-channel emulation.
    nyquist = framerate / 2
    b, a = butter(4, [300 / nyquist, 3000 / nyquist], btype="band")
    filtered = lfilter(b, a, samples)

    # Static + soft clipping for a "radio" character -- fixed seed for reproducible output.
    noise = np.random.default_rng(0).normal(0, 0.02, size=filtered.shape)
    distorted = np.tanh((filtered + noise) * 3.0) / 1.5

    return _to_wav_bytes(distorted, sample_rate=framerate)


# ── Speech-to-text (OPTIONAL, needs openai-whisper) ─────────────────────────────────────
_WHISPER_SAMPLE_RATE = 16000  # Hz -- the fixed input rate Whisper's own model expects
_whisper_models: dict[str, object] = {}  # lazy cache keyed by model_size, loaded on first use


def transcribe_wav_bytes(wav_bytes: bytes, model_size: str = "base.en") -> str | None:
    """Speech-to-text via OpenAI Whisper (offline once the model checkpoint is cached),
    run on CPU to avoid contending with a live Qwen GPU session. Returns None if
    `openai-whisper` isn't installed (see module docstring)."""
    try:
        import whisper
    except ImportError:
        return None

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        framerate = w.getframerate()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if framerate != _WHISPER_SAMPLE_RATE:
        audio = resample(audio, int(len(audio) * _WHISPER_SAMPLE_RATE / framerate)).astype(np.float32)

    if model_size not in _whisper_models:
        _whisper_models[model_size] = whisper.load_model(model_size, device="cpu")
    result = _whisper_models[model_size].transcribe(audio, fp16=False)
    return result["text"].strip()
