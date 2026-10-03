"""VHF Communications Streamlit page (design_vhf_communications.md Sec 12, Phase 2) --
merged 2026-10-03 from the former standalone `VHF Simulator/` app into Basic Simulator's
multipage router (app/streamlit_app.py), as the "Communications" sidebar entry.

Auto-discovered by Streamlit's native multipage convention (a `pages/` folder next to the
main entrypoint, app/streamlit_app.py) -- run `streamlit run app/streamlit_app.py` and
pick "Communications" from the sidebar page list.

Five tabs:
  - "Ask VHF" (Track 1): free-text question -> grounded answer + retrieved sources,
    with a session-scoped question history. Now also answers flag/light-sound questions
    (flag_signals.json/light_sound_signals.json are part of the same RAG index) and can
    use the v6_rag_rerank config once the VHF reranker is trained.
  - "Radio Simulator" (Track 2): pick a scenario, preview it read-only (with a "Play
    reference transmission" distorted-VHF-audio button), then either draft your own
    transmission or receive an incoming hail and reply to it -- by TYPING or by
    RECORDING your voice (st.audio_input -> transcribe_wav_bytes(), openai-whisper,
    optional/graceful -- see vhf_audio.py's module docstring), rendered as a radio-style
    transmission log (st.chat_message bubbles) with a "Play (distorted VHF radio)"
    button per entry (synthesize_vhf_voice_wav(), pyttsx3, also optional/graceful).
  - "OOW -> Comms" (Sec 8.1): draft the radio call for an already-made OOW manoeuvre
    decision -- VHF never decides the manoeuvre itself, only confirms it.
  - "Signals" (Sec 9.1/12.9/12.13): channel-specific renderers for flags (ICS meaning +
    phonetic + Morse + a real flag graphic rendered from Wikipedia's sourced heraldic
    blazon -- see vhf_signals.py's module docstring -- plus a deterministic send/receive
    example), light/sound manoeuvring + restricted-visibility signal patterns (COLREG
    Rules 32-35, real sourced data, now with synthesized blast-tone audio), a Morse
    encoder/decoder (with synthesized Morse-tone audio, standard "PARIS" WPM timing),
    and a 4th "Exchange Practice" sub-tab -- read an incoming flag/light/sound/Morse
    signal, work out your own response, then reveal the correct one (see
    vhf_signals.SIGNAL_EXCHANGE_SCENARIOS). Radio Simulator transmission-log entries
    also get a "Play (distorted VHF radio)" button (see vhf_audio.py's module docstring
    -- optional, needs pyttsx3, Windows-only, gracefully degrades if not installed).
  - "Secret Transmission" (Sec 12.7/12.12): a Caesar-cipher easter egg, plus two REAL
    cryptography sub-tabs -- passphrase-based AES-256-GCM, and RSA-OAEP hybrid encryption
    ("PGP-style", the same scheme OpenPGP itself uses). Still not a GMDSS/VHF procedure,
    kept structurally separate from the real content.

Model: base Qwen3-8B, 4-bit NF4 -- loads lazily on first Ask/Grade/Draft action (same
GPU as the OOW page uses; keep only one page's model loaded at a time on an 8 GB laptop
GPU -- use the sidebar's "Unload model" button before switching pages if you hit an
out-of-memory error).
"""
from __future__ import annotations
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent  # Basic Simulator/app/
ROOT = APP_DIR.parent                             # Basic Simulator/
for p in (ROOT, ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

from app.vhf_agents import (
    ask_vhf_qa, ask_vhf_comms, ask_vhf_from_oow_decision, load_scenarios, preload, unload,
    scenario_playback, MODEL_CONFIGS,
)
from app.vhf_model_variants import DEFAULT_VARIANT
from pipeline.qwen_remote import is_remote_server_up
from app.vhf_crypto import (
    caesar_encode, caesar_decode, encrypt_aes_gcm, decrypt_aes_gcm,
    generate_rsa_keypair, encrypt_rsa_hybrid, decrypt_rsa_hybrid,
)
from app.vhf_signals import (
    PHONETIC_ALPHABET, FLAG_BLAZONS, MORSE_CODE, MANOEUVRING_SIGNALS,
    RESTRICTED_VISIBILITY_SIGNALS, SIGNAL_EXCHANGE_SCENARIOS, load_flag_meanings, flag_svg,
    flag_signal_example, text_to_morse, morse_to_text, pattern_to_symbols,
)
from app.vhf_audio import (
    synthesize_blast_wav, synthesize_morse_wav, synthesize_vhf_voice_wav, transcribe_wav_bytes,
)

st.set_page_config(page_title="VHF Communications Simulator", page_icon="\U0001F4FB", layout="wide")


@st.cache_data(show_spinner=False)
def _cached_transcribe(wav_bytes: bytes) -> str | None:
    """Thin st.cache_data wrapper -- transcribe_wav_bytes() itself stays a plain,
    Streamlit-free function (testable without a running app)."""
    return transcribe_wav_bytes(wav_bytes)


@st.cache_data(show_spinner=False)
def _cached_vhf_voice(text: str) -> bytes | None:
    """Thin st.cache_data wrapper -- synthesize_vhf_voice_wav() itself stays a plain,
    Streamlit-free function (testable without a running app)."""
    return synthesize_vhf_voice_wav(text)


_HEADER_IMAGE = ROOT.parent / "Data" / "VHF" / "VHFProtocol" / "RadioRoomHeader.jpg"
if _HEADER_IMAGE.exists():
    st.image(str(_HEADER_IMAGE), width="stretch")

with st.sidebar:
    st.header("Model")
    st.write(f"Running: **{DEFAULT_VARIANT}**")
    # Loading is lazy/optional either way -- any Ask/Grade/Draft action below loads Qwen3-8B
    # on first use via st.cache_resource if you skip preloading here. This status + the
    # unload button just make that state visible and let you free the GPU again afterward.
    if st.session_state.get("vhf_model_ready", False):
        st.success("\u2705 Qwen3-8B loaded (GPU in use).")
        if st.button("\U0001F6D1 Unload model (free GPU)"):
            with st.spinner("Freeing GPU memory..."):
                unload()
            st.session_state.vhf_model_ready = False
            st.rerun()
    else:
        st.caption("Model not loaded -- GPU free for other processes. Any Ask/Grade/Draft "
                  "action loads it on first use if you skip preloading here.")
        if st.button("Preload model + retrieval index"):
            status = st.empty()
            preload(status_cb=status.write)
            status.write("Ready.")
            st.session_state.vhf_model_ready = True
            st.rerun()
    remote_up = is_remote_server_up()
    st.caption("\U0001F7E2 cloud model reachable" if remote_up else "\U0001F534 cloud model not reachable (SSH tunnel down?)")
    st.session_state.vhf_use_remote = st.toggle(
        "Generate on cloud GPU instead of locally", value=st.session_state.get("vhf_use_remote", False),
        disabled=not remote_up, help="Routes every Ask/Grade/Draft call to cloud/qwen_inference_server.py "
                                     "over the SSH tunnel on port 8801 instead of loading Qwen3-8B locally.",
    )
    st.header("Prompt config (Track 1)")
    config = st.selectbox("Config", list(MODEL_CONFIGS), index=list(MODEL_CONFIGS).index("v1_rag"),
                          format_func=lambda c: MODEL_CONFIGS[c])

tab_qa, tab_sim, tab_oow, tab_signals, tab_crypto = st.tabs([
    "Ask VHF (Track 1)", "Radio Simulator (Track 2)", "OOW \u2192 Comms",
    "\U0001F6A9 Signals", "\U0001F510 Secret Transmission",
])

# ── Tab 1: Ask VHF ───────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Ask a VHF / GMDSS question")
    if "vhf_qa_history" not in st.session_state:
        st.session_state.vhf_qa_history = []  # newest first: [{question, answer, sources, pg_used, config}]

    with st.form("ask_vhf_form", clear_on_submit=False):
        question = st.text_input("Question", placeholder="e.g. What channel do you use to hail a port control station?")
        asked = st.form_submit_button("Ask")

    if asked and question:
        try:
            with st.spinner("Retrieving context + generating answer..."):
                result = ask_vhf_qa(question, config=config)
        except Exception as e:  # noqa: BLE001 -- surface any retrieval/model error to the user, don't crash the app
            st.error(f"Couldn't get an answer: {e}")
        else:
            st.session_state.vhf_model_ready = True
            st.session_state.vhf_qa_history.insert(0, {
                "question": question, "answer": result["answer"],
                "sources": result.get("sources"), "pg_used": result.get("pg_used"), "config": config,
            })

    for i, item in enumerate(st.session_state.vhf_qa_history):
        with st.container(border=True):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            caption_bits = [f"config: {item['config']}"]
            if item.get("pg_used"):
                caption_bits.append(f"PG: {item['pg_used']}")
            st.caption(" · ".join(caption_bits))
            if item.get("sources"):
                with st.expander(f"Sources ({len(item['sources'])} retrieved chunks)"):
                    st.code("\n".join(item["sources"]))

# ── Tab 2: Radio Simulator ───────────────────────────────────────────────
with tab_sim:
    st.subheader("Collision-avoidance radio exchange simulator")
    scenarios = load_scenarios()
    labels = [f"{i}: {s.get('category', '?')} — {s.get('region', '?')}" for i, s in enumerate(scenarios)]
    idx = st.selectbox("Scenario", range(len(scenarios)), format_func=lambda i: labels[i])
    scenario = scenarios[idx]

    col_details, col_spoiler = st.columns(2)
    with col_details:
        with st.expander("Scenario details"):
            st.json(scenario)
    with col_spoiler:
        with st.expander("Expected channel / rules (spoiler)"):
            ch = scenario.get("vhf_channel", {})
            st.write(f"Hailing: **{ch.get('hailing', '?')}** · Working: **{ch.get('working', '?')}**")
            st.write("COLREG rule(s): " + (", ".join(scenario.get("colreg_rules", [])) or "—"))

    mode = st.radio("Mode", ["Preview (read-only playback)", "Transmit (you draft the call)",
                    "Receive (agent hails you first)"], horizontal=True)

    if mode.startswith("Preview"):
        pb = scenario_playback(scenario)
        st.markdown(f"#### {pb['category']} \u2014 {pb['region']}")
        col_own, col_target = st.columns(2)
        with col_own:
            st.markdown("**\U0001F6A2 Own vessel**")
            st.write(pb["own_vessel"])
        with col_target:
            st.markdown("**\U0001F6A4 Target vessel**")
            st.write(pb["target_vessel"])
        st.markdown("**Situation**")
        st.write(pb["situation"])
        st.markdown(
            f"**Channel:** hailing **{pb['channel_hailing']}**, working **{pb['channel_working']}**"
            + (f" \u2014 {pb['channel_settings']}" if pb["channel_settings"] else "")
        )
        if pb["channel_note"]:
            st.caption(pb["channel_note"])
        st.markdown("**Applicable COLREG rule(s):** " + (", ".join(pb["colreg_rules"]) or "\u2014"))
        if pb["reference_transmission"]:
            with st.expander("Reference transmission (spoiler -- compare after you try Transmit/Receive yourself)"):
                st.write(pb["reference_transmission"])
                ref_audio = _cached_vhf_voice(pb["reference_transmission"])
                if ref_audio is not None:
                    st.audio(ref_audio, format="audio/wav")
                else:
                    st.caption("\U0001F50A Audio playback needs pyttsx3 (not installed).")
                if pb["expected_points"]:
                    st.markdown("**Expected points:**")
                    for point in pb["expected_points"]:
                        st.markdown(f"- {point}")
        st.info("This is a read-only preview -- switch to **Transmit** or **Receive** above to "
               "actually play out this scenario with the agent.")

    else:
        log_key = f"vhf_log_{idx}_{mode}"
        if log_key not in st.session_state:
            st.session_state[log_key] = []

        if mode.startswith("Transmit"):
            mic_key = f"transmit_mic_{idx}"
            transcript_key = f"transmit_transcript_{idx}"
            mic_audio = st.audio_input("Or record your transmission", key=mic_key)
            if mic_audio is not None:
                try:
                    transcript = _cached_transcribe(mic_audio.getvalue())
                except Exception as e:  # noqa: BLE001 -- bad/corrupt recording, don't crash the page
                    st.error(f"Couldn't transcribe that recording: {e}")
                else:
                    if transcript is None:
                        st.caption("\U0001F3A4 Speech-to-text needs openai-whisper (not installed).")
                    else:
                        st.session_state[transcript_key] = transcript
                        st.caption(f"Transcribed: \u201c{transcript}\u201d")
            with st.form("transmit_form", clear_on_submit=True):
                user_call = st.text_area(
                    "Your transmission", value=st.session_state.get(transcript_key, ""),
                    placeholder="e.g. \"Gulf Explorer, Gulf Explorer, this is Texas Spirit...\"",
                )
                sent = st.form_submit_button("Grade my transmission")
            if sent and user_call:
                st.session_state.pop(transcript_key, None)
                try:
                    with st.spinner("Grading..."):
                        result = ask_vhf_comms(scenario, mode="transmit", user_text=user_call)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Couldn't grade that transmission: {e}")
                else:
                    st.session_state.vhf_model_ready = True
                    st.session_state[log_key].append({"speaker": "You", "content": user_call})
                    st.session_state[log_key].append({"speaker": "VHF Instructor", "content": result["feedback"]})
        else:
            if not st.session_state[log_key] and st.button("Hail me"):
                try:
                    with st.spinner("Generating incoming hail..."):
                        result = ask_vhf_comms(scenario, mode="receive")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Couldn't generate a hail: {e}")
                else:
                    st.session_state.vhf_model_ready = True
                    st.session_state[log_key] = result["history"]
            if st.session_state[log_key]:
                reply_mic_key = f"receive_mic_{idx}"
                reply_transcript_key = f"receive_transcript_{idx}"
                reply_mic_audio = st.audio_input("Or record your reply", key=reply_mic_key)
                if reply_mic_audio is not None:
                    try:
                        reply_transcript = _cached_transcribe(reply_mic_audio.getvalue())
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Couldn't transcribe that recording: {e}")
                    else:
                        if reply_transcript is None:
                            st.caption("\U0001F3A4 Speech-to-text needs openai-whisper (not installed).")
                        else:
                            st.session_state[reply_transcript_key] = reply_transcript
                            st.caption(f"Transcribed: \u201c{reply_transcript}\u201d")
                with st.form("reply_form", clear_on_submit=True):
                    reply = st.text_input("Your reply", value=st.session_state.get(reply_transcript_key, ""))
                    replied = st.form_submit_button("Send reply")
                if replied and reply:
                    st.session_state.pop(reply_transcript_key, None)
                    try:
                        with st.spinner("..."):
                            result = ask_vhf_comms(scenario, mode="receive", user_text=reply,
                                                   history=st.session_state[log_key])
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Couldn't send that reply: {e}")
                    else:
                        st.session_state.vhf_model_ready = True
                        st.session_state[log_key] = result["history"]

        if st.session_state[log_key] and st.button("Clear log", key=f"clear_{log_key}"):
            st.session_state[log_key] = []
            st.rerun()

        st.markdown("---")
        st.markdown("**Transmission log**")
        if not st.session_state[log_key]:
            st.caption("No transmissions yet.")
        for i, entry in enumerate(st.session_state[log_key]):
            speaker = entry.get("speaker") or ("You" if entry.get("role") == "user" else "Other station")
            is_you = speaker == "You" or entry.get("role") == "user"
            with st.chat_message("user" if is_you else "assistant",
                                 avatar="\U0001F6A2" if is_you else "\U0001F4FB"):
                st.markdown(f"**{speaker}:** {entry['content']}")
                if st.button("\U0001F50A Play (distorted VHF radio)", key=f"play_{log_key}_{i}"):
                    audio_bytes = _cached_vhf_voice(entry["content"])
                    if audio_bytes is None:
                        st.info("Text-to-speech (pyttsx3) isn't installed -- this is an "
                               "optional, local-only feature. See vhf_audio.py's module "
                               "docstring.")
                    else:
                        st.audio(audio_bytes, format="audio/wav")

# ── Tab 3: OOW → Comms ───────────────────────────────────────────────────
with tab_oow:
    st.subheader("Draft a radio call from an OOW decision")
    st.caption("VHF never decides the manoeuvre -- it only confirms a decision the OOW agent "
              "already made (design_vhf_communications.md Sec 8.1).")

    situation = st.text_area(
        "Situation (plain text)",
        value="Own vessel MV Texas Spirit, 150m tanker, course 270 at 12kn. Target vessel "
             "MV Gulf Explorer, 120m research vessel, bearing 090, range 0.5nm, head-on aspect.",
        height=80,
    )
    col_a, col_d, col_e, col_c = st.columns(4)
    with col_a:
        action = st.selectbox("action", ["turn_left", "turn_right", "hold_course", "speed_up", "slow_down", "stop"])
    with col_d:
        degrees = st.number_input("degrees", min_value=0.0, max_value=180.0, value=20.0, step=5.0)
    with col_e:
        encounter_rule = st.text_input("encounter_rule", value="Rule 14")
    with col_c:
        conduct_rule = st.text_input("conduct_rule", value="Rule 14")
    reasoning = st.text_input("reasoning", value="Head-on encounter, both vessels alter to starboard.")

    if st.button("Draft radio call"):
        oow_decision = {
            "action": action, "degrees": degrees if action in ("turn_left", "turn_right") else None,
            "encounter_rule": encounter_rule, "conduct_rule": conduct_rule, "reasoning": reasoning,
        }
        try:
            with st.spinner("Drafting..."):
                result = ask_vhf_from_oow_decision(situation, oow_decision)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't draft that call: {e}")
        else:
            st.session_state.vhf_model_ready = True
            with st.chat_message("assistant", avatar="\U0001F4FB"):
                st.markdown(f"**VHF (confirming OOW decision):** {result['transmission']}")
            with st.expander("OOW decision sent"):
                st.json(oow_decision)

# ── Tab 4: Signals (flags / light / sound / Morse) ────────────────────────
with tab_signals:
    st.subheader("\U0001F6A9 Flags, light/sound signals, and Morse code")
    sig_flags, sig_light_sound, sig_morse, sig_practice = st.tabs(
        ["Flags", "Light & Sound", "Morse", "\U0001F3AD Exchange Practice"]
    )

    with sig_flags:
        st.caption("ICS single-flag meanings are real, sourced text (flag_signals.json, "
                  "also in the RAG index). The flag graphic is rendered from the real "
                  "heraldic blazon (colors + pattern), transcribed from Wikipedia's "
                  "'International maritime signal flags' article (CC BY-SA 4.0) -- exact "
                  "stripe/cross geometry is a simplified approximation, not pixel-perfect "
                  "to the official IMO ICS chart, but colors and pattern type are sourced.")
        letter = st.selectbox("Letter", list(PHONETIC_ALPHABET),
                              format_func=lambda l: f"{l} \u2014 {PHONETIC_ALPHABET[l]}")
        meanings = load_flag_meanings()
        col_flag, col_info = st.columns([1, 3])
        with col_flag:
            st.markdown(flag_svg(letter), unsafe_allow_html=True)
            st.caption(f"Blazon: *{FLAG_BLAZONS[letter]}*")
        with col_info:
            st.markdown(f"**{PHONETIC_ALPHABET[letter]}**")
            st.write(meanings[letter])
            st.code(MORSE_CODE[letter], language=None)
        example = flag_signal_example(letter)
        with st.expander("Example: sending & receiving this flag"):
            st.markdown(f"**Sending:** {example['send']}")
            st.markdown(f"**Receiving:** {example['receive']}")

    with sig_light_sound:
        st.caption("Real, sourced patterns (COLREG Rules 32-35, light_sound_signals.json). "
                  "Light flashes and sound blasts share the same short/prolonged timing "
                  "(Rule 34(b)) -- \u2022 = short, \u2014 = prolonged. Audio is a synthesized "
                  "illustrative tone (not a recording of a real ship's whistle).")
        st.markdown("**Manoeuvring signals (Rule 34)**")
        for sig in MANOEUVRING_SIGNALS:
            st.markdown(f"- `{pattern_to_symbols(sig['pattern'])}` -- {sig['name']} "
                       f"({sig['rule']})")
            st.audio(synthesize_blast_wav(sig["pattern"]), format="audio/wav")
        st.markdown("**Restricted visibility (Rule 35)**")
        for sig in RESTRICTED_VISIBILITY_SIGNALS:
            st.markdown(f"- `{pattern_to_symbols(sig['pattern'])}` -- {sig['name']} "
                       f"({sig['rule']}, {sig['interval']})")
            st.audio(synthesize_blast_wav(sig["pattern"]), format="audio/wav")

    with sig_morse:
        st.caption("A genuine, still-practiced signalling sub-channel (design doc Sec 9 "
                  "scope note) -- not an easter egg. Standard international Morse code.")
        morse_mode = st.radio("Direction", ["Text \u2192 Morse", "Morse \u2192 Text"], horizontal=True)
        if morse_mode.startswith("Text"):
            text_in = st.text_input("Text", placeholder="e.g. SOS")
            if text_in:
                morse_out = text_to_morse(text_in)
                st.code(morse_out, language=None)
                if morse_out:
                    st.audio(synthesize_morse_wav(morse_out), format="audio/wav")
        else:
            morse_in = st.text_input("Morse (letters space-separated, / between words)",
                                     placeholder="e.g. ... --- ...")
            if morse_in:
                st.code(morse_to_text(morse_in), language=None)
                st.audio(synthesize_morse_wav(morse_in), format="audio/wav")

    with sig_practice:
        st.caption("Practice reading an incoming flag/light/sound/Morse signal and "
                  "working out the right response, before revealing the real answer -- "
                  "same sourced content as the other 3 sub-tabs, no model call needed.")
        sc_labels = [f"{i}: {sc['channel']} \u2014 {sc['id']}" for i, sc in enumerate(SIGNAL_EXCHANGE_SCENARIOS)]
        sc_idx = st.selectbox("Scenario", range(len(SIGNAL_EXCHANGE_SCENARIOS)),
                              format_func=lambda i: sc_labels[i], key="practice_scenario")
        sc = SIGNAL_EXCHANGE_SCENARIOS[sc_idx]
        st.markdown(f"**Situation:** {sc['situation']}")

        if sc.get("incoming_flag"):
            st.markdown("**Incoming signal:**")
            st.markdown(flag_svg(sc["incoming_flag"]), unsafe_allow_html=True)
        if sc.get("incoming_pattern"):
            st.markdown(f"**Incoming signal:** `{pattern_to_symbols(sc['incoming_pattern'])}`")
            st.audio(synthesize_blast_wav(sc["incoming_pattern"]), format="audio/wav")

        st.text_input("Your interpretation / response (not graded -- jot it down, then reveal)",
                      key=f"practice_answer_{sc_idx}")

        with st.expander("Reveal"):
            st.write(sc["reveal_meaning"])
            if sc.get("reveal_flag"):
                st.markdown(flag_svg(sc["reveal_flag"]), unsafe_allow_html=True)
            if sc.get("reveal_pattern"):
                st.markdown(f"`{pattern_to_symbols(sc['reveal_pattern'])}`")
                st.audio(synthesize_blast_wav(sc["reveal_pattern"]), format="audio/wav")
            if sc.get("reveal_text"):
                reveal_morse = text_to_morse(sc["reveal_text"])
                st.code(f"{sc['reveal_text']}  ({reveal_morse})", language=None)
                st.audio(synthesize_morse_wav(reveal_morse), format="audio/wav")

# ── Tab 5: Secret Transmission ────────────────────────────────────────────
with tab_crypto:
    st.subheader("\U0001F510 Secret Transmission")
    sub_caesar, sub_aes, sub_rsa = st.tabs([
        "Caesar (easter egg)", "\U0001F511 Passphrase (AES-256-GCM)",
        "\U0001F4DC Asymmetric / PGP-style (RSA-OAEP)",
    ])

    with sub_caesar:
        st.warning("Just for fun -- a Caesar (shift) cipher, **not** a real GMDSS/VHF procedure. "
                  "Never use this for anything safety-related.")
        shift = st.slider("Shift", min_value=1, max_value=25, value=3)
        message = st.text_input("Message", placeholder="e.g. MEET AT BUOY DELTA ONE TWO")
        col_enc, col_dec = st.columns(2)
        with col_enc:
            if st.button("Encode") and message:
                st.code(caesar_encode(message, shift))
        with col_dec:
            if st.button("Decode") and message:
                st.code(caesar_decode(message, shift))

    with sub_aes:
        st.caption("Real encryption -- AES-256-GCM (NIST SP 800-38D), key derived from your "
                  "passphrase via PBKDF2-HMAC-SHA256 (600k iterations). Both sides need the "
                  "same passphrase, shared out-of-band beforehand.")
        passphrase = st.text_input("Shared passphrase", type="password", key="aes_passphrase")
        col_aes_enc, col_aes_dec = st.columns(2)
        with col_aes_enc:
            st.markdown("**Encrypt**")
            aes_plain = st.text_area("Message to encrypt", key="aes_plain")
            if st.button("Encrypt", key="aes_encrypt_btn") and aes_plain and passphrase:
                st.code(encrypt_aes_gcm(aes_plain, passphrase), language="json")
        with col_aes_dec:
            st.markdown("**Decrypt**")
            aes_blob = st.text_area("Encrypted blob (JSON)", key="aes_blob")
            if st.button("Decrypt", key="aes_decrypt_btn") and aes_blob and passphrase:
                try:
                    st.code(decrypt_aes_gcm(aes_blob, passphrase))
                except ValueError as e:
                    st.error(str(e))

    with sub_rsa:
        st.caption("Real asymmetric encryption -- the same hybrid scheme OpenPGP itself uses: "
                  "a random AES-256 session key encrypts your message, then RSA-OAEP(SHA-256) "
                  "encrypts that session key with the recipient's public key. Generate a keypair, "
                  "share the PUBLIC key with whoever should send you secrets, keep the PRIVATE "
                  "key to yourself.")
        if st.button("Generate keypair"):
            priv, pub = generate_rsa_keypair()
            st.session_state.rsa_private_pem = priv
            st.session_state.rsa_public_pem = pub
        if st.session_state.get("rsa_public_pem"):
            with st.expander("My keypair"):
                st.markdown("**Public key** (share this)")
                st.code(st.session_state.rsa_public_pem, language=None)
                st.markdown("**Private key** (keep secret)")
                st.code(st.session_state.rsa_private_pem, language=None)

        col_rsa_enc, col_rsa_dec = st.columns(2)
        with col_rsa_enc:
            st.markdown("**Encrypt to a public key**")
            rsa_pub_in = st.text_area("Recipient's public key (PEM)", key="rsa_pub_in",
                                      value=st.session_state.get("rsa_public_pem", ""))
            rsa_plain = st.text_area("Message to encrypt", key="rsa_plain")
            if st.button("Encrypt", key="rsa_encrypt_btn") and rsa_plain and rsa_pub_in:
                try:
                    st.code(encrypt_rsa_hybrid(rsa_plain, rsa_pub_in), language="json")
                except ValueError as e:
                    st.error(str(e))
        with col_rsa_dec:
            st.markdown("**Decrypt with my private key**")
            rsa_priv_in = st.text_area("My private key (PEM)", key="rsa_priv_in",
                                       value=st.session_state.get("rsa_private_pem", ""))
            rsa_blob = st.text_area("Encrypted blob (JSON)", key="rsa_blob")
            if st.button("Decrypt", key="rsa_decrypt_btn") and rsa_blob and rsa_priv_in:
                try:
                    st.code(decrypt_rsa_hybrid(rsa_blob, rsa_priv_in))
                except ValueError as e:
                    st.error(str(e))
