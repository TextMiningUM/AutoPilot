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
  - "Radio Simulator" (Track 2): pick a scenario, then either draft your own
    transmission (graded by the agent) or receive an incoming hail and reply to it,
    rendered as a radio-style transmission log (st.chat_message bubbles).
  - "OOW -> Comms" (Sec 8.1): draft the radio call for an already-made OOW manoeuvre
    decision -- VHF never decides the manoeuvre itself, only confirms it.
  - "Signals" (Sec 9.1/12.9): channel-specific renderers for flags (ICS meaning +
    phonetic + Morse + a real flag graphic rendered from Wikipedia's sourced heraldic
    blazon -- see vhf_signals.py's module docstring -- plus a deterministic send/receive
    example), light/sound manoeuvring + restricted-visibility signal patterns (COLREG
    Rules 32-35, real sourced data), and a Morse encoder/decoder (a genuine, still-
    practiced signalling sub-channel per Sec 9's scope note, not an easter egg).
  - "Secret Transmission" (Sec 12.7 easter egg): a Caesar-cipher toy, clearly labelled as
    NOT a real GMDSS/VHF procedure, kept structurally separate from the real content.

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
    MODEL_CONFIGS,
)
from app.vhf_model_variants import DEFAULT_VARIANT
from app.vhf_crypto import caesar_encode, caesar_decode
from app.vhf_signals import (
    PHONETIC_ALPHABET, FLAG_BLAZONS, MORSE_CODE, MANOEUVRING_SIGNALS,
    RESTRICTED_VISIBILITY_SIGNALS, load_flag_meanings, flag_svg, flag_signal_example,
    text_to_morse, morse_to_text, pattern_to_symbols,
)

st.set_page_config(page_title="VHF Communications Simulator", page_icon="\U0001F4FB", layout="wide")

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

    mode = st.radio("Mode", ["Transmit (you draft the call)", "Receive (agent hails you first)"],
                    horizontal=True)

    log_key = f"vhf_log_{idx}_{mode}"
    if log_key not in st.session_state:
        st.session_state[log_key] = []

    if mode.startswith("Transmit"):
        with st.form("transmit_form", clear_on_submit=True):
            user_call = st.text_area("Your transmission", placeholder="e.g. \"Gulf Explorer, Gulf Explorer, this is Texas Spirit...\"")
            sent = st.form_submit_button("Grade my transmission")
        if sent and user_call:
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
            with st.form("reply_form", clear_on_submit=True):
                reply = st.text_input("Your reply")
                replied = st.form_submit_button("Send reply")
            if replied and reply:
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
    for entry in st.session_state[log_key]:
        speaker = entry.get("speaker") or ("You" if entry.get("role") == "user" else "Other station")
        is_you = speaker == "You" or entry.get("role") == "user"
        with st.chat_message("user" if is_you else "assistant",
                             avatar="\U0001F6A2" if is_you else "\U0001F4FB"):
            st.markdown(f"**{speaker}:** {entry['content']}")

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
    sig_flags, sig_light_sound, sig_morse = st.tabs(["Flags", "Light & Sound", "Morse"])

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
                  "(Rule 34(b)) -- \u2022 = short, \u2014 = prolonged.")
        st.markdown("**Manoeuvring signals (Rule 34)**")
        for sig in MANOEUVRING_SIGNALS:
            st.markdown(f"- `{pattern_to_symbols(sig['pattern'])}` -- {sig['name']} "
                       f"({sig['rule']})")
        st.markdown("**Restricted visibility (Rule 35)**")
        for sig in RESTRICTED_VISIBILITY_SIGNALS:
            st.markdown(f"- `{pattern_to_symbols(sig['pattern'])}` -- {sig['name']} "
                       f"({sig['rule']}, {sig['interval']})")

    with sig_morse:
        st.caption("A genuine, still-practiced signalling sub-channel (design doc Sec 9 "
                  "scope note) -- not an easter egg. Standard international Morse code.")
        morse_mode = st.radio("Direction", ["Text \u2192 Morse", "Morse \u2192 Text"], horizontal=True)
        if morse_mode.startswith("Text"):
            text_in = st.text_input("Text", placeholder="e.g. SOS")
            if text_in:
                st.code(text_to_morse(text_in), language=None)
        else:
            morse_in = st.text_input("Morse (letters space-separated, / between words)",
                                     placeholder="e.g. ... --- ...")
            if morse_in:
                st.code(morse_to_text(morse_in), language=None)

# ── Tab 5: Secret Transmission (crypto easter egg) ────────────────────────
with tab_crypto:
    st.subheader("\U0001F510 Secret Transmission")
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
