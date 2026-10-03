"""VHF Simulator -- Streamlit interface (MVP, design_vhf_communications.md Sec 12).

Run from the `VHF Simulator/` folder:
    streamlit run app/streamlit_app.py

Two tabs:
  - "Ask VHF" (Track 1): free-text question -> grounded answer + retrieved sources,
    with a session-scoped question history.
  - "Radio Simulator" (Track 2): pick a scenario, then either draft your own
    transmission (graded by the agent) or receive an incoming hail and reply to it,
    rendered as a radio-style transmission log (st.chat_message bubbles).

Flags/light-sound channels, OOW->Comms wiring, and the crypto easter egg are explicitly
out of scope for this MVP (see design doc Sec 12.7) -- not in this file.
"""
from __future__ import annotations
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
REPO_ROOT = ROOT.parent
for p in (ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

from app.vhf_agents import ask_vhf_qa, ask_vhf_comms, load_scenarios, preload, MODEL_CONFIGS
from app.vhf_model_variants import DEFAULT_VARIANT

st.set_page_config(page_title="VHF Communications Simulator", page_icon="\U0001F4FB", layout="wide")

# Same gradient family as Basic Simulator's hero (visual consistency across this
# project's simulator interfaces) -- kept lightweight, no new CSS conventions invented.
st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #0b3d63 0%, #146c94 55%, #19a7ce 100%);
    padding: 1.2rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.hero h1 { color: #f4fbff; margin: 0; font-size: 1.7rem; }
.hero p { color: #d6f0fb; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
</style>
<div class="hero">
<h1>\U0001F4FB VHF Communications Simulator</h1>
<p>Base Qwen3-8B -- no VHF fine-tuning yet (design_vhf_communications.md Sec 9.3)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Model")
    st.write(f"Running: **{DEFAULT_VARIANT}**")
    if st.button("Preload model + retrieval index"):
        status = st.empty()
        preload(status_cb=status.write)
        status.write("Ready.")
    st.header("Prompt config (Track 1)")
    config = st.selectbox("Config", list(MODEL_CONFIGS), index=list(MODEL_CONFIGS).index("v1_rag"),
                          format_func=lambda c: MODEL_CONFIGS[c])

tab_qa, tab_sim = st.tabs(["Ask VHF (Track 1)", "Radio Simulator (Track 2)"])

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
