"""Basic Simulator — multipage entry point.

Routes between the OOW simulator (home.py), Captain Mission, Engine Room, and
Communications (former standalone "VHF Simulator" app, merged 2026-10-03) pages via
st.navigation().

Run from the Basic Simulator/ folder:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

pg = st.navigation([
    st.Page("pages/1_Captain_Mission.py", title="Captain Mission", icon="\U0001F6A2"),
    st.Page("home.py", title="OOW", icon="\U0001F9ED", default=True),
    st.Page("pages/2_Engine_Room.py", title="Engine Room", icon="\U0001F6E2\uFE0F"),
    st.Page("pages/3_Communications.py", title="Communications", icon="\U0001F4FB"),
])

pg.run()
