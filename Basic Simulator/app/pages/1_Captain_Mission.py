"""Captain walking-skeleton Streamlit page -- a minimal UI over the Sec 15.3 debug control
set (step / run-to-next-event / run-to-completion / force-event / mission-state / MPR /
evaluation), using the EXACT SAME `CaptainSkeleton` class as `run_captain_scenario.py` and
`tests/test_captain_skeleton*.py` -- no logic lives in this file, only widgets.

Auto-discovered by Streamlit's native multipage convention (a `pages/` folder next to the
main entrypoint, app/streamlit_app.py) -- run `streamlit run app/streamlit_app.py` and pick
"Captain Mission" from the sidebar page list. Local-safe: pure deterministic simulation, no
GPU/API key needed, same as the rest of app/captain_skeleton.py.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent  # Basic Simulator/app/
ROOT = APP_DIR.parent                             # Basic Simulator/
for p in (ROOT, ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd
import streamlit as st

from app.captain_skeleton import CaptainSkeleton

st.set_page_config(page_title="Captain Mission", page_icon="\U0001F6A2", layout="wide")

SCENARIOS_DIR = ROOT.parent / "Data" / "Captain" / "Scenarios"
SCENARIO_PATHS = {p.stem: p for p in sorted(SCENARIOS_DIR.glob("*.json"))}
MAX_RUN_STEPS = 20_000  # generous safety net for "run to next event"/"run to completion"


def _history_row(skeleton: CaptainSkeleton) -> dict:
    return {"t_h": skeleton.sim.state.elapsed_s / 3600.0,
            "speed_kn": skeleton.sim.state.current_speed_kn,
            "fuel_t": skeleton.sim.state.fuel_tonnes}


def _load(scenario_name: str) -> None:
    st.session_state.captain_skeleton = CaptainSkeleton.from_scenario_file(SCENARIO_PATHS[scenario_name])
    st.session_state.captain_history = [_history_row(st.session_state.captain_skeleton)]
    st.session_state._captain_loaded_scenario = scenario_name


st.title("\U0001F6A2 Captain walking-skeleton")
st.caption("Mission-level brown-envelope decision layer above OOW/VHF -- "
          "design_captain_missions.md Sec 14/15.3's own debug control set, rendered as widgets.")

picked = st.sidebar.selectbox("Scenario", options=list(SCENARIO_PATHS), key="captain_scenario_picked")
if ("captain_skeleton" not in st.session_state
        or st.session_state.get("_captain_loaded_scenario") != picked):
    _load(picked)

skeleton: CaptainSkeleton = st.session_state.captain_skeleton

st.sidebar.divider()
st.sidebar.header("Controls")
if st.sidebar.button("\U0001F504 Reset mission", use_container_width=True):
    _load(picked)
    st.rerun()

n_steps = st.sidebar.number_input("Steps", min_value=1, max_value=500, value=1, step=1)
if st.sidebar.button("\u23E9 Step", use_container_width=True):
    for _ in range(int(n_steps)):
        if skeleton.sim.reached_destination():
            break
        skeleton.step_mission(1)
        st.session_state.captain_history.append(_history_row(skeleton))
    st.rerun()

if st.sidebar.button("\u27A1\uFE0F Run to next event", use_container_width=True):
    n_fired_before = len(skeleton.fired_event_ids)
    for _ in range(MAX_RUN_STEPS):
        if skeleton.sim.reached_destination() or len(skeleton.fired_event_ids) > n_fired_before:
            break
        skeleton.step_mission(1)
        st.session_state.captain_history.append(_history_row(skeleton))
    st.rerun()

if st.sidebar.button("\u23ED\uFE0F Run to completion", use_container_width=True):
    for _ in range(MAX_RUN_STEPS):
        if skeleton.sim.reached_destination():
            break
        skeleton.step_mission(1)
        st.session_state.captain_history.append(_history_row(skeleton))
    st.rerun()

pending = [e for e in skeleton.pending_events if e.event_id not in skeleton.fired_event_ids]
if pending:
    st.sidebar.divider()
    forced_id = st.sidebar.selectbox("Force a pending event", options=[e.event_id for e in pending])
    if st.sidebar.button("\u26A1 Force now", use_container_width=True):
        skeleton.force_event(forced_id)
        st.session_state.captain_history.append(_history_row(skeleton))
        st.rerun()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Elapsed", f"{skeleton.sim.state.elapsed_s / 3600.0:.1f} h")
m2.metric("Distance", f"{skeleton.sim.state.distance_travelled_nm:.1f} / "
                      f"{skeleton.sim.total_route_distance_nm:.1f} nm")
m3.metric("Speed", f"{skeleton.sim.state.current_speed_kn:.1f} kn")
m4.metric("Fuel remaining", f"{skeleton.sim.state.fuel_tonnes:.1f} t")

hazards = [h["type"] for h in skeleton.state.active_hazards if "cleared_at_s" not in h]
st.caption("Active hazards: " + (", ".join(hazards) if hazards else "none") +
          (f"  \u2022  Reached destination: {skeleton.sim.reached_destination()}"))

chart_col, state_col = st.columns([2, 1])
with chart_col:
    history_df = pd.DataFrame(st.session_state.captain_history).set_index("t_h")
    st.markdown("**Speed (kn) over mission time (h)**")
    st.line_chart(history_df[["speed_kn"]])
    st.markdown("**Fuel remaining (t) over mission time (h)**")
    st.line_chart(history_df[["fuel_t"]])

with state_col:
    st.markdown("**Mission State**")
    st.text(skeleton.show_mission_state())
    st.markdown("**Mission Progress Report**")
    st.text(skeleton.show_mpr())

st.divider()
st.markdown("#### Event log")
if skeleton.state.event_log:
    log_df = pd.DataFrame(skeleton.state.event_log)
    for col in ("old", "new"):
        log_df[col] = log_df[col].apply(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
else:
    st.caption("No events logged yet.")

st.divider()
st.markdown("#### Evaluation")
result = skeleton.evaluate()
if not skeleton.sim.reached_destination():
    st.caption("\u26A0\uFE0F Mission not yet complete -- Hard Gate 2 applies (composite capped at 0.2).")
e1, e2, e3, e4 = st.columns(4)
e1.metric("Verdict", result.verdict)
e2.metric("Composite", f"{result.composite_score:.3f}")
e3.metric("Safety", "PASS" if result.safety_passed else "FAIL")
e4.metric("Mission outcome", f"{result.mission_outcome_score:.2f}")
