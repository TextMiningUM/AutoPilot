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
import math
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent  # Basic Simulator/app/
ROOT = APP_DIR.parent                             # Basic Simulator/
for p in (ROOT, ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.captain_skeleton import CaptainSkeleton
from app.mission_sim import position_along_route_nm

st.set_page_config(page_title="Captain Mission", page_icon="\U0001F6A2", layout="wide")

SCENARIOS_DIR = ROOT.parent / "Data" / "Captain" / "Scenarios"
# Recursive glob so the generated/ subdirectory (generate_captain_missions.py's output) is
# selectable here too, not just the 4 hand-authored top-level scenarios.
SCENARIO_PATHS = {p.stem: p for p in sorted(SCENARIOS_DIR.glob("**/*.json"))}
MAX_RUN_STEPS = 20_000  # generous safety net for "run to next event"/"run to completion"
CHART_HEIGHT = 140  # keeps speed/fuel/state/MPR/event-log small enough to fit a 1080p screen


def _history_row(skeleton: CaptainSkeleton) -> dict:
    return {"t_h": skeleton.sim.state.elapsed_s / 3600.0,
            "speed_kn": skeleton.sim.state.current_speed_kn,
            "fuel_t": skeleton.sim.state.fuel_tonnes}


def _mission_preview_figure(skeleton: CaptainSkeleton) -> go.Figure:
    """The Captain-level analogue of OOW's 'Scenario preview' plot -- route, ports,
    exclusion zones, and brown-envelope trigger points (pending vs already-fired) in raw
    lat/lon, since a whole-mission route spans far more than the few-nm window
    pipeline.captain_geo's local metric frame is accurate over (that frame is reserved for
    an encounter window, Sec 13.A.1). y-axis scaled by 1/cos(mean latitude) so the plot
    isn't visibly stretched/squashed at this corridor's own latitude."""
    waypoints = skeleton.order.waypoints
    lats = [p[0] for p in waypoints]
    lons = [p[1] for p in waypoints]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lons, y=lats, mode="lines+markers", name="Route",
                             line=dict(color="gray", dash="dash"), marker=dict(size=7, color="gray")))

    for zone in skeleton.zones:
        if not zone.polygon:
            continue
        zx = [p[1] for p in zone.polygon] + [zone.polygon[0][1]]
        zy = [p[0] for p in zone.polygon] + [zone.polygon[0][0]]
        fig.add_trace(go.Scatter(x=zx, y=zy, mode="lines", fill="toself", name=f"{zone.type}:{zone.id}",
                                 line=dict(color="orange"), fillcolor="rgba(255,165,0,0.2)"))

    if skeleton.ports:
        fig.add_trace(go.Scatter(
            x=[p.position[1] for p in skeleton.ports], y=[p.position[0] for p in skeleton.ports],
            mode="markers", name="Port", marker=dict(size=10, color="blue", symbol="square"),
            text=[p.name for p in skeleton.ports], hoverinfo="text"))

    pending_xy, pending_text, fired_xy, fired_text = [], [], [], []
    for event in skeleton.order.events:
        if event.trigger.get("type") != "distance_along_route_nm":
            continue
        lat, lon = position_along_route_nm(waypoints, event.trigger["value"])
        label = f"{event.event_id}: {event.type} ({event.severity})"
        target = fired_xy if event.event_id in skeleton.fired_event_ids else pending_xy
        target_text = fired_text if event.event_id in skeleton.fired_event_ids else pending_text
        target.append((lon, lat))
        target_text.append(label)
    if pending_xy:
        fig.add_trace(go.Scatter(x=[p[0] for p in pending_xy], y=[p[1] for p in pending_xy],
                                 mode="markers", name="Pending event", text=pending_text, hoverinfo="text",
                                 marker=dict(size=12, color="gold", symbol="diamond")))
    if fired_xy:
        fig.add_trace(go.Scatter(x=[p[0] for p in fired_xy], y=[p[1] for p in fired_xy],
                                 mode="markers", name="Fired event", text=fired_text, hoverinfo="text",
                                 marker=dict(size=12, color="red", symbol="diamond")))

    own_lat, own_lon = position_along_route_nm(waypoints, skeleton.sim.state.distance_travelled_nm)
    fig.add_trace(go.Scatter(x=[own_lon], y=[own_lat], mode="markers", name="Own ship",
                             marker=dict(size=16, color="black", symbol="triangle-up")))

    mean_lat = sum(lats) / len(lats)
    fig.update_layout(title="Mission preview", xaxis_title="Longitude", yaxis_title="Latitude",
                      height=460, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(scaleanchor="x", scaleratio=1 / max(0.1, math.cos(math.radians(mean_lat))))
    return fig


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

plot_col, info_col = st.columns([3, 2])
with plot_col:
    st.plotly_chart(_mission_preview_figure(skeleton), use_container_width=True,
                    key=f"preview_{len(st.session_state.captain_history)}")

with info_col:
    with st.container(height=140):
        st.markdown("**Mission State**")
        st.text(skeleton.show_mission_state())
    with st.container(height=140):
        st.markdown("**Mission Progress Report**")
        st.text(skeleton.show_mpr())
    history_df = pd.DataFrame(st.session_state.captain_history).set_index("t_h")
    st.caption("Speed (kn) / Fuel (t) over mission time (h)")
    sp_col, fu_col = st.columns(2)
    sp_col.line_chart(history_df[["speed_kn"]], height=CHART_HEIGHT)
    fu_col.line_chart(history_df[["fuel_t"]], height=CHART_HEIGHT)

st.markdown("#### Event log")
if skeleton.state.event_log:
    log_df = pd.DataFrame(skeleton.state.event_log)
    for col in ("old", "new"):
        log_df[col] = log_df[col].apply(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    st.dataframe(log_df, use_container_width=True, hide_index=True, height=180)
else:
    st.caption("No events logged yet.")

st.markdown("#### Evaluation")
result = skeleton.evaluate()
if not skeleton.sim.reached_destination():
    st.caption("\u26A0\uFE0F Mission not yet complete -- Hard Gate 2 applies (composite capped at 0.2).")
e1, e2, e3, e4 = st.columns(4)
e1.metric("Verdict", result.verdict)
e2.metric("Composite", f"{result.composite_score:.3f}")
e3.metric("Safety", "PASS" if result.safety_passed else "FAIL")
e4.metric("Mission outcome", f"{result.mission_outcome_score:.2f}")
