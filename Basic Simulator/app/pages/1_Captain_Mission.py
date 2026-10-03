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
from pipeline import captain_memory, captain_tools

st.set_page_config(page_title="Captain Mission", page_icon="\U0001F6A2", layout="wide")

SCENARIOS_DIR = ROOT.parent / "Data" / "Captain" / "Scenarios"
# Recursive glob so the generated/ subdirectory (generate_captain_missions.py's output) is
# selectable here too, not just the 4 hand-authored top-level scenarios.
SCENARIO_PATHS = {p.stem: p for p in sorted(SCENARIOS_DIR.glob("**/*.json"))}
MAX_RUN_STEPS = 20_000  # generous safety net for "run to next event"/"run to completion"
CHART_HEIGHT = 140  # keeps speed/fuel/state/MPR/event-log small enough to fit a 1080p screen

# Narrative rendering for the "Captain decisions" panel -- translates a raw BrownEnvelopeEvent
# (type/severity/params) into a plain-English sentence, keyed by event TYPE (one entry per
# Sec 13.A.2 v1 event).
_EVENT_ICON = {"engine_failure": "\U0001F6E0\uFE0F", "fog": "\U0001F32B\uFE0F", "whale_zone": "\U0001F40B",
              "distress_call": "\U0001F198", "commercial_instruction": "\U0001F4BC"}
_EVENT_NARRATIVE = {
    "engine_failure": lambda e: (f"**Engine failure** ({e.severity}) \u2014 "
                                 f"{e.params.get('fault', 'a machinery fault')}"),
    "fog": lambda e: (f"**Fog** ({e.severity}) \u2014 restricted visibility, Rule 19 safe speed "
                      f"{e.params.get('safe_speed_kn')} kn"),
    "whale_zone": lambda e: (f"**Whale protection zone** ({e.severity}) \u2014 charted speed limit "
                            f"{e.params.get('speed_limit_kn')} kn"),
    "distress_call": lambda e: (f"**Distress call received** ({e.severity}) via VHF \u2014 "
                                f"{e.params.get('detour_distance_nm')} nm detour to reach the scene"),
    "commercial_instruction": lambda e: (f"**Commercial instruction from the company** ({e.severity}) "
                                        f"\u2014 demands {e.params.get('demanded_speed_kn')} kn"),
}

# (narrative sentence, who it's actually communicated to) per Sec 13.B.6 tool name -- the
# "recipient" is honest about today's architecture (see the panel's own caption): a
# speed-affecting decision reaches OOW only as mechanical state, never a rendered instruction.
_DECISION_NARRATIVE = {
    "continue_at_capped_speed": lambda p: (
        f"Continue the voyage at the capped speed of **{p.get('speed_kn')} kn**.", "OOW (speed state)"),
    "request_place_of_refuge": lambda p: (
        f"Divert to the port of refuge **{p.get('port_id')}** for repairs.", "OOW (new destination) + Company/DPA"),
    "reduce_to_safe_speed": lambda p: (
        f"Reduce speed to the Rule 19 safe speed of **{p.get('speed_kn')} kn**.", "OOW (speed state)"),
    "reduce_to_zone_speed_limit": lambda p: (
        f"Reduce speed to the zone's posted limit of **{p.get('speed_kn')} kn**.", "OOW (speed state)"),
    "proceed_to_assist": lambda p: (
        "Divert to render assistance to the vessel in distress.", "OOW (reroute) + VHF (distressed vessel)"),
    "decline_with_logged_reason": lambda p: (
        "Decline to assist \u2014 assisting would itself breach the fuel reserve; reason logged.",
        "DPA / ship's log"),
    "comply_with_instruction": lambda p: (
        f"Comply with the company's instruction, adjusting speed to **{p.get('speed_kn')} kn**.",
        "OOW (speed state) + Company"),
    "refuse_citing_ism_art5": lambda p: (
        "Refuse the instruction, citing **ISM Code Art. 5** (Master's overriding authority).",
        "Company/DPA"),
    "hold": lambda p: ("Hold \u2014 no action taken.", "\u2014"),
}


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


_HEADER_IMAGE = ROOT.parent / "Data" / "Captain" / "CaptainProtocol" / "R2G-LUSV-Open-Sea-StormBanner.png"
if _HEADER_IMAGE.exists():
    st.image(str(_HEADER_IMAGE), width="stretch")

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

m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("Elapsed", f"{skeleton.sim.state.elapsed_s / 3600.0:.1f} h")
m2.metric("Distance", f"{skeleton.sim.state.distance_travelled_nm:.1f} / "
                      f"{skeleton.sim.total_route_distance_nm:.1f} nm")
m3.metric("Speed", f"{skeleton.sim.state.current_speed_kn:.1f} kn")
m4.metric("Fuel remaining", f"{skeleton.sim.state.fuel_tonnes:.1f} t")

plot_col, info_col = st.columns([3, 2])
with plot_col:
    st.plotly_chart(_mission_preview_figure(skeleton), use_container_width=True,
                    key=f"preview_{len(st.session_state.captain_history)}")

with info_col:
    # One de-duplicated block -- elapsed/distance/speed/fuel already live in the metric row
    # above, so only genuinely NEW facts appear here (no repeated mission_id/numbers, no
    # scrollable boxes -- everything visible at once).
    st.markdown(f"**{skeleton.state.mission_id}** \u00b7 {skeleton.order.vessel}")
    st.caption(skeleton.order.goal)
    eta_s = skeleton.sim.eta_s()
    eta_text = f"{eta_s / 3600.0:.1f} h" if eta_s != float("inf") else "n/a (stopped)"
    st.caption(f"ETA {eta_text}  \u2022  as of {skeleton.sim.current_utc().isoformat(timespec='minutes')}"
              f"  \u2022  reached destination: {skeleton.sim.reached_destination()}")
    hazards_text = ", ".join(h["type"] for h in skeleton.state.active_hazards
                             if "cleared_at_s" not in h) or "none"
    monitors_text = ", ".join(skeleton.active_monitors()) or "none"
    st.caption(f"Hazards: {hazards_text}  \u2022  Monitors: {monitors_text}")
    _GOAL_ICON = {"open": "\U0001F518", "met": "\u2705", "abandoned": "\u274C"}
    goals_text = " \u00b7 ".join(f"{_GOAL_ICON.get(g['status'], '\u2022')} {g['goal']} ({g['status']})"
                                for g in skeleton.state.goals)
    st.caption(goals_text)

    history_df = pd.DataFrame(st.session_state.captain_history).set_index("t_h")
    st.caption(f"Speed (kn) / Fuel (t) over time \u2014 {len(skeleton.state.event_log)} log entries below")
    sp_col, fu_col = st.columns(2)
    sp_col.line_chart(history_df[["speed_kn"]], height=CHART_HEIGHT)
    fu_col.line_chart(history_df[["fuel_t"]], height=CHART_HEIGHT)

st.markdown("#### Captain decisions \u2192 command to OOW")
st.caption("There is no text 'instruction' channel yet -- OOW is always the deterministic "
          "`app/oracle_planner.py` planner during Captain training/eval (never an LLM, "
          "Sec 13.C.15), and it only ever decides steering. A Captain decision instead "
          "mechanically sets the mission's own speed (`sim.state.current_speed_kn`), which "
          "the encounter-sim's own-ship inherits every micro-step -- the §3.3 'rendered "
          "constraint line' mechanism is for a future LLM-OOW, not built yet.")
events_by_id = {e.event_id: e for e in skeleton.order.events}
decisions = [d for d in skeleton.state.event_log if d["field_path"] == "captain_decision"]
responses_by_cause = {r["cause"]: r for r in skeleton.state.event_log if r["field_path"] == "captain_response"}
if decisions:
    for d in decisions:
        event = events_by_id.get(d["cause"])
        icon = _EVENT_ICON.get(event.type, "\U0001F4CC") if event else "\U0001F4CC"
        event_text = (_EVENT_NARRATIVE.get(event.type, lambda e: f"**{e.type}**")(event)
                     if event else f"Event `{d['cause']}`")
        tool, params = d["new"]["tool"], d["new"]["params"]
        narrative_fn = _DECISION_NARRATIVE.get(tool)
        decision_text, recipient = narrative_fn(params) if narrative_fn else (f"`{tool}` {params}", "?")
        with st.container(border=True):
            st.markdown(f"{icon} **t = {d['t'] / 3600.0:.1f} h** \u2014 {event_text}")
            st.markdown(f"\U0001F9ED **Captain's decision:** {decision_text}")
            badge = (" \u2022 \U0001F6E1\uFE0F *shield override -- the Captain's own proposal "
                    "breached a safety margin and was substituted*" if d["new"]["shield_substituted"] else "")
            st.caption(f"\U0001F4E1 Communicated to: {recipient}{badge}")

            response = responses_by_cause.get(d["cause"])
            if response is not None:
                plan = response["new"]["plan"]
                with st.expander("\U0001F4CB Plan & reasoning (computed, Sec 16.2/16.4)"):
                    st.markdown(f"**Reasoning:** {response['new']['reasoning']}")
                    st.markdown(f"**Goal achievability:** {plan['resource_note']}")
                    goals_line = " \u00b7 ".join(f"{g['goal']} ({g['status']})" for g in plan["updated_goals"])
                    st.caption(f"Updated goals: {goals_line}")

            if event is not None:
                memory_hits = captain_memory.retrieve(event.type)
                with st.expander(f"\U0001F9E0 Memory consulted ({len(memory_hits)}, Sec 16.1)"):
                    if memory_hits[0].is_placeholder:
                        st.caption("\u26A0\uFE0F Placeholder content -- real RAG/KG/PG not built yet "
                                  "(design_captain_missions.md \u00a716.6 step 1).")
                    for hit in memory_hits:
                        st.markdown(f"- **{hit.source}** (score {hit.score:.2f}): {hit.text}")
else:
    st.caption("No Captain decisions recorded yet.")

with st.expander("\U0001F6E0\uFE0F Tool overview (Sec 16.3/9.1 -- extension point for future tools)"):
    st.caption("The Captain decision layer selects from the 'wired' tools below today. The "
              "'future' rows are a deliberately deferred extension point (Sec 16.3) -- no "
              "inference loop or training format exists for them yet; add new entries to "
              "pipeline/captain_tools.py's TOOL_REGISTRY when one is actually built.")
    tools_df = pd.DataFrame([{"Tool": t.name, "Scope": t.scope, "Status": t.status,
                             "Description": t.description} for t in captain_tools.TOOL_REGISTRY])
    st.dataframe(tools_df, use_container_width=True, hide_index=True)

with st.expander("Event log (full, raw)"):
    if skeleton.state.event_log:
        log_df = pd.DataFrame(skeleton.state.event_log)
        for col in ("old", "new"):
            log_df[col] = log_df[col].apply(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
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
