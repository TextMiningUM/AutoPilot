"""Basic Simulator — OOW COLREG text-to-text simulator (Streamlit).

Run from the Basic Simulator/ folder:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import json
import math
import os
import sys
import textwrap
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
REPO_ROOT = ROOT.parent
for p in (ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st
import streamlit.components.v1 as components

from app.missions import Mission, list_mission_ids, load_mission
from app.simulation import Simulation, VesselConstraints, project_scenario, find_collision
from app.narrate import narrate, contact_line, bearing_and_range, relative_bearing, cpa_tcpa
from app.viz_plotly import trajectory_figure, trajectory_bounds, animated_trajectory_figure
from app.evaluation import score_trajectory

st.set_page_config(page_title="OOW COLREG Simulator", page_icon="\U0001F9ED", layout="wide")

# ── Cool-ish nautical theme ──────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #0b3d63 0%, #146c94 55%, #19a7ce 100%);
    padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.hero h1 { color: #f4fbff; margin: 0; font-size: 1.9rem; }
.hero p { color: #d6f0fb; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
.badge {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
    font-weight: 600; font-size: 0.85rem; margin-right: 0.4rem;
}
.badge-pass { background: #1e7d3220; color: #2fbf50; border: 1px solid #2fbf5060; }
.badge-fail { background: #a0202020; color: #ff5252; border: 1px solid #ff525260; }
.badge-quiet { background: #6c757d20; color: #adb5bd; border: 1px solid #adb5bd50; }
div[data-testid="stMetric"] {
    background: rgba(20,108,148,0.08); border-radius: 10px; padding: 0.6rem 0.8rem;
    border: 1px solid rgba(20,108,148,0.25);
}
/* CPA/TCPA values can list several contacts ("ts1: 120m • ts2: 340m • ..."), and some labels
   ("Goal bearing / dist") are long too -- both used to get clipped with an ellipsis at a
   fixed font-size/single line. Letting them WRAP (box grows taller as needed) is simpler and
   reads better than shrinking the font to force a fit -- a shrunk font on a multi-contact
   line gets hard to read, whereas a taller box costs nothing since these boxes already sit on
   their own row above the plot. The actual `nowrap` culprit turned out to be the inner <p>
   Streamlit renders the text in (not just its stMetricValue/stMetricLabel wrapper divs, which
   is all the first pass here targeted) -- must override that directly too or the wrapper's
   `white-space: normal` has no effect on its non-wrapping child. */
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] p,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    white-space: normal; overflow: visible; text-overflow: unset; overflow-wrap: break-word;
    line-height: 1.3;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.05rem; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] { font-size: 0.75rem; }
/* Cap the native `help=` hover tooltips -- they were wide/tall enough to spill over the
   plot next to the sidebar; keep them small even if some help text is still long. */
div[data-testid="stTooltipContent"] {
    max-width: 260px; font-size: 0.8rem; line-height: 1.25rem;
}
/* Tighten vertical spacing in the sidebar so more controls fit on one normal screen
   without scrolling -- Streamlit's default per-widget gap/margin is generous and adds
   up fast once you stack this many number_input/slider/button rows. */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}
section[data-testid="stSidebar"] div[data-testid="stElementContainer"] {
    margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
    font-size: 0.82rem; margin-bottom: 0.1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Splash screen (shown once per session, then dismissed) ────────────────
if "splash_dismissed" not in st.session_state:
    st.session_state.splash_dismissed = False

if not st.session_state.splash_dismissed:
    st.markdown("""
    <div class="hero" style="margin-top: 8vh; text-align: center;">
      <h1 style="font-size: 2.6rem;">\U0001F9ED Basic Simulator — OOW COLREG Agent</h1>
      <p style="font-size: 1.1rem;">Text-to-text collision-avoidance simulator · Imazu-style missions · base Qwen3-8B (RAG + CoT + Procedural Graph)</p>
    </div>
    """, unsafe_allow_html=True)

    # Load the retrieval index + Qwen3-8B onto the GPU here (once per server process,
    # cached via st.cache_resource) so later "Ask OOW agent" clicks are as fast as
    # possible instead of paying the load cost on the first real question.
    # NOT automatic anymore -- opt-in via the checkbox below. Rationale: browsing
    # missions, Scenario preview, and replaying precomputed LLM-driven runs are all pure
    # JSON/CPU work needing no model at all, but eagerly loading Qwen3-8B here used to grab
    # the whole 8GB GPU regardless -- fatal if a background sweep/training process is
    # already using it. Leaving this unchecked lets you browse/replay freely; "Ask OOW
    # agent" still works either way (it lazily loads the model on first live click if you
    # skipped preloading here).
    st.checkbox(
        "\U0001F9E0 Preload Agent now (loads Qwen3-8B onto the GPU -- only needed for "
        "live 'Ask OOW agent' calls; leave unchecked to just browse missions / replay "
        "precomputed Agent-driven runs without touching the GPU, e.g. while a background "
        "sweep is using it)",
        key="preload_llm",
    )
    if st.session_state.get("preload_llm") and not st.session_state.get("agent_ready", False):
        from app.agents import preload
        with st.status("Loading OOW agent (retrieval index + Qwen3-8B)...", expanded=True) as status:
            preload(status_cb=st.write)
            status.update(label="\u2705 OOW agent ready", state="complete", expanded=False)
        st.session_state.agent_ready = True
    elif st.session_state.get("agent_ready", False):
        st.success("\u2705 OOW agent ready (cached).")
    else:
        st.caption("Agent not loaded -- GPU free for other processes. \"Ask OOW agent\" "
                  "will load it on first use if you skip this.")

    _, center, _ = st.columns([2, 1, 2])
    with center:
        if st.button("\u2693 Enter simulator", type="primary", use_container_width=True):
            st.session_state.splash_dismissed = True
            st.rerun()
    st.stop()

# Available after the splash gate above (which imports app.agents and warms its caches
# on the very first run) -- re-importing here is a cheap sys.modules lookup, not a reload.
from app.agents import MODEL_CONFIGS, SYSTEM_OOW_AGENT, ask_oow
from app.llm_runs import list_runs_for_mission, load_run, checkpoint_at_or_before, list_run_sets, BASE_RUNS_DIR

# ── UI preference persistence ─────────────────────────────────────────────
# Ship-performance/Mission sidebar values persist across app restarts (new browser tab,
# new `streamlit run`, ...) via a small local JSON file -- Streamlit's session_state only
# lives for one browser session, so without this every value silently reset to its
# hardcoded widget default on every restart despite looking "saved" while the app ran.
_UI_PREFS_PATH = ROOT / "Data" / "_ui_prefs.json"
_UI_PREFS_KEYS = [
    "ship_max_speed", "ship_max_turn_rate_pct", "ship_max_accel", "ship_max_decel",
    "ship_turn_rate_deg_s", "mission_min_cpa",
]


def _load_ui_prefs() -> dict:
    if _UI_PREFS_PATH.exists():
        try:
            return json.loads(_UI_PREFS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_ui_prefs() -> None:
    prefs = {k: st.session_state[k] for k in _UI_PREFS_KEYS if k in st.session_state}
    _UI_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _UI_PREFS_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


# Seed session_state from the persisted file BEFORE any widget with these keys renders --
# Streamlit only honours a widget's `value=` default on the very first render of THIS
# session for that key, so this must run once per (browser) session, ahead of the sidebar.
if "_ui_prefs_loaded" not in st.session_state:
    for _k, _v in _load_ui_prefs().items():
        if _k in _UI_PREFS_KEYS:
            st.session_state[_k] = _v
    st.session_state._ui_prefs_loaded = True

# ── Session state ─────────────────────────────────────────────────────────
def build_vessel_constraints(mission: Mission) -> VesselConstraints:
    """Reads the sidebar's Ship performance/Mission widgets straight out of session_state
    (same one-render-lag pattern as wide_plot elsewhere in this file -- on the very first
    run these keys don't exist yet, hence the defaults, which match each widget's own
    `value=`) into one VesselConstraints -- the single source of truth the kinematics layer
    (app/simulation.py) and the OOW agent's prompt both read, instead of each hardcoding its
    own copy of these numbers. cruise_speed_mps always matches `mission`'s own designed
    speed (never a separately user-set value) -- the nominal/rated-speed reference the
    agent's prompt reports must reflect what this mission actually runs at, same reasoning
    as run_llm_scenario.py's CLI path."""
    g = st.session_state.get
    return VesselConstraints(
        max_speed_mps=g("ship_max_speed", 10.0),
        max_rudder_angle_deg=g("ship_max_turn_rate_pct", 30.0),
        max_acceleration_mps2=g("ship_max_accel", 0.2),
        max_deceleration_mps2=g("ship_max_decel", 0.2),
        turn_rate_deg_s=g("ship_turn_rate_deg_s", 3.0),
        cruise_speed_mps=mission.own_ship.speed,
        min_cpa_m=g("mission_min_cpa", 500.0),
        time_step_s=10.0,
    )


# ── Session state ─────────────────────────────────────────────────────────────────────────
mission_ids = list_mission_ids()
if "mission_id" not in st.session_state:
    st.session_state.mission_id = mission_ids[0]
if "sim" not in st.session_state or st.session_state.get("_loaded_mission_id") != st.session_state.mission_id:
    mission = load_mission(st.session_state.mission_id)
    st.session_state.sim = Simulation(mission, build_vessel_constraints(mission))
    st.session_state.mission = mission
    st.session_state._loaded_mission_id = st.session_state.mission_id
    st.session_state.last_decision = None
    st.session_state.last_debug = None
    st.session_state.llm_run_path = None
    st.session_state.llm_compliance_audit = None
    st.session_state.llm_compliance_checked_key = None
    st.session_state._agent_detail_cp_i = None

sim: Simulation = st.session_state.sim
# Refreshed every rerun (not just at mission load/reset) so a sidebar tweak -- e.g. testing
# a slower turn_rate_deg_s -- takes effect on the very next step without losing the ship's
# current position/heading/speed the way constructing a brand-new Simulation would.
sim.constraints = build_vessel_constraints(st.session_state.mission)
mission = st.session_state.mission


_ACTION_TEXT = {
    "turn_left": "Turn to \u2b05\ufe0f port",
    "turn_right": "Turn to \u27a1\ufe0f starboard",
    "hold_course": "\u2b1c Hold course and speed",
    "speed_up": "\u2b06\ufe0f Increase speed",
    "slow_down": "\u2b07\ufe0f Reduce speed",
    "stop": "\U0001F6D1 Stop (all-stop)",
}


def _describe_decision(decision: dict) -> str:
    """Human-readable rendering of a decision dict ({action, degrees, rule_applied,
    reasoning}) for end-user display -- replaces a raw st.json() dump with plain sentences:
    what the helm order actually is (or that nothing changes), which COLREG rule was cited,
    and the model's own reasoning."""
    action = decision.get("action", "hold_course")
    label = _ACTION_TEXT.get(action, action)
    if action in ("turn_left", "turn_right") and decision.get("degrees") is not None:
        label += f" by {float(decision['degrees']):.0f}\u00b0"
    rule = decision.get("rule_applied") or "none"
    lines = [f"**Helm order:** {label}", f"**Rule applied:** {rule}"]
    if decision.get("reasoning"):
        lines.append(f"**Reasoning:** {decision['reasoning']}")
    return "\n\n".join(lines)


def _short_decision(decision: dict) -> str:
    """Decision summary for the IN-PLOT annotation (see _build_decision_by_time): helm
    order + rule on one line, plus the reasoning wrapped onto multiple lines via manual
    `<br>` breaks (Plotly annotations don't auto-wrap text). Only the REASONING travels
    here, not the full situation report (contact list etc.) -- that stays too long/technical
    to read well even wrapped, and is still available via the side panel's Decision log."""
    action = decision.get("action", "hold_course")
    label = _ACTION_TEXT.get(action, action)
    if action in ("turn_left", "turn_right") and decision.get("degrees") is not None:
        label += f" {float(decision['degrees']):.0f}\u00b0"
    rule = decision.get("rule_applied") or "none"
    header = f"<b>{label} \u2022 Rule {rule}</b>"
    reasoning = decision.get("reasoning")
    if not reasoning:
        return header
    wrapped = "<br>".join(textwrap.wrap(reasoning, width=42))
    return f"{header}<br>{wrapped}"


def _describe_params(params: dict) -> str:
    """Human-readable rendering of a precomputed run's `params` block (decision_interval,
    dt, max_steps, enable_thinking, max_new_tokens, k, use_rag, ...) instead of raw JSON."""
    lines = [
        f"**Decisions:** every {params.get('decision_interval', '?')} step(s), "
        f"dt={params.get('dt', '?')}s, max {params.get('max_steps', '?')} steps total",
        f"**Thinking:** {'on' if params.get('enable_thinking') else 'off'}  \u2022  "
        f"**Max new tokens:** {params.get('max_new_tokens', '?')}",
    ]
    if params.get("use_rag"):
        lines.append(f"**RAG context:** on (k={params.get('k', '?')} chunks)")
    else:
        lines.append("**RAG context:** off")
    lines.append("**System prompt:** " +
                ("custom (edited)" if params.get("system_prompt_is_custom") else "default"))
    return "\n\n".join(lines)


def _goal_quickfacts(x: float, y: float, heading: float, speed: float,
                     goal_xy: tuple[float, float]) -> str:
    """One-line summary of bearing/heading-to-goal + ETA, pulled out and shown prominently
    ABOVE the full situation report -- these numbers (added so the model can steer back onto
    a direct course after avoiding a target) were easy to miss buried inside the wall of text."""
    gx, gy = goal_xy
    brg, rng = bearing_and_range(x, y, gx, gy)
    off = relative_bearing(heading, brg)
    side = "starboard" if off > 0 else "port"
    eta = f"\u2248{rng / speed:.0f}s" if speed > 0 else "never (stopped)"
    return (f"\U0001F3AF Goal bearing **{brg:.0f}\u00b0** ({abs(off):.0f}\u00b0 to **{side}** of "
           f"current heading {heading:.0f}\u00b0) \u2022 {rng:.0f}m away \u2022 ETA {eta}")


def _build_metrics_by_time(trajectory: list[dict], mission) -> dict[float, str]:
    """Precomputes the goal-bearing/heading-speed/CPA/TCPA annotation text for EVERY frame
    time in a trajectory, for embedding directly in the native Plotly animation (see
    animated_trajectory_figure's `metrics_by_time`). The native Play button/slider never
    talks back to Streamlit, so a server-rendered metrics panel has no way to know which
    frame the user is currently looking at -- putting the text IN the plot is the only way
    to keep it in sync while scrubbing."""
    times = sorted({r["time"] for r in trajectory})
    out: dict[float, str] = {}
    for t in times:
        own_row = next((r for r in trajectory if r["time"] == t and r["vehicle"] == "own_ship"), None)
        if not own_row:
            continue
        targets = [r for r in trajectory if r["time"] == t and r["vehicle"] != "own_ship"]
        brg, rng = bearing_and_range(own_row["x"], own_row["y"], mission.goal[0], mission.goal[1])
        lines = [f"\U0001F3AF Goal {brg:.0f}\u00b0 \u2022 {rng:.0f}m",
                f"Heading {own_row['heading']:.0f}\u00b0 \u2022 {own_row['speed']:.2f} m/s"]
        for tgt in targets:
            cpa_m, tcpa_s = cpa_tcpa(own_row["x"], own_row["y"], own_row["heading"], own_row["speed"],
                                    tgt["x"], tgt["y"], tgt["heading"], tgt["speed"])
            lines.append(f"{tgt['vehicle']}: CPA {cpa_m:.0f}m \u2022 TCPA {tcpa_s:.0f}s")
        out[t] = "<br>".join(lines)
    return out


def _build_decision_by_time(run_log: dict, times: list[float]) -> dict[float, str]:
    """Precomputes a SHORT one-line decision summary (held constant between checkpoints, via
    checkpoint_at_or_before) for every frame time -- same in-plot-annotation reasoning as
    _build_metrics_by_time. The FULL situation report/reasoning stays in the side panel's
    independent Decision log selector instead of trying to fit it in the plot too."""
    out: dict[float, str] = {}
    for t in times:
        cp = checkpoint_at_or_before(run_log, t)
        out[t] = _short_decision(cp.get("decision", {})) if cp else "No decision yet"
    return out


def _render_plot(trajectory: list[dict], mission, placeholder, title: str,
                 bounds_from: list[dict] | None = None, freeze_at_collision: bool = True) -> None:
    """Plain trajectory render: shows everything recorded so far (or the full precomputed
    path for the no-avoidance preview). `freeze_at_collision` (default True) truncates the
    render to `t <= collision_time` and marks it -- meaningful for a real (Manual helm/LLM
    driven) run that actually stops there. The no-avoidance Scenario preview passes
    `freeze_at_collision=False` instead: it's a hypothetical illustration of the raw
    encounter geometry, so cutting it off at the collision point hid the rest of the
    intended path -- now it shows the full path with the collision point just marked.
    Called directly (no @st.fragment) so it can also be invoked repeatedly, mid-loop, by
    the sidebar's 'Full run' button for a live animated readout (see below)."""
    if not trajectory:
        placeholder.info("Nothing recorded yet.")
        return
    times = sorted({row["time"] for row in trajectory})
    collision = find_collision(trajectory)
    render_traj, render_times = trajectory, times
    if collision is not None and freeze_at_collision:
        render_traj = [row for row in trajectory if row["time"] <= collision["time"]]
        render_times = [t for t in times if t <= collision["time"]]
    x_range, y_range = trajectory_bounds(bounds_from or trajectory, mission)
    placeholder.plotly_chart(
        trajectory_figure(render_traj, mission, title=title, x_range=x_range, y_range=y_range,
                          current_time=render_times[-1] if render_times else None,
                          total_time=times[-1] if times else None, collision=collision),
        # Streamlit requires a unique key per element within a single script run (not just
        # across reruns) -- the Full-run loop below calls this repeatedly in one run, so the
        # key must vary per call. len(trajectory) grows every step, which gives uniqueness
        # "for free" without needing a separate counter.
        use_container_width=True, key=f"chart_render_plot_{len(trajectory)}",
    )
    if collision is not None:
        st.error(
            f"\U0001F4A5 Collision with **{collision['vehicle']}** at t={collision['time']:.0f}s "
            f"(range {collision['range_m']:.0f}m)."
        )


def _animate_preview(trajectory: list[dict], mission, placeholder, title: str,
                     speed_s: float = 0.3, max_frames: int = 80,
                     metrics_by_time: dict[float, str] | None = None,
                     decision_by_time: dict[float, str] | None = None) -> None:
    """Renders ONE native Plotly frames-animation (Play/Pause button + slider) -- this is now
    the ONLY rendering path for Scenario Preview and Play Agent Mission (no more separate
    "static frame + Step button" mode): the native animation plays/scrubs entirely
    client-side with zero Streamlit round-trips, which is strictly better than anything a
    rerun-driven approach can achieve (see repo memory for the MutationObserver-measured
    churn of the rerun-based versions this replaced). `metrics_by_time`/`decision_by_time`
    (see _build_metrics_by_time/_build_decision_by_time) get baked into the figure itself so
    those numbers/decision summary stay in sync with wherever the user scrubs to -- a
    server-side panel has no way to know the native slider's current position."""
    if not trajectory:
        placeholder.info("Nothing recorded yet.")
        return
    times = sorted({row["time"] for row in trajectory})
    collision = find_collision(trajectory)
    x_range, y_range = trajectory_bounds(trajectory, mission)
    render_every = max(1, len(times) // max_frames)
    frame_times = [t for i, t in enumerate(times)
                  if i % render_every == 0 or i == len(times) - 1]
    placeholder.plotly_chart(
        animated_trajectory_figure(trajectory, mission, title=title, x_range=x_range, y_range=y_range,
                                   frame_times=frame_times, collision=collision,
                                   frame_duration_ms=max(int(speed_s * 1000), 60),
                                   metrics_by_time=metrics_by_time, decision_by_time=decision_by_time),
        # Called at most once per mode per rerun -- a stable key is fine.
        use_container_width=True, key="chart_animate_preview",
    )
    if collision is not None:
        st.error(
            f"\U0001F4A5 Collision with **{collision['vehicle']}** at t={collision['time']:.0f}s "
            f"(range {collision['range_m']:.0f}m)."
        )


def _render_metrics_row(placeholders, mission, metrics_row, targets_now) -> None:
    """Fills the 4 goal-bearing/heading/CPA/TCPA metric boxes -- used for Manual helm/Agent
    Real-Time only (Scenario Preview/Play Agent Mission show the same numbers baked directly
    into the plot's own animation frames instead, see _build_metrics_by_time)."""
    b1, b2, b3, b4 = placeholders
    if metrics_row:
        _t, _x, _y, _hdg, _spd = metrics_row
        _goal_brg, _goal_rng = bearing_and_range(_x, _y, mission.goal[0], mission.goal[1])
        b1.metric("Goal bearing / dist", f"{_goal_brg:.0f}\u00b0 / {_goal_rng:.0f}m")
        b2.metric("Our heading / speed", f"{_hdg:.0f}\u00b0 / {_spd:.2f} m/s")
        if targets_now:
            cpas, tcpas = [], []
            for tgt in targets_now:
                cpa_m, tcpa_s = cpa_tcpa(_x, _y, _hdg, _spd, tgt["x"], tgt["y"],
                                        tgt["heading"], tgt["speed"])
                name = tgt.get("vehicle", "?")
                cpas.append(f"{name}: {cpa_m:.0f}m")
                tcpas.append(f"{name}: {tcpa_s:.0f}s")
            b3.metric("CPA", " \u2022 ".join(cpas))
            b4.metric("TCPA", " \u2022 ".join(tcpas))
        else:
            b3.metric("CPA", "no contacts")
            b4.metric("TCPA", "no contacts")
    else:
        for _b in placeholders:
            _b.metric("\u2014", "n/a")


def _render_agent_detail(ph, mission) -> None:
    """Fills the Play Agent Mission "inspect a moment" panel from `_agent_detail_cp_i` (set
    by the "Show details" button near the plot). Always called, every run, regardless of
    whether a pick has been made yet -- an st.empty() placeholder shows NOTHING for a run
    that doesn't write to it, so writing directly from inside the button's own click handler
    would only survive that ONE rerun and then go blank on the next unrelated one."""
    run_path = st.session_state.get("llm_run_path")
    cp_i = st.session_state.get("_agent_detail_cp_i")
    if run_path is None or cp_i is None:
        ph.caption("Drag the slider or click Play in the plot, then use \"Show details\" "
                  "below the plot to see the full situation report/reasoning for a specific "
                  "moment.")
        return
    run_log = load_run(run_path)
    checkpoints = run_log.get("checkpoints") or []
    if cp_i >= len(checkpoints):
        ph.caption("Drag the slider or click Play in the plot, then use \"Show details\" "
                  "below the plot to see the full situation report/reasoning for a specific "
                  "moment.")
        return
    cp = checkpoints[cp_i]
    with ph.container():
        own_row = next((r for r in run_log["trajectory"]
                       if r["time"] == cp["time"] and r["vehicle"] == "own_ship"), None)
        if own_row:
            st.caption(_goal_quickfacts(own_row["x"], own_row["y"], own_row["heading"],
                                        own_row["speed"], mission.goal))
        st.markdown("**Situation report (passed to the Agent)**")
        st.code(cp.get("situation_report", ""), language=None, wrap_lines=True)
        st.markdown("**Agent decision**")
        st.markdown(_describe_decision(cp.get("decision", {})))


# Plot header/mode-radio/placeholder created here (before the sidebar) so the sidebar's
# "Full run" button below can render live progress directly into `chart` as it steps --
# st.columns() containers are positional, not order-dependent, so this still renders in
# the correct (center) column regardless of running before the sidebar in script order.
# The Agent panel always stays beside the plot (never moves below -- two different layout
# experiences was explicitly rejected) with a fixed gray background so its boundary with the
# plot is visible; its width cycles through a few presets via the button at its own top,
# same idea as a resizable panel but without needing a fragile JS drag-handle hack (Streamlit
# has no native mouse-drag column resize).
_PANEL_WIDTH_PRESETS = {"Narrow": [4, 1], "Normal": [3, 2], "Wide": [1, 1]}
_panel_width = st.session_state.get("agent_panel_width", "Normal")
if _panel_width not in _PANEL_WIDTH_PRESETS:
    _panel_width = "Normal"
plot_col, side_col = st.columns(_PANEL_WIDTH_PRESETS[_panel_width], gap="medium")
with side_col:
    side_panel = st.container(key="agent_panel")
st.markdown(
    "<style>.st-key-agent_panel {background-color: rgba(128, 128, 128, 0.12); "
    "border-radius: 0.5rem; padding: 0.75rem;}</style>",
    unsafe_allow_html=True,
)
with side_panel:
    # Header + placeholders created early (before the sidebar/plot) -- `agent_static_ph` holds
    # the Model/config/params info (only changes when switching runs), `agent_detail_ph` holds
    # the full situation report/reasoning for whichever moment the "Show details" button (near
    # the plot) was last used to inspect -- see the Play Agent Mission section in plot_col.
    hdr_col, width_col = st.columns([4, 1])
    hdr_col.markdown("#### \U0001F916 Agent")
    if width_col.button("\u2194\ufe0f", key="cycle_agent_panel_width",
                        help=f"Panel width: {_panel_width} -- click to cycle"):
        names = list(_PANEL_WIDTH_PRESETS)
        st.session_state.agent_panel_width = names[(names.index(_panel_width) + 1) % len(names)]
        st.rerun()
    agent_static_ph = st.empty()
    agent_detail_ph = st.empty()

with plot_col:
    st.caption(f"\U0001F4CB **{mission.name}** ({mission.id}) -- full briefing in the sidebar under Mission.")
    view = st.radio(
        "Mode", options=["Scenario preview (no avoidance)", "Manual helm",
                        "Play Agent Mission", "Agent Real-Time"],
        horizontal=True, label_visibility="collapsed", key="sim_view_mode",
        help="Preview: no-avoidance path. Manual helm: steer it live. Play Agent Mission: "
             "replay a precomputed run. Agent Real-Time: live agent calls.",
    )
    _dt_now = 10.0
    _speed_s = (1500 - (6 - 1) * (1500 - 60) / 9) / 1000  # fixed mid-range animation speed

    # Scenario Preview and Play Agent Mission show goal-bearing/CPA/TCPA (and, for Play Agent
    # Mission, a short decision summary) baked directly into the plot's own native animation
    # instead of a separate metrics row -- there is no single "current frame" outside the
    # plot to compute a row FOR while the user is free-scrubbing its native slider. Manual
    # helm/Agent Real-Time have one live `sim` state, so they keep the row as before.
    if view not in ("Scenario preview (no avoidance)", "Play Agent Mission"):
        b_cols = st.columns(4)
        metric_phs = [c.empty() for c in b_cols]
        metrics_row = (sim.t, sim.own.x, sim.own.y, sim.own.heading, sim.own.speed)
        targets_now = [{"vehicle": v.name, "x": v.x, "y": v.y, "heading": v.heading, "speed": v.speed}
                       for v in sim.targets]
        _render_metrics_row(metric_phs, mission, metrics_row, targets_now)

    chart = st.empty()
    if view == "Scenario preview (no avoidance)":
        preview_traj = project_scenario(mission, dt=_dt_now)
        metrics_by_time = _build_metrics_by_time(preview_traj, mission)
        _animate_preview(preview_traj, mission, chart, "Scenario preview (no avoidance)",
                         speed_s=_speed_s, metrics_by_time=metrics_by_time)
        st.caption("Click **Play** in the chart above (or drag its slider) to move through it.")
    elif view == "Play Agent Mission":
        run_sets = list_run_sets()
        if run_sets:
            picked_run_set = st.selectbox(
                "Run set (folder)", options=run_sets,
                index=run_sets.index(st.session_state.get("llm_run_set", run_sets[0]))
                      if st.session_state.get("llm_run_set", run_sets[0]) in run_sets else 0,
                key="llm_run_set_picker",
                help="Which Data/missions/ subfolder of precomputed runs to browse -- e.g. an "
                     "archived sweep batch (\"MIssions Data V1\", \"Mission No Speed Increase\") "
                     "vs. the live _llm_runs/ folder that new runs are written to by default.",
            )
            if picked_run_set != st.session_state.get("llm_run_set"):
                st.session_state.llm_run_set = picked_run_set
                st.rerun()
            runs_dir = BASE_RUNS_DIR / picked_run_set
        else:
            picked_run_set = None
            runs_dir = BASE_RUNS_DIR / "_llm_runs"
            st.caption("\u26a0\ufe0f No run-log folders found under Data/missions/.")
        available_runs = list_runs_for_mission(mission.id, runs_dir)
        if not available_runs:
            chart.info(
                f"No precomputed Agent run found for **{mission.id}**. Live per-step model calls "
                "were too slow to play interactively, so this mode only plays back runs "
                "computed upfront. Generate one (outside Streamlit):\n\n"
                f"`python -m app.run_llm_scenario --missions {mission.id} --configs v3_rag_cot`"
            )
        else:
            run_labels = {i: f"{r['config']} / tag={r['tag']} -- {r['outcome'].get('verdict', '?')}"
                         for i, r in enumerate(available_runs)}
            picked_run_i = st.selectbox(
                "Precomputed run", options=list(run_labels), format_func=lambda i: run_labels[i],
                key=f"llm_run_picked_i__{picked_run_set}__{mission.id}",
                help="Generated by app/run_llm_scenario.py -- asks the agent every N steps "
                     "(stored per-run) and saves the full trajectory + every situation "
                     "report/recommendation, so playback here never calls the model.",
            )
            picked_run_path = str(available_runs[picked_run_i]["path"])
            if st.session_state.get("llm_run_path") != picked_run_path:
                st.session_state.llm_run_path = picked_run_path
                st.session_state._agent_detail_cp_i = None
            run_log = load_run(picked_run_path)
            run_traj = run_log["trajectory"]
            run_times = sorted({r["time"] for r in run_traj})
            metrics_by_time = _build_metrics_by_time(run_traj, mission)
            decision_by_time = _build_decision_by_time(run_log, run_times)
            _animate_preview(run_traj, mission, chart,
                             f"Agent run ({run_log['config']}/{run_log['tag']})",
                             speed_s=_speed_s, metrics_by_time=metrics_by_time,
                             decision_by_time=decision_by_time)

            # "Show details" -- near the plot (right-aligned, roughly under its bottom-right
            # corner), for inspecting the FULL situation report/reasoning of a specific
            # checkpoint. Can't read the native slider's live position from Python (no
            # bidirectional link without a custom component -- see repo memory), so this is a
            # deliberate separate pick instead of trying to mirror the plot automatically.
            # Just records the pick in session_state -- `_render_agent_detail` (called from
            # side_panel, EVERY run) is what actually renders it, so agent_detail_ph never goes
            # blank on an unrelated rerun (an st.empty() placeholder shows NOTHING for any run
            # that doesn't write to it -- writing directly here would only survive the ONE
            # rerun triggered by this click).
            checkpoints = run_log.get("checkpoints") or []
            if checkpoints:
                _, detail_col = st.columns([2, 1])
                with detail_col:
                    cp_labels = {i: f"t={cp['time']:.0f}s (step {cp['step']})"
                                for i, cp in enumerate(checkpoints)}
                    picked_cp_i = st.selectbox(
                        "Inspect moment", options=list(cp_labels), format_func=lambda i: cp_labels[i],
                        key="pam_inspect_cp_i", label_visibility="collapsed",
                    )
                    if st.button("\U0001F50E Show details in Agent panel", use_container_width=True,
                                key="pam_show_details"):
                        st.session_state._agent_detail_cp_i = picked_cp_i
    else:
        _render_plot(sim.trajectory, mission, chart, "",
                    bounds_from=project_scenario(mission, dt=_dt_now))

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Ship performance")
    st.caption("Feeds the kinematics layer (VesselConstraints) that rate-limits own-ship's "
              "heading/speed changes -- applied over a fixed 10s simulation time step.")
    sp1, sp2 = st.columns(2)
    sp1.number_input("Max speed (m/s)", min_value=0.0, value=10.0, step=0.5, key="ship_max_speed")
    sp2.number_input("Max rudder angle (deg)", min_value=0.0, value=30.0, step=1.0, key="ship_max_turn_rate_pct",
                     help="Informational only -- not yet enforced as a heading-change limit. "
                          "turn_rate_deg_s below is the one actual limit on how fast heading "
                          "can change per step.")
    sp3, sp4 = st.columns(2)
    sp3.number_input("Max acceleration (m/s\u00b2)", min_value=0.0, value=0.2, step=0.05, key="ship_max_accel")
    sp4.number_input("Max deceleration (m/s\u00b2)", min_value=0.0, value=0.2, step=0.05, key="ship_max_decel")
    st.number_input("Turn rate (deg/s)", min_value=0.0, value=3.0, step=0.5, key="ship_turn_rate_deg_s",
                    help="The only actively-enforced limit on how many degrees own-ship's "
                         "heading may change per simulation step.")
    st.divider()

    st.header("Mission")
    st.number_input("Minimal CPA (m)", min_value=0.0, value=500.0, step=50.0, key="mission_min_cpa")
    all_missions = {mid: load_mission(mid) for mid in mission_ids}
    labels = {mid: f"{mid.split('_')[0].upper()} \u2014 {m.name}  ({mid})" for mid, m in all_missions.items()}
    picked = st.selectbox("Scenario", options=mission_ids, format_func=lambda m: labels[m],
                          index=mission_ids.index(st.session_state.mission_id))
    if picked != st.session_state.mission_id:
        st.session_state.mission_id = picked
        st.rerun()

    with st.expander("\U0001F4CB Mission content", expanded=False):
        st.markdown(mission.as_text())

    if st.button("\U0001F504 Reset mission playback", use_container_width=True):
        st.session_state.sim = Simulation(mission, build_vessel_constraints(mission))
        st.session_state.last_decision = None
        st.session_state.last_debug = None
        st.session_state.llm_compliance_audit = None
        st.session_state.llm_compliance_checked_key = None
        st.rerun()

    st.divider()
    dt = 10.0  # fixed simulation time step (s) -- no longer a user-adjustable slider
    sim_mode = st.session_state.get("sim_view_mode", "Manual helm")
    is_preview = sim_mode == "Scenario preview (no avoidance)"
    is_llm_playback = sim_mode == "Play Agent Mission"
    is_frame_scrub = is_preview or is_llm_playback

    if not is_frame_scrub:
        # Scenario Preview/Play Agent Mission no longer need Step/Auto Run controls here --
        # both now play/scrub natively via the Play button + slider embedded in the plot
        # itself. Manual helm/Agent Real-Time still need these since `sim.step()` must stay
        # visible to the Evaluation panel etc. further down the page.
        sb_col, sf_col = st.columns(2)
        if sb_col.button("\u2b05\ufe0f Step back", use_container_width=True, disabled=True):
            pass
        if sf_col.button("\u27a1\ufe0f Step forward", use_container_width=True):
            sim.step(dt)
            st.rerun()

        rs_col1, rs_col2, rs_col3 = st.columns(3)
        n_auto = rs_col2.number_input("Steps", 1, 100, 10, step=1, label_visibility="collapsed")
        if rs_col1.button("\u23ea Steps", use_container_width=True, disabled=True):
            pass
        if rs_col3.button("Steps \u23e9", use_container_width=True):
            for _ in range(int(n_auto)):
                if sim.reached_goal():
                    break
                sim.step(dt)
            st.rerun()

        if st.button("\U0001F680 Auto Run", use_container_width=True):
            # Distance-aware step cap: a fixed cap (e.g. 200) silently cut runs short whenever
            # dt was small and/or the goal was far, which is why "Full run" previously stopped
            # before finishing longer scenarios. Sized from the actual remaining distance/speed.
            straight_dist = math.hypot(mission.goal[0] - sim.own.x, mission.goal[1] - sim.own.y)
            est_speed = max(sim.own.speed, 0.1)
            needed_steps = int((straight_dist / est_speed * 1.5) / dt) + 10
            bounds_from = project_scenario(mission, dt=dt)
            # Play it like a game: redraw the plot as it steps instead of jumping straight
            # to the final frame. Redraws are throttled to ~60 frames total so a long run
            # (many steps at a small dt) doesn't spend most of its time re-rendering Plotly.
            max_steps = min(1200, max(20, needed_steps))
            render_every = max(1, max_steps // 60)
            for i in range(max_steps):
                if sim.reached_goal():
                    break
                sim.step(dt)
                if i % render_every == 0:
                    _render_plot(sim.trajectory, mission, chart, "", bounds_from=bounds_from)
                    time.sleep(0.05)
            _render_plot(sim.trajectory, mission, chart, "", bounds_from=bounds_from)
            st.rerun()
    else:
        st.caption("\u25b6 Use the Play button / slider directly in the plot to move through it.")

    st.divider()
    st.header("Manual helm")
    _max_rudder = sim.constraints.max_rudder_angle_deg
    degrees = st.slider("Turn amount (deg)", 1.0, _max_rudder, min(15.0, _max_rudder), step=1.0)
    st.caption(
        "Each control also advances the simulation by one time step -- steering used to just "
        "change heading/speed without moving, which is why it felt like every click 'stopped' "
        "things. Arrow keys work too (\u2191\u2193\u2190\u2192, S to stop)."
    )
    # Arrow-key / D-pad layout: Speed up on top, Port/Stop/Starboard in the middle row,
    # Slow down on the bottom -- matches the physical arrow-key mental model requested.
    _, up_col, _ = st.columns(3)
    if up_col.button("\u2B06\uFE0F", use_container_width=True):
        sim.speed_up(); sim.step(dt); st.rerun()
    left_col, mid_col, right_col = st.columns(3)
    if left_col.button("\u2B05\uFE0F", use_container_width=True):
        sim.turn_left(degrees); sim.step(dt); st.rerun()
    if mid_col.button("\u23F9\uFE0F", use_container_width=True):
        sim.stop_vessel(); sim.step(dt); st.rerun()
    if right_col.button("\u27A1\uFE0F", use_container_width=True):
        sim.turn_right(degrees); sim.step(dt); st.rerun()
    _, down_col, _ = st.columns(3)
    if down_col.button("\u2B07\uFE0F", use_container_width=True):
        sim.slow_down(); sim.step(dt); st.rerun()

    # Arrow-key bridge: this iframe's JS reaches into the PARENT document (same-origin,
    # since components.html is served from the same Streamlit server) and .click()s the
    # D-pad button matching the key pressed -- so keyboard steering triggers the exact same
    # Python callback as a mouse click (including the step-forward fix above). Guarded by a
    # flag on window.parent so re-injecting this on every rerun doesn't stack up listeners.
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            function clickByText(text) {
                const buttons = doc.querySelectorAll('button');
                for (const b of buttons) {
                    if (b.innerText.trim() === text) { b.click(); return true; }
                }
                return false;
            }
            if (window.parent.__helmKeyListenerAttached) return;
            window.parent.__helmKeyListenerAttached = true;
            doc.addEventListener('keydown', function(e) {
                const map = {
                    'ArrowUp': '\u2B06\uFE0F', 'ArrowDown': '\u2B07\uFE0F',
                    'ArrowLeft': '\u2B05\uFE0F', 'ArrowRight': '\u27A1\uFE0F',
                    's': '\u23F9\uFE0F', 'S': '\u23F9\uFE0F',
                };
                const label = map[e.key];
                if (label && clickByText(label)) {
                    e.preventDefault();
                }
            });
        })();
        </script>
        """,
        height=0,
    )

    st.divider()
    st.caption(
        "Model: base **Qwen3-8B**, 4-bit NF4. 8 selectable RAG/CoT/Procedural-Graph prompt "
        "configs (pick one in the Agent panel). The fine-tuned OOW-QWEN checkpoint isn't "
        "trained yet -- see notebook \u00a713."
    )

    with st.popover("\U0001F4DD System prompt", use_container_width=True):
        st.caption(
            "Edited here, this replaces the system prompt used by every config EXCEPT "
            "**bare_qwen** (which always keeps its own separate, deliberately minimal "
            "prompt as the true baseline)."
        )
        edited_prompt = st.text_area(
            "System prompt", value=st.session_state.get("custom_system_prompt") or SYSTEM_OOW_AGENT,
            height=280, key="system_prompt_editor",
        )
        pc1, pc2 = st.columns(2)
        if pc1.button("\U0001F4BE Save", use_container_width=True):
            st.session_state.custom_system_prompt = edited_prompt
            st.success("Saved.")
        if pc2.button("\u21A9\uFE0F Reset to default", use_container_width=True):
            st.session_state.custom_system_prompt = None
            st.rerun()

    st.divider()
    if st.button("\U0001F6D1 Shut down & free GPU", use_container_width=True):
        from app.agents import unload
        with st.spinner("Freeing GPU memory and shutting down..."):
            unload()
        st.success("\u2705 GPU freed. Server shutting down -- you can close this tab now.")
        os._exit(0)

# ── Single-page, ergonomic layout: plot centered, agent + evaluation stacked on the right ──
# (mission caption + the 5-metric row now render at the top of plot_col, above the plot itself)

with side_panel:
    if sim_mode == "Agent Real-Time":
        # The original live/interactive agent panel -- kept exactly as-is, now scoped to
        # this one mode instead of always showing regardless of which mode was picked.
        with st.expander("Situation report", expanded=False):
            st.caption(_goal_quickfacts(sim.own.x, sim.own.y, sim.own.heading, sim.own.speed,
                                       mission.goal))
            st.code(narrate(mission, sim.own, sim.targets), language=None, wrap_lines=True)

        model_config = st.selectbox(
            "Model / prompt config", options=list(MODEL_CONFIGS),
            format_func=lambda k: MODEL_CONFIGS[k],
            index=list(MODEL_CONFIGS).index("v3_rag_cot"), key="model_config_select",
            help="Same 8 configs as the notebook's \u00a711 prompt ablation study: a bare-Qwen "
                 "baseline plus v0_base..v6_pg_scenario.",
        )

        gc1, gc2 = st.columns(2)
        enable_thinking = gc1.toggle(
            "\U0001F9E0 Thinking", value=False, key="thinking_toggle",
            help="Qwen3's native hidden <think>...</think> reasoning channel. Off by default -- "
                 "it's the single biggest latency cost and isn't needed for a JSON-only answer. "
                 "Turn on to see/allow visible step-by-step reasoning (slower).",
        )
        max_new_tokens = gc2.number_input(
            "Max new tokens", min_value=64, max_value=1024, value=256, step=32, key="max_new_tokens_input",
            help="Hard cap on generated response length. Higher = slower but more room for "
                 "reasoning text before the JSON (relevant mainly with Thinking on).",
        )
        rag_k = st.slider(
            "RAG chunks (k)", min_value=1, max_value=6, value=4, key="rag_k_slider",
            help="Only affects v1_rag/v3_rag_cot. Each retrieved COLREG excerpt adds ~500 tokens "
                 "to the PROMPT (not the response) -- this is why those two configs are slower "
                 "than bare_qwen/v0_base even with generation length unchanged. Lower = faster "
                 "prefill, less context; 6 is the ablation study's default.",
        )
        use_rag = st.toggle(
            "\U0001F4DA Use RAG context", value=True, key="use_rag_toggle",
            help="Only affects v1_rag/v3_rag_cot. Off forces k=0 (no retrieved excerpts injected) "
                 "-- use this for a quick apples-to-apples speed comparison against bare_qwen/v0_base.",
        )
        effective_k = int(rag_k) if use_rag else 0

        if model_config in ("v1_rag", "v3_rag_cot"):
            from app.agents import rag_context_preview
            preview = rag_context_preview(mission, sim.own, sim.targets, k=effective_k)
            if preview["chars"]:
                st.caption(
                    f"\U0001F4CF Context preview: ~{preview['chars']:,} chars "
                    f"(~{preview['chars'] // 4:,} tokens est.) from {preview['chunks']} chunk(s)"
                )
            else:
                st.caption("\U0001F4CF Context preview: RAG off -- no excerpts will be injected.")

        if st.button("\U0001F9E0 Ask OOW agent", type="primary", use_container_width=True):
            with st.spinner(f"Retrieving context + generating ({MODEL_CONFIGS[model_config]})..."):
                decision, debug = ask_oow(mission, sim.own, sim.targets, config=model_config,
                                          system_prompt=st.session_state.get("custom_system_prompt"),
                                          max_new_tokens=int(max_new_tokens), enable_thinking=enable_thinking,
                                          k=effective_k, constraints=sim.constraints)
            st.session_state.last_decision = decision
            st.session_state.last_debug = debug

        decision = st.session_state.last_decision
        debug = st.session_state.last_debug
        if decision:
            st.markdown("**Recommendation**")
            st.markdown(_describe_decision(decision))
            b1, b2 = st.columns(2)
            if b1.button("\u2705 Apply", use_container_width=True):
                sim.apply_action(decision)
                sim.step(dt)
                st.rerun()
            b2.button("\U0001F6AB Ignore", use_container_width=True)

            with st.expander("Retrieval / grounding detail"):
                st.write("**Config used:**", MODEL_CONFIGS.get(debug.get("config"), debug.get("config")))
                st.write("**Prompt sent (user turn):**", f"{debug.get('user_msg_chars', 0):,} chars "
                        f"(~{debug.get('user_msg_chars', 0) // 4:,} tokens est.)")
                st.write("**Retrieved COLREG chunk IDs:**", debug.get("retrieved_chunk_ids"))
                st.write("**Query concepts:**", debug.get("query_concepts"))
                st.write("**Expanded concepts:**", debug.get("expanded_concepts"))
                if debug.get("pg_guidance"):
                    st.markdown("**Procedural-graph guidance**")
                    st.code(debug["pg_guidance"], language=None, wrap_lines=True)
                if debug.get("user_msg"):
                    st.markdown("**Full user turn sent to model** (constraints, situation, "
                               "RAG/PG context -- everything except the system prompt above)")
                    st.code(debug["user_msg"], language=None, wrap_lines=True)
                st.markdown("**Raw model output**")
                st.code(debug.get("raw_response", ""), language=None, wrap_lines=True)
        else:
            st.caption("Click \"Ask OOW agent\" for a recommended manoeuvre.")

    elif sim_mode == "Play Agent Mission":
        # Model/config/params (static, only changes when switching runs) in agent_static_ph.
        # The full situation report/reasoning for a specific moment lives in agent_detail_ph
        # instead, driven by `_agent_detail_cp_i` (set by the "Show details" button next to
        # the plot) -- there is no live "current frame" to follow here since the plot's own
        # Play button/slider never reports its position back to Streamlit.
        run_path = st.session_state.get("llm_run_path")
        if not run_path:
            agent_static_ph.caption("Pick a precomputed run above the plot to see its details here.")
        else:
            run_log = load_run(run_path)
            with agent_static_ph.container():
                st.write("**Model / config:**",
                        MODEL_CONFIGS.get(run_log["config"], run_log["config"]))
                params = run_log.get("params", {})
                st.markdown(_describe_params(params))
                with st.popover("\U0001F4DD View system prompt", use_container_width=True):
                    st.code(params.get("system_prompt", ""), language=None, wrap_lines=True)
        _render_agent_detail(agent_detail_ph, mission)


    else:
        st.caption("Switch to \"Agent Real-Time\" to consult the agent live for the current "
                  "situation, or \"Play Agent Mission\" to review a precomputed run's reasoning.")

with plot_col:
    st.divider()
    st.markdown("#### \u2705 Evaluation")
    # Score the FULL precomputed trajectory for Play LLM Mission (independent of which
    # frame is currently scrubbed to) instead of the live `sim.trajectory`, which stays
    # empty/frozen at mission start in that mode since it never calls sim.step().
    # precomputed_check: the colreg_llm_check ALREADY saved inside the loaded run log by
    # run_llm_scenario.py's run_one() (the sweep) -- shown automatically, no button click
    # needed, since that audit already exists on disk. The live "Check COLREG compliance"
    # button below still lets you force a FRESH check (e.g. for a mode with no precomputed
    # log at all, or to re-verify) -- but a click here only ever updates this browser
    # session's state, it is NEVER written back to the run log file on disk.
    precomputed_check = None
    if sim_mode == "Play Agent Mission":
        _eval_run_path = st.session_state.get("llm_run_path")
        _eval_run_log = load_run(_eval_run_path) if _eval_run_path else None
        eval_traj = _eval_run_log["trajectory"] if _eval_run_log else None
        eval_cache_key = _eval_run_path
        if _eval_run_log:
            precomputed_check = _eval_run_log.get("colreg_llm_check")
    else:
        eval_traj = sim.trajectory
        eval_cache_key = "live"

    if eval_traj is None:
        st.caption("Pick a precomputed run above the plot to see its score here.")
    elif len(eval_traj) < 4:
        st.caption("Run a few steps first -- not enough trajectory yet to score.")
    else:
        if precomputed_check and precomputed_check.get("checked"):
            _pc_score = precomputed_check.get("compliance_score")
            _pc_score_str = f"{_pc_score:.2f}" if isinstance(_pc_score, (int, float)) else "?"
            st.caption(f"\U0001F4BE This run already has a saved COLREG audit from when it was "
                      f"computed (score **{_pc_score_str}**) -- shown below automatically. The "
                      f"button only runs a NEW, separate check for this browser session; it "
                      f"does not overwrite the saved one on disk.")

        ec1, ec2 = st.columns([2, 1])
        with ec2:
            if st.button("\U0001F50D Check COLREG compliance (Claude)", use_container_width=True,
                        help="One-shot AI judge of the full trajectory so far -- runs on demand "
                             "only (a real network call), never during live stepping, so it adds "
                             "no latency to the simulation itself. Result is kept in THIS "
                             "browser session only -- never written back to the run log file."):
                from app.evaluation import llm_compliance_check
                with st.spinner("Asking Claude to audit COLREG compliance..."):
                    try:
                        st.session_state.llm_compliance_audit = llm_compliance_check(eval_traj)
                        st.session_state.llm_compliance_checked_key = (eval_cache_key, len(eval_traj))
                    except Exception as e:
                        st.error(f"Compliance check failed: {e}")
        session_audit = st.session_state.get("llm_compliance_audit")
        stale = st.session_state.get("llm_compliance_checked_key") != (eval_cache_key, len(eval_traj))
        if session_audit is not None and not stale:
            audit, audit_source = session_audit, "this session's live check"
        elif precomputed_check and precomputed_check.get("checked"):
            audit, audit_source = precomputed_check, "saved in the run log"
        else:
            audit, audit_source = None, None
        llm_violations = audit.get("violations") if audit else None
        if session_audit is not None and stale:
            st.caption("\u26A0\uFE0F Trajectory changed since the last compliance check -- re-run for a current result.")
        if audit is not None:
            n_v, n_c = len(audit["violations"]), len(audit["compliant_actions"])
            score = audit.get("compliance_score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            if n_v:
                st.caption(f"\U0001F916 Compliance score **{score_str}** ({audit_source}) -- Claude "
                          f"found {n_v} COLREG violation(s) and {n_c} correctly handled manoeuvre(s).")
            else:
                st.caption(f"\U0001F916 Compliance score **{score_str}** ({audit_source}) -- Claude "
                          f"found no COLREG violations ({n_c} manoeuvre(s) audited as correct).")
            with st.popover("\U0001F4C4 Full COLREG audit"):
                if audit["violations"]:
                    st.markdown("**Violations**")
                    for v in audit["violations"]:
                        st.markdown(f"- {v}")
                if audit["compliant_actions"]:
                    st.markdown("**Correctly handled**")
                    for c in audit["compliant_actions"]:
                        st.markdown(f"- {c}")
                if not audit["violations"] and not audit["compliant_actions"]:
                    st.caption("Nothing to audit -- no manoeuvres/encounters in this trajectory.")
        else:
            st.caption("\u2139\uFE0F Compliance shows a default score of 0.0 (unaudited) until you "
                      "run the Claude check above.")

        result = score_trajectory(
            eval_traj, start_xy=(mission.own_ship.x, mission.own_ship.y),
            goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
            safe_distance_m=sim.constraints.min_cpa_m,
            llm_violations=llm_violations,
            llm_compliance_score=(audit.get("compliance_score") if audit else None),
        )
        verdict = result["verdict"]
        badge_cls = "badge-pass" if verdict == "PASS" else "badge-fail"
        st.markdown(
            f'<span class="badge {badge_cls}">{verdict}</span>'
            f'<span style="font-size:1.1rem; font-weight:700;"> composite = {result["composite_score"]:.3f}</span>',
            unsafe_allow_html=True,
        )
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Safety", result["safety"]["score"],
                  help=f"min CPA {result['safety']['min_cpa_m']} m, passed={result['safety']['passed']}")
        e2.metric("Compl.", result["compliance"]["score"],
                  help=(f"{len(result['compliance']['violations'])} violation(s) -- from Claude's "
                       f"audit ({audit_source})" if audit is not None
                       else "0.0 = not yet audited by Claude (see the button above)"))
        e3.metric("Temp.", result["temporal"].get("temporal_score"))
        e4.metric("Spatial", result["spatial"].get("spatial_score"))
        e5.metric("Man.", result["manoeuvre"].get("manoeuvre_score"),
                  help=f"count={result['manoeuvre'].get('manoeuvre_count')}")
        if result["compliance"]["violations"]:
            with st.expander(f"Compliance violations ({len(result['compliance']['violations'])})"):
                for v in result["compliance"]["violations"]:
                    st.write(f"- {v}")
        with st.expander("Full result JSON"):
            # score_trajectory()'s own result dict has no field distinguishing "compliance
            # score 0 because unaudited" from "compliance score 0 because Claude found it
            # non-compliant" -- both look identical here (score: 0, violations: []) since an
            # audited-clean case is always score 1.0 with empty violations, never 0. Inject
            # this one extra key purely for the JSON dump so that ambiguity can't recur.
            st.json({**result, "compliance_audit_status": audit_source or "NOT audited yet "
                    "(default score 0.0 -- click 'Check COLREG compliance' above)"})

        with st.expander("Degeneracy check (always the same answer?)"):
            st.caption(
                "Per the DTU-paper finding in Design Ideas.docx: small models can score well by "
                "always answering the same thing. \"Quiet\" scenarios (q01/q02) expect "
                "`hold_course`; compare the action distribution across missions here."
            )
            if mission.id.startswith("q"):
                st.write("This is a **quiet** scenario -- correct recommendation is `hold_course`.")
            for tgt in sim.targets:
                c = contact_line(sim.own, tgt)
                if c["quiet"]:
                    st.caption(f"{c['name']}: quiet (CPA {c['cpa_m']:.0f}m, TCPA {c['tcpa_s']:.0f}s)")

# Persist the sidebar's current Ship performance/Mission values to disk on every rerun --
# must run last (after the sidebar widgets above have written this run's values into
# session_state) so a value just changed this run is captured immediately, not one rerun late.
_save_ui_prefs()


