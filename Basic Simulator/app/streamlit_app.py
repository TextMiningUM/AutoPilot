"""Basic Simulator — OOW COLREG text-to-text simulator (Streamlit).

Run from the Basic Simulator/ folder:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import math
import os
import sys
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

from app.missions import list_mission_ids, load_mission
from app.simulation import Simulation, project_scenario, find_collision
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
.side-panel {
    background: rgba(20,108,148,0.05); border-radius: 12px; padding: 0.8rem 1rem;
    border: 1px solid rgba(20,108,148,0.18); margin-bottom: 0.8rem;
}
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
from app.llm_runs import list_runs_for_mission, load_run, checkpoint_at_or_before

# ── Session state ─────────────────────────────────────────────────────────
mission_ids = list_mission_ids()
if "mission_id" not in st.session_state:
    st.session_state.mission_id = mission_ids[0]
if "sim" not in st.session_state or st.session_state.get("_loaded_mission_id") != st.session_state.mission_id:
    mission = load_mission(st.session_state.mission_id)
    st.session_state.sim = Simulation(mission)
    st.session_state.mission = mission
    st.session_state._loaded_mission_id = st.session_state.mission_id
    st.session_state.last_decision = None
    st.session_state.last_debug = None
    st.session_state.preview_frame_idx = 0
    st.session_state.llm_run_frame_idx = 0
    st.session_state.llm_run_path = None
    st.session_state.llm_compliance_violations = None
    st.session_state.llm_compliance_checked_key = None
    st.session_state._llm_autorun = None

sim: Simulation = st.session_state.sim
mission = st.session_state.mission

# Play Agent Mission's Auto Run advances one checkpoint per RERUN rather than looping
# plotly_chart() calls inside a single button click -- each checkpoint gets its own full
# script run, so the chart/metrics/agent panel can each use ONE stable call site per run
# (smooth in-place update, no remount/flicker) instead of several calls forced into one run,
# which both needs a distinct key per call (forcing a remount each time) AND was the exact
# cause of an earlier StreamlitDuplicateElementKey crash. `_autorun_sleep_s`, if not None
# after this, tells the end-of-script block below to sleep then advance + rerun.
_autorun = st.session_state.get("_llm_autorun")
_autorun_sleep_s = None
if _autorun and st.session_state.get("sim_view_mode") == "Play Agent Mission" \
        and st.session_state.get("llm_run_path") == _autorun.get("run_path"):
    _idx_list = _autorun["idx_list"]
    if _autorun["i"] < len(_idx_list):
        st.session_state.llm_run_frame_idx = _idx_list[_autorun["i"]]
        _autorun_sleep_s = _autorun["speed_s"]
    else:
        st.session_state._llm_autorun = None
else:
    st.session_state._llm_autorun = None



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
                     checkpoint_times: list[float] | None = None) -> None:
    """Renders ONE native Plotly frames-animation (Play/Pause button + slider) instead of
    looping placeholder.plotly_chart() in Python. Streamlit requires a unique `key` per
    plotly_chart() call within a single script run, so a Python loop forces a full component
    remount every frame -- that's what caused the flicker/blank-screen-until-the-end bug.
    A single native-frames figure animates entirely client-side (click Play in the chart).
    `speed_s` (seconds/frame) comes straight from the sidebar's "Playback speed" slider.
    `checkpoint_times`, if given (Play LLM Mission), animates through only the actual LLM
    DECISION moments instead of every raw simulation tick -- a 200-step run only makes a new
    decision every `decision_interval` (e.g. 10) steps, so animating every single tick showed
    ~20x more frames than there were actual changes in situation report/LLM decision."""
    if not trajectory:
        placeholder.info("Nothing recorded yet.")
        return
    times = sorted({row["time"] for row in trajectory})
    collision = find_collision(trajectory)
    x_range, y_range = trajectory_bounds(trajectory, mission)
    if checkpoint_times:
        frame_times = sorted(t for t in checkpoint_times if t in set(times))
        if not frame_times or frame_times[-1] != times[-1]:
            frame_times = frame_times + [times[-1]]
    else:
        render_every = max(1, len(times) // max_frames)
        frame_times = [t for i, t in enumerate(times)
                      if i % render_every == 0 or i == len(times) - 1]
    placeholder.plotly_chart(
        animated_trajectory_figure(trajectory, mission, title=title, x_range=x_range, y_range=y_range,
                                   frame_times=frame_times, collision=collision,
                                   frame_duration_ms=max(int(speed_s * 1000), 60)),
        # One mount per Auto-Run click -- a stable key is fine here (unlike a loop).
        use_container_width=True, key="chart_animate_preview",
    )
    st.caption("\u25b6 Click **Play** in the chart (or drag the slider) to animate.")
    if collision is not None:
        st.error(
            f"\U0001F4A5 Collision with **{collision['vehicle']}** at t={collision['time']:.0f}s "
            f"(range {collision['range_m']:.0f}m)."
        )


def _render_preview_frame(trajectory: list[dict], mission, placeholder, title: str, frame_idx: int) -> int:
    """Static (non-animating) single-frame render of the preview at `frame_idx` -- used for
    every normal rerun so the preview doesn't replay from scratch just because some unrelated
    widget elsewhere triggered a rerun. Returns the clamped frame_idx actually rendered.
    Called AT MOST ONCE per script run (Play Agent Mission's Auto Run now advances one
    checkpoint per RERUN instead of looping this in Python -- see `_llm_autorun` above), so a
    single constant `key` is safe: Streamlit patches the SAME Plotly component's data on every
    rerun instead of remounting it, which is what keeps stepping/auto-run flicker-free."""
    times = sorted({row["time"] for row in trajectory})
    if not times:
        placeholder.info("Nothing recorded yet.")
        return 0
    idx = max(0, min(frame_idx, len(times) - 1))
    t = times[idx]
    collision = find_collision(trajectory)
    partial = [row for row in trajectory if row["time"] <= t]
    x_range, y_range = trajectory_bounds(trajectory, mission)
    show_collision = collision if (collision is not None and t >= collision["time"]) else None
    placeholder.plotly_chart(
        trajectory_figure(partial, mission, title=title, x_range=x_range, y_range=y_range,
                          current_time=t, total_time=times[-1], collision=show_collision),
        use_container_width=True, key="chart_preview_frame",
    )
    if show_collision is not None:
        st.error(
            f"\U0001F4A5 Collision with **{show_collision['vehicle']}** at t={show_collision['time']:.0f}s "
            f"(range {show_collision['range_m']:.0f}m)."
        )
    return idx


def _render_metrics_row(placeholders, mission, metrics_row, targets_now) -> None:
    """Fills the 4 goal-bearing/heading/CPA/TCPA metric boxes. Split out of the normal
    top-of-script render so the sidebar's Play Agent Mission auto-run loop can call it again
    per checkpoint -- otherwise these numbers stayed frozen at whatever frame was on screen
    when Auto Run was clicked, out of sync with the animating plot."""
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


def _render_agent_checkpoint(phs: dict, run_log: dict, mission, idx: int) -> None:
    """Fills the situation-report/decision half of the Play Agent Mission panel for the
    trajectory tick at `idx`, one placeholder PER PIECE (header/quickfacts/report/decision)
    instead of one shared `.container()`. Each placeholder is called every run with whatever
    text applies now (an `st.empty()` placeholder shows NOTHING for a run that doesn't write
    to it -- content does not persist from a previous run on its own), but because each piece
    keeps its OWN stable position/identity across reruns, Streamlit only needs to patch the
    text that actually changed rather than tearing down and rebuilding the whole block --
    unlike a single `.container()`, which recreates its entire child-element tree every call."""
    run_times = sorted({r["time"] for r in run_log["trajectory"]})
    idx = max(0, min(idx, len(run_times) - 1))
    cp = checkpoint_at_or_before(run_log, run_times[idx])
    if not cp:
        phs["header"].caption("No decision recorded at/before this frame yet.")
        for key in ("quickfacts", "report_label", "report_code", "decision_label", "decision"):
            phs[key].empty()
        return
    phs["header"].caption(f"Nearest decision: step {cp['step']}, t={cp['time']:.0f}s")
    own_row = next((r for r in run_log["trajectory"]
                   if r["time"] == cp["time"] and r["vehicle"] == "own_ship"), None)
    if own_row:
        phs["quickfacts"].caption(_goal_quickfacts(own_row["x"], own_row["y"], own_row["heading"],
                                                    own_row["speed"], mission.goal))
    else:
        phs["quickfacts"].empty()
    phs["report_label"].markdown("**Situation report (passed to the Agent)**")
    phs["report_code"].code(cp.get("situation_report", ""), language=None, wrap_lines=True)
    phs["decision_label"].markdown("**Agent decision**")
    phs["decision"].markdown(_describe_decision(cp.get("decision", {})))


# Plot header/mode-radio/placeholder created here (before the sidebar) so the sidebar's
# "Full run" button below can render live progress directly into `chart` as it steps --
# st.columns() containers are positional, not order-dependent, so this still renders in
# the correct (center) column regardless of running before the sidebar in script order.
plot_col, side_col = st.columns([3, 2], gap="medium")
with side_col:
    # Header + placeholders created early (before the sidebar) so the sidebar's Play Agent
    # Mission auto-run driver can push live updates as it steps through checkpoints -- same
    # reasoning as `chart`/the metric placeholders below. One placeholder PER PIECE (not one
    # shared `.container()`) so re-rendering only touches the specific text that actually
    # changed instead of tearing down and rebuilding the whole block every single step (which
    # read as the panel "fully redrawing/flickering" on each Step click).
    st.markdown('<div class="side-panel">', unsafe_allow_html=True)
    st.markdown("#### \U0001F916 Agent")
    agent_static_ph = st.empty()
    agent_cp_phs = {
        "header": st.empty(), "quickfacts": st.empty(), "report_label": st.empty(),
        "report_code": st.empty(), "decision_label": st.empty(), "decision": st.empty(),
    }

with plot_col:
    st.caption(f"\U0001F4CB **{mission.name}** ({mission.id}) -- full briefing in the sidebar under Mission.")
    view = st.radio(
        "Mode", options=["Scenario preview (no avoidance)", "Manual helm",
                        "Play Agent Mission", "Agent Real-Time"],
        horizontal=True, label_visibility="collapsed", key="sim_view_mode",
        help="Preview: no-avoidance path. Manual helm: steer it live. Play Agent Mission: "
             "replay a precomputed run. Agent Real-Time: live agent calls.",
    )
    _dt_now = st.session_state.get("dt_slider", 10.0)

    # Resolve the state behind the metrics row below from whatever is ACTUALLY on screen for
    # the current mode -- Scenario preview/Play LLM Mission scrub through a precomputed
    # trajectory that has nothing to do with the live `sim` object (which stays frozen at
    # mission start in those two modes, since they never call sim.step()); reading `sim.*`
    # unconditionally here previously made the metrics look frozen/dead while scrubbing.
    preview_traj = run_log = run_traj = None
    metrics_row = None  # (t, x, y, heading, speed)
    if view == "Scenario preview (no avoidance)":
        preview_traj = project_scenario(mission, dt=_dt_now)
        _times = sorted({r["time"] for r in preview_traj})
        if _times:
            _idx = max(0, min(st.session_state.get("preview_frame_idx", 0), len(_times) - 1))
            _row = next(r for r in preview_traj
                       if r["time"] == _times[_idx] and r["vehicle"] == "own_ship")
            metrics_row = (_times[_idx], _row["x"], _row["y"], _row["heading"], _row["speed"])
    elif view == "Play Agent Mission":
        _run_path = st.session_state.get("llm_run_path")
        if _run_path:
            run_log = load_run(_run_path)
            run_traj = run_log["trajectory"]
            _times = sorted({r["time"] for r in run_traj})
            if _times:
                _idx = max(0, min(st.session_state.get("llm_run_frame_idx", 0), len(_times) - 1))
                _row = next(r for r in run_traj
                           if r["time"] == _times[_idx] and r["vehicle"] == "own_ship")
                metrics_row = (_times[_idx], _row["x"], _row["y"], _row["heading"], _row["speed"])
    else:
        metrics_row = (sim.t, sim.own.x, sim.own.y, sim.own.heading, sim.own.speed)

    # Resolve target snapshots (name/x/y/heading/speed) at the SAME resolved time as
    # metrics_row, for the CPA/TCPA boxes below -- same live-vs-precomputed split as above.
    targets_now: list[dict] = []
    if view == "Scenario preview (no avoidance)" and preview_traj and metrics_row:
        targets_now = [r for r in preview_traj
                       if r["time"] == metrics_row[0] and r["vehicle"] != "own_ship"]
    elif view == "Play Agent Mission" and run_traj and metrics_row:
        targets_now = [r for r in run_traj
                       if r["time"] == metrics_row[0] and r["vehicle"] != "own_ship"]
    elif metrics_row:
        targets_now = [{"vehicle": v.name, "x": v.x, "y": v.y, "heading": v.heading, "speed": v.speed}
                       for v in sim.targets]

    b_cols = st.columns(4)
    metric_phs = [c.empty() for c in b_cols]
    _render_metrics_row(metric_phs, mission, metrics_row, targets_now)

    chart = st.empty()
    st.session_state.setdefault("_animated_active_mode", None)
    if view == "Scenario preview (no avoidance)":
        st.session_state.setdefault("preview_frame_idx", 0)
        if st.session_state._animated_active_mode != view:
            _render_preview_frame(preview_traj, mission, chart,
                                  "Scenario preview (no avoidance)",
                                  st.session_state.preview_frame_idx)
            n_preview_frames = len(sorted({r["time"] for r in preview_traj}))
            st.caption(f"Frame {min(st.session_state.preview_frame_idx, n_preview_frames - 1) + 1}/{n_preview_frames} "
                      "-- use the sidebar's Step (advance one frame) or Auto Run (play once) to move through it.")
        else:
            _animate_preview(preview_traj, mission, chart,
                             "Scenario preview (no avoidance)",
                             speed_s=st.session_state.get("_animated_speed_s", 0.3))
            st.caption("Click **Play** in the chart above (or drag its slider) to watch it -- use Step/Reset to go back to frame-by-frame.")
    elif view == "Play Agent Mission":
        available_runs = list_runs_for_mission(mission.id)
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
                key="llm_run_picked_i",
                help="Generated by app/run_llm_scenario.py -- asks the agent every N steps "
                     "(stored per-run) and saves the full trajectory + every situation "
                     "report/recommendation, so playback here never calls the model. See the "
                     "Agent panel for the model/params/situation-report/decision at this frame.",
            )
            picked_run_path = str(available_runs[picked_run_i]["path"])
            if st.session_state.get("llm_run_path") != picked_run_path:
                st.session_state.llm_run_path = picked_run_path
                st.session_state.llm_run_frame_idx = 0
                st.session_state._animated_active_mode = None
            run_log = load_run(picked_run_path)
            run_traj = run_log["trajectory"]
            run_times = sorted({r["time"] for r in run_traj})
            st.session_state.setdefault("llm_run_frame_idx", 0)
            clamped_idx = _render_preview_frame(
                run_traj, mission, chart,
                f"Agent run ({run_log['config']}/{run_log['tag']})",
                st.session_state.llm_run_frame_idx,
            )
            st.session_state.llm_run_frame_idx = clamped_idx
            st.caption(f"Frame {clamped_idx + 1}/{len(run_times)} -- t={run_times[clamped_idx]:.0f}s. "
                      "Use the sidebar's Step/Run steps/Auto Run to move through it. See the "
                      "Agent panel (right) for what was passed to/decided by the model here.")
    else:
        _render_plot(sim.trajectory, mission, chart, "",
                    bounds_from=project_scenario(mission, dt=_dt_now))

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Ship performance")
    st.caption("Not wired into the physics yet -- stored here to bring into the simulation later.")
    sp1, sp2 = st.columns(2)
    sp1.number_input("Max speed (m/s)", min_value=0.0, value=10.0, step=0.5, key="ship_max_speed")
    sp2.number_input("Max turn rate (%)", min_value=0.0, value=30.0, step=1.0, key="ship_max_turn_rate_pct")
    sp3, sp4 = st.columns(2)
    sp3.number_input("Max acceleration (m/s\u00b2)", min_value=0.0, value=0.2, step=0.05, key="ship_max_accel")
    sp4.number_input("Max deceleration (m/s\u00b2)", min_value=0.0, value=0.2, step=0.05, key="ship_max_decel")
    st.number_input("Turn rate (deg/s)", min_value=0.0, value=3.0, step=0.5, key="ship_turn_rate_deg_s")
    st.divider()

    st.header("Mission")
    m1, m2 = st.columns(2)
    m1.number_input("Cruise speed (m/s)", min_value=0.0, value=10.0, step=0.5, key="mission_cruise_speed")
    m2.number_input("Minimal CPA (m)", min_value=0.0, value=500.0, step=50.0, key="mission_min_cpa")
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
        st.session_state.sim = Simulation(mission)
        st.session_state.last_decision = None
        st.session_state.last_debug = None
        st.session_state.preview_frame_idx = 0
        st.session_state.llm_run_frame_idx = 0
        st.session_state._animated_active_mode = None
        st.session_state.llm_compliance_violations = None
        st.session_state.llm_compliance_checked_key = None
        st.rerun()

    st.divider()
    st.header("Simulation")
    dt = st.slider("Time step (s)", 1.0, 60.0, 10.0, step=1.0, key="dt_slider")
    sim_mode = st.session_state.get("sim_view_mode", "Manual helm")
    is_preview = sim_mode == "Scenario preview (no avoidance)"
    is_llm_playback = sim_mode == "Play Agent Mission" and st.session_state.get("llm_run_path")
    is_frame_scrub = is_preview or is_llm_playback
    _frame_key = "preview_frame_idx" if is_preview else "llm_run_frame_idx"

    sb_col, sf_col = st.columns(2)
    if sb_col.button("\u2b05\ufe0f Step back", use_container_width=True, disabled=not is_frame_scrub):
        st.session_state[_frame_key] = max(0, st.session_state.get(_frame_key, 0) - 1)
        st.session_state._animated_active_mode = None
        st.rerun()
    if sf_col.button("\u27a1\ufe0f Step forward", use_container_width=True):
        if is_frame_scrub:
            st.session_state[_frame_key] = st.session_state.get(_frame_key, 0) + 1
            st.session_state._animated_active_mode = None
        else:
            sim.step(dt)
        st.rerun()

    rs_col1, rs_col2, rs_col3 = st.columns(3)
    n_auto = rs_col2.number_input("Steps", 1, 100, 10, step=1, label_visibility="collapsed")
    if rs_col1.button("\u23ea Steps", use_container_width=True, disabled=not is_frame_scrub):
        st.session_state[_frame_key] = max(0, st.session_state.get(_frame_key, 0) - int(n_auto))
        st.session_state._animated_active_mode = None
        st.rerun()
    if rs_col3.button("Steps \u23e9", use_container_width=True):
        if is_frame_scrub:
            st.session_state[_frame_key] = st.session_state.get(_frame_key, 0) + int(n_auto)
            st.session_state._animated_active_mode = None
        else:
            for _ in range(int(n_auto)):
                if sim.reached_goal():
                    break
                sim.step(dt)
        st.rerun()

    speed_level = st.slider("Playback speed", 1, 10, 6, disabled=not is_frame_scrub)
    playback_ms = int(1500 - (speed_level - 1) * (1500 - 60) / 9)  # 1=slowest (1500ms), 10=fastest (60ms)
    if st.button("\U0001F680 Auto Run", use_container_width=True):
        if is_preview:
            # Runs exactly once, only on this click -- not on every unrelated rerun, which is
            # what made the preview feel like it "kept on playing" before this fix. No agent
            # panel to keep in sync here, so the native client-side Plotly animation is fine.
            frame_traj = project_scenario(mission, dt=dt)
            frame_title = "Scenario preview (no avoidance)"
            _animate_preview(frame_traj, mission, chart, frame_title, speed_s=playback_ms / 1000)
            st.session_state._animated_active_mode = sim_mode
            st.session_state._animated_speed_s = playback_ms / 1000
        elif is_llm_playback:
            # Sets up the job and reruns immediately -- the early "Session state" block at the
            # top of the script (see `_llm_autorun`) advances one checkpoint per RERUN and the
            # end-of-script block sleeps/reruns again, so the chart/metrics/Agent panel each
            # render via their normal single-call-per-run path (smooth in-place update) instead
            # of being looped here, which forced a remount every checkpoint (distinct keys
            # needed within one run) and was the exact cause of an earlier
            # StreamlitDuplicateElementKey crash.
            frame_log = load_run(st.session_state.llm_run_path)
            frame_times = sorted({r["time"] for r in frame_log["trajectory"]})
            checkpoints = frame_log.get("checkpoints") or [{"time": frame_times[-1]}]
            idx_list = [frame_times.index(cp["time"]) if cp["time"] in frame_times else len(frame_times) - 1
                       for cp in checkpoints]
            st.session_state._llm_autorun = {
                "run_path": st.session_state.llm_run_path, "idx_list": idx_list, "i": 0,
                "speed_s": max(playback_ms / 1000, 0.4),
            }
            st.session_state._animated_active_mode = None
        elif sim_mode == "Play Agent Mission":
            st.warning("No precomputed Agent run loaded for this mission -- pick one above the "
                      "plot, or generate one via app/run_llm_scenario.py (see its docstring).")
        else:
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

    st.divider()
    st.header("Manual helm")
    degrees = st.slider("Turn amount (deg)", 1.0, 90.0, 15.0, step=1.0)
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

with side_col:
    if sim_mode == "Agent Real-Time":
        # The original live/interactive agent panel -- kept exactly as-is, now scoped to
        # this one mode instead of always showing regardless of which mode was picked.
        with st.expander("Situation report", expanded=False):
            st.caption(_goal_quickfacts(sim.own.x, sim.own.y, sim.own.heading, sim.own.speed,
                                       mission.goal))
            st.code(narrate(mission, sim.own), language=None, wrap_lines=True)

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
            preview = rag_context_preview(mission, sim.own, k=effective_k)
            if preview["chars"]:
                st.caption(
                    f"\U0001F4CF Context preview: ~{preview['chars']:,} chars "
                    f"(~{preview['chars'] // 4:,} tokens est.) from {preview['chunks']} chunk(s)"
                )
            else:
                st.caption("\U0001F4CF Context preview: RAG off -- no excerpts will be injected.")

        if st.button("\U0001F9E0 Ask OOW agent", type="primary", use_container_width=True):
            with st.spinner(f"Retrieving context + generating ({MODEL_CONFIGS[model_config]})..."):
                decision, debug = ask_oow(mission, sim.own, config=model_config,
                                          system_prompt=st.session_state.get("custom_system_prompt"),
                                          max_new_tokens=int(max_new_tokens), enable_thinking=enable_thinking,
                                          k=effective_k)
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
                st.markdown("**Raw model output**")
                st.code(debug.get("raw_response", ""), language=None, wrap_lines=True)
        else:
            st.caption("Click \"Ask OOW agent\" for a recommended manoeuvre.")

    elif sim_mode == "Play Agent Mission":
        # Read-only playback detail for the precomputed run currently loaded/scrubbed in the
        # plot -- model+params (static) in agent_static_ph, situation report/decision for the
        # nearest checkpoint (advanced by the auto-run rerun-driver too) in agent_cp_phs.
        # Both always re-render: an st.empty() placeholder shows NOTHING on a run that doesn't
        # write to it, so a "skip if unchanged" shortcut here previously blanked the whole
        # panel on some reruns instead of just avoiding a redundant draw.
        run_path = st.session_state.get("llm_run_path")
        if not run_path:
            agent_static_ph.caption("Pick a precomputed run above the plot to see its details here.")
        else:
            run_log = load_run(run_path)
            idx = st.session_state.get("llm_run_frame_idx", 0)
            with agent_static_ph.container():
                st.write("**Model / config:**",
                        MODEL_CONFIGS.get(run_log["config"], run_log["config"]))
                params = run_log.get("params", {})
                st.markdown(_describe_params(params))
                with st.popover("\U0001F4DD View system prompt", use_container_width=True):
                    st.code(params.get("system_prompt", ""), language=None, wrap_lines=True)
            _render_agent_checkpoint(agent_cp_phs, run_log, mission, idx)

    else:
        st.caption("Switch to \"Agent Real-Time\" to consult the agent live for the current "
                  "situation, or \"Play Agent Mission\" to review a precomputed run's reasoning.")

    st.markdown('</div>', unsafe_allow_html=True)

with plot_col:
    st.divider()
    st.markdown("#### \u2705 Evaluation")
    # Score the FULL precomputed trajectory for Play LLM Mission (independent of which
    # frame is currently scrubbed to) instead of the live `sim.trajectory`, which stays
    # empty/frozen at mission start in that mode since it never calls sim.step().
    if sim_mode == "Play Agent Mission":
        _eval_run_path = st.session_state.get("llm_run_path")
        eval_traj = load_run(_eval_run_path)["trajectory"] if _eval_run_path else None
        eval_cache_key = _eval_run_path
    else:
        eval_traj = sim.trajectory
        eval_cache_key = "live"

    if eval_traj is None:
        st.caption("Pick a precomputed run above the plot to see its score here.")
    elif len(eval_traj) < 4:
        st.caption("Run a few steps first -- not enough trajectory yet to score.")
    else:
        ec1, ec2 = st.columns([2, 1])
        with ec2:
            if st.button("\U0001F50D Check COLREG compliance (Claude)", use_container_width=True,
                        help="One-shot AI judge of the full trajectory so far -- runs on demand "
                             "only (a real network call), never during live stepping, so it adds "
                             "no latency to the simulation itself."):
                from app.evaluation import llm_compliance_check
                with st.spinner("Asking Claude to audit COLREG compliance..."):
                    try:
                        st.session_state.llm_compliance_violations = llm_compliance_check(eval_traj)
                        st.session_state.llm_compliance_checked_key = (eval_cache_key, len(eval_traj))
                    except Exception as e:
                        st.error(f"Compliance check failed: {e}")
        llm_violations = st.session_state.get("llm_compliance_violations")
        stale = st.session_state.get("llm_compliance_checked_key") != (eval_cache_key, len(eval_traj))
        if llm_violations is not None:
            if stale:
                st.caption("\u26A0\uFE0F Trajectory changed since the last compliance check -- re-run for a current result.")
            elif llm_violations:
                st.caption(f"\U0001F916 Claude found {len(llm_violations)} COLREG violation(s).")
            else:
                st.caption("\U0001F916 Claude found no COLREG violations in this run.")
        else:
            st.caption("\u2139\uFE0F Compliance shows a default score of 1.0 until you run the Claude check above.")

        result = score_trajectory(
            eval_traj, start_xy=(mission.own_ship.x, mission.own_ship.y),
            goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
            llm_violations=llm_violations if not stale else None,
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
                  help=f"{len(result['compliance']['violations'])} violation(s)")
        e3.metric("Temp.", result["temporal"].get("temporal_score"))
        e4.metric("Spatial", result["spatial"].get("spatial_score"))
        e5.metric("Man.", result["manoeuvre"].get("manoeuvre_score"),
                  help=f"count={result['manoeuvre'].get('manoeuvre_count')}")
        if result["compliance"]["violations"]:
            with st.expander(f"Compliance violations ({len(result['compliance']['violations'])})"):
                for v in result["compliance"]["violations"]:
                    st.write(f"- {v}")
        with st.expander("Full result JSON"):
            st.json(result)

        with st.expander("Degeneracy check (always the same answer?)"):
            st.caption(
                "Per the DTU-paper finding in Design Ideas.docx: small models can score well by "
                "always answering the same thing. \"Quiet\" scenarios (q01/q02) expect "
                "`hold_course`; compare the action distribution across missions here."
            )
            if mission.id.startswith("q"):
                st.write("This is a **quiet** scenario -- correct recommendation is `hold_course`.")
            for tgt in mission.targets:
                c = contact_line(sim.own, tgt)
                if c["quiet"]:
                    st.caption(f"{c['name']}: quiet (CPA {c['cpa_m']:.0f}m, TCPA {c['tcpa_s']:.0f}s)")

# Play Agent Mission Auto Run driver: everything above has already rendered this checkpoint
# (chart/metrics/Agent panel), so the user gets to actually SEE it for `_autorun_sleep_s`
# before advancing -- sleeping any earlier would just delay the render instead.
if _autorun_sleep_s is not None:
    time.sleep(_autorun_sleep_s)
    st.session_state._llm_autorun["i"] += 1
    st.rerun()


