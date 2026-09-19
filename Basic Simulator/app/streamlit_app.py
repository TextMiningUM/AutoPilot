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
from app.narrate import narrate, contact_line
from app.viz_plotly import trajectory_figure, trajectory_bounds
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
.side-panel {
    background: rgba(20,108,148,0.05); border-radius: 12px; padding: 0.8rem 1rem;
    border: 1px solid rgba(20,108,148,0.18); margin-bottom: 0.8rem;
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
    from app.agents import preload
    if not st.session_state.get("agent_ready", False):
        with st.status("Loading OOW agent (retrieval index + Qwen3-8B)...", expanded=True) as status:
            preload(status_cb=st.write)
            status.update(label="\u2705 OOW agent ready", state="complete", expanded=False)
        st.session_state.agent_ready = True
    else:
        st.success("\u2705 OOW agent ready (cached).")

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

sim: Simulation = st.session_state.sim
mission = st.session_state.mission


def _next_chart_key(prefix: str = "chart") -> str:
    """Plotly charts need a unique `key` per st.plotly_chart() call within a script run --
    _animate_preview()/the Full-run loop call it many times per run (once per animation
    frame), and without a distinct key each call Streamlit raises
    StreamlitDuplicateElementId (auto-generated IDs collide when params match). A simple
    incrementing counter in session_state guarantees uniqueness regardless of how many
    times any of the render functions below are called in a single run."""
    n = st.session_state.get("_chart_key_counter", 0) + 1
    st.session_state["_chart_key_counter"] = n
    return f"{prefix}_{n}"


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
        use_container_width=True, key=_next_chart_key("render_plot"),
    )
    if collision is not None:
        st.error(
            f"\U0001F4A5 Collision with **{collision['vehicle']}** at t={collision['time']:.0f}s "
            f"(range {collision['range_m']:.0f}m)."
        )


def _animate_preview(trajectory: list[dict], mission, placeholder, title: str,
                     speed_s: float = 0.05, max_frames: int = 80) -> None:
    """One-shot playback from t=0 for the (static, precomputed) Scenario preview -- without
    this, _render_plot draws the WHOLE trajectory in a single call, which looks like it
    'jumps straight to the end' since the current-position arrow lands on the final point
    immediately. Redraws are throttled to `max_frames` total so a long preview doesn't spend
    most of its time re-rendering Plotly."""
    if not trajectory:
        placeholder.info("Nothing recorded yet.")
        return
    times = sorted({row["time"] for row in trajectory})
    collision = find_collision(trajectory)
    x_range, y_range = trajectory_bounds(trajectory, mission)
    render_every = max(1, len(times) // max_frames)
    for i, t in enumerate(times):
        if i % render_every != 0 and i != len(times) - 1:
            continue
        partial = [row for row in trajectory if row["time"] <= t]
        show_collision = collision if (collision is not None and t >= collision["time"]) else None
        placeholder.plotly_chart(
            trajectory_figure(partial, mission, title=title, x_range=x_range, y_range=y_range,
                              current_time=t, total_time=times[-1], collision=show_collision),
            use_container_width=True, key=_next_chart_key("animate_preview"),
        )
        time.sleep(speed_s)
    if collision is not None:
        st.error(
            f"\U0001F4A5 Collision with **{collision['vehicle']}** at t={collision['time']:.0f}s "
            f"(range {collision['range_m']:.0f}m)."
        )


def _render_preview_frame(trajectory: list[dict], mission, placeholder, title: str, frame_idx: int) -> int:
    """Static (non-animating) single-frame render of the preview at `frame_idx` -- used for
    every normal rerun so the preview doesn't replay from scratch just because some unrelated
    widget elsewhere triggered a rerun. Returns the clamped frame_idx actually rendered."""
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
        use_container_width=True, key=_next_chart_key("preview_frame"),
    )
    if show_collision is not None:
        st.error(
            f"\U0001F4A5 Collision with **{show_collision['vehicle']}** at t={show_collision['time']:.0f}s "
            f"(range {show_collision['range_m']:.0f}m)."
        )
    return idx


# Plot header/mode-radio/placeholder created here (before the sidebar) so the sidebar's
# "Full run" button below can render live progress directly into `chart` as it steps --
# st.columns() containers are positional, not order-dependent, so this still renders in
# the correct (center) column regardless of running before the sidebar in script order.
plot_col, side_col = st.columns([3, 2], gap="medium")
with plot_col:
    st.markdown("#### \U0001F4C8 Plot")
    view = st.radio(
        "Mode", options=["Scenario preview (no avoidance)", "Manual helm", "LLM driven"],
        horizontal=True, label_visibility="collapsed", key="sim_view_mode",
        help="Scenario preview projects the whole mission forward at current heading/speed "
             "with no steering -- the raw encounter geometry. Manual helm shows the live "
             "trajectory as you steer it with the sidebar's Port/Starboard/Speed controls. "
             "LLM driven also shows the live trajectory, but 'Full run' (sidebar) drives it "
             "by repeatedly asking the OOW agent instead of you steering manually.",
    )
    chart = st.empty()
    _dt_now = st.session_state.get("dt_slider", 10.0)
    if view == "Scenario preview (no avoidance)":
        preview_traj = project_scenario(mission, dt=_dt_now)
        st.session_state.setdefault("preview_frame_idx", 0)
        _render_preview_frame(preview_traj, mission, chart,
                              f"{mission.name} -- scenario preview (no avoidance)",
                              st.session_state.preview_frame_idx)
        n_preview_frames = len(sorted({r["time"] for r in preview_traj}))
        st.caption(f"Frame {min(st.session_state.preview_frame_idx, n_preview_frames - 1) + 1}/{n_preview_frames} "
                  "-- use the sidebar's Step (advance one frame) or Full run (play once) to move through it.")
    elif view == "LLM driven":
        available_runs = list_runs_for_mission(mission.id)
        if not available_runs:
            chart.info(
                f"No precomputed LLM run found for **{mission.id}**. Live per-step model calls "
                "were too slow to play interactively, so LLM driven mode only plays back runs "
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
                     "report/recommendation, so playback here never calls the model.",
            )
            picked_run_path = str(available_runs[picked_run_i]["path"])
            if st.session_state.get("llm_run_path") != picked_run_path:
                st.session_state.llm_run_path = picked_run_path
                st.session_state.llm_run_frame_idx = 0
            run_log = load_run(picked_run_path)
            run_traj = run_log["trajectory"]
            run_times = sorted({r["time"] for r in run_traj})
            st.session_state.setdefault("llm_run_frame_idx", 0)
            clamped_idx = _render_preview_frame(
                run_traj, mission, chart,
                f"{mission.name} -- LLM run ({run_log['config']}/{run_log['tag']})",
                st.session_state.llm_run_frame_idx,
            )
            st.session_state.llm_run_frame_idx = clamped_idx
            st.caption(f"Frame {clamped_idx + 1}/{len(run_times)} -- t={run_times[clamped_idx]:.0f}s. "
                      "Use the sidebar's Step/Run steps/Full run to move through it.")
            cp = checkpoint_at_or_before(run_log, run_times[clamped_idx])
            if cp:
                with st.expander(f"\U0001F9E0 Agent decision at t={cp['time']:.0f}s (step {cp['step']})",
                                 expanded=True):
                    st.code(cp.get("situation_report", ""), language=None)
                    st.json(cp.get("decision", {}), expanded=False)
    else:
        _render_plot(sim.trajectory, mission, chart, mission.name,
                    bounds_from=project_scenario(mission, dt=_dt_now))

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mission")
    all_missions = {mid: load_mission(mid) for mid in mission_ids}
    labels = {mid: f"{m.name}  ({mid})" for mid, m in all_missions.items()}
    picked = st.selectbox("Scenario", options=mission_ids, format_func=lambda m: labels[m],
                          index=mission_ids.index(st.session_state.mission_id))
    if picked != st.session_state.mission_id:
        st.session_state.mission_id = picked
        st.rerun()

    with st.expander(f"\U0001F4CB {mission.name} briefing", expanded=False):
        st.markdown(mission.as_text())

    if st.button("\U0001F504 Reset mission", use_container_width=True):
        st.session_state.sim = Simulation(mission)
        st.session_state.last_decision = None
        st.session_state.last_debug = None
        st.rerun()

    st.divider()
    st.header("Simulation")
    dt = st.slider("Time step (s)", 1.0, 60.0, 10.0, step=1.0, key="dt_slider")
    sim_mode = st.session_state.get("sim_view_mode", "Manual helm")
    is_preview = sim_mode == "Scenario preview (no avoidance)"
    is_llm_playback = sim_mode == "LLM driven" and st.session_state.get("llm_run_path")
    is_frame_scrub = is_preview or is_llm_playback
    _frame_key = "preview_frame_idx" if is_preview else "llm_run_frame_idx"

    if st.button("\u25B6 Step", use_container_width=True,
                help="Advances one frame of the loaded run/preview instead of stepping the "
                     "real simulation." if is_frame_scrub else None):
        if is_frame_scrub:
            st.session_state[_frame_key] = st.session_state.get(_frame_key, 0) + 1
        else:
            sim.step(dt)
        st.rerun()
    rs_col1, rs_col2 = st.columns([1, 2])
    n_auto = rs_col1.number_input("Steps", 1, 100, 10, step=1)
    if rs_col2.button("\u25B6\u25B6\u25B6 Run steps", use_container_width=True):
        if is_frame_scrub:
            st.session_state[_frame_key] = st.session_state.get(_frame_key, 0) + int(n_auto)
        else:
            for _ in range(int(n_auto)):
                if sim.reached_goal():
                    break
                sim.step(dt)
        st.rerun()

    if st.button("\U0001F680 Full run", use_container_width=True,
                help="Scenario preview / LLM driven: plays the loaded (precomputed) "
                     "trajectory through once, start to finish -- no model calls, since "
                     "LLM driven only plays back runs computed upfront (see app/"
                     "run_llm_scenario.py). Manual helm: dead-reckons to completion (goal "
                     "reached, collision, or a step cap), animating the plot live as it goes."):
        if is_frame_scrub:
            # Runs exactly once, only on this click -- not on every unrelated rerun, which is
            # what made the preview feel like it "kept on playing" before this fix.
            if is_preview:
                frame_traj = project_scenario(mission, dt=dt)
                frame_title = f"{mission.name} -- scenario preview (no avoidance)"
            else:
                frame_log = load_run(st.session_state.llm_run_path)
                frame_traj = frame_log["trajectory"]
                frame_title = f"{mission.name} -- LLM run ({frame_log['config']}/{frame_log['tag']})"
            _animate_preview(frame_traj, mission, chart, frame_title)
            st.session_state[_frame_key] = len(sorted({r["time"] for r in frame_traj})) - 1
        elif sim_mode == "LLM driven":
            st.warning("No precomputed LLM run loaded for this mission -- pick one above the "
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
                    _render_plot(sim.trajectory, mission, chart, mission.name, bounds_from=bounds_from)
                    time.sleep(0.05)
            _render_plot(sim.trajectory, mission, chart, mission.name, bounds_from=bounds_from)
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
    if up_col.button("\u2B06\uFE0F", use_container_width=True, help="Speed up + step"):
        sim.speed_up(); sim.step(dt); st.rerun()
    left_col, mid_col, right_col = st.columns(3)
    if left_col.button("\u2B05\uFE0F", use_container_width=True, help="Port + step"):
        sim.turn_left(degrees); sim.step(dt); st.rerun()
    if mid_col.button("\u23F9\uFE0F", use_container_width=True, help="Stop + step"):
        sim.stop_vessel(); sim.step(dt); st.rerun()
    if right_col.button("\u27A1\uFE0F", use_container_width=True, help="Starboard + step"):
        sim.turn_right(degrees); sim.step(dt); st.rerun()
    _, down_col, _ = st.columns(3)
    if down_col.button("\u2B07\uFE0F", use_container_width=True, help="Slow down + step"):
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
    if st.button("\U0001F6D1 Shut down & free GPU", use_container_width=True,
                help="Unloads the model + retrieval index and terminates this server process "
                     "so the GPU is fully released -- closing the browser tab alone does NOT "
                     "do this, the Streamlit process keeps running (and keeps the VRAM) until "
                     "killed."):
        from app.agents import unload
        with st.spinner("Freeing GPU memory and shutting down..."):
            unload()
        st.success("\u2705 GPU freed. Server shutting down -- you can close this tab now.")
        os._exit(0)

# ── Single-page, ergonomic layout: plot centered, agent + evaluation stacked on the right ──
st.caption(f"\U0001F4CB **{mission.name}** ({mission.id}) -- full briefing in the sidebar under Mission.")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Time elapsed", f"{sim.t:.0f} s")
m2.metric("Heading", f"{sim.own.heading:.1f}\u00b0")
m3.metric("Speed", f"{sim.own.speed:.2f} m/s")
cpa_now = sim.min_cpa_now()
m4.metric("Closest range now", "n/a" if cpa_now == float("inf") else f"{cpa_now:.0f} m")
m5.metric("Reached goal?", "Yes" if sim.reached_goal() else "No")

with side_col:
    st.markdown('<div class="side-panel">', unsafe_allow_html=True)
    st.markdown("#### \U0001F916 Agent")
    with st.expander("Situation report", expanded=False):
        st.code(narrate(mission, sim.own), language=None)

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
        st.json(decision, expanded=False)
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
                st.code(debug["pg_guidance"], language=None)
            st.markdown("**Raw model output**")
            st.code(debug.get("raw_response", ""), language=None)
    else:
        st.caption("Click \"Ask OOW agent\" for a recommended manoeuvre.")
    st.markdown('</div>', unsafe_allow_html=True)

with plot_col:
    st.divider()
    st.markdown("#### \u2705 Evaluation")
    if len(sim.trajectory) < 4:
        st.caption("Run a few steps first -- not enough trajectory yet to score.")
    else:
        ec1, ec2 = st.columns([2, 1])
        with ec2:
            if st.button("\U0001F50D Check COLREG compliance (Claude)", use_container_width=True,
                        help="One-shot LLM judge of the full trajectory so far -- runs on demand "
                             "only (a real network call), never during live stepping, so it adds "
                             "no latency to the simulation itself."):
                from app.evaluation import llm_compliance_check
                with st.spinner("Asking Claude to audit COLREG compliance..."):
                    try:
                        st.session_state.llm_compliance_violations = llm_compliance_check(sim.trajectory)
                        st.session_state.llm_compliance_checked_len = len(sim.trajectory)
                    except Exception as e:
                        st.error(f"LLM compliance check failed: {e}")
        llm_violations = st.session_state.get("llm_compliance_violations")
        stale = st.session_state.get("llm_compliance_checked_len") != len(sim.trajectory)
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
            sim.trajectory, start_xy=(mission.own_ship.x, mission.own_ship.y),
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

