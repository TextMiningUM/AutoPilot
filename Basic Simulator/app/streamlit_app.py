"""Basic Simulator — OOW COLREG text-to-text simulator (Streamlit).

Run from the Basic Simulator/ folder:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
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

from app.missions import list_mission_ids, load_mission
from app.simulation import Simulation, project_scenario
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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>\U0001F9ED Basic Simulator — OOW COLREG Agent</h1>
  <p>Text-to-text collision-avoidance simulator · Imazu-style missions · base Qwen3-8B (RAG + CoT + Procedural Graph)</p>
</div>
""", unsafe_allow_html=True)

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
    for prefix in ("preview_play", "live_play"):
        st.session_state[f"{prefix}_idx"] = 0
        st.session_state[f"{prefix}_playing"] = False

sim: Simulation = st.session_state.sim
mission = st.session_state.mission

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

    if st.button("\U0001F504 Reset mission", use_container_width=True):
        st.session_state.sim = Simulation(mission)
        st.session_state.last_decision = None
        st.session_state.last_debug = None
        st.rerun()

    st.divider()
    st.header("Simulation")
    dt = st.slider("Time step (s)", 1.0, 60.0, 10.0, step=1.0)
    if st.button("\u23ED\uFE0F Step", use_container_width=True):
        sim.step(dt)
        st.rerun()
    n_auto = st.number_input("Auto-run steps", 1, 100, 10, step=1)
    if st.button("\u25B6\uFE0F Run steps", use_container_width=True):
        for _ in range(int(n_auto)):
            if sim.reached_goal():
                break
            sim.step(dt)
        st.rerun()

    st.divider()
    st.header("Manual helm")
    degrees = st.slider("Turn amount (deg)", 1.0, 90.0, 15.0, step=1.0)
    c1, c2 = st.columns(2)
    if c1.button("\u2B05\uFE0F Port", use_container_width=True):
        sim.turn_left(degrees); st.rerun()
    if c2.button("Starboard \u27A1\uFE0F", use_container_width=True):
        sim.turn_right(degrees); st.rerun()
    c3, c4 = st.columns(2)
    if c3.button("\U0001F40C Slow", use_container_width=True):
        sim.slow_down(); st.rerun()
    if c4.button("\U0001F680 Speed up", use_container_width=True):
        sim.speed_up(); st.rerun()
    if st.button("\u26D4 Stop", use_container_width=True):
        sim.stop_vessel(); st.rerun()

    st.divider()
    st.caption(
        "Model: base **Qwen3-8B**, 4-bit NF4, combined RAG + CoT + Procedural-Graph prompt. "
        "The fine-tuned OOW-QWEN checkpoint isn't trained yet -- see notebook §13."
    )

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_mission, tab_plot, tab_agent, tab_eval = st.tabs(
    ["\U0001F4CB Mission", "\U0001F4C8 Plot", "\U0001F916 Agent", "\u2705 Evaluation"]
)

with tab_mission:
    left, right = st.columns([2, 1])
    with left:
        st.markdown(mission.as_text())
    with right:
        st.metric("Time elapsed", f"{sim.t:.0f} s")
        st.metric("Own-ship heading", f"{sim.own.heading:.1f}\u00b0")
        st.metric("Own-ship speed", f"{sim.own.speed:.2f} m/s")
        cpa_now = sim.min_cpa_now()
        st.metric("Closest target range now", "n/a" if cpa_now == float("inf") else f"{cpa_now:.0f} m")
        st.metric("Reached goal?", "Yes" if sim.reached_goal() else "No")


@st.fragment
def _playback(trajectory: list[dict], mission, placeholder, title: str, state_prefix: str,
               speed_s: float = 0.35, bounds_from: list[dict] | None = None) -> None:
    """Discrete-rerun playback (one time-step advanced per rerun, via st.rerun()) instead
    of a single blocking Python loop -- a blocking loop never gives Streamlit a chance to
    process a Pause click in between frames, which is why Pause didn't work before.
    Wrapped in @st.fragment so each rerun only re-renders THIS chart/buttons, not the
    whole page (sidebar, tabs, CSS) -- that full-page re-render was what caused the
    remaining blink. Play position is kept in session_state so Pause/Resume/Restart all
    work correctly across reruns."""
    times = sorted({row["time"] for row in trajectory})
    if not times:
        st.info("Nothing recorded yet.")
        return
    idx_key, playing_key = f"{state_prefix}_idx", f"{state_prefix}_playing"
    st.session_state.setdefault(idx_key, 0)
    st.session_state.setdefault(playing_key, False)

    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("\u25B6\uFE0F Play", key=f"{state_prefix}_play_btn", use_container_width=True):
        if st.session_state[idx_key] >= len(times) - 1:
            st.session_state[idx_key] = 0
        st.session_state[playing_key] = True
    if c2.button("\u23F8\uFE0F Pause", key=f"{state_prefix}_pause_btn", use_container_width=True):
        st.session_state[playing_key] = False
    if c3.button("\u23EA\uFE0F Restart", key=f"{state_prefix}_restart_btn", use_container_width=True):
        st.session_state[idx_key] = 0
        st.session_state[playing_key] = False

    idx = min(st.session_state[idx_key], len(times) - 1)
    t = times[idx]
    partial = [row for row in trajectory if row["time"] <= t]
    x_range, y_range = trajectory_bounds(bounds_from or trajectory, mission)
    placeholder.plotly_chart(
        trajectory_figure(partial, mission, title=title, x_range=x_range, y_range=y_range,
                           current_time=t, total_time=times[-1]),
        use_container_width=True, key=f"{state_prefix}_chart",
    )
    st.progress((idx + 1) / len(times))

    if st.session_state[playing_key]:
        if idx < len(times) - 1:
            st.session_state[idx_key] = idx + 1
            time.sleep(speed_s)
            st.rerun()
        else:
            st.session_state[playing_key] = False


with tab_plot:
    show_preview = st.checkbox(
        "\U0001F52D Scenario preview (straight-line, no avoidance)", value=(sim.t == 0.0),
        help="Projects the whole mission forward at current heading/speed with no steering -- "
             "shows the raw encounter geometry the scenario was built to create, independent of "
             "your live (steerable) run below.",
    )
    if show_preview:
        preview_traj = project_scenario(mission, dt=dt)
        preview_chart = st.empty()
        _playback(preview_traj, mission, preview_chart, f"{mission.name} -- scenario preview (no avoidance)",
                   state_prefix="preview_play")
        st.divider()
    st.markdown("**Live run** (reflects manual helm / applied agent decisions so far)")
    live_chart = st.empty()
    _playback(sim.trajectory, mission, live_chart, mission.name, state_prefix="live_play",
               bounds_from=project_scenario(mission, dt=dt))

with tab_agent:
    st.subheader("Situation report")
    situation = narrate(mission, sim.own)
    st.code(situation, language=None)

    if st.button("\U0001F9E0 Ask OOW agent", type="primary"):
        from app.agents import ask_oow
        with st.spinner("Retrieving context + generating..."):
            decision, debug = ask_oow(mission, sim.own)
        st.session_state.last_decision = decision
        st.session_state.last_debug = debug

    decision = st.session_state.last_decision
    debug = st.session_state.last_debug
    if decision:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**OOW recommendation**")
            st.json(decision)
        with col_b:
            st.markdown("**Apply to own-ship**")
            if st.button("\u2705 Apply recommendation"):
                sim.apply_action(decision)
                sim.step(dt)
                st.rerun()
            if st.button("\U0001F6AB Ignore (I'll steer manually)"):
                pass

        with st.expander("Retrieval / grounding detail"):
            st.write("**Retrieved COLREG chunk IDs:**", debug.get("retrieved_chunk_ids"))
            st.write("**Query concepts:**", debug.get("query_concepts"))
            st.write("**Expanded concepts:**", debug.get("expanded_concepts"))
            if debug.get("pg_guidance"):
                st.markdown("**Procedural-graph guidance**")
                st.code(debug["pg_guidance"], language=None)
            st.markdown("**Raw model output**")
            st.code(debug.get("raw_response", ""), language=None)
    else:
        st.info("Click \"Ask OOW agent\" to get a recommended manoeuvre for the current situation.")

with tab_eval:
    st.subheader("COLREGS compliance validation (Woerner/Benjamin/Novitzky/Leonard-style)")
    st.caption(
        "Two hard gates first (collision -> 0.0, never-arrived -> capped at 0.2), then a weighted "
        "blend of compliance/temporal/spatial/manoeuvre/smoothness for runs that both stayed safe "
        "and reached the goal. Reused unchanged from `Evaluation Functions/evaluate_run.py`."
    )
    if len(sim.trajectory) < 4:
        st.info("Run a few steps first -- not enough trajectory yet to score.")
    else:
        result = score_trajectory(
            sim.trajectory, start_xy=(mission.own_ship.x, mission.own_ship.y),
            goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
        )
        verdict = result["verdict"]
        badge_cls = "badge-pass" if verdict == "PASS" else "badge-fail"
        st.markdown(
            f'<span class="badge {badge_cls}">{verdict}</span>'
            f'<span style="font-size:1.3rem; font-weight:700;"> composite = {result["composite_score"]:.3f}</span>',
            unsafe_allow_html=True,
        )
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Safety", result["safety"]["score"],
                  help=f"min CPA {result['safety']['min_cpa_m']} m, passed={result['safety']['passed']}")
        c2.metric("Compliance", result["compliance"]["score"],
                  help=f"{len(result['compliance']['violations'])} violation(s)")
        c3.metric("Temporal", result["temporal"].get("temporal_score"))
        c4.metric("Spatial", result["spatial"].get("spatial_score"))
        c5.metric("Manoeuvre", result["manoeuvre"].get("manoeuvre_score"),
                  help=f"count={result['manoeuvre'].get('manoeuvre_count')}")
        with st.expander("Full result JSON"):
            st.json(result)

        st.divider()
        st.subheader("Degeneracy check (does the agent always say the same thing?)")
        st.caption(
            "Per the DTU-paper finding in Design Ideas.docx: small models can score well on "
            "Imazu-style benchmarks by always answering the same thing. This mission suite includes "
            "\"quiet\" scenarios (q01/q02) where the correct answer is hold_course, alongside 12 "
            "genuine-threat scenarios -- run the agent across several missions and compare the "
            "action distribution here."
        )
        if mission.id.startswith("q"):
            st.write("This is a **quiet** scenario -- the correct recommendation is `hold_course` "
                     "(no real risk of collision).")
        for tgt in mission.targets:
            c = contact_line(sim.own, tgt)
            if c["quiet"]:
                st.caption(f"{c['name']}: quiet (CPA {c['cpa_m']:.0f}m, TCPA {c['tcpa_s']:.0f}s) -- "
                          "no action expected for this contact.")
