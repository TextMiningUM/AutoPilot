"""Chief Engineer "Engine Room" Streamlit page -- condition-monitoring dashboard over
pipeline/chief_engineer_agent_spec.py's deterministic evaluation core + a real (small-scale)
retrieval layer over pipeline/chief_engineer_memory.py. No logic lives in this file beyond
widgets/session-state plumbing -- the same split as 1_Captain_Mission.py.

Styled after a real marine SCADA engine-control-room mimic (dark theme, analog gauges,
ACK/MUTE alarm banner, subsystem nav) while staying deliberately compact / "submarine
control room" (little screen space, dense information, small charts) -- see the tightened
CSS block and CHART_HEIGHT below. IMPORTANT, deliberately NOT a twin-screw mimic: the
reference vessel (design_chief_engineer.md Sec 4.1, Sawada et al. 2021's 106 m cargo
vessel) has ONE main propulsion engine (MAN B&W S-series slow-speed two-stroke crosshead)
direct-coupled to ONE fixed-pitch propeller/shaft line -- unlike a twin-screw CPP ferry/
naval layout, there is no second shaft and no propeller-pitch gauge to show.

Auto-discovered by Streamlit's native multipage convention. Local-safe: pure deterministic
simulation + a small CPU sentence-embedder (no GPU/API key needed) -- first "Ask" query is
slow (embedder load), subsequent ones are fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent  # Basic Simulator/app/
ROOT = APP_DIR.parent                             # Basic Simulator/
for p in (ROOT, ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import datetime as _dt
import json

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from pipeline.chief_engineer_agent_spec import (
    ConditionLimit, ConditionReading, ConditionVerdict,
    all_known_limits, evaluate_condition, recommend_maintenance,
    generate_degradation_trace, generate_synthetic_incident, compute_engine_status,
)
from pipeline import chief_engineer_memory
from pipeline import chief_engineer_remote

st.set_page_config(page_title="Engine Room", page_icon="\U0001F6E2\uFE0F", layout="wide")

_HEADER_IMAGE = ROOT.parent / "Data" / "ChiefEngineer" / "ChiefEngineerManuals" / "man_project_guides" / "EngineRoom.jpg"
if _HEADER_IMAGE.exists():
    st.image(str(_HEADER_IMAGE), width="stretch")

# ── SCADA-mimic dark theme, compact "submarine control room" layout ────────────────────
st.markdown("""
<style>.block-container { padding-top: 2.5rem; padding-bottom: 1rem; }
.ce-clock { color:#7fe3a3; font-family:monospace; font-size:0.95rem; text-align:right; }
div[data-testid="stMetric"] { background: rgba(20,60,40,0.07); border-radius: 6px;
    padding: 0.25rem 0.5rem; border: 1px solid rgba(20,60,40,0.18); }
div[data-testid="stMetricValue"] { font-size: 1.1rem; }
div[data-testid="stMetricLabel"] { font-size: 0.72rem; }
.sev-badge { display:inline-block; padding:0.08rem 0.5rem; border-radius:999px;
    font-weight:600; font-size:0.78rem; }
.sev-nominal { background:#1e7d3220; color:#2fbf50; border:1px solid #2fbf5060; }
.sev-watch { background:#146c9420; color:#19a7ce; border:1px solid #19a7ce60; }
.sev-warning { background:#b4760020; color:#e08700; border:1px solid #e0870060; }
.sev-critical { background:#a0202020; color:#ff5252; border:1px solid #ff525260; }
div[data-testid="stButton"] button { border:1px solid #444; font-weight:700; font-size:0.72rem; }
.ce-alarm-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
.ce-alarm-table th { text-align:left; color:#8aa0b0; font-weight:600; padding:0.2rem 0.5rem;
    border-bottom:1px solid #2a2f35; }
.ce-alarm-table td { padding:0.2rem 0.5rem; border-bottom:1px solid #1d2024; }
.ce-alarm-row-new { background:#2a1414; }
.ce-alarm-row-ack { background:#15181b; color:#8aa0b0; }
hr { margin: 0.4rem 0; }
</style>
""", unsafe_allow_html=True)

COMPONENTS: list[ConditionLimit] = all_known_limits()
N_POINTS = 24
DT_HOURS = 2.0
RATED_SPEED_KN = 12.0  # Sawada et al. (2021) 106m reference vessel's design speed (pipeline/nomoto.py)
RATED_RPM = 105  # illustrative MAN B&W S-series-class MCR shaft rpm, not a cited manual value
CHART_HEIGHT = 110  # tighter than Captain's CHART_HEIGHT=140 -- "little space"
_SEV_ICON = {"nominal": "\U0001F7E2", "watch": "\U0001F535", "warning": "\U0001F7E0", "critical": "\U0001F534"}
_SEV_COLOR = {"nominal": "#2fbf50", "watch": "#19a7ce", "warning": "#e08700", "critical": "#ff5252"}

DRAWINGS_DIR = ROOT.parent / "Data" / "ChiefEngineer" / "ChiefEngineerManuals" / "Drawings"
ENGINE_IMAGE_PATH = (ROOT.parent / "Data" / "ChiefEngineer" / "ChiefEngineerManuals"
                     / "man_project_guides" / "man engine abstract.jpg")


@st.cache_resource
def _load_engine_image() -> Image.Image | None:
    """The user-supplied engine illustration, loaded once per server process."""
    return Image.open(ENGINE_IMAGE_PATH) if ENGINE_IMAGE_PATH.exists() else None


# Normalised (0-1, origin bottom-left) position of each monitored parameter's value chip,
# hand-placed to sit near that part's real label/location in man_project_guides/
# "man engine abstract.jpg" (cylinder heads / turbo inlet / crankcase / fuel / lube oil
# cooler are all labelled in that image; cooling water has no explicit label there, so its
# chip sits near the top-left piping as the closest visual proxy).
CHIP_POSITIONS: dict[tuple[str, str], tuple[float, float]] = {
    ("cylinder_unit", "exhaust_temp_spread_c"): (0.40, 0.72),
    ("turbocharger", "exhaust_gas_temp_c"): (0.66, 0.75),
    ("main_bearing", "temperature_c"): (0.80, 0.45),
    ("fuel_injection", "viscosity_cst"): (0.63, 0.14),
    ("lubricating_oil", "tbn"): (0.51, 0.14),
    ("lubricating_oil", "iron_content_ppm"): (0.46, 0.09),
    ("cooling_water", "temperature_c"): (0.22, 0.68),
}

# Subsystem grouping for the sidebar nav -- mirrors a real engine-control-room menu
# (STATUS / NAPĘD / SG LB / ... in a twin-screw SCADA mimic) scaled down to the single
# engine / single shaft line this reference installation actually has.
SUBSYSTEMS: dict[str, list[tuple[str, str]]] = {
    "MAIN ENGINE": [("main_bearing", "temperature_c"), ("cylinder_unit", "exhaust_temp_spread_c"),
                    ("turbocharger", "exhaust_gas_temp_c")],
    "FUEL": [("fuel_injection", "viscosity_cst")],
    "LUBE OIL": [("lubricating_oil", "tbn"), ("lubricating_oil", "iron_content_ppm")],
    "COOLING": [("cooling_water", "temperature_c")],
}


def _component_key(limit: ConditionLimit) -> tuple[str, str]:
    return (limit.component_id, limit.parameter)


def _seeded_component_offset(limit: ConditionLimit, seed: int) -> int:
    """Stable (non-randomized) per-component seed offset -- Python's built-in hash() is
    randomized per-process for str, so a plain sum-of-codepoints is used instead to keep
    traces reproducible run-to-run for the same (seed, component) pair."""
    key = "".join(_component_key(limit))
    return seed + sum(map(ord, key)) % 1000


def _init_state(seed: int, fault_key: tuple[str, str] | None) -> None:
    traces: dict[tuple[str, str], list[ConditionReading]] = {}
    for limit in COMPONENTS:
        key = _component_key(limit)
        will_fail = key == fault_key
        traces[key] = generate_degradation_trace(
            limit, seed=_seeded_component_offset(limit, seed), n_points=N_POINTS,
            dt_hours=DT_HOURS, will_fail=will_fail,
        )
    st.session_state.ce_traces = traces
    st.session_state.ce_reveal = 1
    st.session_state.ce_seed = seed
    st.session_state.ce_fault_key = fault_key
    st.session_state.ce_ack = set()  # acknowledged (component_id, parameter) alarm keys
    st.session_state.ce_muted = False


def _load() -> None:
    if "ce_traces" not in st.session_state:
        _init_state(seed=42, fault_key=_component_key(COMPONENTS[0]))
    st.session_state.setdefault("ce_subsystem", "ALL")


def _current_verdicts() -> dict[tuple[str, str], ConditionVerdict]:
    out = {}
    for limit in COMPONENTS:
        key = _component_key(limit)
        hist = st.session_state.ce_traces[key][: st.session_state.ce_reveal]
        out[key] = evaluate_condition(hist[-1], limit, hist)
    return out


_load()
verdicts = _current_verdicts()
elapsed_h = (st.session_state.ce_reveal - 1) * DT_HOURS
engine_status = compute_engine_status(list(verdicts.values()), rated_speed_kn=RATED_SPEED_KN, reported_at=elapsed_h)
worst = max(verdicts.values(), key=lambda v: ("nominal", "watch", "warning", "critical").index(v.severity))
speed_fraction = engine_status.max_speed_kn / RATED_SPEED_KN
alarm_keys = [k for k, v in verdicts.items() if v.severity != "nominal"]
n_alerts = len(alarm_keys)
n_unacked = sum(1 for k in alarm_keys if k not in st.session_state.ce_ack)

# ── Dark SCADA-style header bar: vessel/engine identity + ACK/MUTE + clock ──────────────
h_title, h_ack, h_mute, h_clock = st.columns([5, 1, 1, 1.4])
with h_title:
    st.subheader("MAIN PROPULSION")
    st.caption("Single screw \u2022 1 \u00d7 MAN B&W S-series main engine \u2192 1 \u00d7 fixed-pitch propeller"
              " (not twin-screw -- design_chief_engineer.md Sec 4.1)")
with h_ack:
    if st.button(f"ACK ({n_unacked})", use_container_width=True, disabled=n_unacked == 0):
        st.session_state.ce_ack |= set(alarm_keys)
        st.rerun()
with h_mute:
    if st.button("\U0001F507 MUTE" if not st.session_state.ce_muted else "\U0001F515 MUTED",
                 use_container_width=True):
        st.session_state.ce_muted = not st.session_state.ce_muted
        st.rerun()
with h_clock:
    st.markdown(f'<div class="ce-clock">{_dt.datetime.now():%H:%M:%S}<br>'
                f'sim +{elapsed_h:.0f}h</div>', unsafe_allow_html=True)

# ── Sidebar: subsystem nav + drill controls ─────────────────────────────────────────────
with st.sidebar:
    st.caption("Subsystem")
    st.session_state.ce_subsystem = st.radio(
        "Subsystem", ["ALL"] + list(SUBSYSTEMS), index=(["ALL"] + list(SUBSYSTEMS)).index(st.session_state.ce_subsystem),
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Engine Room \u2014 drill controls")
    seed_in = st.number_input("Scenario seed", min_value=0, max_value=9999, value=st.session_state.ce_seed, step=1)
    fault_labels = ["(none \u2014 all nominal)"] + [f"{l.component_id}/{l.parameter}" for l in COMPONENTS]
    fault_keys: list[tuple[str, str] | None] = [None] + [_component_key(l) for l in COMPONENTS]
    cur_idx = fault_keys.index(st.session_state.ce_fault_key) if st.session_state.ce_fault_key in fault_keys else 1
    fault_choice = st.selectbox("Component under stress", fault_labels, index=cur_idx)
    chosen_fault_key = fault_keys[fault_labels.index(fault_choice)]
    if st.button("\U0001F504 Reset / apply", use_container_width=True):
        _init_state(seed=int(seed_in), fault_key=chosen_fault_key)
        st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("\u23ED\uFE0F +2h", use_container_width=True):
        st.session_state.ce_reveal = min(st.session_state.ce_reveal + 1, N_POINTS)
        st.rerun()
    if c2.button("\u23ED\uFE0F\u23ED\uFE0F +8h", use_container_width=True):
        st.session_state.ce_reveal = min(st.session_state.ce_reveal + 4, N_POINTS)
        st.rerun()
    if st.button("\u23ED Jump to end of drill", use_container_width=True):
        st.session_state.ce_reveal = N_POINTS
        st.rerun()

_SEV_ORDER = ("nominal", "watch", "warning", "critical")


def _schematic_figure(verdicts: dict[tuple[str, str], ConditionVerdict], illustrative_rpm: int) -> go.Figure:
    """The real engine illustration (man_project_guides/man engine abstract.jpg) as a
    background image, with every monitored parameter's live value annotated in the
    vicinity of its real equipment location on that picture and coloured by severity --
    so a problem is visible directly on the drawing, not only in a separate table. Falls
    back to a plain message if the image file isn't present on disk."""
    fig = go.Figure()
    img = _load_engine_image()
    fig.update_xaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, 1], fixedrange=True)

    if img is None:
        fig.add_annotation(x=0.5, y=0.5, text=f"Engine image not found:<br>{ENGINE_IMAGE_PATH}",
                            showarrow=False, font=dict(size=12, color="#ff5252"))
        fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    fig.add_layout_image(dict(source=img, xref="x", yref="y", x=0, y=1, sizex=1, sizey=1,
                               xanchor="left", yanchor="top", sizing="stretch", layer="below"))

    def value_chip(x, y, v: ConditionVerdict):
        fig.add_annotation(x=x, y=y, text=f"{v.parameter}<br><b>{v.value:.1f} {v.limit.unit}</b>",
                            showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor="#000",
                            ax=0, ay=-28,
                            font=dict(size=10, color="#0b0f12"),
                            bgcolor=_SEV_COLOR[v.severity], bordercolor="#000", borderwidth=1, borderpad=3)

    worst_main_engine = max((verdicts[k] for k in SUBSYSTEMS["MAIN ENGINE"]), key=lambda v: _SEV_ORDER.index(v.severity))
    if worst_main_engine.severity != "nominal":
        fig.add_shape(type="rect", x0=0.06, y0=0.18, x1=0.98, y1=0.90,
                      line=dict(color=_SEV_COLOR[worst_main_engine.severity], width=4, dash="dot"),
                      fillcolor="rgba(0,0,0,0)")

    for key, (cx, cy) in CHIP_POSITIONS.items():
        value_chip(cx, cy, verdicts[key])

    fig.add_annotation(x=0.07, y=0.30, text=f"<b>{illustrative_rpm} RPM</b>", showarrow=False,
                        font=dict(size=11, color="#143c28"),
                        bgcolor="#bcdccb", bordercolor="#000", borderwidth=1, borderpad=3)

    fig.update_layout(height=460, width=660, margin=dict(l=6, r=6, t=6, b=6),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def _percent_of_critical(v: ConditionVerdict) -> float:
    """Illustrative 0-130% reading of how far `v.value` sits toward its critical bound --
    only used to drive the ring-gauge meters below, not a value agent_spec itself computes."""
    limit = v.limit
    if limit.direction == "above":
        crit = limit.critical_threshold
        pct = (v.value / crit) * 100 if crit else 0.0
    elif limit.direction == "below":
        crit = limit.critical_threshold
        pct = (crit / v.value) * 100 if v.value else 100.0
    else:
        warn_lo, warn_hi = limit.warning_threshold  # type: ignore[misc]
        crit_lo, crit_hi = limit.critical_threshold  # type: ignore[misc]
        mid = (warn_lo + warn_hi) / 2
        if v.value >= mid:
            pct = (v.value - mid) / (crit_hi - mid) * 100 if crit_hi != mid else 0.0
        else:
            pct = (mid - v.value) / (mid - crit_lo) * 100 if mid != crit_lo else 0.0
    return max(0.0, min(130.0, pct))


def _ring_gauge(v: ConditionVerdict) -> go.Figure:
    """A small analog-meter-style donut ring (percent-of-critical, severity-coloured) --
    an "engine room meter" look per the ring/donut widgets the user pointed to."""
    pct = min(100.0, _percent_of_critical(v))
    fig = go.Figure(go.Pie(values=[pct, max(0.0, 100 - pct)], hole=0.72, sort=False,
                           marker=dict(colors=[_SEV_COLOR[v.severity], "#1d2024"]),
                           textinfo="none", showlegend=False))
    fig.update_layout(height=92, margin=dict(l=2, r=2, t=2, b=2), paper_bgcolor="rgba(0,0,0,0)",
                       annotations=[dict(text=f"{pct:.0f}%", x=0.5, y=0.5, font_size=11, showarrow=False,
                                          font_color="#eef3f7")])
    return fig


# ── Status metrics ──────────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Elapsed", f"{elapsed_h:.0f} h")
m2.metric("Worst status", f"{_SEV_ICON[worst.severity]} {worst.severity}")
m3.metric("Max speed", f"{engine_status.max_speed_kn:.1f} / {RATED_SPEED_KN:.0f} kn")
m4.metric("Active alerts", str(n_alerts))
if engine_status.fault:
    st.caption(f"\u26A0\uFE0F {engine_status.fault}")

tab_monitor, tab_chat = st.tabs(["\U0001F5A5\uFE0F Monitor", "\U0001F4AC Ask the Chief Engineer"])

# ── Schematic (left) + 2x4 ring-gauge meter bank (right) ────────────────────────────────
_visible = SUBSYSTEMS.get(st.session_state.ce_subsystem) if st.session_state.ce_subsystem != "ALL" else None
VISIBLE_COMPONENTS = [l for l in COMPONENTS if _visible is None or _component_key(l) in _visible]

with tab_monitor:
    schem_col, gauge_col, detail_col = st.columns([3, 1.6, 2.4])
    with schem_col:
        illustrative_rpm = round(RATED_RPM * max(0.0, min(1.0, speed_fraction)))
        st.plotly_chart(_schematic_figure(verdicts, illustrative_rpm), use_container_width=False,
                        config={"displayModeBar": False})
        st.caption("Mimic diagram, not to scale -- chip colour = severity at that measurement point; "
                   "shaft RPM is illustrative (derived from the speed cap), not a modelled engine simulation.")
    with gauge_col:
        st.caption("Meter bank \u2014 % of critical limit")
        grid = VISIBLE_COMPONENTS[:8]  # 4 rows x 2 columns -- narrower bank, room for detail_col alongside
        for row_start in (0, 2, 4, 6):
            row_cols = st.columns(2)
            for col, limit in zip(row_cols, grid[row_start:row_start + 2]):
                v = verdicts[_component_key(limit)]
                col.plotly_chart(_ring_gauge(v), use_container_width=True, config={"displayModeBar": False})
                col.caption(f"{limit.component_id}  \n{v.value:.1f} {limit.unit}")

    # ── Component detail (trend chart drill-down) -- sits beside the meter bank ─────────
    with detail_col:
        st.caption("Component detail")
        sel_label = st.selectbox("", [f"{l.component_id}/{l.parameter}" for l in VISIBLE_COMPONENTS], label_visibility="collapsed")
        sel_limit = VISIBLE_COMPONENTS[[f"{l.component_id}/{l.parameter}" for l in VISIBLE_COMPONENTS].index(sel_label)]
        sel_key = _component_key(sel_limit)
        sel_hist = st.session_state.ce_traces[sel_key][: st.session_state.ce_reveal]
        sel_verdict = verdicts[sel_key]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r.timestamp_s / 3600 for r in sel_hist], y=[r.value for r in sel_hist],
                                  mode="lines+markers", line=dict(width=2), marker=dict(size=4), showlegend=False))
        if sel_limit.direction == "range":
            lo, hi = sel_limit.warning_threshold  # type: ignore[misc]
            fig.add_hrect(y0=lo, y1=hi, fillcolor="green", opacity=0.08, line_width=0)
        else:
            fig.add_hline(y=sel_limit.warning_threshold, line_dash="dot", line_color="orange", line_width=1)
            fig.add_hline(y=sel_limit.critical_threshold, line_dash="dot", line_color="red", line_width=1)
        fig.update_layout(height=CHART_HEIGHT, margin=dict(l=4, r=4, t=4, b=4),
                           xaxis_title=None, yaxis_title=sel_limit.unit)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        badge_class = f"sev-{sel_verdict.severity}"
        st.markdown(f'<span class="sev-badge {badge_class}">{sel_verdict.severity}</span> '
                    f'trend {sel_verdict.trend_slope_per_h:+.2f}/h', unsafe_allow_html=True)
        rec = recommend_maintenance(sel_verdict)
        if rec:
            st.caption(f"\u2192 {rec.action}")
        st.caption(f"Source: {sel_limit.source_citation}")

        st.caption("Alarm list" + (" \U0001F507 (muted)" if st.session_state.ce_muted else ""))
        if not alarm_keys:
            st.markdown('<span class="sev-badge sev-nominal">no active alarms</span>', unsafe_allow_html=True)
        else:
            rows_html = []
            for key in sorted(alarm_keys, key=lambda k: -("nominal", "watch", "warning", "critical").index(verdicts[k].severity)):
                v = verdicts[key]
                rec = recommend_maintenance(v)
                acked = key in st.session_state.ce_ack
                row_class = "ce-alarm-row-ack" if acked else "ce-alarm-row-new"
                rows_html.append(
                    f'<tr class="{row_class}"><td><span class="sev-badge sev-{v.severity}">{v.severity}</span></td>'
                    f'<td>{key[0]}/{key[1]}</td><td>{rec.action if rec else ""}</td>'
                    f'<td>{elapsed_h:.0f}h</td><td>{"ACK" if acked else "NEW"}</td></tr>'
                )
            st.markdown(
                '<table class="ce-alarm-table"><tr><th>Sev</th><th>Source</th><th>Message</th>'
                '<th>Time</th><th>Status</th></tr>' + "".join(rows_html) + "</table>",
                unsafe_allow_html=True,
            )

        # ── Synthetic incident drill (Phase 2) ──────────────────────────────────────────
        with st.expander("\U0001F9EA Synthetic incident drill", expanded=False):
            ic1, ic2 = st.columns(2)
            inc_seed = ic1.number_input("Incident seed", min_value=0, max_value=9999, value=1, key="inc_seed")
            caught = ic2.toggle("Caught in time", value=True, key="inc_caught")
            missed = generate_synthetic_incident(sel_limit, seed=int(inc_seed), caught_in_time=False)
            caught_inc = generate_synthetic_incident(sel_limit, seed=int(inc_seed), caught_in_time=True)
            shown = caught_inc if caught else missed
            st.caption(shown.narrative)

        # ── Drawings (illustrative, honest provenance -- design_chief_engineer.md Sec 4.3) ──
        with st.expander("\U0001F4D0 Reference drawings (illustrative)", expanded=False):
            manifest_path = DRAWINGS_DIR / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                st.caption(manifest.get("_comment", "")[:220] + "\u2026")
                cols = st.columns(len(manifest["files"]))
                for col, entry in zip(cols, manifest["files"]):
                    img_path = DRAWINGS_DIR / entry["file"]
                    if img_path.exists():
                        col.image(str(img_path), use_container_width=True)
                    col.caption(f"{entry['title']} \u2014 *{entry['source_vessel']}* (illustrative)")
            else:
                st.caption("No drawings manifest found yet.")

# ── Ask the Chief Engineer (real retrieval + optional real generation via the cloud) ───
with tab_chat:
    remote_up = chief_engineer_remote.is_remote_server_up()
    status = "\U0001F7E2 cloud model reachable" if remote_up else "\U0001F534 cloud model not reachable (SSH tunnel down?)"
    st.caption(status)
    use_remote = st.toggle("Generate answer with fine-tuned model (cloud)", value=remote_up, disabled=not remote_up)
    query = st.text_input("Question", placeholder="e.g. what causes main bearing overheating?",
                           label_visibility="collapsed")
    if query:
        if use_remote and remote_up:
            with st.spinner("Asking the fine-tuned Chief Engineer model (cloud)..."):
                try:
                    answer = chief_engineer_remote.ask_chief_engineer_remote(query)
                    st.markdown(answer)
                except (ConnectionError, RuntimeError) as e:
                    st.error(str(e))
        with st.expander("Retrieved known-issue sources", expanded=not (use_remote and remote_up)):
            hits = chief_engineer_memory.retrieve(query, k=3)
            for h in hits:
                tag = "placeholder" if h.is_placeholder else f"score {h.score:.2f}"
                st.markdown(f"**{h.source}** _({tag})_")
                st.caption(h.text)

