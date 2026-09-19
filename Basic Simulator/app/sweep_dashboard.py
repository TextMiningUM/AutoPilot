"""Dashboard for app/sweep_llm_params.py. Reads Data/missions/_llm_runs/_sweep_summary.json
(updated after every completed job by the sweep script) and redraws on demand via a manual
"Refresh" button -- entirely decoupled from the sweep process itself (read-only, safe to run
alongside it on a different port while the sweep keeps computing). Deliberately NOT an
auto-refresh (neither a <meta http-equiv="refresh"> full page reload, which destroys the
whole browser session and collapses every expander, nor a timed st.fragment(run_every=...),
which was too noisy at a 5-10s cadence) -- the user just clicks Refresh when they want the
latest state.

Run (separate terminal/port from the main app):
    streamlit run app/sweep_dashboard.py --server.port 8510
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (ROOT, ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.missions import list_mission_ids, load_mission

# Kept as a plain literal (matching app.agents.MODEL_CONFIGS's keys) instead of importing
# app.agents itself -- that module pulls in torch/transformers/sentence-transformers at
# import time, which this read-only dashboard has no need for.
CONFIG_NAMES = [
    "bare_qwen", "v0_base", "v1_rag", "v2_cot", "v3_rag_cot",
    "v4_pg", "v5_pg_incident", "v6_pg_scenario",
]

SUMMARY_FILE = ROOT / "Data" / "missions" / "_llm_runs" / "_sweep_summary.json"

st.set_page_config(page_title="LLM sweep dashboard", layout="wide")
title_cols = st.columns([5, 1])
with title_cols[0]:
    st.title("\U0001F4CA LLM config sweep \u2014 progress")
with title_cols[1]:
    st.button("\U0001F504 Refresh", width="stretch")

MISSIONS = list_mission_ids()
CONFIGS = CONFIG_NAMES
TOTAL_JOBS = len(MISSIONS) * len(CONFIGS)
# Same (mission-outer, config-inner) order sweep_llm_params.py's sweep() iterates in, so the
# first (mission, config) pair missing from the summary IS the job currently in flight.
JOB_ORDER = [(m, c) for m in MISSIONS for c in CONFIGS]
MISSION_OBJS = {m: load_mission(m) for m in MISSIONS}  # cheap: just json + dataclasses


def _describe_run(r: dict) -> str:
    """Turns evaluate_run.py's full result dict into a short, plain-text readout of what
    actually happened -- verdict/safety/compliance/efficiency/manoeuvres -- for the details
    popup, instead of dumping the raw nested JSON."""
    safety, compliance = r["safety"], r["compliance"]
    temporal, spatial, man = r["temporal"], r["spatial"], r["manoeuvre"]
    lines = [f"**{r['verdict']}** (composite score **{r['composite_score']:.2f}**)", ""]

    if safety["passed"]:
        lines.append(f"- No collision \u2014 closest approach to any target was "
                    f"{safety['min_cpa_m']:.0f}m.")
    else:
        lines.append(f"- \u26A0\uFE0F Collision occurred (closest approach "
                    f"{safety['min_cpa_m']:.0f}m).")

    if compliance["violations"]:
        lines.append(f"- {len(compliance['violations'])} COLREG violation(s) flagged: "
                    + "; ".join(compliance["violations"]))
    else:
        lines.append("- No COLREG violations flagged.")

    if temporal["arrived"]:
        lines.append(f"- Reached the goal in {temporal['time_actual_s']:.0f}s "
                    f"({temporal['time_ratio']:.2f}\u00d7 the direct-line time), covering "
                    f"{spatial['path_length_m']:.0f}m ({spatial['path_ratio']:.2f}\u00d7 the "
                    f"direct distance).")
    else:
        lines.append(f"- Did **not** reach the goal within the step budget (travelled "
                    f"{spatial['path_length_m']:.0f}m).")

    if man["manoeuvre_count"] == 0:
        lines.append("- Held course/speed the whole run \u2014 no manoeuvres.")
    else:
        lines.append(f"- {man['manoeuvre_count']} course/speed change(s), smoothness "
                    f"{man['smoothness_score']:.2f} (avg heading rate "
                    f"{man['mean_abs_heading_rate']:.2f}\u00b0/s, avg speed rate "
                    f"{man['mean_abs_speed_rate']:.3f} m/s\u00b2).")
    return "\n".join(lines)


def _render() -> None:
    summary: dict[str, list[dict]] = (
        json.loads(SUMMARY_FILE.read_text(encoding="utf-8")) if SUMMARY_FILE.exists() else {}
    )
    rows_by_mission = {m: {r["config"]: r for r in summary.get(m, [])} for m in MISSIONS}
    done = sum(len(rows) for rows in rows_by_mission.values())

    current_job = next(
        ((m, c) for m, c in JOB_ORDER if c not in rows_by_mission.get(m, {})), None
    )

    top_cols = st.columns([3, 1])
    with top_cols[0]:
        st.progress(min(1.0, done / TOTAL_JOBS) if TOTAL_JOBS else 0.0,
                   text=f"{done}/{TOTAL_JOBS} jobs complete")
    with top_cols[1]:
        if current_job:
            st.metric("In progress", f"{current_job[0]} / {current_job[1]}")
        else:
            st.metric("In progress", "\u2014 (all done)")

    st.caption(f"Reads {SUMMARY_FILE.relative_to(ROOT)} \u2022 read-only \u2022 click "
              "Refresh above for the latest state")

    st.divider()
    st.subheader("Leaderboard (best config per mission so far)")
    leaderboard = []
    for mission_id in MISSIONS:
        rows = rows_by_mission[mission_id]
        if not rows:
            leaderboard.append({"mission": mission_id, "done": "0/8", "best_config": "\u2014",
                                "composite": None, "verdict": "\u2014"})
            continue
        best = max(rows.values(), key=lambda r: r["composite_score"])
        leaderboard.append({
            "mission": mission_id, "done": f"{len(rows)}/{len(CONFIGS)}",
            "best_config": best["config"], "composite": best["composite_score"],
            "verdict": best["verdict"],
        })
    st.dataframe(leaderboard, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Per-mission detail (all variations)")
    for mission_id in MISSIONS:
        rows = rows_by_mission[mission_id]
        # Explicit `key=` (stable across reruns) instead of relying on the auto-key derived
        # from the label -- the label's "(done/8)" count changes as jobs complete, which
        # would otherwise make Streamlit treat it as a brand-new expander each time and
        # collapse it back shut.
        with st.expander(f"{mission_id}  ({len(rows)}/{len(CONFIGS)} configs done)",
                         expanded=False, key=f"exp_{mission_id}"):
            # Nested st.expander isn't allowed inside another expander, so mission brief and
            # per-config detail below use a checkbox/selectbox to reveal on click instead.
            if st.checkbox("\U0001F4CB Show mission brief", key=f"brief_{mission_id}"):
                st.markdown(MISSION_OBJS[mission_id].as_text())
            st.divider()
            table = []
            for config in CONFIGS:
                r = rows.get(config)
                is_current = current_job == (mission_id, config)
                if r is None:
                    status = "\U0001F504 running" if is_current else "\u23F3 pending"
                    table.append({"config": config, "status": status, "composite": None,
                                 "verdict": None})
                else:
                    table.append({"config": config, "status": "\u2705 done",
                                 "composite": r["composite_score"], "verdict": r["verdict"]})
            st.dataframe(table, width="stretch", hide_index=True)

            done_configs = [c for c in CONFIGS if rows.get(c) is not None]
            if done_configs:
                picked = st.selectbox("View details for:", done_configs,
                                      key=f"detail_pick_{mission_id}")
                r = rows[picked]
                with st.popover(f"\U0001F4C4 {picked} \u2014 details"):
                    m_cols = st.columns(5)
                    m_cols[0].metric("Safety", r["safety"]["score"])
                    m_cols[1].metric("Compliance", r["compliance"]["score"])
                    m_cols[2].metric("Temporal", r["temporal"]["temporal_score"])
                    m_cols[3].metric("Spatial", r["spatial"]["spatial_score"])
                    m_cols[4].metric("Manoeuvre", r["manoeuvre"]["manoeuvre_score"])
                    st.markdown(_describe_run(r))


_render()
