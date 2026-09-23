"""Dashboard for app/sweep_llm_params.py. Builds its leaderboard/detail tables by scanning
Data/missions/_llm_runs/ directly for every {mission_id}__{config}[__{tag}].json run log
(the mission id is a fixed prefix, config/tag is the variable part at the end -- exactly
app.llm_runs.run_log_path's own naming scheme) and scoring each on the spot with the same
composite evaluate_run.py/sweep_llm_params.py use elsewhere, rather than trusting
_sweep_summary.json's cache -- that cache only ever gains rows (sweep_llm_params.py's
_save_row merges, never removes), so it kept showing long-deleted/moved run logs and could
lag behind ones already sitting on disk. Also reads _sweep_status.json (updated right
before every job starts, cleared when the sweep finishes) for whichever (mission, config) is
actually in flight right now. Redraws on demand via a manual "Refresh" button -- entirely
decoupled from the sweep process itself (read-only, safe to run alongside it on a different
port while the sweep keeps computing). Deliberately NOT an auto-refresh (neither a
<meta http-equiv="refresh"> full page reload, which destroys the whole browser session and
collapses every expander, nor a timed st.fragment(run_every=...), which was too noisy at a
5-10s cadence) -- the user just clicks Refresh when they want the latest state.

Run (separate terminal/port from the main app):
    streamlit run app/sweep_dashboard.py --server.port 8510
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (ROOT, ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.evaluation import score_trajectory
from app.llm_runs import parse_run_filename
from app.missions import list_mission_ids, load_mission
from app.simulation import VesselConstraints

# Kept as a plain literal (matching app.agents.MODEL_CONFIGS's keys) instead of importing
# app.agents itself -- that module pulls in torch/transformers/sentence-transformers at
# import time, which this read-only dashboard has no need for. Must be kept in sync by
# hand whenever a new config is added to app.agents.MODEL_CONFIGS (2026-09-23: was
# missing v7_super_rag/v8_super_cot_pg/v9_super_all -- those runs existed on disk but
# never showed up here).
CONFIG_NAMES = [
    "bare_qwen", "v0_base", "v1_rag", "v2_cot", "v3_rag_cot",
    "v4_pg", "v5_pg_incident", "v6_pg_scenario",
    "v7_super_rag", "v8_super_cot_pg", "v9_super_all",
    "v1_rag_simple", "v7_simple_rag",
]

RUNS_DIR = ROOT / "Data" / "missions" / "_llm_runs"
STATUS_FILE = RUNS_DIR / "_sweep_status.json"
_DEFAULT_MIN_CPA_M = VesselConstraints().min_cpa_m

st.set_page_config(page_title="LLM sweep dashboard", layout="wide")
title_cols = st.columns([5, 1])
with title_cols[0]:
    st.title("\U0001F4CA LLM config sweep \u2014 progress")
with title_cols[1]:
    st.button("\U0001F504 Refresh", width="stretch")

MISSIONS = list_mission_ids()
CONFIGS = CONFIG_NAMES
TOTAL_JOBS = len(MISSIONS) * len(CONFIGS)
MISSION_OBJS = {m: load_mission(m) for m in MISSIONS}  # cheap: just json + dataclasses


def _generated_at_index() -> list[tuple[datetime, Path]]:
    """Peeks just the top-level "generated_at" out of every run log in RUNS_DIR (across ALL
    missions/configs, not just one), sorted chronologically -- used to ESTIMATE wall-clock
    latency for older logs that predate run_llm_scenario.py recording latency_s directly.
    Within one sequential sweep (one job computed after another on the same host, writing
    its log the instant it finishes), the gap between a job's own generated_at and the
    immediately PRECEDING job's generated_at approximates that job's own compute time.
    "generated_at" is embedded IN the JSON content itself, so unlike file mtime it survives
    being scp'd between machines unchanged -- but the estimate is still only valid for
    consecutive entries actually produced back-to-back by the SAME sequential process (see
    the sanity cap in _estimate_latency below)."""
    entries: list[tuple[datetime, Path]] = []
    for path in RUNS_DIR.glob("*.json"):
        if path.name.startswith("_sweep_"):
            continue
        try:
            gen_at = json.loads(path.read_text(encoding="utf-8")).get("generated_at")
            ts = datetime.fromisoformat(gen_at) if gen_at else None
        except (json.JSONDecodeError, OSError, ValueError):
            ts = None
        if ts is not None:
            entries.append((ts, path))
    entries.sort(key=lambda e: e[0])
    return entries


def _estimate_latency(path: Path, index: list[tuple[datetime, Path]]) -> float | None:
    """Gap to the immediately preceding entry in _generated_at_index()'s chronological
    order -- None if `path` is the very first entry overall, or if the gap is implausibly
    large (> 1h, almost certainly a different sweep session/host rather than this job's own
    compute time) or non-positive (clock skew between hosts)."""
    for i, (ts, p) in enumerate(index):
        if p != path:
            continue
        if i == 0:
            return None
        delta = (ts - index[i - 1][0]).total_seconds()
        return delta if 0 < delta <= 3600 else None
    return None


def _score_log(mission_id: str, log: dict, tag: str, path: Path,
               gen_at_index: list[tuple[datetime, Path]], weights: str = "W0_base") -> dict:
    """Same composite scoring sweep_llm_params.py's score_one() applies to a freshly
    computed run -- duplicated here (rather than imported) because sweep_llm_params.py
    pulls in app.run_llm_scenario -> app.agents -> torch/transformers, which this
    read-only, always-import-light dashboard must never load.

    Prefers the log's OWN embedded "evaluation" (written by run_llm_scenario.py's run_one()
    at save time) over recomputing -- faster (skips the CSV round-trip through
    evaluate_run.py) and only recomputes for older logs from before that field existed."""
    mission = MISSION_OBJS[mission_id]
    result = log.get("evaluation") or score_trajectory(
        log["trajectory"], start_xy=(mission.own_ship.x, mission.own_ship.y),
        goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
        safe_distance_m=_DEFAULT_MIN_CPA_M,
        llm_violations=(log.get("colreg_llm_check") or {}).get("violations"),
        llm_compliance_score=(log.get("colreg_llm_check") or {}).get("compliance_score"),
    )
    latency_s = log.get("latency_s")
    latency_is_estimate = latency_s is None
    if latency_is_estimate:
        latency_s = _estimate_latency(path, gen_at_index)
    return {
        "config": log.get("config"), "weights": log.get("weights") or weights,
        "tag": log.get("tag", tag),
        "composite_score": result["composite_score"], "verdict": result["verdict"],
        "safety": result["safety"], "compliance": result["compliance"],
        "temporal": result["temporal"], "spatial": result["spatial"],
        "manoeuvre": result["manoeuvre"], "latency_s": latency_s,
        "latency_is_estimate": latency_is_estimate,
        "colreg_llm_check": log.get("colreg_llm_check"),
    }


def _scan_mission_runs(mission_id: str, gen_at_index: list[tuple[datetime, Path]]) -> dict[str, dict]:
    """Globs RUNS_DIR for every {mission_id}__*.json and scores each file directly -- the
    single source of truth for what's actually on disk RIGHT NOW, instead of
    _sweep_summary.json's append-only cache. Filenames are parsed via
    app.llm_runs.parse_run_filename(), which understands both the current
    {mission}__{config}__{weights}__{tag}.json form and the older 2/3-segment forms
    written before the weights axis existed. When more than one tag produced a log for the
    same config, the most recently modified file wins (whatever's actually current)."""
    prefix = f"{mission_id}__"
    latest_mtime: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for path in RUNS_DIR.glob(f"{prefix}*.json"):
        parsed = parse_run_filename(path)
        config, weights, tag = parsed["config"], parsed["weights"], parsed["tag"]
        if config not in CONFIGS:
            continue
        mtime = path.stat().st_mtime
        if config in latest_mtime and latest_mtime[config] >= mtime:
            continue
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        latest_mtime[config] = mtime
        rows[config] = _score_log(mission_id, log, tag, path, gen_at_index, weights)
    return rows



def _colreg_status(r: dict) -> str:
    """Narrow-column status for the end-of-mission Anthropic Claude COLREG compliance
    check (app.evaluation.llm_compliance_check, run automatically by run_llm_scenario.py's
    run_one() -- see its colreg_llm_check field): compact enough for a dataframe cell, full
    violations/compliant-actions text lives in the "View details" popover instead
    (_describe_run) since it can be long. "--" covers both older logs generated before this
    check existed at all, and logs where it was explicitly skipped (--no-colreg-check) --
    NOT the same as a checked 0.0 (audited and found non-compliant)."""
    check = r.get("colreg_llm_check")
    if not check or not check.get("checked"):
        return "\u2014"
    score = check.get("compliance_score")
    violations = check.get("violations") or []
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
    return f"\u2705 {score_str}" if not violations else f"\u26A0\uFE0F {score_str} ({len(violations)})"


def _axis_cols(r: dict) -> dict:
    """Flattens one scored row's per-axis breakdown into the handful of columns the
    leaderboard/per-mission tables both show next to the composite score -- safety/
    compliance/temporal/spatial/manoeuvre are each already a 0-1 axis score. latency is the
    whole run's wall-clock compute time: exact (from run_llm_scenario.py's own latency_s)
    for logs generated after that field was added, or a "~"-prefixed ESTIMATE (derived from
    the gap to the previous log's generated_at, see _estimate_latency) for older ones, or
    None if no estimate was possible either (first log ever, or an implausible gap). colreg
    is the narrow Claude-compliance-check status -- see _colreg_status."""
    latency = r.get("latency_s")
    if latency is None:
        latency_str = None
    elif r.get("latency_is_estimate"):
        latency_str = f"~{latency:.0f}s"
    else:
        latency_str = f"{latency:.0f}s"
    return {
        "safety": r["safety"]["score"], "compliance": r["compliance"]["score"],
        "temporal": r["temporal"]["temporal_score"], "spatial": r["spatial"]["spatial_score"],
        "manoeuvre": r["manoeuvre"]["manoeuvre_score"], "latency": latency_str,
        "colreg": _colreg_status(r),
    }


def _read_current_job() -> tuple[tuple[str, str] | None, str]:
    """Reads STATUS_FILE (written by sweep_llm_params.py's sweep() right before each job,
    cleared when the whole sweep finishes) for whichever (mission, config) is ACTUALLY being
    computed right now, plus a human-readable "Xs/Xm ago" age string -- replaces the old
    approach of guessing the current job as the first (mission, config) gap in strict
    q01-first order, which pointed at a long-finished mission (or one nobody on THIS host is
    even running) as soon as jobs complete out of order, e.g. a local run covering only
    s11/s12 while a separate cloud process works through q01, q02, ... in parallel, each
    writing its own copy of SUMMARY_FILE/STATUS_FILE. Returns (None, "") if no sweep is
    currently running on this host (file absent -- most recent sweep finished or none ever
    ran here)."""
    if not STATUS_FILE.exists():
        return None, ""
    try:
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, ""
    age_s = max(0.0, time.time() - STATUS_FILE.stat().st_mtime)
    age = f"{age_s:.0f}s ago" if age_s < 60 else f"{age_s / 60:.0f}m ago"
    return (status["mission"], status["config"]), age



def _colreg_audit_markdown(r: dict) -> str:
    """Standalone COLREG-audit-only text (score + every violation/compliant-action
    explanation) -- factored out of _describe_run() so the leaderboard can show it in its
    own compact popover (r['colreg'] narrow-column status is the summary, this is the full
    text) without dragging in the rest of the safety/temporal/spatial/manoeuvre readout."""
    lines = ["**Claude COLREG compliance check** (end-of-mission, Anthropic API -- full audit)"]
    check = r.get("colreg_llm_check")
    if not check or not check.get("checked"):
        reason = (check or {}).get("error") or "not run for this log (older log, or --no-colreg-check)"
        lines.append(f"- Not checked -- {reason}.")
        return "\n".join(lines)
    violations = check.get("violations") or []
    compliant_actions = check.get("compliant_actions") or []
    score = check.get("compliance_score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
    if not violations:
        lines.append(f"- Compliance score **{score_str}** -- Claude found no COLREG "
                    f"violations ({len(compliant_actions)} manoeuvre(s) audited as correct).")
    else:
        lines.append(f"- Compliance score **{score_str}** -- Claude flagged "
                    f"{len(violations)} violation(s):")
        for v in violations:
            lines.append(f"  - {v}")
    if compliant_actions:
        lines.append(f"- Correctly handled ({len(compliant_actions)}):")
        for c in compliant_actions:
            lines.append(f"  - {c}")
    return "\n".join(lines)


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

    lines.append("")
    lines.append(_colreg_audit_markdown(r))
    return "\n".join(lines)


def _render() -> None:
    gen_at_index = _generated_at_index()
    rows_by_mission = {m: _scan_mission_runs(m, gen_at_index) for m in MISSIONS}
    done = sum(len(rows) for rows in rows_by_mission.values())

    current_job, current_job_age = _read_current_job()

    top_cols = st.columns([3, 1])
    with top_cols[0]:
        st.progress(min(1.0, done / TOTAL_JOBS) if TOTAL_JOBS else 0.0,
                   text=f"{done}/{TOTAL_JOBS} jobs complete")
    with top_cols[1]:
        if current_job:
            st.metric("In progress", f"{current_job[0]} / {current_job[1]}",
                      help=f"last updated {current_job_age}")
        else:
            st.metric("In progress", "\u2014 (none running on this host)")

    st.caption(f"Scans {RUNS_DIR.relative_to(ROOT)} directly for "
              "{mission}__{config}[__{tag}].json \u2022 read-only \u2022 click Refresh "
              "above for the latest state")

    st.divider()
    st.subheader("Leaderboard (best config per mission so far)")
    leaderboard = []
    best_row_by_mission: dict[str, dict] = {}
    for mission_id in MISSIONS:
        rows = rows_by_mission[mission_id]
        if not rows:
            leaderboard.append({
                "mission": mission_id, "done": "0/8", "best_config": "\u2014",
                "composite": None, "verdict": "\u2014", "safety": None, "compliance": None,
                "temporal": None, "spatial": None, "manoeuvre": None, "latency": None,
                "colreg": "\u2014",
            })
            continue
        best = max(rows.values(), key=lambda r: r["composite_score"])
        best_row_by_mission[mission_id] = best
        leaderboard.append({
            "mission": mission_id, "done": f"{len(rows)}/{len(CONFIGS)}",
            "best_config": best["config"], "composite": best["composite_score"],
            "verdict": best["verdict"], **_axis_cols(best),
        })
    st.dataframe(leaderboard, width="stretch", hide_index=True)

    # st.dataframe has no per-cell popover, so the "colreg" column's full audit text (can be
    # long -- see _colreg_audit_markdown) lives in a compact strip of buttons right below the
    # table instead, one per mission that actually has a checked audit -- clicking one pops
    # up that mission's best-config score + every violation/compliant-action explanation.
    checked_missions = [m for m in MISSIONS
                       if ((best_row_by_mission.get(m) or {}).get("colreg_llm_check") or {}).get("checked")]
    if checked_missions:
        st.caption("\U0001F4C4 View full COLREG audit (best config per mission):")
        audit_cols = st.columns(min(len(checked_missions), 7))
        for i, mission_id in enumerate(checked_missions):
            with audit_cols[i % len(audit_cols)]:
                best = best_row_by_mission[mission_id]
                score = (best.get("colreg_llm_check") or {}).get("compliance_score")
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
                with st.popover(f"{mission_id}  {score_str}"):
                    st.markdown(_colreg_audit_markdown(best))
    elif best_row_by_mission:
        # Otherwise this whole section just silently disappears with no explanation --
        # looks like a broken/missing feature rather than "no logs have been audited yet".
        st.caption("\U0001F4C4 No mission has a saved COLREG audit yet -- these logs predate "
                  "the check, or were generated with --no-colreg-check. Re-run with the audit "
                  "enabled (the default) to populate it.")

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
                    table.append({
                        "config": config, "status": status, "composite": None,
                        "verdict": None, "safety": None, "compliance": None,
                        "temporal": None, "spatial": None, "manoeuvre": None,
                        "latency": None, "colreg": "\u2014",
                    })
                else:
                    table.append({
                        "config": config, "status": "\u2705 done",
                        "composite": r["composite_score"], "verdict": r["verdict"],
                        **_axis_cols(r),
                    })
            st.dataframe(table, width="stretch", hide_index=True)

            done_configs = [c for c in CONFIGS if rows.get(c) is not None]
            if done_configs:
                picked = st.selectbox("View details for:", done_configs,
                                      key=f"detail_pick_{mission_id}")
                r = rows[picked]
                with st.popover(f"\U0001F4C4 {picked} \u2014 details"):
                    m_cols = st.columns(7)
                    m_cols[0].metric("Safety", r["safety"]["score"])
                    m_cols[1].metric("Compliance", r["compliance"]["score"])
                    m_cols[2].metric("Temporal", r["temporal"]["temporal_score"])
                    m_cols[3].metric("Spatial", r["spatial"]["spatial_score"])
                    m_cols[4].metric("Manoeuvre", r["manoeuvre"]["manoeuvre_score"])
                    latency = r.get("latency_s")
                    if latency is None:
                        latency_str = "\u2014"
                    elif r.get("latency_is_estimate"):
                        latency_str = f"~{latency:.0f}s"
                    else:
                        latency_str = f"{latency:.0f}s"
                    m_cols[5].metric("Latency", latency_str)
                    m_cols[6].metric("COLREG (Claude)", _colreg_status(r))
                    st.markdown(_describe_run(r))


_render()
