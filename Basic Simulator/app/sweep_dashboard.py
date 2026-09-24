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
import hashlib
import json
import subprocess
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

from app.evaluation import compliance_finding_parts, score_trajectory
from app.llm_runs import parse_run_filename
from app.missions import list_mission_ids, load_mission
from app.simulation import VesselConstraints
from core import review_path, safe_write_jsonl

# Kept as a plain literal (matching app.agents.MODEL_CONFIGS's keys) instead of importing
# app.agents itself -- that module pulls in torch/transformers/sentence-transformers at
# import time, which this read-only dashboard has no need for. Must be kept in sync by
# hand whenever a new config is added to app.agents.MODEL_CONFIGS.
CONFIG_NAMES = [
    "bare_qwen", "v0_base", "v1_rag", "v2_cot", "v3_rag_cot",
    "v4_pg", "v5_pg_incident", "v6_pg_scenario",
    "v7_super_rag", "v8_super_cot_pg", "v9_super_all",
    "v10_super_colreg_rag", "v11_super_colreg_rag_cot",
]

RUNS_DIR = ROOT / "Data" / "missions" / "_llm_runs"
STATUS_FILE = RUNS_DIR / "_sweep_status.json"
_DEFAULT_MIN_CPA_M = VesselConstraints().min_cpa_m

AUDIT_SCRIPT = ROOT / "_analysis" / "audit_runs.py"
AUDIT_OUT_DIR = ROOT / "_analysis" / "audit"

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
        checkpoints=log.get("checkpoints"),
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
    """Narrow-column status for the optional Anthropic Claude plain-language explanation
    (app.evaluation.llm_compliance_check, compliance-rebuild STAP 4 -- opt-in via
    run_llm_scenario.py's --explain flag, off by default) of the deterministic compliance
    findings already scored in r['compliance']['score']. "--" covers both older logs
    generated before this field existed and logs where --explain was not passed."""
    check = r.get("colreg_llm_check")
    if not check or not check.get("checked"):
        return "\u2014"
    n = len(check.get("explanations") or [])
    return f"\U0001F4AC {n}" if n else "\u2705 none"


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
    lines = ["**Claude explanation** (compliance-rebuild STAP 4 -- plain-language, opt-in, "
            "never affects the score)"]
    check = r.get("colreg_llm_check")
    if not check or not check.get("checked"):
        reason = (check or {}).get("error") or "not requested for this log (older log, or no --explain)"
        lines.append(f"- Not checked -- {reason}.")
        return "\n".join(lines)
    explanations = check.get("explanations") or []
    if not explanations:
        lines.append("- No findings to explain (deterministic compliance score was clean).")
    else:
        lines.append(f"- {len(explanations)} finding(s) explained:")
        for e in explanations:
            lines.append(f"  - {e}")
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

    if compliance["breakdown"]:
        parts = [compliance_finding_parts(e) for e in compliance["breakdown"]]
        lines.append(f"- {len(compliance['breakdown'])} deterministic compliance finding(s): "
                    + "; ".join(f"{label} ({code}) @ {at}" for code, label, at, _ in parts))
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
        st.caption("\U0001F4AC View Claude explanation (best config per mission):")
        audit_cols = st.columns(min(len(checked_missions), 7))
        for i, mission_id in enumerate(checked_missions):
            with audit_cols[i % len(audit_cols)]:
                best = best_row_by_mission[mission_id]
                n = len((best.get("colreg_llm_check") or {}).get("explanations") or [])
                with st.popover(f"{mission_id}  {n} finding(s)"):
                    st.markdown(_colreg_audit_markdown(best))
    elif best_row_by_mission:
        # Otherwise this whole section just silently disappears with no explanation --
        # looks like a broken/missing feature rather than "no logs have been explained yet".
        st.caption("\U0001F4AC No mission has a saved Claude explanation yet -- these logs predate "
                  "the field, or were generated without --explain (the default). Re-run with "
                  "--explain to populate it -- it never affects the deterministic score above.")

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


# ─────────────────────────────────────────────────────────────────────────────
# Audit tab -- thin read-only front-end for _analysis/audit_runs.py. ALL audit logic
# (schema detection, every BLOCKER/ERROR/WARN check, aggregation) lives there and only
# there -- this file just lets the user pick a tag/config/mission subset, runs that
# script as a subprocess, and renders the JSON it writes to _analysis/audit/<tag>/.
# ─────────────────────────────────────────────────────────────────────────────
def _available_tags() -> list[str]:
    """Tags are read from the FILENAME (parse_run_filename), not by opening every run's
    JSON -- same fast, read-only convention _scan_mission_runs already relies on."""
    tags = {parse_run_filename(p)["tag"] for p in RUNS_DIR.glob("*.json")
           if not p.name.startswith("_sweep_")}
    return sorted(tags)


def _tag_run_files(tag: str, configs: list[str], missions: list[str]) -> list[Path]:
    files = []
    for path in RUNS_DIR.glob("*.json"):
        if path.name.startswith("_sweep_"):
            continue
        parsed = parse_run_filename(path)
        if parsed["tag"] != tag:
            continue
        if configs and parsed["config"] not in configs:
            continue
        if missions and parsed["mission_id"] not in missions:
            continue
        files.append(path)
    return sorted(files)


def _cache_key(files: list[Path], configs: list[str], missions: list[str], baseline_tag: str | None) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([sorted(configs), sorted(missions), baseline_tag], sort_keys=True).encode())
    for p in files:
        h.update(f"{p.name}:{p.stat().st_mtime}".encode())
    return h.hexdigest()[:16]


def _meta_path(out_dir: Path) -> Path:
    return out_dir / "_dashboard_meta.json"


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _run_audit_subprocess(tag: str, configs: list[str], missions: list[str],
                          out_dir: Path, baseline_tag: str | None) -> tuple[int, str, str]:
    args = [sys.executable, str(AUDIT_SCRIPT), "--tag", tag, "--out-dir", str(out_dir)]
    if configs:
        args += ["--configs", *configs]
    if missions:
        args += ["--missions", *missions]
    if baseline_tag:
        args += ["--baseline-tag", baseline_tag]
    proc = subprocess.run(args, capture_output=True, text=True, cwd=str(ROOT))
    return proc.returncode, proc.stdout, proc.stderr


def _render_audit_tab() -> None:
    st.subheader("\U0001F50D Run auditor")
    st.caption("Deterministic, read-only checks (BLOCKER/ERROR/WARN) over precomputed run "
              "logs -- see _analysis/audit_runs.py. This tab only triggers it and shows its "
              "output; no checks run in Streamlit itself.")

    tags = _available_tags()
    if not tags:
        st.info("Geen runs gevonden in Data/missions/_llm_runs -- niets te auditen.")
        return

    sel_cols = st.columns([2, 2, 2, 2])
    with sel_cols[0]:
        tag = st.selectbox("Tag", tags, key="audit_tag")
    with sel_cols[1]:
        configs = st.multiselect("Configs (leeg = alle)", CONFIGS, key="audit_configs")
    with sel_cols[2]:
        missions = st.multiselect("Missions (leeg = alle)", MISSIONS, key="audit_missions")
    with sel_cols[3]:
        baseline_choices = ["(geen)"] + [t for t in tags if t != tag]
        baseline_pick = st.selectbox("Baseline-tag", baseline_choices, key="audit_baseline_tag")
        baseline_tag = None if baseline_pick == "(geen)" else baseline_pick

    out_dir = AUDIT_OUT_DIR / tag
    files = _tag_run_files(tag, configs, missions)
    current_key = _cache_key(files, configs, missions, baseline_tag)
    meta = _load_json(_meta_path(out_dir))
    summary_path = out_dir / "audit_summary.json"
    is_cached = meta is not None and meta.get("cache_key") == current_key and summary_path.exists()

    run_clicked = st.button("\u25B6\uFE0F Audit draaien", type="primary")
    if run_clicked or (not is_cached and not summary_path.exists()):
        if not files:
            st.warning("Geen runs voor deze tag/selectie.")
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            with st.spinner(f"Audit draait over {len(files)} run(s)..."):
                code, out, err = _run_audit_subprocess(tag, configs, missions, out_dir, baseline_tag)
            _meta_path(out_dir).write_text(json.dumps({
                "cache_key": current_key, "audited_at": datetime.now().isoformat(),
                "exit_code": code,
            }), encoding="utf-8")
            with st.expander("stdout / stderr", expanded=(code != 0 and bool(files))):
                st.code(out or "(geen stdout)")
                if err:
                    st.code(err)
            meta = _load_json(_meta_path(out_dir))

    summary = _load_json(summary_path)
    if summary is None:
        st.info("Nog niet geaudit -- klik op 'Audit draaien'.")
        return
    if summary.get("n_runs", 0) == 0:
        st.info("Geen runs voor deze tag.")
        return

    audited_at = (meta or {}).get("audited_at") or summary.get("generated_at")
    st.caption(f"Laatst geaudit: {audited_at}, {summary['n_runs']} runs \u2014 schema's: "
              f"{summary.get('schema_versions')}")

    # 1. BLOCKER banner
    blockers = summary.get("blockers") or []
    if blockers:
        st.error(f"\U0001F6D1 {len(blockers)} BLOCKER(s)")
        for b in blockers[:50]:
            st.markdown(f"- `{b['code']}` step={b['step']}: {b['message']}")
    else:
        st.success("\u2705 Geen blockers")

    # 2. Primary metrics per config x weights
    st.markdown("### Primaire metrieken per config \u00d7 weights")
    by_config_rows = [{"config::weights": k, **v} for k, v in (summary.get("by_config") or {}).items()]
    if by_config_rows:
        st.dataframe(by_config_rows, width="stretch", hide_index=True)

    # 3. Same table per mission-type, canary highlighted
    st.markdown("### Per mission-type")
    by_type_rows = [{"type": k, **v} for k, v in (summary.get("by_mission_type") or {}).items()]
    if by_type_rows:
        st.dataframe(by_type_rows, width="stretch", hide_index=True)
    if summary.get("canary"):
        c = summary["canary"]
        st.metric("\U0001F426 Kanarie (UM01/UM02) A-rate -- hoort ~0 te zijn", f"{c['A_rate']:.2%}")

    # 4. Filters + filtered checkpoints (from audit_summary.json's own embedded
    # checkpoint_findings -- no per-run copy files are written into the audit folder)
    st.markdown("### Checkpoints")
    all_rows = summary.get("checkpoint_findings") or []
    if not all_rows:
        st.caption("Geen checkpoint-bevindingen.")
    else:
        f_cols = st.columns(3)
        with f_cols[0]:
            f_config = st.selectbox("Config", ["(alle)"] + sorted({r["config"] for r in all_rows}),
                                    key="audit_filter_config")
        with f_cols[1]:
            f_mission = st.selectbox("Mission", ["(alle)"] + sorted({r["mission_id"] for r in all_rows}),
                                     key="audit_filter_mission")
        with f_cols[2]:
            f_codes = st.multiselect("Code(s) (leeg = alle)", sorted({r["code"] for r in all_rows}),
                                     key="audit_filter_codes")
        filtered = [r for r in all_rows
                   if (f_config == "(alle)" or r["config"] == f_config)
                   and (f_mission == "(alle)" or r["mission_id"] == f_mission)
                   and (not f_codes or r["code"] in f_codes)]
        st.caption(f"{len(filtered)} bevinding(en)")
        by_checkpoint: dict[tuple[str, int], list[dict]] = {}
        for r in filtered:
            by_checkpoint.setdefault((r["run_path"], r["step"]), []).append(r)
        for (run_path, step), fs in list(by_checkpoint.items())[:200]:
            run = _load_json(Path(run_path))
            cp = next((c for c in (run or {}).get("checkpoints", []) if c["step"] == step), None)
            codes = ", ".join(f"`{r['code']}`" for r in fs)
            with st.expander(f"{fs[0]['mission_id']} / {fs[0]['config']} / step={step} \u2014 {codes}"):
                st.code(run_path, language=None)
                if cp:
                    st.text_area("situation_report", cp.get("situation_report", ""),
                                height=150, key=f"sr_{run_path}_{step}")
                    st.json(cp.get("decision") or {})
                for r in fs:
                    st.markdown(f"- **{r['severity']}** `{r['code']}`: {r['message']}")
                    if r["details"]:
                        st.json(r["details"])
        if by_checkpoint and st.button("\U0001F4E4 Exporteer gefilterde checkpoints als DPO-rejected-kandidaten",
                                       key="audit_export_filtered"):
            rows = []
            for (run_path, step), fs in by_checkpoint.items():
                run = _load_json(Path(run_path))
                cp = next((c for c in (run or {}).get("checkpoints", []) if c["step"] == step), None) if run else None
                if not cp:
                    continue
                rows.append({
                    "mission_id": fs[0]["mission_id"], "config": fs[0]["config"],
                    "weights": fs[0].get("weights"), "step": step,
                    "codes": sorted({r["code"] for r in fs}),
                    "situation_report": cp.get("situation_report"), "decision": cp.get("decision"),
                })
            code_tag = "_".join(sorted(f_codes)) if f_codes else "filtered"
            target = review_path(RUNS_DIR / f"dpo_rejected_candidates_{tag}_{code_tag}.jsonl")
            safe_write_jsonl(rows, target, overwrite=True)
            st.success(f"{len(rows)} kandidaten geschreven naar {target}")

    # 5. Top-10 worst checkpoints + DPO-rejected export
    st.markdown("### Top-10 slechtste checkpoints")
    top_worst = summary.get("top_worst") or []
    for w in top_worst:
        st.markdown(f"- weight={w['weight']} `{Path(w['path']).name}` step={w['step']}: {w['codes']}")
    if top_worst and st.button("\U0001F4E4 Exporteer als DPO-rejected-kandidaten"):
        rows = []
        for w in top_worst:
            run = _load_json(Path(w["path"]))
            if not run:
                continue
            cp = next((c for c in run.get("checkpoints", []) if c["step"] == w["step"]), None)
            if not cp:
                continue
            rows.append({
                "mission_id": run.get("mission_id"), "config": run.get("config"),
                "weights": run.get("weights"), "tag": run.get("tag"), "step": w["step"],
                "codes": w["codes"], "situation_report": cp.get("situation_report"),
                "decision": cp.get("decision"),
            })
        target = review_path(RUNS_DIR / f"dpo_rejected_candidates_{tag}.jsonl")
        safe_write_jsonl(rows, target, overwrite=True)
        st.success(f"{len(rows)} kandidaten geschreven naar {target}")

    # 6. Baseline comparison
    st.markdown("### Baseline-vergelijking")
    if baseline_tag is None:
        st.caption("Geen baseline-tag geselecteerd.")
    elif summary.get("baseline_reason"):
        st.warning(f"niet vergelijkbaar: {summary['baseline_reason']}")
    elif summary.get("baseline"):
        base = summary["baseline"]
        diff_rows = []
        for key, m in (summary.get("by_config") or {}).items():
            bm = (base.get("by_config") or {}).get(key)
            if bm:
                diff_rows.append({"config::weights": key, "A-rate nu": m["A_rate"],
                                 "A-rate baseline": bm["A_rate"],
                                 "delta": m["A_rate"] - bm["A_rate"]})
        if diff_rows:
            st.dataframe(diff_rows, width="stretch", hide_index=True)
    else:
        st.caption("Nog niet vergeleken -- klik op 'Audit draaien'.")


tab_sweep, tab_audit = st.tabs(["Sweep", "Audit"])
with tab_sweep:
    _render()
with tab_audit:
    _render_audit_tab()
