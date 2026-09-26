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
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (ROOT, ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.evaluation import compliance_finding_parts, score_trajectory
from app.llm_runs import parse_run_filename
from app.missions import list_mission_ids, load_mission
from app.model_variants import MODEL_VARIANTS, variant_label
from app.simulation import VesselConstraints
from app.baselines import BASELINE_CONFIGS
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

# ── UI preference persistence ─────────────────────────────────────────────
# Same local-JSON-file pattern as app/streamlit_app.py's _UI_PREFS_PATH -- Streamlit's
# session_state only lives for one browser session, so without this the Baseline tab's
# tag selection silently reset to its hardcoded default on every dashboard restart.
_UI_PREFS_PATH = ROOT / "Data" / "_sweep_dashboard_ui_prefs.json"
_UI_PREFS_KEYS = ["comparison_baseline_tag"]


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


# Seed session_state from the persisted file BEFORE the widget with this key renders --
# Streamlit only honours a widget's `index=`/`value=` default on the very first render of
# THIS session for that key, so this must run once per (browser) session, ahead of the tabs.
if "_ui_prefs_loaded" not in st.session_state:
    for _k, _v in _load_ui_prefs().items():
        if _k in _UI_PREFS_KEYS:
            st.session_state[_k] = _v
    st.session_state._ui_prefs_loaded = True

st.set_page_config(page_title="LLM sweep dashboard", layout="wide")
title_cols = st.columns([5, 1])
with title_cols[0]:
    st.title("\U0001F4CA LLM config sweep \u2014 progress")
with title_cols[1]:
    st.button("\U0001F504 Refresh", width="stretch")

MISSIONS = list_mission_ids()
CONFIGS = CONFIG_NAMES
# Total jobs assumes ONE model variant (qwen_base, the only one with full history); a
# second variant doubles the real total, but this is only used for a rough top-of-page
# progress bar, not for anything score-bearing -- see the model-variant multiselect below
# for the real per-variant breakdown.
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
    weights_value = log.get("weights") or weights
    model_variant = log.get("model_variant") or weights_value
    return {
        "config": log.get("config"), "weights": weights_value,
        "model_variant": model_variant, "model_label": variant_label(model_variant),
        "tag": log.get("tag", tag),
        # Missing in older logs predating this param -- run_llm_scenario.py's own CLI
        # default was always "kinematics" before --kinematics-model existed, so a missing
        # field means that default was used, never "unknown".
        "kinematics_model": (log.get("params") or {}).get("kinematics_model") or "kinematics",
        "composite_score": result["composite_score"], "verdict": result["verdict"],
        "safety": result["safety"], "compliance": result["compliance"],
        "explanation_compliance": result.get("explanation_compliance") or {"score": None, "breakdown": []},
        "temporal": result["temporal"], "spatial": result["spatial"],
        "manoeuvre": result["manoeuvre"], "latency_s": latency_s,
        "latency_is_estimate": latency_is_estimate,
        "colreg_llm_check": log.get("colreg_llm_check"),
    }


def _scan_mission_runs(mission_id: str, gen_at_index: list[tuple[datetime, Path]]) -> dict[tuple[str, str], dict]:
    """Globs RUNS_DIR for every {mission_id}__*.json and scores each file directly -- the
    single source of truth for what's actually on disk RIGHT NOW, instead of
    _sweep_summary.json's append-only cache. Filenames are parsed via
    app.llm_runs.parse_run_filename(), which understands both the current
    {mission}__{config}__{weights}__{tag}.json form and the older 2/3-segment forms
    written before the weights axis existed. Keyed by (config, model_variant) -- NOT just
    config -- so two different model variants sharing a config never silently overwrite
    each other (the old config-only keying did exactly that, discarding whichever variant
    wasn't most-recently-modified). When more than one tag produced a log for the SAME
    (config, model_variant) pair, the most recently modified file wins (whatever's actually
    current)."""
    prefix = f"{mission_id}__"
    latest_mtime: dict[tuple[str, str], float] = {}
    rows: dict[tuple[str, str], dict] = {}
    for path in RUNS_DIR.glob(f"{prefix}*.json"):
        parsed = parse_run_filename(path)
        config, weights, tag = parsed["config"], parsed["weights"], parsed["tag"]
        if config not in CONFIGS:
            continue
        mtime = path.stat().st_mtime
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        model_variant = log.get("model_variant") or weights
        key = (config, model_variant)
        if key in latest_mtime and latest_mtime[key] >= mtime:
            continue
        latest_mtime[key] = mtime
        rows[key] = _score_log(mission_id, log, tag, path, gen_at_index, weights)
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
    compliance/explanation/temporal/spatial/manoeuvre are each already a 0-1 axis score.
    compliance is manoeuvre-only (was the physical action safe/COLREG-correct);
    explanation is its separate counterpart (did the model's own stated encounter_rule/
    conduct_rule/risk claim match ground truth) -- see evaluate_run.py's COMPLIANCE_CATEGORY
    split; only compliance feeds the composite score, explanation is informational only.
    latency is the whole run's wall-clock compute time: exact (from run_llm_scenario.py's
    own latency_s) for logs generated after that field was added, or a "~"-prefixed
    ESTIMATE (derived from the gap to the previous log's generated_at, see
    _estimate_latency) for older ones, or None if no estimate was possible either (first
    log ever, or an implausible gap). colreg is the narrow Claude-compliance-check status
    -- see _colreg_status."""
    latency = r.get("latency_s")
    if latency is None:
        latency_str = None
    elif r.get("latency_is_estimate"):
        latency_str = f"~{latency:.0f}s"
    else:
        latency_str = f"{latency:.0f}s"
    return {
        "safety": r["safety"]["score"], "compliance": r["compliance"]["score"],
        "explanation": r["explanation_compliance"]["score"],
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
        lines.append(f"- {len(compliance['breakdown'])} manoeuvre compliance finding(s): "
                    + "; ".join(f"{label} ({code}) @ {at}" for code, label, at, _ in parts))
    else:
        lines.append("- No manoeuvre/COLREG violations flagged.")

    explanation = r.get("explanation_compliance") or {"score": None, "breakdown": []}
    if explanation["breakdown"]:
        parts = [compliance_finding_parts(e) for e in explanation["breakdown"]]
        lines.append(f"- {len(explanation['breakdown'])} explanation/citation finding(s): "
                    + "; ".join(f"{label} ({code}) @ {at}" for code, label, at, _ in parts))
    else:
        lines.append("- No explanation/citation mismatches flagged.")

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


def _resolve_baseline_tag(baseline_tags: list[str]) -> str | None:
    """Single shared baseline-tag resolution (the SAME `comparison_baseline_tag` session
    key the Baseline tab's own selectbox reads/writes, persisted across restarts) -- used
    everywhere else a baseline reference is shown too, so picking e.g. a Nomoto-generated
    vs. an old-kinematics-generated baseline tag there is reflected in the Sweep tab's
    Leaderboard/per-mission tables as well, instead of each place tracking it separately."""
    if not baseline_tags:
        return None
    current = st.session_state.get("comparison_baseline_tag")
    if current in baseline_tags:
        return current
    return "baseline" if "baseline" in baseline_tags else baseline_tags[0]


def _render() -> None:
    gen_at_index = _generated_at_index()
    rows_by_mission = {m: _scan_mission_runs(m, gen_at_index) for m in MISSIONS}
    done = sum(len(rows) for rows in rows_by_mission.values())

    baseline_tags = _available_baseline_tags()
    active_baseline_tag = _resolve_baseline_tag(baseline_tags)
    baseline_rows_by_mission = ({m: _scan_baseline_runs(m, active_baseline_tag) for m in MISSIONS}
                                if active_baseline_tag else {})

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

    if active_baseline_tag:
        st.caption(f"\u2696\uFE0F Baseline reference: **{active_baseline_tag}** (ship-dynamics model: "
                  f"**{_baseline_tag_kinematics_model(active_baseline_tag)}**) -- change on the "
                  "Baseline tab (shared selection, applies here too). \"best_llm_vs_baseline\" below "
                  "is restricted to LLM runs using that SAME ship-dynamics model, never mixed with "
                  "the other one.")

    st.caption(f"Scans {RUNS_DIR.relative_to(ROOT)} directly for "
              "{mission}__{config}[__{tag}].json \u2022 read-only \u2022 click Refresh "
              "above for the latest state")

    # Model-variant axis: every (config, model_variant) key seen anywhere in the current
    # data, not just the registered ones -- an ad-hoc weights combo still shows up (labeled
    # via variant_label()'s unregistered fallback) instead of silently vanishing.
    all_variants = sorted({key[1] for rows in rows_by_mission.values() for key in rows},
                          key=lambda v: (v not in MODEL_VARIANTS, v))
    filt_cols = st.columns([2, 2])
    with filt_cols[0]:
        picked_variants = st.multiselect(
            "Model variant(s)", options=all_variants,
            default=all_variants, format_func=variant_label, key="variant_filter",
            help="Which model checkpoint(s) answered -- see app.model_variants.MODEL_VARIANTS. "
                 "A run's filename/'model_variant' field carries this; unregistered ad-hoc "
                 "weights combos still show up here (labeled as-is).")
    with filt_cols[1]:
        picked_setup = st.selectbox(
            "Setup variant (config) for the comparison grid", ["(best across configs)"] + CONFIGS,
            key="config_for_grid",
            help="The comparison grid below is mission \u00d7 model-variant -- pick ONE prompt "
                 "config to see that config's score in every cell, or '(best across configs)' "
                 "to see each (mission, variant)'s own best-scoring config instead.")

    st.divider()
    st.subheader("\U0001F4CA Comparison grid: mission \u00d7 model variant")
    if not picked_variants:
        st.info("Selecteer minstens \u00e9\u00e9n model variant hierboven.")
    else:
        grid_rows = []
        for mission_id in MISSIONS:
            rows = rows_by_mission[mission_id]
            row_out = {"mission": mission_id}
            for variant in picked_variants:
                if picked_setup == "(best across configs)":
                    candidates = [r for (cfg, v), r in rows.items() if v == variant]
                    cell = max(candidates, key=lambda r: r["composite_score"]) if candidates else None
                else:
                    cell = rows.get((picked_setup, variant))
                label = variant_label(variant)
                if cell is None:
                    row_out[label] = "\u2014"
                else:
                    cfg_suffix = f" ({cell['config']})" if picked_setup == "(best across configs)" else ""
                    row_out[label] = f"{cell['composite_score']:.3f}{cfg_suffix}"
            grid_rows.append(row_out)
        st.dataframe(grid_rows, width="stretch", hide_index=True)
        st.caption("Cel = composite score" +
                  (" van de best-scorende config voor die (missie, variant)-combinatie."
                   if picked_setup == "(best across configs)"
                   else f" voor config={picked_setup!r}.") +
                  " \u2014 zie de per-missie detail-tabellen hieronder voor de volledige "
                  "per-config/per-variant breakdown.")

    st.divider()
    st.subheader("Leaderboard (best config \u00d7 variant per mission so far)")
    leaderboard = []
    best_row_by_mission: dict[str, dict] = {}
    wanted_km = _baseline_tag_kinematics_model(active_baseline_tag) if active_baseline_tag else None
    for mission_id in MISSIONS:
        base_rows = baseline_rows_by_mission.get(mission_id) or {}
        best_base = max(base_rows.values(), key=lambda r: r["composite_score"], default=None)
        all_rows = rows_by_mission[mission_id]
        # Restricted to the SAME ship-dynamics model as the active baseline tag -- comparing
        # against the best LLM run regardless of physics model would be apples-to-oranges.
        matching_rows = ({k: r for k, r in all_rows.items() if r["kinematics_model"] == wanted_km}
                         if wanted_km else {})
        best_matching = max(matching_rows.values(), key=lambda r: r["composite_score"], default=None)
        base_cols = {
            "best_baseline": BASELINE_CONFIGS.get(best_base["config"], best_base["config"])
                            if best_base else "\u2014",
            "baseline_composite": best_base["composite_score"] if best_base else None,
            "best_llm_vs_baseline": best_matching["composite_score"] if best_matching else None,
        }
        rows = {k: r for k, r in rows_by_mission[mission_id].items() if k[1] in picked_variants}
        if not rows:
            leaderboard.append({
                "mission": mission_id, "done": f"0/{len(CONFIGS) * max(len(picked_variants), 1)}",
                "best_config": "\u2014", "best_variant": "\u2014",
                "composite": None, "verdict": "\u2014", "safety": None, "compliance": None,
                "explanation": None, "temporal": None, "spatial": None, "manoeuvre": None,
                "latency": None, "colreg": "\u2014", **base_cols,
            })
            continue
        best = max(rows.values(), key=lambda r: r["composite_score"])
        best_row_by_mission[mission_id] = best
        leaderboard.append({
            "mission": mission_id, "done": f"{len(rows)}/{len(CONFIGS) * len(picked_variants)}",
            "best_config": best["config"], "best_variant": best["model_label"],
            "composite": best["composite_score"],
            "verdict": best["verdict"], **_axis_cols(best), **base_cols,
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
        rows = {k: r for k, r in rows_by_mission[mission_id].items() if k[1] in picked_variants}
        n_expected = len(CONFIGS) * max(len(picked_variants), 1)
        # Explicit `key=` (stable across reruns) instead of relying on the auto-key derived
        # from the label -- the label's "(done/8)" count changes as jobs complete, which
        # would otherwise make Streamlit treat it as a brand-new expander each time and
        # collapse it back shut.
        with st.expander(f"{mission_id}  ({len(rows)}/{n_expected} config\u00d7variant done)",
                         expanded=False, key=f"exp_{mission_id}"):
            # Nested st.expander isn't allowed inside another expander, so mission brief and
            # per-config detail below use a checkbox/selectbox to reveal on click instead.
            if st.checkbox("\U0001F4CB Show mission brief", key=f"brief_{mission_id}"):
                st.markdown(MISSION_OBJS[mission_id].as_text())
            st.divider()
            table = []
            for config in CONFIGS:
                for variant in picked_variants:
                    r = rows.get((config, variant))
                    is_current = current_job == (mission_id, config)
                    if r is None:
                        status = "\U0001F504 running" if is_current else "\u23F3 pending"
                        table.append({
                            "config": config, "variant": variant_label(variant),
                            "status": status, "composite": None,
                            "verdict": None, "safety": None, "compliance": None,
                            "explanation": None, "temporal": None, "spatial": None,
                            "manoeuvre": None, "latency": None, "colreg": "\u2014",
                        })
                    else:
                        table.append({
                            "config": config, "variant": r["model_label"], "status": "\u2705 done",
                            "composite": r["composite_score"], "verdict": r["verdict"],
                            **_axis_cols(r),
                        })
            for base_cfg, r in (baseline_rows_by_mission.get(mission_id) or {}).items():
                table.append({
                    "config": BASELINE_CONFIGS.get(base_cfg, base_cfg), "variant": "deterministic baseline",
                    "status": "\u2705 done", "composite": r["composite_score"], "verdict": r["verdict"],
                    **_axis_cols(r),
                })
            st.dataframe(table, width="stretch", hide_index=True)

            done_keys = [k for k in rows if rows.get(k) is not None]
            if done_keys:
                key_labels = {k: f"{k[0]} / {variant_label(k[1])}" for k in done_keys}
                picked = st.selectbox("View details for:", done_keys, format_func=lambda k: key_labels[k],
                                      key=f"detail_pick_{mission_id}")
                r = rows[picked]
                with st.popover(f"\U0001F4C4 {key_labels[picked]} \u2014 details"):
                    m_cols = st.columns(8)
                    m_cols[0].metric("Safety", r["safety"]["score"])
                    m_cols[1].metric("Compliance", r["compliance"]["score"])
                    m_cols[2].metric("Explanation", r["explanation_compliance"]["score"])
                    m_cols[3].metric("Temporal", r["temporal"]["temporal_score"])
                    m_cols[4].metric("Spatial", r["spatial"]["spatial_score"])
                    m_cols[5].metric("Manoeuvre", r["manoeuvre"]["manoeuvre_score"])
                    latency = r.get("latency_s")
                    if latency is None:
                        latency_str = "\u2014"
                    elif r.get("latency_is_estimate"):
                        latency_str = f"~{latency:.0f}s"
                    else:
                        latency_str = f"{latency:.0f}s"
                    m_cols[6].metric("Latency", latency_str)
                    m_cols[7].metric("COLREG (Claude)", _colreg_status(r))
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


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tab -- deterministic baselines (app/baselines/) vs. the best LLM-agent run
# per mission, on the SAME missions, scored with the SAME evaluate_run.py composite as
# every other tab here. Read-only: only scans+scores existing run logs, never runs
# anything itself (unlike the Audit tab's subprocess call).
# ─────────────────────────────────────────────────────────────────────────────
def _available_baseline_tags() -> list[str]:
    tags = {parse_run_filename(p)["tag"] for p in RUNS_DIR.glob("*.json")
           if not p.name.startswith("_sweep_") and parse_run_filename(p)["config"] in BASELINE_CONFIGS}
    return sorted(tags)


def _baseline_tag_kinematics_model(tag: str) -> str:
    """Baseline runs never record their own params.kinematics_model (app.run_baseline_
    scenario writes no "params" block at all, deterministic baselines need no prompt/model
    params) -- the ONLY signal for which ship-dynamics model generated a baseline tag's
    runs is the tag NAME itself, by convention ("baseline_Nomoto_all" vs plain "baseline").
    Matches _score_log()'s own "missing means kinematics" default for LLM runs."""
    return "nomoto" if "nomoto" in tag.lower() else "kinematics"


def _scan_baseline_runs(mission_id: str, tag: str) -> dict[str, dict]:
    """{config: scored_row} for every deterministic baseline run log matching this
    mission_id + tag (see app.baselines.BASELINE_CONFIGS) -- same on-disk scan+scoring
    convention as _scan_mission_runs(), just keyed by config alone since baselines have no
    model-variant axis (weights is always the fixed literal "deterministic", see
    app.run_baseline_scenario.WEIGHTS)."""
    rows: dict[str, dict] = {}
    for path in RUNS_DIR.glob(f"{mission_id}__*.json"):
        parsed = parse_run_filename(path)
        if parsed["config"] not in BASELINE_CONFIGS or parsed["tag"] != tag:
            continue
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows[parsed["config"]] = _score_log(mission_id, log, tag, path, [], parsed["weights"])
    return rows


_FULL_GREEN, _NEAR_GREEN = "#2ecc71", "#a9dfbf"
_FULL_RED, _NEAR_RED = "#e74c3c", "#f5b7b1"


def _tiered_highlight(s: pd.Series, band_frac: float = 0.05, higher_is_better: bool = True) -> list[str]:
    """Per-row/column text-color tiers: the single BEST value gets full green, the single
    WORST gets full red, and any OTHER value within `band_frac` (default 5%) of the
    row/column's range (max-min) from either extreme gets a LIGHTER shade of the same
    color -- so a near-tie for best/worst is visually distinguishable from the middle of
    the pack, not just the single winner/loser. `higher_is_better=False` flips which
    extreme counts as "best" -- e.g. for POINTS tables (regatta-style ranking below),
    where a LOWER number is the better placing."""
    vals = s.dropna()
    if vals.empty:
        return ["" for _ in s]
    vmin, vmax = float(vals.min()), float(vals.max())
    band = band_frac * (vmax - vmin)
    styles = []
    for v in s:
        if pd.isna(v):
            styles.append("")
            continue
        is_best = (v >= vmax) if higher_is_better else (v <= vmin)
        is_worst = (v <= vmin) if higher_is_better else (v >= vmax)
        near_best = (v >= vmax - band) if higher_is_better else (v <= vmin + band)
        near_worst = (v <= vmin + band) if higher_is_better else (v >= vmax - band)
        if is_best:
            styles.append(f"color:{_FULL_GREEN};font-weight:700;")
        elif is_worst:
            styles.append(f"color:{_FULL_RED};font-weight:700;")
        elif near_best:
            styles.append(f"color:{_NEAR_GREEN};font-weight:600;")
        elif near_worst:
            styles.append(f"color:{_NEAR_RED};font-weight:600;")
        else:
            styles.append("")
    return styles



@st.cache_data(ttl=30, show_spinner="Scanning baseline + LLM run logs...")
def _load_comparison_data(tag: str) -> tuple[list[dict], dict[tuple[str, str], dict], dict[str, str]]:
    """Scans+scores every mission's baseline and LLM run logs ONCE, cached for 30s --
    the per-mission grid used to re-run this full 35-mission scan on EVERY interaction
    (including a slow, glitchy native cell-click selection), making even a slider drag
    noticeably slow. A 30s TTL means new runs written while the dashboard is open still
    show up within half a minute, without re-scanning on every widget interaction."""
    gen_at_index = _generated_at_index()
    baseline_configs = list(BASELINE_CONFIGS)
    wanted_km = _baseline_tag_kinematics_model(tag)
    detail_lookup: dict[tuple[str, str], dict] = {}
    best_llm_label: dict[str, str] = {}
    table_rows = []
    for mission_id in MISSIONS:
        base_rows = _scan_baseline_runs(mission_id, tag)
        all_llm_rows = _scan_mission_runs(mission_id, gen_at_index)
        # Only compare against LLM runs generated under the SAME ship-dynamics model as
        # this baseline tag (Nomoto-vs-Nomoto or legacy-kinematics-vs-legacy-kinematics) --
        # mixing them would be an apples-to-oranges physics comparison.
        llm_rows = {k: r for k, r in all_llm_rows.items() if r["kinematics_model"] == wanted_km}
        row_out = {"mission": mission_id}
        for cfg in baseline_configs:
            r = base_rows.get(cfg)
            short = cfg.replace("baseline_", "")
            row_out[short] = r["composite_score"] if r else None
            if r:
                detail_lookup[(mission_id, short)] = r
        if llm_rows:
            best = max(llm_rows.values(), key=lambda r: r["composite_score"])
            row_out["best_llm"] = best["composite_score"]
            detail_lookup[(mission_id, "best_llm")] = best
            best_llm_label[mission_id] = f"{best['config']} / {best['model_label']}"
        else:
            row_out["best_llm"] = None
        table_rows.append(row_out)
    return table_rows, detail_lookup, best_llm_label


def _render_comparison_tab() -> None:
    st.subheader("\u2696\uFE0F Baselines vs. LLM agent")
    st.caption("Compares the deterministic baselines (`app/baselines/` -- Rule tree, "
              "Velocity Obstacle, Artificial Potential Field, Dynamic Window Approach, "
              "MPC, Sawada et al. reconstruction) against the best LLM-agent run per "
              "mission, on the same 35 missions and the same `evaluate_run.py` composite "
              "score as the Sweep tab. Read-only -- runs nothing itself, only scans "
              "existing run logs (see `app.run_baseline_scenario` to generate new "
              "baseline runs). Pick a mission + system below the grid and click for its "
              "full evaluation breakdown.")

    baseline_tags = _available_baseline_tags()
    if not baseline_tags:
        st.info("No baseline runs found yet -- run e.g. `python -m "
               "app.run_baseline_scenario` first (see Basic Simulator/app/baselines/).")
        return
    default_idx = baseline_tags.index("baseline") if "baseline" in baseline_tags else 0
    # A persisted tag from a previous session that no longer exists on disk must not reach
    # the widget (Streamlit raises if session_state[key] isn't in options) -- drop it and
    # fall back to default_idx instead.
    if st.session_state.get("comparison_baseline_tag") not in baseline_tags:
        st.session_state.pop("comparison_baseline_tag", None)
    tag = st.selectbox("Baseline tag", baseline_tags, index=default_idx, key="comparison_baseline_tag")
    st.caption(f"\"best_llm\" below is restricted to LLM runs using the SAME ship-dynamics "
              f"model as this tag (**{_baseline_tag_kinematics_model(tag)}**) -- never the best "
              "LLM run overall, which could otherwise have used the other physics model.")

    baseline_configs = list(BASELINE_CONFIGS)
    system_keys = baseline_configs + ["best_llm"]
    system_labels = {**BASELINE_CONFIGS, "best_llm": "Best LLM-agent run per mission (any config/variant)"}
    short_to_system = {cfg.replace("baseline_", ""): cfg for cfg in baseline_configs}
    short_to_system["best_llm"] = "best_llm"
    axis_names = ["safety", "compliance", "explanation", "temporal", "spatial", "manoeuvre"]

    table_rows, detail_lookup, best_llm_label = _load_comparison_data(tag)
    # Per-system aggregates (n/reached/collision/composite/axes), derived from the SAME
    # cached detail_lookup -- kept out of the cached function itself so a code change to
    # this aggregation doesn't require waiting out the cache TTL to see effect.
    agg = {k: {"n": 0, "reached": 0, "collision": 0, "composite": [], "axes": []} for k in system_keys}
    for (mission_id, short), r in detail_lookup.items():
        cfg = short_to_system.get(short)
        if cfg is None:
            continue
        a = agg[cfg]
        a["n"] += 1
        a["composite"].append(r["composite_score"])
        a["axes"].append(_axis_cols(r))
        a["reached"] += int(bool(r["temporal"]["arrived"]))
        a["collision"] += int(not r["safety"]["passed"])


    st.markdown("### Summary")
    summary_rows = []
    for key in system_keys:
        a = agg[key]
        mean_c = sum(a["composite"]) / len(a["composite"]) if a["composite"] else None
        std_c = statistics.pstdev(a["composite"]) if len(a["composite"]) > 1 else (0.0 if a["composite"] else None)
        row = {
            "system": system_labels[key], "n_missions": a["n"], "reached_goal": a["reached"],
            "collision": a["collision"],
            "mean_composite": round(mean_c, 3) if mean_c is not None else None,
            "std_composite": round(std_c, 3) if std_c is not None else None,
        }
        for axis in axis_names:
            vals = [ax[axis] for ax in a["axes"] if ax.get(axis) is not None]
            row[f"mean_{axis}"] = round(sum(vals) / len(vals), 3) if vals else None
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("system")
    numeric_summary_cols = [c for c in summary_df.columns if c != "n_missions"]
    count_cols = ["reached_goal", "collision"]
    float_cols = [c for c in numeric_summary_cols if c not in count_cols]
    # std_composite (spread) and collision (count) are both "lower is better" -- the
    # opposite sense of every other column's higher-is-better default.
    inverse_cols = ["std_composite", "collision"]
    normal_cols = [c for c in numeric_summary_cols if c not in inverse_cols]
    summary_styler = (summary_df.style
                      .format("{:.3f}", subset=float_cols, na_rep="\u2014")
                      .format("{:.0f}", subset=count_cols, na_rep="\u2014")
                      .apply(_tiered_highlight, axis=0, subset=normal_cols)
                      .apply(_tiered_highlight, axis=0, subset=inverse_cols, higher_is_better=False))
    st.dataframe(summary_styler, width="stretch")
    st.caption("Per column: greenest = best, reddest = worst (lighter shade = within 5% "
              "of that extreme) -- higher is better for every column except \"collision\" "
              "and \"std_composite\" (spread), where lower is better.")

    st.markdown("### Strengths / weaknesses per algorithm (relative to the other techniques)")
    # Per-axis MEAN per system, then RANK systems against each other on that SAME axis --
    # a system's "strength" is the axis where it ranks best among the OTHER systems, not
    # just the axis where it happens to score highest against its OWN other axes (the
    # previous version compared each algorithm only to itself, e.g. "MPC is better at
    # safety than at manoeuvre" -- true but useless for picking a technique, since it never
    # said how MPC's safety compares to VO's or the LLM's safety).
    axis_system_means: dict[str, dict[str, float]] = {axis: {} for axis in axis_names}
    for key in system_keys:
        a = agg[key]
        for axis in axis_names:
            vals = [ax[axis] for ax in a["axes"] if ax.get(axis) is not None]
            if vals:
                axis_system_means[axis][key] = sum(vals) / len(vals)
    axis_ranks: dict[str, dict[str, int]] = {}
    axis_field_avg: dict[str, float] = {}
    for axis, means_by_system in axis_system_means.items():
        if not means_by_system:
            continue
        ordered = sorted(means_by_system.items(), key=lambda kv: kv[1], reverse=True)
        axis_ranks[axis] = {sys_key: rank + 1 for rank, (sys_key, _) in enumerate(ordered)}
        axis_field_avg[axis] = sum(means_by_system.values()) / len(means_by_system)

    for key in system_keys:
        per_axis = {axis: (axis_ranks[axis][key], axis_system_means[axis][key])
                   for axis in axis_names if axis in axis_ranks and key in axis_ranks[axis]}
        if not per_axis:
            continue
        best_axis = min(per_axis, key=lambda ax: per_axis[ax][0])   # rank 1 = best
        worst_axis = max(per_axis, key=lambda ax: per_axis[ax][0])  # highest rank = worst
        best_rank, best_val = per_axis[best_axis]
        worst_rank, worst_val = per_axis[worst_axis]
        st.markdown(
            f"- **{system_labels[key]}**: relatively strongest at *{best_axis}* "
            f"(ranks #{best_rank}/{len(axis_ranks[best_axis])}, {best_val:.2f} vs. field "
            f"average {axis_field_avg[best_axis]:.2f}), relatively weakest at *{worst_axis}* "
            f"(ranks #{worst_rank}/{len(axis_ranks[worst_axis])}, {worst_val:.2f} vs. field "
            f"average {axis_field_avg[worst_axis]:.2f}).")

    st.markdown("### Per mission")
    default_rows = min(len(table_rows), 20)
    visible_rows = st.slider(
        "Table height (visible rows)", min_value=5, max_value=len(table_rows),
        value=default_rows, key="per_mission_table_rows",
        help="Drag to show more missions at once, or shrink for a smaller screen.",
    )
    row_height_px, header_height_px = 35, 38  # Streamlit's default dataframe row/header height
    per_mission_df = pd.DataFrame(table_rows).set_index("mission")
    per_mission_styler = (per_mission_df.style
                          .format("{:.3f}", na_rep="\u2014")
                          .apply(_tiered_highlight, axis=1))
    # Plain, non-interactive grid (no on_select) -- native cell-click selection was both
    # slow (triggered a full script rerun+rescan per click) and glitchy (a second click on
    # an already-selected cell could enter glide-data-grid's inline text-edit mode, even
    # though this dataframe is read-only). A dedicated mission/system picker below is a
    # faster, unambiguous replacement: no full-grid rerun, no accidental edit state.
    st.dataframe(per_mission_styler, width="stretch", hide_index=False,
                height=header_height_px + visible_rows * row_height_px)
    st.caption("Per row (mission): greenest cell = best system for that mission, reddest = worst "
              "(lighter shade = within 5% of that extreme) -- usually the LLM agent on the "
              "green end.")

    st.markdown("### Ranking (regatta-style: 1st place = 1 point, lower total = better)")
    st.caption("Sailing/regatta \"low-point\" scoring using real World Sailing outcome "
              "codes: finishers (reached the goal, no genuine right-of-way violation, no "
              "collision) are ranked by composite score among themselves (1st = 1 point, "
              "2nd = 2, ...; ties share the better rank). **DNF** (Did Not Finish), **DSQ** "
              "(Disqualified -- a genuine give-way/passing-side violation, see below), and "
              "**DNE** (Disqualification Not Excludable -- an actual collision) all score "
              "the SAME fixed points value: (number of systems compared) + 1 -- always "
              "worse than every finisher, matching the standard regatta convention that "
              "non-finishers share one \"worse than the fleet\" score rather than an "
              "escalating penalty. This is an EXPLICIT outcome-based tiering (never "
              "trusting the raw composite score's own 0.0/\u22640.2 gates to sort these "
              "correctly on their own). No run at all for a mission counts as a DNF too "
              "(never attempted = never finished). Only a REAL manoeuvre violation counts "
              "toward DSQ -- failing to give way (no action despite acute risk), cutting "
              "across a contact's bow, or passing on the wrong side; a merely-mislabelled "
              "rule citation or a bare CPA/safe-distance shortfall is ignored for this "
              "purpose, and that run is scored as a normal finisher instead. Points are "
              "summed across all missions -- LOWER total is better. The per-axis columns "
              "(safety_points etc.) are NOT outcome-gated this way -- they rank purely by "
              "that axis's own continuous score, since e.g. a DNF run's manoeuvre/temporal "
              "scores are still meaningful to compare.")

    def _regatta_points(values: dict[str, float | None]) -> dict[str, int]:
        """Generic continuous-score low-point ranking -- used for the per-AXIS columns
        only (see _regatta_points_overall below for the outcome-tiered "overall" column)."""
        present = {k: v for k, v in values.items() if v is not None}
        if not present:
            return dict.fromkeys(values, len(values) + 1)
        ranks = pd.Series(present).rank(ascending=False, method="min").astype(int).to_dict()
        dnf_points = len(present) + 1
        return {k: ranks.get(k, dnf_points) for k in values}

    # Only these manoeuvre-category codes are a genuine give-way/passing-side violation
    # (per user request) -- a wrong-direction turn, an oversized turn request, an early
    # stand-on action, or a bare CPA/safe-distance shortfall are all left out: none of
    # them means own-ship actually failed to give way or passed on the wrong side.
    _REAL_VIOLATION_CODES = {"D_no_action_when_required", "P_port_toward_contact", "P_wrong_side_pass"}

    def _classify_mission_outcomes(mission_id: str) -> tuple[dict[str, float], list[str], list[str], list[str]]:
        """Classify every system's run for one mission into finisher/DNF/DSQ/DNE buckets,
        using real World-Sailing-style codes: finisher = reached goal, no genuine
        right-of-way violation, no collision; DNF (no run, or ran but never reached the
        goal, otherwise clean); DSQ (a genuine give-way/passing-side violation logged --
        see _REAL_VIOLATION_CODES -- but no actual collision); DNE (an actual collision,
        the worst tier). Checked in this exact precedence order (collision first) so a run
        that BOTH collided AND had other compliance findings logged is classified DNE, not
        DSQ."""
        finishers: dict[str, float] = {}
        dnf: list[str] = []
        dsq: list[str] = []
        dne: list[str] = []
        for short, cfg in short_to_system.items():
            r = detail_lookup.get((mission_id, short))
            real_violation = r is not None and any(
                compliance_finding_parts(e)[0] in _REAL_VIOLATION_CODES
                for e in r["compliance"]["breakdown"])
            if r is None:
                dnf.append(cfg)
            elif not r["safety"]["passed"]:
                dne.append(cfg)
            elif real_violation:
                dsq.append(cfg)
            elif not r["temporal"]["arrived"]:
                dnf.append(cfg)
            else:
                finishers[cfg] = r["composite_score"]
        return finishers, dnf, dsq, dne

    def _regatta_points_overall(mission_id: str) -> dict[str, int]:
        """Outcome-tiered "overall" points for one mission: finishers ranked by composite
        score among themselves; DNF/DSQ/DNE all share the SAME fixed points value
        (len(system_keys) + 1) -- the standard regatta convention (non-finishers all score
        one worse than the whole fleet, not an escalating per-tier penalty)."""
        finishers, dnf, dsq, dne = _classify_mission_outcomes(mission_id)
        points: dict[str, int] = {}
        if finishers:
            points.update(pd.Series(finishers).rank(ascending=False, method="min").astype(int).to_dict())
        worst_points = len(system_keys) + 1
        for cfg in dnf + dsq + dne:
            points[cfg] = worst_points
        return points

    overall_points = {key: 0 for key in system_keys}
    axis_points = {axis: {key: 0 for key in system_keys} for axis in axis_names}
    for mission_id in MISSIONS:
        for key, pts in _regatta_points_overall(mission_id).items():
            overall_points[key] += pts
        for axis in axis_names:
            axis_values = {}
            for short, cfg in short_to_system.items():
                r = detail_lookup.get((mission_id, short))
                axis_values[cfg] = _axis_cols(r)[axis] if r else None
            for key, pts in _regatta_points(axis_values).items():
                axis_points[axis][key] += pts

    ranking_rows = []
    for key in system_keys:
        rrow = {"system": system_labels[key], "overall_points": overall_points[key]}
        for axis in axis_names:
            rrow[f"{axis}_points"] = axis_points[axis][key]
        ranking_rows.append(rrow)
    ranking_cols = ["overall_points"] + [f"{axis}_points" for axis in axis_names]
    ranking_df = pd.DataFrame(ranking_rows).set_index("system").sort_values("overall_points")
    ranking_styler = (ranking_df.style
                      .format("{:.0f}", subset=ranking_cols)
                      .apply(_tiered_highlight, axis=0, subset=ranking_cols, higher_is_better=False))
    st.dataframe(ranking_styler, width="stretch")
    st.caption("Lower = better in every column -- these are POINTS (regatta placings summed "
              "over 35 missions), not scores, so greenest = fewest points.")

    st.markdown("#### Per-mission points detail (where the DNF / DSQ / DNE outcomes were)")
    default_detail_rows = min(len(MISSIONS), 20)
    visible_detail_rows = st.slider(
        "Table height (visible rows)", min_value=5, max_value=len(MISSIONS),
        value=default_detail_rows, key="points_detail_table_rows",
        help="Drag to show more missions at once, or shrink for a smaller screen.",
    )
    # Text-only colour (no background fill) for all three non-finisher tiers -- a solid
    # fill for 30+ rows read as visually "loud"; a coloured number is enough to spot them.
    _TIER_STYLE = {
        "DNE": f"color:{_FULL_RED};", "DSQ": f"color:{_FULL_RED};", "DNF": f"color:{_FULL_RED};",
    }
    system_to_short = {cfg: short for short, cfg in short_to_system.items()}
    points_rows, tier_rows = [], []
    for mission_id in MISSIONS:
        finishers, dnf, dsq, dne = _classify_mission_outcomes(mission_id)
        pts = _regatta_points_overall(mission_id)
        tier_by_cfg = {cfg: "DNF" for cfg in dnf}
        tier_by_cfg.update({cfg: "DSQ" for cfg in dsq})
        tier_by_cfg.update({cfg: "DNE" for cfg in dne})
        prow, trow = {"mission": mission_id}, {"mission": mission_id}
        for key in system_keys:
            # Short column header (e.g. "ruletree", "best_llm") -- the full paper-reference
            # labels used elsewhere are far too wide for a 7-column-wide grid like this one.
            label = system_to_short[key]
            tier = tier_by_cfg.get(key, "finisher")
            prow[label] = f"{tier} ({pts[key]})" if tier != "finisher" else str(pts[key])
            trow[label] = tier
        points_rows.append(prow)
        tier_rows.append(trow)
    points_detail_df = pd.DataFrame(points_rows).set_index("mission")
    tier_df = pd.DataFrame(tier_rows).set_index("mission")

    def _style_points_detail(_: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [[_TIER_STYLE.get(tier_df.loc[idx, col], f"color:{_FULL_GREEN};")
              for col in points_detail_df.columns] for idx in points_detail_df.index],
            index=points_detail_df.index, columns=points_detail_df.columns)

    st.dataframe(
        points_detail_df.style.apply(_style_points_detail, axis=None),
        width="stretch", hide_index=False,
        height=header_height_px + visible_detail_rows * row_height_px,
        column_config={col: st.column_config.TextColumn(width="small")
                      for col in points_detail_df.columns})
    st.caption("One cell per mission x system (columns use the same short config names as "
              "the tag/system pickers below): plain green number = that system's regatta "
              "points for that mission (finisher, ranked by composite score); red **DNF** = "
              "ran but never reached the goal; red **DSQ** = a genuine give-way/passing-side "
              "violation was logged (no collision); red **DNE** = an actual collision. The "
              "number in parentheses is the fixed points value non-finishers share for that "
              "mission (number of systems compared + 1).")

    st.markdown("#### Inspect one run")
    pick_cols = st.columns([2, 3, 1])
    with pick_cols[0]:
        picked_mission = st.selectbox("Mission", MISSIONS, key="detail_pick_mission")
    with pick_cols[1]:
        picked_short = st.selectbox(
            "System", list(short_to_system), key="detail_pick_system",
            format_func=lambda short: system_labels[short_to_system[short]])
    with pick_cols[2]:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)  # align with the selectboxes
        show_clicked = st.button("\U0001F50D Click for details", key="detail_pick_button")

    if show_clicked:
        r = detail_lookup.get((picked_mission, picked_short))
        system_key = short_to_system[picked_short]
        if r is None:
            st.info(f"No run found for {picked_mission} / {system_labels[system_key]}.")
        else:
            @st.dialog(f"{picked_mission} \u2014 {system_labels[system_key]}", width="large")
            def _show_picked_detail(picked_mission=picked_mission, system_key=system_key, r=r) -> None:
                if system_key == "best_llm":
                    st.caption(f"Winning run: {best_llm_label.get(picked_mission, '?')}")
                st.markdown(_describe_run(r))
            _show_picked_detail()


tab_sweep, tab_audit, tab_comparison = st.tabs(["Sweep", "Audit", "Baseline"])
with tab_sweep:
    _render()
with tab_audit:
    _render_audit_tab()
with tab_comparison:
    _render_comparison_tab()

_save_ui_prefs()
