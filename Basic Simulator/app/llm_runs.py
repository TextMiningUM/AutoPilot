"""Shared schema/IO for precomputed LLM-driven mission runs.

app/run_llm_scenario.py (CLI, run offline/in the background) WRITES these logs by calling
the OOW agent only every `decision_interval` steps (dead-reckoning at the last decision in
between) and saving the full trajectory + every situation report/recommendation + every
agent parameter used. streamlit_app.py's "LLM driven" playback mode READS them -- Step/Run
steps/Full run just scrub through the precomputed trajectory (instant, no model calls),
which is why this split exists: a live per-step model call was too slow to play interactively.

Storing the full `params` block per log means multiple agentic variations of the SAME
mission (different config, thinking, k, system prompt, ...) can be run under different
`tag`s and compared side by side later -- that's the whole point of keeping them as
separate, fully self-describing files instead of overwriting one "the" run per mission.
"""
from __future__ import annotations
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
BASE_RUNS_DIR = ROOT / "Data" / "missions"
RUNS_DIR = BASE_RUNS_DIR / "_llm_runs"


def run_log_path(mission_id: str, config: str, tag: str = "default") -> Path:
    """One JSON file per (mission, config) in the common case -- `default` tag is omitted
    from the filename entirely. Only a non-default `tag` (an explicit variation, e.g. a
    different thinking/k/system-prompt setup for the same mission+config) adds a suffix,
    so exploring variations never clutters the normal one-file-per-(mission,config) case."""
    if tag == "default":
        return RUNS_DIR / f"{mission_id}__{config}.json"
    return RUNS_DIR / f"{mission_id}__{config}__{tag}.json"


def list_run_sets() -> list[str]:
    """Subfolder names directly under Data/missions/ that contain at least one precomputed
    run-log JSON ("{mission_id}__{config}[__{tag}].json") -- selectable in the sidebar's
    run-folder picker so archived batches (e.g. "Mission No Speed Increase", a legacy sweep
    kept for comparison) and the live `_llm_runs/` folder can all be browsed for playback."""
    sets = []
    for p in sorted(BASE_RUNS_DIR.iterdir()):
        if p.is_dir() and any(p.glob("*__*.json")):
            sets.append(p.name)
    return sets


def list_runs_for_mission(mission_id: str, runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Returns [{"config","tag","path","generated_at","outcome"}] for every precomputed
    run available for this mission, newest first. Corrupt/unreadable files are skipped."""
    if not runs_dir.exists():
        return []
    out = []
    for p in runs_dir.glob(f"{mission_id}__*.json"):
        try:
            log = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "config": log.get("config"), "tag": log.get("tag", "default"),
            "path": p, "generated_at": log.get("generated_at"),
            "outcome": log.get("outcome", {}),
        })
    out.sort(key=lambda r: r.get("generated_at") or "", reverse=True)
    return out


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_at_or_before(log: dict, t: float) -> dict | None:
    """Latest checkpoint whose time <= t -- what the Agent panel shows during playback,
    since that's the most recent moment the agent actually made a decision."""
    best = None
    for cp in log.get("checkpoints", []):
        if cp["time"] <= t and (best is None or cp["time"] > best["time"]):
            best = cp
    return best
