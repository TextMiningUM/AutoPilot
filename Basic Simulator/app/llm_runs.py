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


def run_log_path(mission_id: str, config: str, weights: str = "W0_base", tag: str = "default") -> Path:
    """One JSON file per (mission, config, weights) in the common case. Filename ALWAYS
    carries all 4 segments -- {mission_id}__{config}__{weights}__{tag}.json -- since
    `weights` (2026-09-22) is a genuinely separate axis from `config`: config selects the
    PROMPT (retrieval/CoT/PG ingredients), weights selects which checkpoint answers it
    (W0_base for now; W1_sft/W2_sft_dpo/W3_sft_dpo_reflect once the fine-tuned-checkpoint
    loading path exists -- main plan phase F4). See parse_run_filename() for reading this
    back, including the OLD 2/3-segment names (no weights segment -- all implicitly
    W0_base) written before this axis existed."""
    return RUNS_DIR / f"{mission_id}__{config}__{weights}__{tag}.json"


def parse_run_filename(path: Path) -> dict:
    """{"mission_id", "config", "weights", "tag"} from a run log's filename, understanding
    both the current 4-segment form (mission__config__weights__tag.json) and the two older
    forms written before the weights axis existed: 3-segment (mission__config__tag.json,
    e.g. the units_v1 archive) and 2-segment (mission__config.json, tag="default"). Every
    run ever written used base Qwen3-8B (no fine-tuned-checkpoint loading path exists yet
    in agents.py), so any name without an explicit weights segment implies "W0_base"."""
    parts = path.stem.split("__")
    if len(parts) >= 4:
        mission_id, config, weights, tag = parts[0], parts[1], parts[2], "__".join(parts[3:])
    elif len(parts) == 3:
        mission_id, config, tag = parts
        weights = "W0_base"
    elif len(parts) == 2:
        mission_id, config = parts
        weights, tag = "W0_base", "default"
    else:
        raise ValueError(f"Unrecognized run-log filename: {path.name!r}")
    return {"mission_id": mission_id, "config": config, "weights": weights, "tag": tag}


def list_run_sets() -> list[str]:
    """Subfolder names directly under Data/missions/ that contain at least one precomputed
    run-log JSON ("{mission_id}__{config}[__{weights}]__{tag}.json" or the older 2/3-segment
    forms) -- selectable in the sidebar's run-folder picker so archived batches (e.g.
    "Mission No Speed Increase", a legacy sweep kept for comparison) and the live
    `_llm_runs/` folder can all be browsed for playback."""
    sets = []
    for p in sorted(BASE_RUNS_DIR.iterdir()):
        if p.is_dir() and any(p.glob("*__*.json")):
            sets.append(p.name)
    return sets


def list_runs_for_mission(mission_id: str, runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Returns [{"config","weights","tag","path","generated_at","outcome"}] for every
    precomputed run available for this mission, newest first. Corrupt/unreadable files are
    skipped. `weights` falls back to parse_run_filename() for older logs written before
    that field existed in the JSON body (all implicitly "W0_base")."""
    if not runs_dir.exists():
        return []
    out = []
    for p in runs_dir.glob(f"{mission_id}__*.json"):
        try:
            log = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "config": log.get("config"),
            "weights": log.get("weights") or parse_run_filename(p)["weights"],
            "tag": log.get("tag", "default"),
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
