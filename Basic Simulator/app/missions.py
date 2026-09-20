"""Mission loading: Data/missions/<set>/*.json -> in-memory Mission objects. Missions live
in named subfolders ("sets", e.g. "MIssions Data V1") under MISSIONS_DIR rather than loose
in MISSIONS_DIR itself, so multiple mission batches can coexist and the app can let the user
pick which folder to load from (see list_mission_sets())."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MISSIONS_DIR = ROOT / "Data" / "missions"
# Folders that hold something other than mission-definition JSONs (precomputed run logs,
# sweep backups, ...) -- never offered as a selectable mission set even if they contain
# `.json` files, since load_mission() would fail on their (very different) schema.
_NON_MISSION_SET_NAMES = {"_llm_runs", "Mission No Speed Increase"}


@dataclass
class Vessel:
    name: str
    x: float
    y: float
    heading: float
    speed: float


@dataclass
class Mission:
    id: str
    name: str
    rule_refs: list[str]
    own_ship_role: str
    description: str
    pass_criteria: list[str]
    own_ship: Vessel
    goal: tuple[float, float]
    targets: list[Vessel] = field(default_factory=list)

    def as_text(self) -> str:
        """Human-readable mission brief -- the "missie in tekst" panel."""
        lines = [
            f"## {self.name}  ({self.id})",
            f"**Own-ship role:** {self.own_ship_role}",
            f"**Applicable rules:** {', '.join(self.rule_refs) or '(none -- no give-way/stand-on situation)'}",
            "",
            self.description,
            "",
            "**Pass criteria:**",
        ]
        lines += [f"- {c}" for c in self.pass_criteria]
        return "\n".join(lines)


def _vessel(name: str, d: dict) -> Vessel:
    return Vessel(name=name, x=float(d["x"]), y=float(d["y"]),
                   heading=float(d["heading"]), speed=float(d["speed"]))


def _is_mission_file(path: Path) -> bool:
    """A mission-definition JSON has these top-level keys -- distinguishes it from a
    precomputed run log (mission_id/config/trajectory/...) that might live alongside it."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(d, dict) and "own_ship" in d and "goal" in d


def list_mission_sets(base_dir: Path = MISSIONS_DIR) -> list[str]:
    """Subfolder names directly under `base_dir` that contain at least one real mission
    JSON -- these are the selectable "mission sets" in the sidebar's folder picker."""
    sets = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir() or p.name in _NON_MISSION_SET_NAMES or p.name.startswith("_"):
            continue
        if any(_is_mission_file(f) for f in p.glob("*.json")):
            sets.append(p.name)
    return sets


def _resolve_missions_dir(missions_dir: Path | None) -> Path:
    """`None` (the default for every function below) resolves to the first mission SET
    folder (e.g. "MIssions Data V1") rather than MISSIONS_DIR itself -- mission JSONs no
    longer live loose in MISSIONS_DIR, so this keeps every caller that doesn't explicitly
    pass a folder (CLI scripts like run_llm_scenario.py/sweep_llm_params.py/
    sweep_dashboard.py) working unchanged against whichever set is first alphabetically."""
    if missions_dir is not None:
        return missions_dir
    sets = list_mission_sets()
    return (MISSIONS_DIR / sets[0]) if sets else MISSIONS_DIR


def load_mission(mission_id: str, missions_dir: Path | None = None) -> Mission:
    missions_dir = _resolve_missions_dir(missions_dir)
    path = missions_dir / f"{mission_id}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return Mission(
        id=d["id"], name=d["name"], rule_refs=d["rule_refs"],
        own_ship_role=d["own_ship_role"], description=d["description"],
        pass_criteria=d["pass_criteria"],
        own_ship=_vessel("own_ship", d["own_ship"]),
        goal=(float(d["goal"]["x"]), float(d["goal"]["y"])),
        targets=[_vessel(t["name"], t) for t in d["targets"]],
    )


def list_mission_ids(missions_dir: Path | None = None) -> list[str]:
    missions_dir = _resolve_missions_dir(missions_dir)
    return sorted(p.stem for p in missions_dir.glob("*.json"))


def load_all_missions(missions_dir: Path | None = None) -> dict[str, Mission]:
    missions_dir = _resolve_missions_dir(missions_dir)
    return {mid: load_mission(mid, missions_dir) for mid in list_mission_ids(missions_dir)}

