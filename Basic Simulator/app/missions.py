"""Mission loading: Data/missions/*.json -> in-memory Mission objects.

Mission JSON is authored in nautical miles/knots (x_nm/y_nm/heading_deg/speed_kn -- see
app/units.py and generate_imazu_missions.py), matching Sawada et al. (2021), the canonical
Imazu-problem reference paper. Conversion to this project's INTERNAL physics/kinematics
units (metres, m/s -- Simulation, VesselConstraints, narrate.cpa_tcpa/classify_encounter,
...) happens ONCE, right here at load time, via app.units -- the physics core itself is
never rewritten in NM/kt. Older mission files (pre-dating this convention) that still use
bare x/y/heading/speed fields are accepted unchanged (already in metres/m-s, no double
conversion) for backward compatibility during the migration."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.units import kn_to_mps, nm_to_m

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MISSIONS_DIR = ROOT / "Data" / "missions"


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
    """x_nm/y_nm/heading_deg/speed_kn (NM/kt, Sawada-convention missions) -> metres/m-s;
    bare x/y/heading/speed (older missions, pre-dating this convention) are already in
    metres/m-s and pass through unconverted."""
    if "x_nm" in d:
        return Vessel(name=name, x=nm_to_m(float(d["x_nm"])), y=nm_to_m(float(d["y_nm"])),
                      heading=float(d["heading_deg"]), speed=kn_to_mps(float(d["speed_kn"])))
    return Vessel(name=name, x=float(d["x"]), y=float(d["y"]),
                   heading=float(d["heading"]), speed=float(d["speed"]))


def _goal_xy(d: dict) -> tuple[float, float]:
    if "x_nm" in d:
        return nm_to_m(float(d["x_nm"])), nm_to_m(float(d["y_nm"]))
    return float(d["x"]), float(d["y"])


def mission_from_dict(d: dict) -> Mission:
    """Same schema load_mission() reads off disk (Data/missions/{id}.json) -- shared so a
    mission embedded in a run log (see run_llm_scenario.py's mission_to_dict()) can be
    reconstructed identically, without needing the original scenario file to still exist
    at its expected path (e.g. after a run log gets moved/archived into a subfolder on its
    own)."""
    return Mission(
        id=d["id"], name=d["name"], rule_refs=d["rule_refs"],
        own_ship_role=d["own_ship_role"], description=d["description"],
        pass_criteria=d["pass_criteria"],
        own_ship=_vessel("own_ship", d["own_ship"]),
        goal=_goal_xy(d["goal"]),
        targets=[_vessel(t["name"], t) for t in d["targets"]],
    )


def mission_to_dict(mission: Mission) -> dict:
    """Inverse of mission_from_dict() -- same shape as the on-disk mission JSON schema, so
    it round-trips through mission_from_dict(mission_to_dict(m)) unchanged. Used to embed
    the full mission definition inside a run log (see run_llm_scenario.py) instead of just
    a bare mission_id string."""
    return {
        "id": mission.id, "name": mission.name, "rule_refs": mission.rule_refs,
        "own_ship_role": mission.own_ship_role, "description": mission.description,
        "pass_criteria": mission.pass_criteria,
        "own_ship": {"x": mission.own_ship.x, "y": mission.own_ship.y,
                     "heading": mission.own_ship.heading, "speed": mission.own_ship.speed},
        "goal": {"x": mission.goal[0], "y": mission.goal[1]},
        "targets": [{"name": t.name, "x": t.x, "y": t.y, "heading": t.heading, "speed": t.speed}
                   for t in mission.targets],
    }


def load_mission(mission_id: str, missions_dir: Path = MISSIONS_DIR) -> Mission:
    path = missions_dir / f"{mission_id}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return mission_from_dict(d)


def list_mission_ids(missions_dir: Path = MISSIONS_DIR) -> list[str]:
    return sorted(p.stem for p in missions_dir.glob("*.json"))


def load_all_missions(missions_dir: Path = MISSIONS_DIR) -> dict[str, Mission]:
    return {mid: load_mission(mid, missions_dir) for mid in list_mission_ids(missions_dir)}

