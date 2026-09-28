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

from app.units import kn_to_mps, mps_to_kn, nm_to_m

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
    # Live-simulation-only bookkeeping (mirrors Simulation.target_heading, see its own
    # docstring for why goal_course_action()/goal_course_check_line() need this to avoid
    # a heading-runaway bug under slow-responding kinematics models) -- always None for
    # a Vessel built directly from mission JSON (Track-2 generators, tests), so every
    # existing static caller is completely unaffected.
    target_heading: float | None = None


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
    # Scripted, NON-reactive target course/speed changes -- models a give-way vessel that
    # never yields, turns the wrong way, or a target whose aspect wavers, WITHOUT any
    # intelligence/decision-making on the target's side (see Docs/nomoto_dynamics_design_
    # and_verification.md §12 for why this matters under Nomoto). Each dict:
    # {"target": name, "trigger_time_s": float, "new_heading_deg": float|None,
    # "new_speed_mps": float|None (already converted from new_speed_kn at load time)}.
    # Applied by Simulation, once per entry, the first step sim.t >= trigger_time_s.
    target_maneuvers: list[dict] = field(default_factory=list)

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
        target_maneuvers=[
            {
                "target": tm["target"], "trigger_time_s": float(tm["trigger_time_s"]),
                "new_heading_deg": float(tm["new_heading_deg"]) if tm.get("new_heading_deg") is not None else None,
                "new_speed_mps": kn_to_mps(float(tm["new_speed_kn"])) if tm.get("new_speed_kn") is not None else None,
            }
            for tm in d.get("target_maneuvers", [])
        ],
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
        "target_maneuvers": [
            {
                "target": tm["target"], "trigger_time_s": tm["trigger_time_s"],
                "new_heading_deg": tm["new_heading_deg"],
                "new_speed_kn": mps_to_kn(tm["new_speed_mps"]) if tm["new_speed_mps"] is not None else None,
            }
            for tm in mission.target_maneuvers
        ],
    }


def load_mission(mission_id: str, missions_dir: Path = MISSIONS_DIR) -> Mission:
    path = missions_dir / f"{mission_id}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return mission_from_dict(d)


def duplicate_of(mission_id: str, missions_dir: Path = MISSIONS_DIR) -> str | None:
    """Returns the mission id this one is a KNOWN geometric duplicate of (e.g. Imazu06/
    Imazu12 -- same two targets, listed in reversed order, in Sawada et al.'s own Table 4,
    see generate_imazu_missions.py's own docstring), or None -- an optional, purely
    informational `"duplicate_of"` key on the raw on-disk JSON (mission_from_dict()/Mission
    deliberately never carries it, so it never round-trips into a run log's embedded
    mission). 2026-09-28 addition, so tabel-level aggregates (app/sweep_dashboard.py) can
    exclude a duplicate from cross-mission averages without deleting/renaming the file
    (which would break any already-generated run log referencing it). EXACT duplicates
    (Imazu08 of Imazu05, Imazu22 of Imazu15) are instead removed entirely -- see
    generate_imazu_missions.py's REMOVED_CASES."""
    path = missions_dir / f"{mission_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("duplicate_of")


def list_mission_ids(missions_dir: Path = MISSIONS_DIR) -> list[str]:
    return sorted(p.stem for p in missions_dir.glob("*.json"))


def load_all_missions(missions_dir: Path = MISSIONS_DIR) -> dict[str, Mission]:
    return {mid: load_mission(mid, missions_dir) for mid in list_mission_ids(missions_dir)}

