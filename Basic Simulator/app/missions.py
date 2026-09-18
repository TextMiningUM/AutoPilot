"""Mission loading: Data/missions/*.json -> in-memory Mission objects."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

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
    return Vessel(name=name, x=float(d["x"]), y=float(d["y"]),
                   heading=float(d["heading"]), speed=float(d["speed"]))


def load_mission(mission_id: str) -> Mission:
    path = MISSIONS_DIR / f"{mission_id}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return Mission(
        id=d["id"], name=d["name"], rule_refs=d["rule_refs"],
        own_ship_role=d["own_ship_role"], description=d["description"],
        pass_criteria=d["pass_criteria"],
        own_ship=_vessel("own_ship", d["own_ship"]),
        goal=(float(d["goal"]["x"]), float(d["goal"]["y"])),
        targets=[_vessel(t["name"], t) for t in d["targets"]],
    )


def list_mission_ids() -> list[str]:
    return sorted(p.stem for p in MISSIONS_DIR.glob("*.json"))


def load_all_missions() -> dict[str, Mission]:
    return {mid: load_mission(mid) for mid in list_mission_ids()}
