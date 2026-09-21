"""One-off generator: Data/missions/Imazu01.json .. Imazu22.json from the canonical
Imazu benchmark's own MOOS-IvP reference files (Data/Imazu Missions/imazu/imazu/NN/NN.ini).

Positions/headings/speeds are taken EXACTLY as given in each .ini (own-ship + every
IMAZU_TSn vehicle) -- NOT rescaled to this project's other missions' speed/distance
convention -- so results on these missions are directly comparable to other papers that
use the same published Imazu problem set. Only the heading sign convention is normalised
(.ini uses signed degrees, e.g. -90; our schema uses 0-360).
"""
import json
import re
from pathlib import Path

IMAZU_DIR = Path("Basic Simulator/Data/Imazu Missions/imazu/imazu")
OUT_DIR = Path("Basic Simulator/Data/missions")

RULE_MAP = {
    "head_on": ["Rule 14"],
    "crossing_stbd": ["Rule 15", "Rule 16"],
    "crossing_port": ["Rule 15", "Rule 17"],
    "overtaking_give_way": ["Rule 13"],
    "overtaking_stand_on": ["Rule 13", "Rule 17"],
    "parallel_no_risk": [],
}
ROLE_MAP = {
    "head_on": "mutual (both alter to starboard)",
    "crossing_stbd": "give-way",
    "crossing_port": "stand-on",
    "overtaking_give_way": "give-way (overtaking)",
    "overtaking_stand_on": "stand-on (being overtaken)",
    "parallel_no_risk": "none (no give-way/stand-on situation)",
}
NAME_MAP = {
    "head_on": "head-on",
    "crossing_stbd": "crossing from starboard",
    "crossing_port": "crossing from port",
    "overtaking_give_way": "overtaking (own-ship give-way)",
    "overtaking_stand_on": "overtaking (own-ship stand-on)",
    "parallel_no_risk": "parallel, no risk",
}
SUMMARY_MAP = {
    "head_on": "{t} approaches head-on",
    "crossing_stbd": "{t} crosses from starboard ({fine})",
    "crossing_port": "{t} crosses from port ({fine})",
    "overtaking_give_way": "{t} is a slower target ahead that own-ship must overtake",
    "overtaking_stand_on": "{t} is a faster target overtaking own-ship from astern",
    "parallel_no_risk": "{t} shares own-ship's course and speed with no closing risk",
}
CRITERIA_MAP = {
    "head_on": "Own-ship alters course to starboard (not port) for the head-on target(s).",
    "crossing_stbd": "Own-ship gives way early and substantially (normally to starboard) "
                     "for the target(s) crossing from starboard, and avoids crossing ahead "
                     "if circumstances allow.",
    "crossing_port": "Own-ship holds course/speed as stand-on for the target(s) crossing "
                     "from port, only acting under Rule 17(b) if the other vessel clearly "
                     "fails to act.",
    "overtaking_give_way": "Own-ship keeps clear of the slower target it is overtaking and "
                           "does not cut in front until finally past and clear.",
    "overtaking_stand_on": "Own-ship holds course and speed while being overtaken, absent "
                           "evidence the target is not keeping clear.",
    "parallel_no_risk": "Own-ship recognises the same-course/same-speed target creates no "
                       "closing risk and does not manoeuvre unnecessarily for it.",
}


def parse_ini(path: Path) -> dict:
    section = None
    default_speed = None
    vehicles: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"\[(.+)\]", line)
        if m:
            section = m.group(1)
            if section.startswith("vehicle:"):
                vehicles[section.split(":", 1)[1]] = {}
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if section == "DEFAULT" and key == "speed":
            default_speed = float(val)
        elif section and section.startswith("vehicle:"):
            vname = section.split(":", 1)[1]
            if key in ("start", "waypoint"):
                sub = {}
                for part in val.split(","):
                    k2, _, v2 = part.strip().partition("=")
                    sub[k2.strip()] = float(v2.strip())
                vehicles[vname][key] = sub
            elif key == "speed":
                vehicles[vname]["speed"] = float(val)
    for v in vehicles.values():
        v.setdefault("speed", default_speed)
    return vehicles


def bearing_from(ox: float, oy: float, tx: float, ty: float) -> float:
    import math
    return math.degrees(math.atan2(tx - ox, ty - oy))


def classify(bearing: float, heading: float, speed: float, own_speed: float) -> str:
    rel_heading = heading % 360
    ahead = -30 <= bearing <= 30
    behind = bearing >= 150 or bearing <= -150
    reciprocal = 150 <= rel_heading <= 210
    same_dir = rel_heading <= 30 or rel_heading >= 330
    if ahead and reciprocal:
        return "head_on"
    if ahead and same_dir:
        return "overtaking_give_way" if speed < own_speed - 0.01 else "parallel_no_risk"
    if behind and same_dir:
        return "overtaking_stand_on" if speed > own_speed + 0.01 else "parallel_no_risk"
    return "crossing_stbd" if bearing > 0 else "crossing_port"


def fineness(bearing: float) -> str:
    a = abs(bearing)
    if a < 30:
        return "fine"
    if a < 75:
        return "broad"
    return "abeam"


def build_mission(case_num: int) -> dict:
    nn = f"{case_num:02d}"
    ini_path = IMAZU_DIR / nn / f"{nn}.ini"
    vehicles = parse_ini(ini_path)
    own = vehicles["LLM_SHIP"]
    ox, oy = own["start"]["x"], own["start"]["y"]
    own_speed = own["speed"]
    target_names = sorted(k for k in vehicles if k != "LLM_SHIP")

    types, summaries, targets_json = [], [], []
    for tname in target_names:
        t = vehicles[tname]
        tx, ty, theading, tspeed = t["start"]["x"], t["start"]["y"], t["start"]["heading"], t["speed"]
        bearing = bearing_from(ox, oy, tx, ty)
        ttype = classify(bearing, theading, tspeed, own_speed)
        types.append(ttype)
        label = tname.replace("IMAZU_TS", "ts").lower()
        fill = {"t": label, "fine": fineness(bearing)}
        summaries.append(SUMMARY_MAP[ttype].format(**fill))
        targets_json.append({
            "name": label, "x": round(tx, 3), "y": round(ty, 3),
            "heading": round(theading % 360, 1), "speed": tspeed,
        })

    unique_types = list(dict.fromkeys(types))  # first-seen order, de-duplicated
    name_parts = " + ".join(NAME_MAP[t] for t in unique_types)
    n = len(target_names)
    name = f"Imazu {case_num:02d} -- {name_parts}" + (f" ({n} targets)" if n > 1 else "")

    rule_refs = []
    for t in unique_types:
        for r in RULE_MAP[t]:
            if r not in rule_refs:
                rule_refs.append(r)
    if n > 1 and "Rule 8(d)" not in rule_refs:
        rule_refs.append("Rule 8(d)")

    if n == 1:
        own_ship_role = ROLE_MAP[types[0]]
    else:
        own_ship_role = " / ".join(
            f"{ROLE_MAP[t]} re: {targets_json[i]['name']}" if t == "parallel_no_risk"
            else f"{ROLE_MAP[t]} to {targets_json[i]['name']}"
            for i, t in enumerate(types))

    criteria = [CRITERIA_MAP[t] for t in unique_types]
    if n > 1:
        criteria.append("Own-ship's manoeuvre for one target does not create a new "
                        "close-quarters situation with another (Rule 8(d)).")
    criteria.append("Minimum CPA to ALL targets stays above the safe-distance threshold.")

    description = (
        f"Canonical Imazu problem {case_num} (Sawada/Zhai Appendix D reconstruction), "
        f"reproduced with the exact own-ship/target start positions, headings, and speeds "
        f"from the reference scenario in Data/Imazu Missions/imazu/imazu/{nn}/{nn}.ini -- "
        f"kept geometrically identical to the published benchmark so results can be "
        f"compared directly against other papers using the same Imazu case. "
        + "; ".join(summaries) + "."
    )

    return {
        "id": f"Imazu{nn}", "name": name, "rule_refs": rule_refs,
        "own_ship_role": own_ship_role, "description": description,
        "pass_criteria": criteria,
        "own_ship": {"x": ox, "y": oy, "heading": own["start"]["heading"], "speed": own_speed},
        "goal": {"x": own["waypoint"]["x"], "y": own["waypoint"]["y"]},
        "targets": targets_json,
    }


if __name__ == "__main__":
    for case_num in range(1, 23):
        mission = build_mission(case_num)
        out_path = OUT_DIR / f"Imazu{case_num:02d}.json"
        out_path.write_text(json.dumps(mission, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"wrote {out_path.name}: {mission['name']}")
