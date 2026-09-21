"""Generator: Data/missions/Imazu01.json .. Imazu22.json from Sawada et al. (2021)'s own
Table 1/Table 4 (J. Mar. Sci. Technol. 26(2), 509-524, DOI 10.1007/s00773-020-00755-0) --
the canonical Imazu-problem reference paper. Positions are in nautical miles and speeds in
knots, matching the paper exactly, NOT rescaled to this project's other missions' metre/m-s
convention -- fields are unit-labelled (x_nm/y_nm/heading_deg/speed_kn) rather than bare
x/y/speed specifically to avoid the kind of unit ambiguity that caused the mission.targets
staleness bug earlier in this project. These files are NOT yet loadable by app/missions.py
-- that conversion (NM/kt -> the physics engine's internal m/m/s at load time, via
app/units.py) is wired up in the next phase.

COORDINATE CONVENTION: Sawada's (X, Y) has X = the own-ship transit axis (north-equivalent,
own-ship's heading 0 moves along +X) and Y = the perpendicular axis (east-equivalent) --
confirmed by checking that own-ship's own track (X: -6 -> +6, Y=0, heading=0) only makes
sense if a heading-0 velocity is purely +X. Mapped directly onto this project's own
(x=east, y=north) schema as x_nm=Y, y_nm=X (i.e. just relabelling which axis is called x
vs y, not an actual geometric transform -- physical north/east values are preserved 1:1).

HEADING CONVENTION: independently verified (see repo history/PR for the full 50-target-
instance check) that Sawada's heading is a standard MATH angle -- velocity = (cos(heading),
sin(heading)) in (X, Y) -- which, once mapped onto this project's (x=east, y=north) axes,
numerically equals this project's own compass-bearing heading (0=north, clockwise) with NO
transformation needed. Verified against the paper's own explicit design rule ("each target
ship is positioned such that it collides with own-ship at the origin, at TCPA=30 minutes,
if course/speed are held") for EVERY target in all 22 cases: computing each target's
required bearing to the origin and comparing it to the heading-implied bearing gives an
EXACT match (within transcription rounding) for 49 of the 50 target instances. The single
exception is Case 2's ts1, published as heading=90.0 -- required bearing analysis gives
exactly 270.0 deg (the opposite direction) instead, a 180-degree discrepancy with no
intermediate cases anywhere else in the table (ruling out a systematic error) -- treated
as a one-off correction of a published typo, see that target's own "source_note" field.

SPEED CONVENTION: Sawada's own design rule (target distance from origin = speed x 30
minutes) gives an unambiguous, paper-internal way to identify which targets get the
8.4kt "overtaken vessel" exception instead of the 12.0kt nominal speed: every single
target in Table 4 sits at EXACTLY 6.000 NM (12.0kt x 0.5h) or EXACTLY 4.200 NM (8.4kt x
0.5h) from the origin, with no target at any intermediate distance -- so the exception
list (ts1 in cases 3, 7, 15, 16, 17, 20, 22) is read directly off the table's own
geometry, not guessed from which cases "look like" overtaking in the literature.
"""
import json
import math
from pathlib import Path

OUT_DIR = Path("Basic Simulator/Data/missions")

FAST_KT = 12.0
SLOW_KT = 8.4
OWN_START_NM = (0.0, -6.0)   # (x_nm=east, y_nm=north) -- Sawada (X=-6, Y=0)
GOAL_NM = (0.0, 6.0)         # Sawada (X=6, Y=0)
OWN_HEADING_DEG = 0.0

# Sawada et al. (2021) Table 4 -- (X, Y, heading_deg) per target, per case, exactly as
# published EXCEPT Case 2 ts1's heading (see module docstring + that target's source_note).
TABLE4 = {
    1:  [(6.000, 0.000, 180.0)],
    2:  [(0.000, 6.000, 270.0)],   # published 90.0 -- corrected, see source_note below
    3:  [(-4.200, 0.000, 0.0)],
    4:  [(-4.243, -4.243, 45.0)],
    5:  [(6.000, 0.000, 180.0), (0.000, 6.000, -90.0)],
    6:  [(-5.909, 1.042, -10.0), (-4.243, 4.243, -45.0)],
    7:  [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0)],
    8:  [(6.000, 0.000, 180.0), (0.000, 6.000, -90.0)],
    9:  [(-5.196, 3.000, -30.0), (0.000, 6.000, -90.0)],
    10: [(0.000, 6.000, -90.0), (-5.796, -1.553, 15.0)],
    11: [(0.000, -6.000, 90.0), (-5.196, 3.000, -30.0)],
    12: [(-4.243, 4.243, -45.0), (-5.909, 1.042, -10.0)],
    13: [(6.000, 0.000, 180.0), (-5.909, -1.042, 10.0), (-4.243, -4.243, 45.0)],
    14: [(-5.909, 1.042, -10.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
    15: [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
    16: [(-2.970, -2.970, 45.0), (0.000, -6.000, 90.0), (0.000, 6.000, -90.0)],
    17: [(-4.200, 0.000, 0.0), (-5.909, -1.042, 10.0), (-4.243, 4.243, -45.0)],
    18: [(4.243, 4.243, -135.0), (-5.796, 1.553, -15.0), (-5.196, 3.000, -30.0)],
    19: [(-5.796, -1.553, 15.0), (-5.796, 1.553, -15.0), (4.243, 4.243, -135.0)],
    20: [(-4.200, 0.000, 0.0), (-5.796, 1.553, -15.0), (0.000, 6.000, -90.0)],
    21: [(-5.796, 1.553, -15.0), (-5.796, -1.553, 15.0), (0.000, 6.000, -90.0)],
    22: [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
}
# ts1 in exactly these cases sits at 4.200 NM from the origin (8.4kt x 0.5h), every other
# target at 6.000 NM (12.0kt x 0.5h) -- see module docstring.
SLOW_TARGET_CASES = {3, 7, 15, 16, 17, 20, 22}

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


def bearing_from(ox: float, oy: float, tx: float, ty: float) -> float:
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
    ox, oy = OWN_START_NM
    own_speed = FAST_KT
    slow_here = case_num in SLOW_TARGET_CASES

    types, summaries, targets_json = [], [], []
    for i, (sx, sy, heading) in enumerate(TABLE4[case_num], start=1):
        # Sawada (X, Y) -> our (x_nm=east=Y, y_nm=north=X) -- see module docstring.
        tx, ty = sy, sx
        tspeed = SLOW_KT if (slow_here and i == 1) else FAST_KT
        bearing = bearing_from(ox, oy, tx, ty)
        ttype = classify(bearing, heading, tspeed, own_speed)
        types.append(ttype)
        label = f"ts{i}"
        summaries.append(SUMMARY_MAP[ttype].format(t=label, fine=fineness(bearing)))
        target = {"name": label, "x_nm": round(tx, 3), "y_nm": round(ty, 3),
                  "heading_deg": heading % 360, "speed_kn": tspeed}
        if case_num == 2 and i == 1:
            target["source_note"] = ("heading corrected from Table 4's published 90.0 deg "
                                     "to 270.0 deg -- required-bearing-to-origin analysis "
                                     "(own-ship's TCPA=30min design rule) gives an exact "
                                     "180 deg mismatch at the published value, the only "
                                     "such mismatch across all 50 target instances in "
                                     "Table 4 (all 49 others match exactly); treated as a "
                                     "one-off correction of a published typo.")
        targets_json.append(target)

    unique_types = list(dict.fromkeys(types))
    name_parts = " + ".join(NAME_MAP[t] for t in unique_types)
    n = len(types)
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
        f"Canonical Imazu problem {case_num}, reproduced from Sawada et al. (2021) Table 4 "
        f"(J. Mar. Sci. Technol. 26(2), 509-524) -- exact published own-ship/target start "
        f"positions (NM), headings (deg), and speeds (kn), so results are directly "
        f"comparable against other papers using the same published Imazu problem set. "
        + "; ".join(summaries) + "."
    )

    return {
        "id": f"Imazu{nn}", "name": name, "rule_refs": rule_refs,
        "own_ship_role": own_ship_role, "description": description,
        "pass_criteria": criteria,
        "own_ship": {"x_nm": OWN_START_NM[0], "y_nm": OWN_START_NM[1],
                    "heading_deg": OWN_HEADING_DEG, "speed_kn": own_speed},
        "goal": {"x_nm": GOAL_NM[0], "y_nm": GOAL_NM[1]},
        "targets": targets_json,
    }


if __name__ == "__main__":
    for case_num in range(1, 23):
        mission = build_mission(case_num)
        out_path = OUT_DIR / f"Imazu{case_num:02d}.json"
        out_path.write_text(json.dumps(mission, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"wrote {out_path.name}: {mission['name']}")
