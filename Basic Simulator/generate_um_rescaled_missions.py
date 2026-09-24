"""Regenerates UM01.._rescaled..UM13_rescaled at Imazu scale (12 kt own-ship, targets on a
collision course with own-ship's track at TCPA=30 min, matching generate_imazu_missions.py's
own construction exactly) -- fixes the geometric-feasibility bug found in the original
small-craft-scale UM missions (own speed ~2.5 m/s / target range 600-1100 m / 500 m safe
distance left no real margin for a 30 deg-per-command turn issued once per decision
interval to open the required separation; UM04 needs >=90 deg on the FIRST command, UM09
collides by decision 2). Read-only against the Imazu*.json files/generate_imazu_missions.py
-- writes ONLY new UM*_rescaled.json files, never touches the originals.

Per-target construction (identical in kind to every one of the 22 canonical Imazu
missions, which this whole project has run successfully throughout this session):
target heading is preserved UNCHANGED (fully encodes the encounter type -- head-on/
crossing-side/overtaking -- via the same bearing-heading relationship Sawada's own table
uses), target speed is rescaled to preserve the ORIGINAL target/own speed RATIO (12.0 kt
* ratio), and the target start position is placed at distance=(new_speed_kt*0.5) NM from
the origin along the reverse of its heading -- guaranteeing it reaches the origin at
T=1800s (30 min), the SAME instant own-ship (starting 6 NM back, also at 12 kt) reaches
the origin, exactly matching Imazu's own "everyone collides at the origin at TCPA=30min"
design rule. UM01/UM02 are deliberately "quiet" (no real risk) missions in the original
set -- kept quiet here by placing the target far enough out that it never nears own-ship's
track, rather than forcing a collision-course construction that would defeat their purpose.
"""
import json
import math
from pathlib import Path

MISSIONS_DIR = Path("Basic Simulator/Data/missions")
OWN_START_NM = (0.0, -6.0)
GOAL_NM = (0.0, 6.0)
OWN_SPEED_KT = 12.0
QUIET_MISSIONS = {1, 2}  # UM01/UM02 -- keep genuinely no-risk, never force a collision course
QUIET_RANGE_NM = 9.0  # comfortably beyond the 6 NM transit -- stays clear throughout


def bearing_from(ox: float, oy: float, tx: float, ty: float) -> float:
    return math.degrees(math.atan2(tx - ox, ty - oy))


def rescale_target(heading_deg: float, ratio: float, quiet: bool, orig_bearing_deg: float = 0.0) -> dict:
    speed_kt = round(ratio * OWN_SPEED_KT, 2)
    rad = math.radians(heading_deg)
    if quiet:
        # Placed along the ORIGINAL bearing from own-ship's start (not a heading-reversal
        # placement, which can land exactly on own-ship's own track for a reciprocal
        # heading, e.g. UM02's heading=180) -- same heading preserves the "crossing/
        # passing far away" character, but the target stays well off own-ship's track.
        brg_rad = math.radians(orig_bearing_deg)
        x = OWN_START_NM[0] + QUIET_RANGE_NM * math.sin(brg_rad)
        y = OWN_START_NM[1] + QUIET_RANGE_NM * math.cos(brg_rad)
    else:
        dist_nm = speed_kt * 0.5  # reaches the origin at T=30min, matching own-ship exactly
        x, y = -dist_nm * math.sin(rad), -dist_nm * math.cos(rad)
    return {"x_nm": round(x, 3), "y_nm": round(y, 3), "heading_deg": heading_deg, "speed_kn": speed_kt}


def rescale_mission(mission_num: int) -> dict:
    src_path = MISSIONS_DIR / f"UM{mission_num:02d}.json"
    d = json.loads(src_path.read_text(encoding="utf-8"))
    own_speed_orig = d["own_ship"]["speed_kn"]
    quiet = mission_num in QUIET_MISSIONS

    new_targets = []
    for t in d["targets"]:
        ratio = t["speed_kn"] / own_speed_orig
        orig_brg = bearing_from(d["own_ship"]["x_nm"], d["own_ship"]["y_nm"], t["x_nm"], t["y_nm"])
        new_t = rescale_target(t["heading_deg"], ratio, quiet, orig_brg)
        new_t["name"] = t["name"]
        new_targets.append(new_t)

    d["own_ship"] = {"x_nm": OWN_START_NM[0], "y_nm": OWN_START_NM[1],
                     "heading_deg": 0.0, "speed_kn": OWN_SPEED_KT}
    d["goal"] = {"x_nm": GOAL_NM[0], "y_nm": GOAL_NM[1]}
    d["targets"] = new_targets
    d["description"] = (d["description"] + "\n\nRescaled 2026-09-24 to Imazu scale (12 kt "
                        "own-ship, 6 NM/30 min construction) -- the original small-craft "
                        "scale (own speed ~2.5 m/s, 600-1100 m target range) left no "
                        "geometric margin for a 30 deg-per-command turn to open the 500 m "
                        "safe distance before impact (e.g. UM04 needed >=90 deg on the "
                        "very first command). Bearings/ranges below are a byproduct of "
                        "preserving each target's original heading + speed ratio under "
                        "this construction, not independently chosen.")
    return d


def main() -> None:
    for i in range(1, 14):
        mission = rescale_mission(i)
        out_path = MISSIONS_DIR / f"UM{i:02d}_rescaled.json"
        out_path.write_text(json.dumps(mission, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        own = mission["own_ship"]
        report = []
        for t in mission["targets"]:
            brg = bearing_from(own["x_nm"], own["y_nm"], t["x_nm"], t["y_nm"])
            brg = brg if brg <= 180 else brg - 360
            rng = math.hypot(t["x_nm"] - own["x_nm"], t["y_nm"] - own["y_nm"])
            report.append(f"{t['name']}: hdg={t['heading_deg']:.1f} spd={t['speed_kn']:.2f}kt "
                         f"bearing_from_own={brg:.1f} range={rng:.3f}NM")
        print(f"wrote {out_path.name}: " + "; ".join(report))


if __name__ == "__main__":
    main()
