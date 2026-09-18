"""One-off generator: writes Data/missions/*.json from the same scenario
geometry as Brain Storming/generate_moos_scenarios.py (single/two/three-target
Imazu-style buckets), plus two "quiet" scenarios (long TCPA / wide CPA, correct
answer = hold_course) per the DTU-paper finding in Design Ideas.docx that
Imazu-only benchmarks never test whether an agent correctly does nothing.

Run with: python -m app.build_missions   (from the Basic Simulator/ folder)
"""
from __future__ import annotations
import json
from pathlib import Path

from app.geometry import compute_target, same_line_target, bearing_range_to_xy

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MISSIONS_DIR = ROOT / "Data" / "missions"

V_OS = 2.5  # m/s, ~4.9 kn, own-ship nominal speed -- same as generate_moos_scenarios.py


def mission(sid, name, rule_refs, os_role, description, pass_criteria,
            targets, mission_len_m=2400, v_os=V_OS):
    return {
        "id": sid,
        "name": name,
        "rule_refs": rule_refs,
        "own_ship_role": os_role,
        "description": description,
        "pass_criteria": pass_criteria,
        "own_ship": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": v_os},
        "goal": {"x": 0.0, "y": mission_len_m},
        "targets": [
            {"name": f"ts{i+1}", "x": t["start_x"], "y": t["start_y"],
             "heading": t["heading"], "speed": t["speed"]}
            for i, t in enumerate(targets)
        ],
    }


MISSIONS = [
    mission(
        "s01_head_on", "Head-on", ["Rule 14"], "mutual (both alter to starboard)",
        "Own-ship meets a single target nearly dead ahead on a reciprocal course.",
        ["Own-ship alters course to starboard (not port).",
         "Minimum CPA stays above the configured safe-distance threshold.",
         "Own-ship returns to her original track/heading after the target is past and clear."],
        [compute_target(bearing_from_os_deg=4, range_m=1100, v_ts=2.5, v_os=V_OS)],
    ),
    mission(
        "s02_crossing_stbd_fine", "Crossing - target fine on starboard bow",
        ["Rule 15", "Rule 16"], "give-way",
        "Target is on own-ship's starboard bow at a fine angle (~20 deg relative). Own-ship must give way.",
        ["Own-ship is give-way: she alters course (normally to starboard) and/or speed early.",
         "Own-ship avoids crossing ahead of the target if circumstances allow.",
         "Minimum CPA stays above the safe-distance threshold."],
        [compute_target(bearing_from_os_deg=20, range_m=900, v_ts=2.2, v_os=V_OS)],
    ),
    mission(
        "s03_crossing_stbd_broad", "Crossing - target broad on starboard bow",
        ["Rule 15", "Rule 16"], "give-way",
        "Target is broad on own-ship's starboard bow (~65 deg relative). Own-ship must give way.",
        ["Own-ship is give-way: early, substantial action to keep well clear.",
         "Minimum CPA stays above the safe-distance threshold."],
        [compute_target(bearing_from_os_deg=65, range_m=900, v_ts=2.6, v_os=V_OS)],
    ),
    mission(
        "s04_crossing_stbd_abeam", "Crossing - target near starboard beam",
        ["Rule 15", "Rule 16"], "give-way",
        "Target is nearly abeam on own-ship's starboard side (~90 deg relative). Own-ship must give way.",
        ["Own-ship is give-way and passes astern of the target.",
         "Minimum CPA stays above the safe-distance threshold."],
        [compute_target(bearing_from_os_deg=90, range_m=800, v_ts=3.2, v_os=V_OS)],
    ),
    mission(
        "s05_crossing_port_broad", "Crossing - target broad on port bow (own-ship stand-on)",
        ["Rule 15", "Rule 17"], "stand-on",
        "Target is broad on own-ship's port bow (~60 deg relative to port). The target does not "
        "give way, so own-ship must hold course/speed under Rule 17(a)(i) and only act under "
        "Rule 17(a)(ii)/(b) if the target clearly fails to act and collision becomes otherwise unavoidable.",
        ["Own-ship holds course and speed for a meaningful initial period (does not manoeuvre prematurely).",
         "If the target never gives way, own-ship eventually takes late avoiding action under Rule 17(b) "
         "rather than colliding.",
         "Minimum CPA stays above the safe-distance threshold."],
        [compute_target(bearing_from_os_deg=300, range_m=900, v_ts=2.2, v_os=V_OS)],
    ),
    mission(
        "s06_crossing_port_fine", "Crossing - target fine on port bow (own-ship stand-on)",
        ["Rule 15", "Rule 17"], "stand-on",
        "Target is fine on own-ship's port bow (~20 deg relative to port). Own-ship is stand-on.",
        ["Own-ship holds course and speed initially.",
         "Own-ship does not alter to port toward the target if she must eventually act (Rule 17(c)).",
         "Minimum CPA stays above the safe-distance threshold."],
        [compute_target(bearing_from_os_deg=340, range_m=900, v_ts=2.2, v_os=V_OS)],
    ),
    mission(
        "s07_overtaking_os_gives_way", "Overtaking - own-ship overtakes a slower target",
        ["Rule 13"], "give-way (overtaking)",
        "A slower target is on own-ship's track ahead of her. Own-ship closes from astern and must "
        "keep clear as the overtaking vessel until finally past and clear.",
        ["Own-ship's initial approach is recognisably from more than 22.5 deg abaft the target's beam.",
         "Own-ship keeps clear of the target throughout the pass (does not cut in front).",
         "Minimum CPA stays above the safe-distance threshold."],
        [same_line_target(range_ahead_m=700, v_ts=1.0, v_os=V_OS, astern=False)],
    ),
    mission(
        "s08_overtaking_os_stands_on", "Overtaking - own-ship is overtaken (stand-on)",
        ["Rule 13", "Rule 17"], "stand-on (being overtaken)",
        "A faster target approaches from astern of own-ship and must keep clear as the overtaking vessel.",
        ["Own-ship holds course and speed while being overtaken, absent evidence the target is not "
         "keeping clear.",
         "Minimum CPA stays above the safe-distance threshold."],
        [same_line_target(range_ahead_m=600, v_ts=3.5, v_os=V_OS, astern=True)],
    ),
    mission(
        "s09_double_crossing_squeeze", "Two-target: crossing from both sides",
        ["Rule 15", "Rule 16", "Rule 17", "Rule 8(d)"], "give-way to TS1 / stand-on to TS2",
        "TS1 crosses from own-ship's starboard bow (give-way) while TS2 simultaneously crosses "
        "from own-ship's port bow (stand-on). Own-ship must satisfy both encounters with one "
        "coherent manoeuvre plan.",
        ["Own-ship's manoeuvre for TS1 (starboard give-way) does not create a new close-quarters "
         "situation with TS2 (Rule 8(d)/8(c)).",
         "Minimum CPA to BOTH targets stays above the safe-distance threshold."],
        [compute_target(bearing_from_os_deg=45, range_m=900, v_ts=2.2, v_os=V_OS),
         compute_target(bearing_from_os_deg=315, range_m=900, v_ts=2.2, v_os=V_OS)],
    ),
    mission(
        "s10_headon_plus_crossing", "Two-target: head-on plus crossing from starboard",
        ["Rule 14", "Rule 15", "Rule 16", "Rule 8(d)"], "mutual with TS1 / give-way to TS2",
        "TS1 is head-on; TS2 crosses from own-ship's starboard bow at the same time. Both encounters "
        "call for a starboard alteration, testing whether a single coherent manoeuvre satisfies both.",
        ["Own-ship alters to starboard (satisfying both Rule 14 and Rule 15/16 simultaneously).",
         "Minimum CPA to BOTH targets stays above the safe-distance threshold."],
        [compute_target(bearing_from_os_deg=0, range_m=1100, v_ts=2.5, v_os=V_OS),
         compute_target(bearing_from_os_deg=60, range_m=900, v_ts=2.3, v_os=V_OS)],
    ),
    mission(
        "s11_overtake_plus_crossing", "Two-target: overtaking plus crossing from starboard",
        ["Rule 13", "Rule 15", "Rule 16"], "give-way to both (different rules)",
        "A slow target is ahead on own-ship's track (overtaking case) while a second target "
        "simultaneously crosses from own-ship's starboard bow (crossing case).",
        ["Own-ship keeps clear of the overtaken target (Rule 13) while also giving way to the "
         "crossing target (Rule 15/16) without conflating the two into a single miscategorised manoeuvre.",
         "Minimum CPA to BOTH targets stays above the safe-distance threshold."],
        [same_line_target(range_ahead_m=700, v_ts=1.0, v_os=V_OS, astern=False),
         compute_target(bearing_from_os_deg=50, range_m=900, v_ts=2.0, v_os=V_OS)],
    ),
    mission(
        "s12_converging_cluster", "Three-target: converging cluster",
        ["Rule 14", "Rule 15", "Rule 16", "Rule 17", "Rule 8(d)"],
        "mixed give-way/stand-on across three simultaneous encounters",
        "TS1 crosses from starboard, TS2 crosses from port, and TS3 is head-on, all converging on "
        "own-ship at roughly the same time - the hardest single-agent test in this set.",
        ["Own-ship finds a single trajectory that keeps a safe CPA from all three targets simultaneously.",
         "Own-ship does not alter to port toward TS2 while manoeuvring for TS1/TS3.",
         "Minimum CPA to ALL THREE targets stays above the safe-distance threshold."],
        [compute_target(bearing_from_os_deg=40, range_m=850, v_ts=2.0, v_os=V_OS),
         compute_target(bearing_from_os_deg=320, range_m=850, v_ts=2.0, v_os=V_OS),
         compute_target(bearing_from_os_deg=0, range_m=1200, v_ts=2.3, v_os=V_OS)],
    ),
    # --- "Quiet" scenarios (NEW): correct answer is hold_course, no rule violated by
    # inaction -- per Design Ideas.docx's DTU-paper finding that real AIS data shows
    # 86% of encounters need no action, and Imazu-only benchmarks never test this.
    mission(
        "q01_quiet_long_tcpa", "Quiet - distant target, very long TCPA",
        [], "none (no give-way/stand-on situation yet)",
        "A target is far away and on a track that will not create risk of collision for a long time. "
        "The correct response is to hold course/speed and simply monitor -- not manoeuvre.",
        ["Own-ship holds course and speed (no manoeuvre).",
         "Own-ship does not treat this as an imminent-risk encounter."],
        [{"start_x": bearing_range_to_xy(30, 8000)[0], "start_y": bearing_range_to_xy(30, 8000)[1],
          "heading": 90.0, "speed": 1.0}],
        mission_len_m=4000,
    ),
    mission(
        "q02_quiet_wide_cpa", "Quiet - crossing target with a naturally wide CPA",
        [], "none (risk of collision does not exist)",
        "A target crosses well ahead/astern with a geometry that already gives a wide closest-point-"
        "of-approach without any action -- Rule 7(a)/(d) risk-of-collision does not exist here.",
        ["Own-ship holds course and speed (no manoeuvre).",
         "Own-ship does not over-react to a target that poses no real risk."],
        [{"start_x": -3000.0, "start_y": 400.0, "heading": 180.0, "speed": 2.0}],
        mission_len_m=2400,
    ),
]


def main() -> None:
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for m in MISSIONS:
        out = MISSIONS_DIR / f"{m['id']}.json"
        out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"Wrote {len(MISSIONS)} mission files to {MISSIONS_DIR}")


if __name__ == "__main__":
    main()
