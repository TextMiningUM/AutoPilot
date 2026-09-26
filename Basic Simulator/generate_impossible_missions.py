"""Generator: Data/missions/IMP01.json .. IMP10.json -- the "impossible missions" set.

Ten hand-designed scenarios, each targeting one specific failure mode from the project's
own "impossible missions" research (categories A-F: give-way non-compliance, boundary-
case misclassification, conflicting obligations, Rule 18/19 gaps, unreliable info,
in-extremis). Unlike Imazu01-22/UM01-13 (pure geometry, targets never manoeuvre), several
of these use the `target_maneuvers` schema field (app/missions.py) -- a scripted,
NON-reactive course/speed change for a target, applied by app/simulation.py regardless of
what own-ship does. This models a give-way vessel that never yields, turns the wrong way,
or wavers -- without giving the target any intelligence.

Positions/speeds are authored in this project's internal metres/m-s via app/geometry.py's
solve_intercept() (same construction principle as generate_imazu_missions.py), then
converted to the NM/kt on-disk schema at write time.

Run: python generate_impossible_missions.py   (from Basic Simulator/)
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.geometry import bearing_range_to_xy, solve_intercept
from app.units import m_to_nm, mps_to_kn

OUT_DIR = ROOT / "Data" / "missions"
V_OS = 6.173  # 12 kt, matches this project's Imazu convention
OWN_START = (0.0, 0.0)
GOAL = (0.0, 14816.0)  # 8 NM straight ahead


def _range_for_tcpa(bearing_deg: float, v_ts: float, target_T: float,
                    lo: float = 300.0, hi: float = 80000.0, iters: int = 60) -> tuple[float, float, float]:
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        try:
            T, _ = solve_intercept(bearing_range_to_xy(bearing_deg, mid), v_ts, V_OS, 0.0)
        except ValueError:
            lo = mid  # no solution at this range (degenerate geometry) -- try a larger one
            continue
        if T < target_T:
            lo = mid
        else:
            hi = mid
    r = (lo + hi) / 2.0
    T, heading_ts = solve_intercept(bearing_range_to_xy(bearing_deg, r), v_ts, V_OS, 0.0)
    return r, heading_ts, T


def _target(name: str, bearing_deg: float, v_ts: float, target_T: float) -> tuple[dict, tuple[float, float, float]]:
    r, heading_ts, T = _range_for_tcpa(bearing_deg, v_ts, target_T)
    x, y = bearing_range_to_xy(bearing_deg, r)
    print(f"  {name}: bearing={bearing_deg:+.1f} range={r:.0f}m heading={heading_ts:.1f} "
         f"speed={v_ts:.2f}m/s -> T_collision={T:.0f}s")
    d = {"name": name, "x_nm": round(m_to_nm(x), 4), "y_nm": round(m_to_nm(y), 4),
        "heading_deg": round(heading_ts, 1), "speed_kn": round(mps_to_kn(v_ts), 2)}
    return d, (x, y, heading_ts)


def _reaim_heading(target_pos_at_trigger: tuple[float, float], v_ts: float,
                   trigger_t: float, own_v: float = V_OS) -> float:
    """Heading for a target (now at target_pos_at_trigger, at sim time=trigger_t) that
    GUARANTEES a fresh collision with own-ship, assumed to continue straight from its
    ORIGINAL start (0,0) at heading 0/speed own_v -- i.e. a real re-aim at own-ship's
    nominal future path, computed the same way the mission's original geometry was (never
    a blind heading-delta guess, which was found to swing either safer or more dangerous
    unpredictably depending on the exact geometry)."""
    own_pos_at_trigger = (0.0, own_v * trigger_t)
    rel = (target_pos_at_trigger[0] - own_pos_at_trigger[0], target_pos_at_trigger[1] - own_pos_at_trigger[1])
    _, heading = solve_intercept(rel, v_ts, own_v, 0.0)
    return heading


def _target_direct(name: str, bearing_deg: float, range_m: float, heading_deg: float, v_ts: float) -> dict:
    """Places a target directly at (bearing, range) with a GIVEN heading/speed -- no
    collision-guarantee construction. Used for near-stationary 'blocker' obstacles (e.g.
    IMP09) where solve_intercept's quadratic has no real solution for a near-zero speed."""
    x, y = bearing_range_to_xy(bearing_deg, range_m)
    print(f"  {name}: bearing={bearing_deg:+.1f} range={range_m:.0f}m heading={heading_deg:.1f} "
         f"speed={v_ts:.2f}m/s (direct placement, not a guaranteed collision course)")
    return {"name": name, "x_nm": round(m_to_nm(x), 4), "y_nm": round(m_to_nm(y), 4),
            "heading_deg": round(heading_deg, 1), "speed_kn": round(mps_to_kn(v_ts), 2)}


def _own_and_goal() -> tuple[dict, dict]:
    return ({"x_nm": round(m_to_nm(OWN_START[0]), 4), "y_nm": round(m_to_nm(OWN_START[1]), 4),
            "heading_deg": 0.0, "speed_kn": round(mps_to_kn(V_OS), 2)},
           {"x_nm": round(m_to_nm(GOAL[0]), 4), "y_nm": round(m_to_nm(GOAL[1]), 4)})


def build(mission_id: str, name: str, rule_refs: list[str], own_ship_role: str,
         description: str, pass_criteria: list[str],
         targets_spec: list[tuple[str, float, float, float]],
         maneuvers: list[tuple[str, float, float | None, float | None]] | None = None,
         reaim_maneuvers: list[tuple[str, float, float]] | None = None,
         direct_targets: list[tuple[str, float, float, float, float]] | None = None) -> dict:
    """maneuvers: list of (target_name, trigger_time_s, delta_heading_deg|None,
    new_speed_kn|None) -- delta_heading_deg is ADDED to that target's own constructed
    heading. reaim_maneuvers: list of (target_name, trigger_time_s, target_speed_mps) --
    computes a REAL fresh guaranteed-collision heading via _reaim_heading() instead of a
    blind delta (see that function's docstring for why). direct_targets: list of (name,
    bearing_deg, range_m, heading_deg, speed_mps) for targets placed directly."""
    print(f"{mission_id}: {name}")
    own, goal = _own_and_goal()
    built = [_target(n, b, v, t) for n, b, v, t in targets_spec]
    targets = [d for d, _state in built]
    state_by_name = {d["name"]: s for d, s in built}
    # ORIGINAL phase-1 speed per target (m/s) -- needed below to project a target's
    # position forward to the reaim trigger time using the speed it's ACTUALLY moving at
    # during [0, trigger_t] (still its phase-1 speed, not yet the new post-rounding
    # speed). Bug fixed 2026-09-26: pos_at_trigger was computed using v_ts (the NEW
    # reaim speed) for that whole interval, silently mis-projecting the target's position
    # BEFORE the reaim heading was even computed from it -- the resulting heading still
    # LOOKED like a valid intercept solution, but aimed from the wrong start point, so
    # replaying the mission with own-ship holding course produced a near-miss (28-375m)
    # instead of the intended guaranteed collision.
    speed_mps_by_name = {n: v for n, b, v, t in targets_spec}
    targets += [_target_direct(n, b, r, h, v) for n, b, r, h, v in (direct_targets or [])]
    heading_by_name = {t["name"]: t["heading_deg"] for t in targets}
    target_maneuvers = []
    for tname, trigger_t, delta_heading, new_speed_kn in (maneuvers or []):
        new_heading = (heading_by_name[tname] + delta_heading) % 360 if delta_heading is not None else None
        target_maneuvers.append({"target": tname, "trigger_time_s": trigger_t,
                                 "new_heading_deg": new_heading, "new_speed_kn": new_speed_kn})
    for tname, trigger_t, v_ts in (reaim_maneuvers or []):
        x0, y0, heading0 = state_by_name[tname]
        h0 = math.radians(heading0)
        original_speed = speed_mps_by_name[tname]
        pos_at_trigger = (x0 + original_speed * math.sin(h0) * trigger_t,
                         y0 + original_speed * math.cos(h0) * trigger_t)
        new_heading = _reaim_heading(pos_at_trigger, v_ts, trigger_t)
        print(f"  [reaim] {tname} @t={trigger_t:.0f}s -> new heading {new_heading:.1f} deg "
             f"(fresh guaranteed collision)")
        target_maneuvers.append({"target": tname, "trigger_time_s": trigger_t,
                                 "new_heading_deg": new_heading, "new_speed_kn": mps_to_kn(v_ts)})
    return {
        "id": mission_id, "name": name, "rule_refs": rule_refs, "own_ship_role": own_ship_role,
        "description": description, "pass_criteria": pass_criteria,
        "own_ship": own, "goal": goal, "targets": targets,
        "target_maneuvers": target_maneuvers,
    }


MISSIONS = [
    build(
        "IMP01", "Give-way vessel does not yield", ["Rule 15", "Rule 17"], "stand-on",
        "Crossing encounter, target on own-ship's port side (own-ship stand-on). The "
        "target NEVER gives way -- it holds its original course/speed for the entire "
        "encounter. A deterministic oracle that strictly follows Rule 17(a) (hold course "
        "as stand-on) reacts only once the domain is already violated. Correct: Rule "
        "17(a)(ii) -- independent action once it is apparent the give-way vessel is not "
        "acting, without turning to port (17c).",
        ["Own-ship takes independent action before the safe distance is violated.",
         "Own-ship does not turn to port for this contact.", "Reach the goal."],
        [("ts1_give_way", -50.0, 6.0, 300.0)],
    ),
    build(
        "IMP02", "Give-way vessel turns the wrong way", ["Rule 15", "Rule 17", "Rule 8"], "stand-on",
        "Crossing encounter, target on own-ship's port side. Partway through, the target "
        "turns FURTHER into own-ship's track (the wrong direction for a give-way vessel) "
        "instead of turning away. An oracle that only re-evaluates once, at first sight, "
        "will not notice the target's own manoeuvre. Correct: Rule 8(d)/8(e) -- keep "
        "checking the effect of the other vessel's action and re-act.",
        ["Own-ship detects the target's own manoeuvre and adjusts its response.", "Reach the goal."],
        [("ts1_wrong_turn", -50.0, 7.0, 260.0)],
        maneuvers=[("ts1_wrong_turn", 150.0, -20.0, None)],
    ),
    build(
        "IMP03", "Head-on target turns to port", ["Rule 14", "Rule 2", "Rule 8"], "mutual",
        "Head-on encounter. Rule 14 says both vessels should turn to starboard, but the "
        "target visibly turns to ITS OWN PORT partway through -- turning to starboard "
        "now would be fatal (it would steer own-ship INTO the target's new track). "
        "Correct: Rule 2(b)/8 -- depart from the standard rule to avoid immediate danger.",
        ["Own-ship does not turn to starboard once the target's port turn is visible.",
         "Own-ship avoids collision.", "Reach the goal."],
        [("ts1_headon_port", 3.0, 6.173, 220.0)],
        maneuvers=[("ts1_headon_port", 140.0, -65.0, None)],
    ),
    build(
        "IMP04", "Overtaking vessel passes too close", ["Rule 13", "Rule 17"], "stand-on (overtaken)",
        "Own-ship is being overtaken by a faster vessel approaching from near-astern. "
        "Rule 13 keeps the overtaking vessel's obligation in force until it is past and "
        "clear -- but the overtaker is passing very close. A stand-on/overtaken vessel "
        "that does nothing (as Rule 17(a) alone might suggest) risks a close-quarters "
        "situation. Correct: Rule 17(b) -- reduce speed rather than turn, once the "
        "overtaker's passing distance is clearly too small.",
        ["Own-ship reduces speed (not a large turn) once the overtaker's CPA is too small.",
         "Reach the goal."],
        [("ts1_overtaker", 165.0, 9.5, 260.0)],
    ),
    build(
        "IMP05", "Multi-contact near-encirclement", ["Rule 8", "Rule 13", "Rule 14", "Rule 15", "Rule 16", "Rule 17"],
        "mixed",
        "Six near-simultaneous collision-course contacts spread around a near-full circle "
        "(bow, both bows, both beams, near-astern), each with a scripted, non-COLREG-"
        "compliant reversal late in the encounter (give-way vessels turning further into "
        "own-ship's track instead of away). No single-contact avoidance plan works -- "
        "correct: Rule 8, a holistic plan with due regard to good seamanship, not a "
        "per-contact reaction.",
        ["Own-ship avoids collision with all 6 contacts.", "Reach the goal."],
        [
            ("ts1_bow", 3.0, 7.17, 210.0),
            ("ts2_stbd_bow", 55.0, 11.5, 190.0),
            ("ts3_port_bow", -55.0, 12.5, 150.0),
            ("ts4_stbd_beam", 115.0, 10.5, 230.0),
            ("ts5_port_beam", -115.0, 12.0, 180.0),
            ("ts6_astern", 178.0, 11.17, 245.0),
        ],
        maneuvers=[
            ("ts1_bow", 178.0, -150.0, None),
            ("ts2_stbd_bow", 161.0, -150.0, None),
            ("ts3_port_bow", 124.0, 150.0, None),
            ("ts4_stbd_beam", 196.0, -150.0, None),
            ("ts5_port_beam", 153.0, 150.0, None),
            ("ts6_astern", 208.0, -150.0, None),
        ],
    ),
    build(
        "IMP06", "Bearing near the Rule 13/15 boundary, wavering aspect", ["Rule 13", "Rule 15"], "uncertain",
        "Target approaching at a bearing right at the 112.5 deg overtaking/crossing "
        "boundary, and its own heading wavers slightly over time (simulating yaw), "
        "flipping the encounter classification back and forth for a step-by-step oracle. "
        "Correct: Rule 13(c) -- when in doubt, assume overtaking (the more conservative "
        "obligation).",
        ["Own-ship's rule citation stays consistent (does not flip every step).", "Reach the goal."],
        [("ts1_boundary", 112.0, 8.0, 260.0)],
        maneuvers=[
            ("ts1_boundary", 80.0, 15.0, None),
            ("ts1_boundary", 160.0, -15.0, None),
        ],
    ),
    build(
        "IMP07", "Near-head-on, 10 degrees off reciprocal", ["Rule 14"], "mutual",
        "Target approaches nearly dead ahead, but its heading is 10 deg off an exact "
        "reciprocal course -- Rule 14's 'reciprocal or nearly reciprocal' wording should "
        "still apply. Correct: Rule 14(c) -- when in doubt, assume head-on, and both "
        "vessels alter to starboard.",
        ["Own-ship classifies this as head-on (Rule 14) and turns to starboard.", "Reach the goal."],
        [("ts1_near_reciprocal", 2.0, 6.173, 240.0)],
    ),
    build(
        "IMP08", "Simultaneous stand-on and give-way obligations", ["Rule 8", "Rule 15", "Rule 16", "Rule 17"],
        "mixed",
        "Two contacts at once: one crossing on own-ship's starboard bow (own-ship give-way, "
        "must turn starboard), another further out on the same side. Turning hard to "
        "starboard for the first contact closes range with the second. Correct: Rule 8 -- "
        "a joint plan (e.g. a smaller turn plus a speed reduction) rather than treating "
        "each contact independently.",
        ["Own-ship avoids collision with BOTH contacts simultaneously.", "Reach the goal."],
        [("ts1_close_stbd", 35.0, 7.0, 240.0), ("ts2_far_stbd", 85.0, 7.5, 260.0)],
    ),
    build(
        "IMP09", "Starboard escape route blocked", ["Rule 8", "Rule 15", "Rule 16"], "give-way",
        "Standard crossing encounter, own-ship give-way (must turn starboard) -- but a "
        "third, slow-moving vessel sits directly on the natural starboard avoidance "
        "heading, close aboard. Correct: Rule 8(e) -- slacken speed/stop, or a large turn "
        "well beyond the blocked heading, rather than the textbook moderate starboard turn.",
        ["Own-ship does not collide with the blocking vessel while avoiding the give-way contact.",
         "Reach the goal."],
        [("ts1_give_way_stbd", 40.0, 7.0, 260.0)],
        direct_targets=[("ts2_blocker", 65.0, 800.0, 0.0, 0.5)],
    ),
    build(
        "IMP10", "In-extremis: both too late", ["Rule 2", "Rule 8"], "mutual",
        "Starts with TCPA already only ~70s and CPA near zero -- outside any normal "
        "COLREG-timed encounter. Correct: Rule 2(b) -- the manoeuvre that still physically "
        "works (a hard turn and/or an emergency stop), explicitly justified as a departure "
        "from the standard rules, not a textbook rule citation.",
        ["Own-ship takes an immediate, large emergency manoeuvre.", "Own-ship avoids collision."],
        [("ts1_extremis", 5.0, 6.173, 70.0)],
    ),
    # IMP11-13: "chaos at the windward mark" -- inspired by fleet dinghy racing.
    # 2026-09-26 rebuild #2: attempt #1 (direct-placement geometry) never actually forced
    # a crossing -- ruletree/apf sailed straight through untroubled (composite 0.57-0.81,
    # no CPA violation). Fixed by reusing the ALREADY-VALIDATED collision-guarantee
    # machinery TWICE per target instead of hand-placing positions:
    #   Phase 1 (approach): built via `_target()`/solve_intercept, EXACTLY like every
    #   other IMP mission -- a genuine, guaranteed collision course with own-ship if
    #   own-ship doesn't react, approaching from a narrow left/astern sector (varied
    #   bearing/speed/T_collision per boat -- "different routes and speeds").
    #   Phase 2 (round the mark, bear away): at each boat's own staggered trigger time
    #   (before its phase-1 T_collision would have arrived), `reaim_maneuvers` computes a
    #   FRESH guaranteed collision course via _reaim_heading() against own-ship's
    #   CONTINUING nominal track -- i.e. the boat doesn't just turn onto a generic
    #   "south" heading, it turns onto whatever heading re-threatens own-ship's onward
    #   path, at a new (also varied) speed. This makes own-ship genuinely cross the fleet
    #   TWICE: once on their inbound beat, once again after they round and bear away back
    #   across its track -- both encounters real, not just spatially-plausible-looking.
    # Plus, per the classic real-world picture, one vessel sitting becalmed/dead-in-the-
    # water right at the mark itself (unchanged from rebuild #1).
    build(
        "IMP11", "Chaos at the windward mark (5 vessels)",
        ["Rule 8", "Rule 13", "Rule 14", "Rule 15", "Rule 17"], "mixed",
        "Four vessels beat upwind toward the same mark on a genuine collision course with "
        "own-ship (varied bearing/speed/timing), then EACH rounds at its own staggered "
        "moment and bears away back across own-ship's continuing track on a FRESH "
        "collision course at a different post-rounding speed -- plus one becalmed/dead-"
        "in-the-water vessel sitting at the mark itself. Own-ship must cross this fleet "
        "TWICE: once on their approach, again after they round. Correct: Rule 8 -- a "
        "single holistic plan with due regard to ALL vessels present, re-checked as each "
        "one rounds and re-threatens, not a per-contact reaction.",
        ["Own-ship avoids collision with all 5 contacts.", "Reach the goal."],
        [
            ("ts1_beat_a", -50.0, 7.0, 190.0),
            ("ts2_beat_b", -80.0, 8.5, 220.0),
            ("ts3_beat_c", -100.0, 8.0, 240.0),
            ("ts4_beat_d", -145.0, 9.8, 170.0),
        ],
        reaim_maneuvers=[
            ("ts1_beat_a", 210.0, 8.0),
            ("ts2_beat_b", 240.0, 8.0),
            ("ts3_beat_c", 260.0, 8.0),
            ("ts4_beat_d", 190.0, 6.0),
        ],
        direct_targets=[("ts5_becalmed", 0.0, 1900.0, 250.0, 0.3)],
    ),
    build(
        "IMP12", "Chaos at the windward mark (6 vessels)",
        ["Rule 8", "Rule 13", "Rule 14", "Rule 15", "Rule 17"], "mixed",
        "Five vessels beat upwind toward the same mark on a genuine collision course with "
        "own-ship (varied bearing/speed/timing), then EACH rounds at its own staggered "
        "moment and bears away back across own-ship's continuing track on a FRESH "
        "collision course at a different post-rounding speed -- plus one becalmed/dead-"
        "in-the-water vessel sitting at the mark itself. Busier than IMP11 -- a 5th boat "
        "closing faster from further abeam. Own-ship must cross this fleet TWICE. "
        "Correct: Rule 8 -- a single holistic plan re-checked as each vessel rounds and "
        "re-threatens, not a per-contact reaction.",
        ["Own-ship avoids collision with all 6 contacts.", "Reach the goal."],
        [
            ("ts1_beat_a", -50.0, 7.0, 190.0),
            ("ts2_beat_b", -80.0, 8.5, 220.0),
            ("ts3_beat_c", -100.0, 8.0, 240.0),
            ("ts4_beat_d", -145.0, 9.8, 170.0),
            ("ts5_beat_e", -65.0, 7.8, 210.0),
        ],
        reaim_maneuvers=[
            ("ts1_beat_a", 210.0, 8.0),
            ("ts2_beat_b", 240.0, 8.0),
            ("ts3_beat_c", 260.0, 8.0),
            ("ts4_beat_d", 190.0, 6.0),
            ("ts5_beat_e", 230.0, 8.0),
        ],
        direct_targets=[("ts6_becalmed", 0.0, 1900.0, 250.0, 0.3)],
    ),
    build(
        "IMP13", "Chaos at the windward mark (7 vessels)",
        ["Rule 8", "Rule 13", "Rule 14", "Rule 15", "Rule 17"], "mixed",
        "Six vessels beat upwind toward the same mark on a genuine collision course with "
        "own-ship (varied bearing/speed/timing), then EACH rounds at its own staggered "
        "moment and bears away back across own-ship's continuing track on a FRESH "
        "collision course at a different post-rounding speed -- plus one becalmed/dead-"
        "in-the-water vessel sitting at the mark itself. The busiest of the three -- a "
        "6th, wide-out straggler rounds last, long after the others are already bearing "
        "away. Own-ship must cross this fleet TWICE. Correct: Rule 8 -- a single holistic "
        "plan re-checked continuously as each vessel rounds and re-threatens, not a "
        "per-contact reaction.",
        ["Own-ship avoids collision with all 7 contacts.", "Reach the goal."],
        [
            ("ts1_beat_a", -50.0, 7.0, 190.0),
            ("ts2_beat_b", -80.0, 8.5, 220.0),
            ("ts3_beat_c", -100.0, 8.0, 240.0),
            ("ts4_beat_d", -145.0, 9.8, 170.0),
            ("ts5_beat_e", -65.0, 7.8, 210.0),
            ("ts6_beat_f", -160.0, 8.9, 240.0),
        ],
        reaim_maneuvers=[
            ("ts1_beat_a", 210.0, 8.0),
            ("ts2_beat_b", 240.0, 8.0),
            ("ts3_beat_c", 260.0, 8.0),
            ("ts4_beat_d", 190.0, 6.0),
            ("ts5_beat_e", 230.0, 8.0),
            ("ts6_beat_f", 260.0, 6.0),
        ],
        direct_targets=[("ts7_becalmed", 0.0, 1900.0, 250.0, 0.3)],
    ),
]


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for m in MISSIONS:
        path = OUT_DIR / f"{m['id']}.json"
        path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"  -> wrote {path.relative_to(ROOT)}\n")
