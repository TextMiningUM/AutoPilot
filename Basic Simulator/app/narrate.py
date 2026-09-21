"""Encounter classification + CPA/TCPA + situation narration, ported from
Brain Storming/pilot_agents.ipynb (already includes the Rule-13 bugfix
described in Design Ideas.docx: overtaking is defined by the bearing of
OWN-SHIP as seen from the TARGET, not the other way around)."""
from __future__ import annotations
import math

from app.missions import Mission, Vessel


def bearing_and_range(ox: float, oy: float, tx: float, ty: float) -> tuple[float, float]:
    dx, dy = tx - ox, ty - oy
    return math.degrees(math.atan2(dx, dy)) % 360.0, math.hypot(dx, dy)


def relative_bearing(own_heading: float, true_bearing: float) -> float:
    return (true_bearing - own_heading + 540) % 360 - 180


def cpa_tcpa(ox: float, oy: float, ohdg: float, ospd: float,
             tx: float, ty: float, thdg: float, tspd: float) -> tuple[float, float]:
    """Closest point of approach (m) and time-to-CPA (s), clamped to t>=0
    (a closing CPA that already happened in the past isn't a live risk)."""
    oh, th = math.radians(ohdg), math.radians(thdg)
    vox, voy = ospd * math.sin(oh), ospd * math.cos(oh)
    vtx, vty = tspd * math.sin(th), tspd * math.cos(th)
    dx, dy = tx - ox, ty - oy
    dvx, dvy = vtx - vox, vty - voy
    rel_sq = dvx ** 2 + dvy ** 2
    if rel_sq < 1e-6:
        return math.hypot(dx, dy), 0.0
    t = max(0.0, -(dx * dvx + dy * dvy) / rel_sq)
    return math.hypot(dx + dvx * t, dy + dvy * t), t


def recommended_decision_interval(mission: Mission, dt: float = 10.0) -> int:
    """How many simulation steps should pass between LLM decision calls for this mission,
    based on how urgent its closest encounter is at t=0 -- a fast-closing contact (short
    TCPA) keeps the same cadence used before this existed (10 steps), while a quiet mission
    (no contacts, or a slow-developing one) can safely go longer between (expensive) LLM
    calls without missing anything, cutting how many calls a full run needs. Never shorter
    than 10 -- this is meant to REDUCE call volume for calm missions, not add extra calls to
    tight ones. There is no simulator-level min_cpa backstop -- meeting min_cpa is entirely
    the agent's own responsibility (see SYSTEM_OOW_AGENT), same as reaching the goal, so a
    sparser decision cadence genuinely means fewer chances to react, not just fewer LLM calls
    with a safety net underneath. Callers should still let a user/CLI override this, never treat
    it as mandatory."""
    if not mission.targets:
        return 20
    min_tcpa = min(
        cpa_tcpa(mission.own_ship.x, mission.own_ship.y, mission.own_ship.heading, mission.own_ship.speed,
                t.x, t.y, t.heading, t.speed)[1]
        for t in mission.targets
    )
    if min_tcpa < 250:
        return 10
    if min_tcpa < 600:
        return 15
    return 20


def classify_encounter(own_x: float, own_y: float, own_hdg: float,
                        tgt_x: float, tgt_y: float, tgt_hdg: float) -> tuple[str, list[str], float]:
    """Correct Rule 13 check: overtaking is defined by the bearing of OWN-SHIP
    as seen from the TARGET (>112.5 deg abaft the target's beam), not by the
    bearing of the target as seen from own-ship -- using only the latter
    misclassifies real overtaking cases as ordinary crossing whenever the
    closing angle is fine/moderate rather than near-dead-astern."""
    brg_own_to_tgt, _ = bearing_and_range(own_x, own_y, tgt_x, tgt_y)
    rel_from_own = relative_bearing(own_hdg, brg_own_to_tgt)

    brg_tgt_to_own, _ = bearing_and_range(tgt_x, tgt_y, own_x, own_y)
    rel_from_tgt = relative_bearing(tgt_hdg, brg_tgt_to_own)

    course_diff = (tgt_hdg - own_hdg + 540) % 360 - 180
    if abs(rel_from_own) <= 6 and abs(abs(course_diff) - 180) <= 20:
        return "head_on", ["Rule 14"], rel_from_own
    if abs(rel_from_tgt) > 112.5:
        return "we_are_overtaking_target", ["Rule 13"], rel_from_own
    if abs(rel_from_own) > 112.5:
        return "target_is_overtaking_us", ["Rule 13"], rel_from_own
    if rel_from_own > 0:
        return "crossing_target_on_starboard", ["Rule 15", "Rule 16"], rel_from_own
    return "crossing_target_on_port", ["Rule 15", "Rule 17"], rel_from_own


# Below these thresholds a target isn't a live collision-avoidance concern
# (per Design Ideas.docx's DTU-paper finding: most real encounters need no
# action) -- purely descriptive labels for the narration, no behaviour change.
QUIET_TCPA_S = 600.0
QUIET_CPA_M = 300.0


def contact_line(own: Vessel, tgt: Vessel) -> dict:
    _, rng = bearing_and_range(own.x, own.y, tgt.x, tgt.y)
    cpa, tcpa = cpa_tcpa(own.x, own.y, own.heading, own.speed, tgt.x, tgt.y, tgt.heading, tgt.speed)
    # Same dot-product sign check as inside cpa_tcpa(), duplicated here rather than
    # changing cpa_tcpa()'s 2-tuple return -- streamlit_app.py's metric boxes call
    # cpa_tcpa() directly and only need the numbers. Tells us whether the RAW (unclamped)
    # time-to-closest-approach was negative, i.e. the closest point already happened and
    # the vessels are now diverging, vs still closing -- a bare "TCPA 0s" doesn't
    # distinguish these, and the model was repeatedly misreading the former as "collision
    # imminent" even with an explicit system-prompt clarification (see basic_simulator.md).
    oh, th = math.radians(own.heading), math.radians(tgt.heading)
    vox, voy = own.speed * math.sin(oh), own.speed * math.cos(oh)
    vtx, vty = tgt.speed * math.sin(th), tgt.speed * math.cos(th)
    dx, dy = tgt.x - own.x, tgt.y - own.y
    dvx, dvy = vtx - vox, vty - voy
    rel_sq = dvx ** 2 + dvy ** 2
    closing = rel_sq >= 1e-6 and -(dx * dvx + dy * dvy) > 0
    enc, rules, rel = classify_encounter(own.x, own.y, own.heading, tgt.x, tgt.y, tgt.heading)
    quiet = tcpa > QUIET_TCPA_S or cpa > QUIET_CPA_M
    return {
        "name": tgt.name, "range_m": rng, "rel_bearing_deg": rel,
        "heading": tgt.heading, "speed": tgt.speed,
        "cpa_m": cpa, "tcpa_s": tcpa, "closing": closing, "encounter": enc, "rules": rules, "quiet": quiet,
    }


def narrate(mission: Mission, own: Vessel, cruise_speed_mps: float | None = None) -> str:
    """Situation report text handed to the OOW agent -- own-ship state, goal, and every
    target's range/bearing/heading/speed/CPA/TCPA. Deliberately leaves out encounter
    classification, applicable rule(s), and the quiet tag: those are exactly what the
    agent is being asked to work out, so handing them over in the prompt would leak the
    answer instead of testing whether the model can derive it. contact_line() still
    computes/returns them (used elsewhere, e.g. the Evaluation panel's degeneracy check).
    Also reports a nominal/rated speed reference (mission.own_ship.speed by default -- the
    mission's ORIGINAL starting speed, distinct from `own.speed`, which mutates as the run
    progresses -- or `cruise_speed_mps` when given, e.g. from the live simulator's
    VesselConstraints/sidebar "Cruise speed" setting, so a user-configured resume speed
    actually reaches the agent instead of always falling back to whatever speed the mission
    file happened to start at) and an ETA-at-current-speed: without a speed reference point
    or any sense of time cost, the model had no basis to ever consider speed_up over
    hold_course (observed: 0/1697 decisions across the full sweep ever chose speed_up).
    Also reports the goal BEARING and how many degrees off-course the current heading is:
    observed runs (e.g. s06_crossing_port_fine) where the model correctly avoided a target
    then held that avoidance heading forever, drifting thousands of metres past the goal --
    only the goal's DISTANCE was ever given, never a bearing/heading number the model could
    act on to steer back once clear."""
    gx, gy = mission.goal
    goal_brg, goal_rng = bearing_and_range(own.x, own.y, gx, gy)
    off_course = relative_bearing(own.heading, goal_brg)
    nominal_speed = cruise_speed_mps if cruise_speed_mps is not None else mission.own_ship.speed
    lines = [f"Own-ship at ({own.x:.1f}, {own.y:.1f}), heading {own.heading:.1f}, "
             f"speed {own.speed:.2f} m/s (nominal/rated speed for this mission: "
             f"{nominal_speed:.2f} m/s)."]
    lines.append(f"Mission goal at ({gx:.1f}, {gy:.1f}), {goal_rng:.0f}m away, bearing "
                f"{goal_brg:.1f} deg.")
    # A separate, unmistakable line for the ONE number that drives the goal-correction
    # decision -- observed the model repeatedly grabbing a CONTACT's rel.bearing instead
    # (e.g. reasoning "the goal is 82 deg off" while quoting a target's rel.bearing of
    # -82.4, when the goal line itself said "0 deg to port") when this was buried mid-
    # sentence in the goal line above ("42 deg to starboard of current heading" reads too
    # much like a contact's rel.bearing phrasing). This line is ONLY ever about the goal,
    # never about a contact, and spells out the exact action to copy when off course.
    if abs(off_course) <= 10:
        lines.append("GOAL COURSE CHECK: heading is ALREADY on the goal bearing (within "
                     "10 deg) -- no turn needed for the goal.")
    else:
        side = "starboard" if off_course > 0 else "port"
        turn_action = "turn_right" if off_course > 0 else "turn_left"
        lines.append(f"GOAL COURSE CHECK: heading is {abs(off_course):.0f} deg off the goal "
                    f"bearing, to {side} -- to correct, use action \"{turn_action}\" with "
                    f"degrees={abs(off_course):.0f} (unless a target poses a real collision "
                    f"risk, which takes precedence).")
    if own.speed > 0:
        eta = goal_rng / own.speed
        lines.append(f"At current speed, ETA to goal \u2248 {eta:.0f}s if heading straight there.")
    else:
        lines.append("At current speed (stopped), the goal will never be reached.")
    if not mission.targets:
        lines.append("No other ships tracked.")
    else:
        n = len(mission.targets)
        lines.append(f"{n} other ship{'s' if n != 1 else ''}:")
        for tgt in mission.targets:
            c = contact_line(own, tgt)
            tcpa_note = "" if c["closing"] else " (already past closest point, ranges now increasing)"
            lines.append(
                f'  - Ship named "{c["name"]}": range {c["range_m"]:.0f}m, rel.bearing '
                f"{c['rel_bearing_deg']:.1f} deg, heading {c['heading']:.1f}, speed {c['speed']:.2f} m/s, "
                f"CPA {c['cpa_m']:.0f}m, TCPA {c['tcpa_s']:.0f}s{tcpa_note}"
            )
    return "\n".join(lines)
