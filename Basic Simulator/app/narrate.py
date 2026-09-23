"""Encounter classification + CPA/TCPA + situation narration, ported from
Brain Storming/pilot_agents.ipynb (already includes the Rule-13 bugfix
described in Design Ideas.docx: overtaking is defined by the bearing of
OWN-SHIP as seen from the TARGET, not the other way around).

All geometry/kinematics here (bearing_and_range, cpa_tcpa, classify_encounter,
contact_line) stays in this project's INTERNAL units (metres, m/s) -- only narrate()'s
final TEXT output converts to nautical miles/knots (via app.units) at the point of
display, matching Sawada et al. (2021)'s own units. See app/units.py's docstring for why
this project converts only at the edges, never in the physics core."""
from __future__ import annotations
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # Auto Pilot/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.missions import Mission, Vessel
from app.units import m_to_nm, mps_to_kn
# bearing_and_range/relative_bearing/classify_encounter/goal_course_check_line live in
# pipeline/oow_agent_spec.py (dependency-free, shared with the Track-2 training-data
# generators) so this calculation is never re-implemented a second time and silently
# drifts -- see that module's docstring.
from pipeline.oow_agent_spec import (
    bearing_and_range, relative_bearing, classify_encounter, goal_course_check_line,
)


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
    it as mandatory.

    Additionally capped so the mission's own total transit time (start to goal, at own-ship's
    nominal speed) always gets at least a handful of decision points -- the TCPA-based
    thresholds above were tuned against the original missions' ~2.5 m/s speed scale, where
    the WHOLE mission takes 1000s+ regardless of how "urgent" its encounter looked. Imazu's
    20 m/s missions break that assumption: Imazu01's encounter has min_tcpa=25s (still
    classified "urgent" -> 10 steps = 100s under the logic above), but the ENTIRE mission
    is only 50s long -- so the single decision at t=0 was the only one ever made, before
    both the encounter AND the whole mission had already finished. This cap only binds when
    a mission's own transit time is short relative to its TCPA-based cadence (i.e. fast,
    short missions like Imazu); for the original slower/longer missions transit time is
    already generous relative to their cadence, so this never changes their behaviour."""
    if not mission.targets:
        base = 20
    else:
        min_tcpa = min(
            cpa_tcpa(mission.own_ship.x, mission.own_ship.y, mission.own_ship.heading, mission.own_ship.speed,
                    t.x, t.y, t.heading, t.speed)[1]
            for t in mission.targets
        )
        if min_tcpa < 250:
            base = 10
        elif min_tcpa < 600:
            base = 15
        else:
            base = 20
    own = mission.own_ship
    gx, gy = mission.goal
    transit_s = math.hypot(gx - own.x, gy - own.y) / own.speed if own.speed > 0 else float("inf")
    min_decisions_across_transit = 5
    transit_cap = max(1, math.floor(transit_s / min_decisions_across_transit / dt))
    return max(1, min(base, transit_cap))


def recommended_max_steps(mission: Mission, dt: float = 10.0) -> int:
    """How many simulation steps a run needs to have a realistic chance of reaching the
    goal, sized from the mission's OWN straight-line transit distance/speed rather than one
    fixed number for every mission (same reasoning as recommended_decision_interval's
    transit_cap above). A fixed 200-step budget (run_llm_scenario.py's old default) was
    tuned against the original ~2.5 m/s-scale missions; Sawada-scale Imazu missions (12 NM
    transit at 12/8.4 kt) need ~3600-5100s just to go straight there, which a 200*10s=2000s
    budget can never reach -- every job silently scored max_steps_reached/0.2 regardless of
    how well it avoided collisions. *1.5 leaves headroom for avoidance detours (same
    multiplier streamlit_app.py's Auto-Run button already uses for the same reason);
    +10/min 20 floor keeps very short/quiet missions from getting an unreasonably tiny cap;
    1200 ceiling bounds worst-case sweep runtime."""
    own = mission.own_ship
    gx, gy = mission.goal
    straight_dist = math.hypot(gx - own.x, gy - own.y)
    est_speed = max(own.speed, 0.1)
    needed_steps = int((straight_dist / est_speed * 1.5) / dt) + 10
    return min(1200, max(20, needed_steps))


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


def narrate(mission: Mission, own: Vessel, targets: list[Vessel], cruise_speed_mps: float | None = None) -> str:
    """Situation report text handed to the OOW agent -- own-ship state, goal, and every
    target's range/bearing/heading/speed/CPA/TCPA. `targets` MUST be the simulation's live,
    currently-moving contact list (Simulation.targets) -- NEVER `mission.targets`, which is
    the static definition frozen at t=0 and never mutated as the run progresses (bug fixed
    2026-09-21: this function used to read mission.targets directly, so every contact's
    range/bearing/CPA/TCPA was computed against its ORIGINAL position forever, silently
    diverging from reality by kilometres over a long run while own-ship's own side of the
    same calculation was correctly live).
    Deliberately leaves out encounter classification, applicable rule(s), and the quiet
    tag: those are exactly what the agent is being asked to work out, so handing them over
    in the prompt would leak the answer instead of testing whether the model can derive it.
    contact_line() still computes/returns them (used elsewhere, e.g. the Evaluation panel's
    degeneracy check).
    Also reports a nominal/rated speed reference (mission.own_ship.speed by default -- the
    mission's ORIGINAL starting speed, distinct from `own.speed`, which mutates as the run
    progresses -- or `cruise_speed_mps` when given, which now always equals that same
    starting speed too -- see build_vessel_constraints() in streamlit_app.py and run_one()
    in run_llm_scenario.py -- since each mission's own speed is treated as its intended
    cruising speed, not a slow start to accelerate away from) and an ETA-at-current-speed: without a speed reference point
    or any sense of time cost, the model had no basis to ever consider speed_up over
    hold_course (observed: 0/1697 decisions across the full sweep ever chose speed_up).
    Also reports the goal BEARING and how many degrees off-course the current heading is:
    observed runs (e.g. s06_crossing_port_fine) where the model correctly avoided a target
    then held that avoidance heading forever, drifting thousands of metres past the goal --
    only the goal's DISTANCE was ever given, never a bearing/heading number the model could
    act on to steer back once clear."""
    gx, gy = mission.goal
    goal_brg, goal_rng = bearing_and_range(own.x, own.y, gx, gy)
    nominal_speed = cruise_speed_mps if cruise_speed_mps is not None else mission.own_ship.speed
    lines = [f"Own-ship at ({m_to_nm(own.x):.3f}, {m_to_nm(own.y):.3f}) NM, heading "
             f"{own.heading:.1f}, speed {mps_to_kn(own.speed):.2f} kt (nominal/rated speed "
             f"for this mission: {mps_to_kn(nominal_speed):.2f} kt)."]
    lines.append(f"Mission goal at ({m_to_nm(gx):.3f}, {m_to_nm(gy):.3f}) NM, "
                f"{m_to_nm(goal_rng):.3f} NM away, bearing {goal_brg:.1f} deg.")
    # A separate, unmistakable line for the ONE number that drives the goal-correction
    # decision -- observed the model repeatedly grabbing a CONTACT's rel.bearing instead
    # (e.g. reasoning "the goal is 82 deg off" while quoting a target's rel.bearing of
    # -82.4, when the goal line itself said "0 deg to port") when this was buried mid-
    # sentence in the goal line above ("42 deg to starboard of current heading" reads too
    # much like a contact's rel.bearing phrasing). This line is ONLY ever about the goal,
    # never about a contact, and spells out the exact action to copy when off course.
    # goal_course_check_line() (pipeline/oow_agent_spec.py) is the SAME function the
    # Track-2 training-data generators call -- never a second, independently-drifting copy.
    lines.append(goal_course_check_line(own.x, own.y, own.heading, gx, gy))
    if own.speed > 0:
        eta = goal_rng / own.speed
        lines.append(f"At current speed, ETA to goal \u2248 {eta:.0f}s if heading straight there.")
    else:
        lines.append("At current speed (stopped), the goal will never be reached.")
    if not targets:
        lines.append("No other ships tracked.")
    else:
        n = len(targets)
        lines.append(f"{n} other ship{'s' if n != 1 else ''}:")
        for tgt in targets:
            c = contact_line(own, tgt)
            tcpa_note = "" if c["closing"] else " (already past closest point, ranges now increasing)"
            lines.append(
                f'  - Ship named "{c["name"]}": range {m_to_nm(c["range_m"]):.3f} NM, rel.bearing '
                f"{c['rel_bearing_deg']:.1f} deg, heading {c['heading']:.1f}, speed "
                f"{mps_to_kn(c['speed']):.2f} kt, CPA {m_to_nm(c['cpa_m']):.3f} NM, "
                f"TCPA {c['tcpa_s']:.0f}s{tcpa_note}"
            )
    return "\n".join(lines)
