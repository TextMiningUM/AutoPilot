"""Quality-review STAP 1 (blocking bug, 2026-09-23): real_risk() single-source gate.

Covers: the shared pipeline.oow_agent_spec.real_risk() edge cases, the concrete
give_way/we_are_overtaking_contact example the B1/B2 review found (CPA 308m, TCPA 82s,
Leo risk label "low" -- previously silently dropped to hold_course/none), representative
fixtures across the three wrong-outcome buckets the review counted (hold_course/
speed_up/goal-turn), and a FULL real-dataset regression test over all 7928 Leo frames.
"""
import json

from core import AgentPaths
from pipeline.oow_agent_spec import real_risk, RISK_HORIZON_S
from pipeline.track2.build_oow_scenarios_leo import leo_choose_action, _real_risk, SAFE_CPA_M, LEO_FILE

CACHE = AgentPaths.oow().cache_dir


# ── real_risk() edge cases (STAP 1's required test list) ──────────────────────────────
def test_real_risk_true_when_cpa_below_safe_and_tcpa_within_horizon() -> None:
    assert real_risk(300.0, 100.0, 500.0) is True


def test_real_risk_false_when_tcpa_zero_and_cpa_above_safe() -> None:
    assert real_risk(600.0, 0.0, 500.0) is False


def test_real_risk_false_when_tcpa_negative_even_if_cpa_below_safe() -> None:
    """TCPA < 0 = already past the closest point -- never a risk, however small CPA was."""
    assert real_risk(100.0, -30.0, 500.0) is False


def test_real_risk_false_when_tcpa_beyond_horizon() -> None:
    assert real_risk(100.0, RISK_HORIZON_S + 1, 500.0) is False


def test_real_risk_false_when_cpa_or_tcpa_missing() -> None:
    assert real_risk(None, 100.0, 500.0) is False
    assert real_risk(100.0, None, 500.0) is False


# ── minimal Leo-state fixture builder ──────────────────────────────────────────────────
def _leo_state(own_role: str, encounter_type: str, cpa_m: float, tcpa_s: float,
              risk_label: str = "low", own_speed: float = 10.0, target_speed: float = 10.0,
              goal_xy: tuple[float, float] = (0.0, -1000.0)) -> dict:
    """A minimal-but-schema-complete Leo `state` dict -- own-ship heading 0/north, goal
    BEHIND own-ship by default (off-course, so GOAL COURSE CHECK alone would want a turn)
    so a give_way/stand_on real-risk contact must visibly override it to prove the fix,
    not just coincidentally agree with it."""
    return {
        "own_ship": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": own_speed,
                    "target_speed": target_speed, "movement_status": "underway",
                    "colregs_vessel_type": "power_driven", "name": "LLM_SHIP",
                    "paused": False, "stopped": False},
        "mission": {"x": goal_xy[0], "y": goal_xy[1]},
        "contacts": [{
            "name": "ts1", "own_role": own_role, "encounter_type": encounter_type,
            "cpa_distance_m": cpa_m, "tcpa_s": tcpa_s, "risk": risk_label,
            "range_m": 1000.0, "relative_bearing_deg": 20.0, "heading": 200.0,
            "speed": 8.0, "closing_speed": 3.0, "colregs_vessel_type": "power_driven",
            "active_encounter_rules": [13], "standing_rules": [2, 5, 6, 7],
        }],
        "conditions": {"visibility_condition": "clear"},
    }


# ── the concrete B1/B2-review example: give_way overtaking, Leo risk label "low" ──────
def test_give_way_overtaking_low_risk_label_now_yields_action_and_rule() -> None:
    """Evidence from the review: risk=low, own_role=give_way,
    encounter_type=we_are_overtaking_contact, CPA 308m, TCPA 82s -- previously labelled
    hold_course/none (the risk-label gate silently vetoed a genuine CPA+TCPA risk)."""
    state = _leo_state("give_way", "we_are_overtaking_contact", cpa_m=308.0, tcpa_s=82.0,
                       risk_label="low")
    decision = leo_choose_action(state)
    assert decision["action"] != "hold_course"
    assert decision["encounter_rule"] != "none"
    assert decision["conduct_rule"] != "none"
    assert decision["decisive_contact_name"] == "ts1"


# ── the three wrong-outcome buckets the review counted (hold_course/speed_up/goal-turn) ──
def test_give_way_head_on_low_risk_label_overrides_hold_course_default() -> None:
    state = _leo_state("give_way", "head_on", cpa_m=250.0, tcpa_s=60.0, risk_label="low",
                       own_speed=10.0, target_speed=10.0, goal_xy=(0.0, 1000.0))  # goal ahead -> would be hold_course
    decision = leo_choose_action(state)
    assert decision["action"] not in ("hold_course", None)
    assert decision["encounter_rule"] != "none" and decision["conduct_rule"] != "none"


def test_give_way_crossing_low_risk_label_overrides_speed_up_default() -> None:
    # own-ship below target speed AND on the goal bearing -- would fall through to
    # speed_up if the give-way contact were (wrongly) filtered out by the risk label.
    state = _leo_state("give_way", "crossing", cpa_m=350.0, tcpa_s=120.0, risk_label="low",
                       own_speed=8.0, target_speed=10.0, goal_xy=(0.0, 1000.0))
    decision = leo_choose_action(state)
    assert decision["action"] != "speed_up"
    assert decision["encounter_rule"] != "none" and decision["conduct_rule"] != "none"


def test_give_way_crossing_low_risk_label_overrides_goal_turn_default() -> None:
    # goal off-course (behind/to the side) -- would fall through to a GOAL COURSE CHECK
    # turn if the give-way contact were (wrongly) filtered out by the risk label.
    state = _leo_state("give_way", "crossing", cpa_m=400.0, tcpa_s=200.0, risk_label="low",
                       own_speed=10.0, target_speed=10.0, goal_xy=(-1000.0, 0.0))
    decision = leo_choose_action(state)
    assert decision["category"] != "leo_clear_goal_turn"
    assert decision["encounter_rule"] != "none" and decision["conduct_rule"] != "none"


# ── full real-dataset regression (STAP 1's required "volledige-dataset-test") ─────────
def test_never_speed_up_when_any_give_way_or_stand_on_contact_has_real_risk_full_dataset() -> None:
    """Runs leo_choose_action() over ALL 7928 real Leo frames -- whenever ANY contact
    leo_choose_action() actually recognises as give-way/stand-on (own_role in give_way/
    both_give_way/stand_on) has real_risk()==True, the decision must never be speed_up (a
    give-way or stand-on obligation always overrides mission-progress speed changes).
    Scoped to the roles leo_choose_action() currently handles -- see
    test_stationary_contact_real_risk_not_yet_handled_flagged_for_design_decision() below
    for a SEPARATE, NOT-yet-in-scope gap this same full-dataset scan surfaced."""
    violations = []
    with LEO_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = rec["state"]
            if state["own_ship"].get("paused") or state["own_ship"].get("stopped"):
                continue
            any_real_risk = any(_real_risk(c) for c in state["contacts"]
                                if c["own_role"] in ("give_way", "both_give_way", "stand_on"))
            if not any_real_risk:
                continue
            decision = leo_choose_action(state)
            if decision["action"] == "speed_up":
                violations.append(rec["id"])
    assert not violations, f"{len(violations)} frame(s) chose speed_up despite a real-risk give-way/stand-on contact: {violations[:10]}"


def test_stationary_contact_real_risk_not_yet_handled_flagged_for_design_decision() -> None:
    """NOT a regression test -- a documented, KNOWN gap surfaced by writing the
    full-dataset scan above: leo_choose_action() has no branch at all for
    own_role=="not_applicable" (encounter_type=="stationary_contact", e.g. an anchored
    vessel/fixed object) -- COLREG give-way/stand-on rules don't apply to a stationary
    object the same way, so these contacts fall through this generator's give_way/
    stand_on checks entirely and can still reach speed_up even with a real CPA/TCPA risk.
    Affects 99/7928 frames (~1.2%) at time of writing. This assertion documents the
    CURRENT (gap-having) count so a future fix changes this number deliberately, not
    silently -- it is NOT a claim that 99 is correct, see quality-review STOP 1 report."""
    n_stationary_speed_up_despite_risk = 0
    with LEO_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = rec["state"]
            if state["own_ship"].get("paused") or state["own_ship"].get("stopped"):
                continue
            any_stationary_risk = any(_real_risk(c) for c in state["contacts"]
                                      if c["own_role"] == "not_applicable")
            if not any_stationary_risk:
                continue
            decision = leo_choose_action(state)
            if decision["action"] == "speed_up":
                n_stationary_speed_up_despite_risk += 1
    assert n_stationary_speed_up_despite_risk == 99
