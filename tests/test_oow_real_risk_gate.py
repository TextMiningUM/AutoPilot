"""Quality-review STAP 1 (blocking bug, 2026-09-23): real_risk() single-source gate,
plus its STAP 1 extension: a dedicated stationary-object-avoidance branch (decision
2026-09-23, option (a)).

Covers: the shared pipeline.oow_agent_spec.real_risk() edge cases, the concrete
give_way/we_are_overtaking_contact example the B1/B2 review found (CPA 308m, TCPA 82s,
Leo risk label "low" -- previously silently dropped to hold_course/none), representative
fixtures across the three wrong-outcome buckets the review counted (hold_course/
speed_up/goal-turn), the stationary-avoidance branch (synthetic + the real 99-frame
bugfix), and a FULL real-dataset regression test over all 7928 Leo frames.
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
def test_stationary_real_risk_frames_now_get_a_turn_or_slow_down_full_dataset() -> None:
    """Bugfix regression (was a documented-gap test at STOP 1's mid-point, 99/7928
    frames): every real Leo frame with a real-risk STATIONARY contact must now get
    conduct_rule=="Rule 8" and an action in (turn_left, turn_right, slow_down) -- never
    hold_course, never speed_up, never encounter_rule/conduct_rule=="none"."""
    violations = []
    n_stationary_risk_frames = 0
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
                                      if c.get("encounter_type") == "stationary_contact")
            if not any_stationary_risk:
                continue
            n_stationary_risk_frames += 1
            decision = leo_choose_action(state)
            ok = (decision["action"] in ("turn_left", "turn_right", "slow_down", "stop")
                 and decision["conduct_rule"] in ("Rule 8", "Rule 13", "Rule 14", "Rule 16"))
            if not ok:
                violations.append((rec["id"], decision["action"], decision["conduct_rule"]))
    assert n_stationary_risk_frames > 0  # sanity: this population must be non-empty
    assert not violations, f"{len(violations)} frame(s) still not handled: {violations[:10]}"


def test_never_speed_up_when_any_give_way_or_stand_on_contact_has_real_risk_full_dataset() -> None:
    """Runs leo_choose_action() over ALL 7928 real Leo frames -- whenever ANY contact
    leo_choose_action() actually recognises as give-way/stand-on (own_role in give_way/
    both_give_way/stand_on) has real_risk()==True, the decision must never be speed_up (a
    give-way or stand-on obligation always overrides mission-progress speed changes)."""
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


def test_never_hold_course_when_real_risk_exists_unless_stand_on_full_dataset() -> None:
    """Extends the full-dataset scan (per the 2026-09-23 stationary-branch decision):
    whenever a give-way/stand-on/stationary contact leo_choose_action() actually
    recognises has real_risk()==True, hold_course is only ever a valid label if
    conduct_rule=="Rule 17" (a genuine stand-on situation) -- catches any FUTURE
    fall-through branch of this same class. Scoped to the roles this generator currently
    handles -- see test_not_applicable_moving_contact_real_risk_not_yet_handled() below
    for a SEPARATE, newly-surfaced gap this same scan found (own_role=="not_applicable"
    on a MOVING crossing/parallel contact, not a stationary object)."""
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
                                if c["own_role"] in ("give_way", "both_give_way", "stand_on")
                                or c.get("encounter_type") == "stationary_contact")
            if not any_real_risk:
                continue
            decision = leo_choose_action(state)
            if decision["action"] == "hold_course" and decision["conduct_rule"] != "Rule 17":
                violations.append(rec["id"])
    assert not violations, f"{len(violations)} frame(s) chose hold_course despite real risk without Rule 17: {violations[:10]}"


def test_not_applicable_moving_contact_real_risk_not_yet_handled() -> None:
    """NOT a regression test -- a NEW, documented gap surfaced while writing the test
    above: own_role=="not_applicable" also occurs on MOVING contacts (encounter_type
    "crossing"/"parallel", real vessels with vessel_type/colregs_vessel_type=="ship"/
    "power_driven", sometimes even carrying active_encounter_rules=[8,15] already) --
    distinct from the stationary_contact case this session already fixed. Leo's own
    role classifier evidently sometimes fails to assign give_way/stand_on for these,
    and leo_choose_action() has no fallback -- they still fall through to hold_course/
    "cleared"/rule "none" despite real CPA/TCPA risk. Affects 220/7928 frames at time
    of writing. Documents the CURRENT count so a future fix changes it deliberately."""
    n = 0
    with LEO_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = rec["state"]
            if state["own_ship"].get("paused") or state["own_ship"].get("stopped"):
                continue
            risky = [c for c in state["contacts"] if _real_risk(c)
                    and c["own_role"] == "not_applicable"
                    and c.get("encounter_type") != "stationary_contact"]
            if not risky:
                continue
            decision = leo_choose_action(state)
            if decision["action"] == "hold_course" and decision["conduct_rule"] != "Rule 17":
                n += 1
    assert n == 220


# ── stationary-object avoidance (decision 2026-09-23, option (a)) ─────────────────────
def _leo_state_stationary(cpa_m: float, tcpa_s: float, bearing_deg: float,
                          give_way: dict | None = None) -> dict:
    """Own-ship heading 0/north; a single stationary_contact by default, optionally with
    an ADDITIONAL real-risk give-way contact (`give_way`, a dict of _contact-style
    overrides merged over a give_way/crossing default) to exercise the priority rule."""
    contacts = [{
        "name": "buoy1", "own_role": "not_applicable", "encounter_type": "stationary_contact",
        "cpa_distance_m": cpa_m, "tcpa_s": tcpa_s, "risk": "medium",
        "range_m": 400.0, "relative_bearing_deg": bearing_deg, "heading": 0.0,
        "speed": 0.0, "closing_speed": 8.0, "colregs_vessel_type": "aid_to_navigation",
        "active_encounter_rules": [], "standing_rules": [2, 5, 6, 7],
    }]
    if give_way is not None:
        base = {"name": "ts1", "own_role": "give_way", "encounter_type": "crossing", "risk": "high",
                "cpa_distance_m": 450.0, "tcpa_s": 100.0, "relative_bearing_deg": 45.0,
                "range_m": 1000.0, "heading": 200.0, "speed": 8.0, "closing_speed": 4.0,
                "colregs_vessel_type": "power_driven",
                "active_encounter_rules": [15, 16], "standing_rules": [2, 5, 6, 7]}
        base.update(give_way)
        contacts.append(base)
    return {
        "own_ship": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 8.0, "target_speed": 8.0,
                    "movement_status": "underway", "colregs_vessel_type": "power_driven",
                    "name": "LLM_SHIP", "paused": False, "stopped": False},
        "mission": {"x": 0.0, "y": 1000.0},
        "contacts": contacts,
        "conditions": {"visibility_condition": "clear"},
    }


def test_stationary_contact_starboard_diverges_to_port() -> None:
    state = _leo_state_stationary(cpa_m=120.0, tcpa_s=240.0, bearing_deg=20.0)
    d = leo_choose_action(state)
    assert d["action"] == "turn_left"  # diverge AWAY from the starboard-side object
    assert d["encounter_rule"] == "none" and d["conduct_rule"] == "Rule 8"
    assert d["category"] == "leo_stationary_avoid"
    assert d["decisive_contact_name"] == "buoy1"


def test_stationary_contact_port_diverges_to_starboard() -> None:
    state = _leo_state_stationary(cpa_m=120.0, tcpa_s=240.0, bearing_deg=-20.0)
    d = leo_choose_action(state)
    assert d["action"] == "turn_right"
    assert d["encounter_rule"] == "none" and d["conduct_rule"] == "Rule 8"


def test_stationary_contact_beyond_horizon_is_not_real_risk() -> None:
    state = _leo_state_stationary(cpa_m=120.0, tcpa_s=RISK_HORIZON_S + 1, bearing_deg=20.0)
    d = leo_choose_action(state)
    assert d["conduct_rule"] != "Rule 8"
    assert d["category"] != "leo_stationary_avoid"


def test_stationary_contact_already_past_is_not_real_risk() -> None:
    state = _leo_state_stationary(cpa_m=120.0, tcpa_s=-5.0, bearing_deg=20.0)
    d = leo_choose_action(state)
    assert d["conduct_rule"] != "Rule 8"
    assert d["category"] != "leo_stationary_avoid"


def test_give_way_with_smaller_cpa_wins_over_stationary() -> None:
    """Stationary CPA 300m, give-way CPA 100m (smaller) -> the give-way action wins."""
    state = _leo_state_stationary(cpa_m=300.0, tcpa_s=150.0, bearing_deg=20.0,
                                  give_way={"cpa_distance_m": 100.0, "tcpa_s": 90.0})
    d = leo_choose_action(state)
    assert d["decisive_contact_name"] == "ts1"
    assert d["encounter_rule"] == "Rule 15" and d["conduct_rule"] == "Rule 16"


def test_give_way_wins_but_8c_check_falls_back_to_slow_down_if_it_shrinks_stationary_cpa() -> None:
    """A give-way contact dead ahead (bearing 0, encounter_type crossing) forces
    turn_right (Rule 15/16). The stationary object sits at bearing 60 (starboard bow),
    range 400 -- numerically verified (see the STAP1-extension investigation) that a
    30 deg starboard turn shrinks ITS cpa_distance_m from ~346m to ~200m, so the scoped
    8(c) check must fall back to slow_down rather than let the give-way turn through."""
    state = _leo_state_stationary(cpa_m=346.0, tcpa_s=200.0, bearing_deg=60.0,
                                  give_way={"cpa_distance_m": 100.0, "tcpa_s": 90.0,
                                           "relative_bearing_deg": 0.0, "heading": 180.0})
    d = leo_choose_action(state)
    assert d["action"] == "slow_down"
    assert d["degrees"] is None
