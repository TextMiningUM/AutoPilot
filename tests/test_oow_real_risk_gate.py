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


def test_never_speed_up_when_any_contact_has_real_risk_full_dataset() -> None:
    """Quality-review STAP 2 (2026-09-23), point 4: NO role/type qualification -- runs
    leo_choose_action() over ALL 7928 real Leo frames, and whenever ANY contact
    (regardless of own_role/encounter_type) has real_risk()==True, the decision must
    never be speed_up. The not_applicable-moving-contact fix (geometric fallback or
    frame exclusion) is what makes this fully unqualified check pass green."""
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
            if not any(_real_risk(c) for c in state["contacts"]):
                continue
            decision = leo_choose_action(state)
            if decision["action"] == "speed_up":
                violations.append(rec["id"])
    assert not violations, f"{len(violations)} frame(s) chose speed_up despite a real-risk contact: {violations[:10]}"


def test_never_hold_course_when_real_risk_exists_unless_stand_on_full_dataset() -> None:
    """Quality-review STAP 2, point 4: NO role/type qualification -- whenever ANY contact
    has real_risk()==True, hold_course is only ever valid if conduct_rule=="Rule 17"."""
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
            if not any(_real_risk(c) for c in state["contacts"]):
                continue
            decision = leo_choose_action(state)
            if decision["action"] == "hold_course" and decision["conduct_rule"] != "Rule 17":
                violations.append(rec["id"])
    assert not violations, f"{len(violations)} frame(s) chose hold_course despite real risk without Rule 17: {violations[:10]}"


# ── not_applicable moving contacts (decision 2026-09-23: "nu fixen, geometrisch, gescoped") ──
def test_not_applicable_moving_contacts_now_get_a_role_or_are_excluded() -> None:
    """Bugfix regression (was a documented-gap test at STOP 1, 220/7928 frames): every
    real Leo frame with a real-risk, MOVING, own_role=="not_applicable" contact must now
    either (a) get a geometrically-derived give_way/both_give_way/stand_on role and a
    non-None action/rule, or (b) be explicitly EXCLUDED (action=None, role="excluded") --
    never silently hold_course/speed_up with rule "none"."""
    violations = []
    n_population = 0
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
                    and c.get("encounter_type") != "stationary_contact"
                    and (c.get("speed") or 0) > 0.5]
            if not risky:
                continue
            n_population += 1
            decision = leo_choose_action(state)
            ok = (decision["role"] == "excluded"
                 or (decision["action"] is not None and decision["conduct_rule"] != "none"))
            if not ok:
                violations.append((rec["id"], decision["action"], decision["role"], decision["conduct_rule"]))
    assert n_population > 0
    assert not violations, f"{len(violations)} frame(s) still not handled: {violations[:10]}"


def test_not_applicable_moving_contact_breakdown_report() -> None:
    """Reports (not asserts a specific split -- see chat for the STOP 1 numbers) the
    resulting role/action breakdown of the 220-frame population, for the STOP 1 report."""
    from collections import Counter
    role_counter = Counter()
    action_counter = Counter()
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
                    and c.get("encounter_type") != "stationary_contact"
                    and (c.get("speed") or 0) > 0.5]
            if not risky:
                continue
            decision = leo_choose_action(state)
            role_counter[decision["role"]] += 1
            action_counter[decision["action"]] += 1
    print(f"\nnot_applicable-moving breakdown: roles={dict(role_counter)} actions={dict(action_counter)}")


def test_geometric_role_head_on_yields_both_give_way_mutual() -> None:
    state = _leo_state("not_applicable", "head_on", cpa_m=300.0, tcpa_s=100.0, risk_label="low")
    state["contacts"][0]["relative_bearing_deg"] = 2.0
    state["contacts"][0]["heading"] = 180.0
    state["contacts"][0]["closing_speed"] = 5.0
    d = leo_choose_action(state)
    assert d["encounter_rule"] == "Rule 14" and d["conduct_rule"] == "Rule 14"
    assert d["action"] == "turn_right"


def test_geometric_role_crossing_target_on_starboard_is_give_way() -> None:
    state = _leo_state("not_applicable", "crossing", cpa_m=300.0, tcpa_s=100.0, risk_label="low")
    state["contacts"][0]["relative_bearing_deg"] = 45.0
    state["contacts"][0]["heading"] = 270.0
    state["contacts"][0]["closing_speed"] = 5.0
    d = leo_choose_action(state)
    assert d["encounter_rule"] == "Rule 15" and d["conduct_rule"] == "Rule 16"
    assert d["action"] == "turn_right"


def test_geometric_role_crossing_target_on_port_is_stand_on() -> None:
    # tcpa_s=200 is within RISK_HORIZON_S (300) but above STAND_ON_TCPA_S (180) --
    # a real risk that is NOT yet imminent enough to trigger 17(a)(ii)/(b)'s own turn.
    state = _leo_state("not_applicable", "crossing", cpa_m=300.0, tcpa_s=200.0, risk_label="low")
    state["contacts"][0]["relative_bearing_deg"] = -45.0
    state["contacts"][0]["heading"] = 90.0
    state["contacts"][0]["closing_speed"] = 5.0
    d = leo_choose_action(state)
    assert d["encounter_rule"] == "Rule 15" and d["conduct_rule"] == "Rule 17"
    assert d["action"] == "hold_course"


def test_not_converging_not_applicable_contact_excludes_the_frame() -> None:
    """closing_speed<=0 despite CPA/TCPA real_risk -- too ambiguous to trust, must
    EXCLUDE the whole frame (action=None, role="excluded"), never hold_course."""
    state = _leo_state("not_applicable", "crossing", cpa_m=300.0, tcpa_s=100.0, risk_label="low")
    state["contacts"][0]["relative_bearing_deg"] = 45.0
    state["contacts"][0]["heading"] = 270.0
    state["contacts"][0]["closing_speed"] = 0.0  # NOT converging
    d = leo_choose_action(state)
    assert d["action"] is None
    assert d["role"] == "excluded"
    assert d["category"] == "leo_excluded_ambiguous_geometry"




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


def test_stationary_contact_beyond_horizon_is_early_band_and_still_avoids() -> None:
    """Quality-review STOP-1-blocking-bug fix (2026-09-23), SUPERSEDES this test's
    original expectation: CPA is below the safe distance but TCPA sits beyond the
    horizon -- risk_band() says "early", not "safe": a stationary object on a collision
    course does not wait for a horizon either, so this is STILL Rule 8 avoidance, just
    bucketed "early_stationary" instead of "stationary"."""
    state = _leo_state_stationary(cpa_m=120.0, tcpa_s=RISK_HORIZON_S + 1, bearing_deg=20.0)
    d = leo_choose_action(state)
    assert d["conduct_rule"] == "Rule 8" and d["encounter_rule"] == "none"
    assert d["category"] == "leo_early_stationary_avoid"
    assert d["bucket"] == "early_stationary"


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
