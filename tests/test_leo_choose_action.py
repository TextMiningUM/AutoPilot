"""Synthetic-state tests for pipeline/track2/build_oow_scenarios_leo.py's
leo_choose_action() -- Fase B1 of the RAG-rebuild-v2 plan (rewrite of the Leo MOOS
scenario action-labeling from role-only to a Rule-7 real-risk gate). Each test builds a
minimal, hand-constructed `state` dict (not real Leo data) so the exact branch under test
is unambiguous."""
from pipeline.track2.build_oow_scenarios_leo import leo_choose_action, SAFE_CPA_M, CRITICAL_RANGE_M


def _own(speed=10.0, target_speed=10.0, paused=False, stopped=False,
        x=0.0, y=0.0, heading=0.0):
    return {"speed": speed, "target_speed": target_speed, "paused": paused, "stopped": stopped,
           "x": x, "y": y, "heading": heading}


def _contact(own_role="give_way", encounter_type="crossing", risk="high",
            cpa_distance_m=200.0, tcpa_s=300.0, relative_bearing_deg=45.0,
            range_m=800.0, closing_speed=5.0,
            active_encounter_rules=(15, 16), standing_rules=(2, 5, 6, 7)):
    return {
        "own_role": own_role, "encounter_type": encounter_type, "risk": risk,
        "cpa_distance_m": cpa_distance_m, "tcpa_s": tcpa_s,
        "relative_bearing_deg": relative_bearing_deg, "range_m": range_m,
        "closing_speed": closing_speed,
        "active_encounter_rules": list(active_encounter_rules),
        "standing_rules": list(standing_rules),
    }


def _state(contacts, own=None, mission=None):
    # Default mission waypoint sits dead ahead of own-ship's default heading (0 deg,
    # due north from the origin) so tests that don't care about goal-course-following
    # get the "already on bearing" branch (hold_course/speed_up) exactly like before
    # goal-course-following was added to the no-real-risk branch.
    return {"own_ship": own or _own(), "contacts": contacts,
           "mission": mission or {"x": 0.0, "y": 1000.0}}



def test_low_risk_give_way_holds_course() -> None:
    """A give-way contact with CPA at/above the safe distance is not a real risk (Rule 7)
    -- must not trigger a manoeuvre, regardless of role."""
    c = _contact(own_role="give_way", risk="low", cpa_distance_m=SAFE_CPA_M + 50)
    d = leo_choose_action(_state([c]))
    assert d["action"] == "hold_course"
    assert d["rule_applied"] == "none"


def test_tcpa_zero_with_large_cpa_is_not_real_risk() -> None:
    """TCPA=0 alone must never drive the decision -- with a large CPA (already passed
    clear, or was never going to be close), there is no real risk."""
    c = _contact(own_role="give_way", risk="medium", cpa_distance_m=SAFE_CPA_M + 300, tcpa_s=0.0)
    d = leo_choose_action(_state([c]))
    assert d["action"] == "hold_course"
    assert d["rule_applied"] == "none"


def test_head_on_give_way_turns_starboard() -> None:
    c = _contact(own_role="both_give_way", encounter_type="head_on", risk="high",
                cpa_distance_m=100.0)
    d = leo_choose_action(_state([c]))
    assert d["action"] == "turn_right"
    assert d["rule_applied"] == "Rule 14"
    assert d["degrees"] is not None and 15.0 <= d["degrees"] <= 30.0


def test_crossing_give_way_turns_starboard_and_scales_degrees_with_cpa_shortfall() -> None:
    close = _contact(own_role="give_way", encounter_type="crossing", risk="high", cpa_distance_m=50.0)
    far = _contact(own_role="give_way", encounter_type="crossing", risk="medium", cpa_distance_m=450.0)
    d_close = leo_choose_action(_state([close]))
    d_far = leo_choose_action(_state([far]))
    assert d_close["action"] == d_far["action"] == "turn_right"
    assert d_close["rule_applied"] == "Rule 15"
    # A bigger CPA shortfall must never produce a SMALLER turn.
    assert d_close["degrees"] > d_far["degrees"]
    assert 15.0 <= d_far["degrees"] <= 30.0
    assert 15.0 <= d_close["degrees"] <= 30.0


def test_rule13_overtaking_permits_either_turn_direction_by_geometry() -> None:
    """Rule 13 (overtaking) allows EITHER side, unlike Rule 14/15/16's mandatory
    starboard -- direction is derived from which side the contact currently sits on
    (diverge away from it), not a fixed direction. Confirms turn_left IS a reachable
    outcome for this category (never hardcoded to starboard like head-on/crossing)."""
    starboard_contact = _contact(own_role="give_way", encounter_type="we_are_overtaking_contact",
                                 risk="high", cpa_distance_m=100.0, relative_bearing_deg=20.0)
    port_contact = _contact(own_role="give_way", encounter_type="we_are_overtaking_contact",
                            risk="high", cpa_distance_m=100.0, relative_bearing_deg=-20.0)
    d_stbd = leo_choose_action(_state([starboard_contact]))
    d_port = leo_choose_action(_state([port_contact]))
    assert d_stbd["rule_applied"] == d_port["rule_applied"] == "Rule 13"
    # Diverge-away convention: turn AWAY from whichever side the contact is currently on.
    assert d_stbd["action"] == "turn_left"
    assert d_port["action"] == "turn_right"
    assert {d_stbd["action"], d_port["action"]} == {"turn_left", "turn_right"}


def test_stand_on_holds_course_without_17b_trigger() -> None:
    c = _contact(own_role="stand_on", encounter_type="crossing", risk="high",
                cpa_distance_m=100.0, tcpa_s=600.0)  # real risk, but NOT imminent
    d = leo_choose_action(_state([c]))
    assert d["action"] == "hold_course"
    assert d["rule_applied"] == "Rule 17"


def test_stand_on_rule17b_triggers_own_action_when_imminent() -> None:
    c = _contact(own_role="stand_on", encounter_type="crossing", risk="critical",
                cpa_distance_m=100.0, tcpa_s=60.0, relative_bearing_deg=30.0)  # real risk AND imminent
    d = leo_choose_action(_state([c]))
    assert d["action"] in ("turn_left", "turn_right")
    assert d["rule_applied"] == "Rule 17"
    assert d["degrees"] is not None


def test_stop_only_on_close_range_still_closing_critical_risk() -> None:
    c = _contact(own_role="give_way", encounter_type="crossing", risk="critical",
                cpa_distance_m=50.0, range_m=CRITICAL_RANGE_M - 50, closing_speed=3.0)
    d = leo_choose_action(_state([c]))
    assert d["action"] == "stop"


def test_stop_not_triggered_merely_by_short_tcpa() -> None:
    """The original bug's stop-default fired on TCPA<1.5min OR risk=='critical' alone.
    A critical-risk contact that is NOT actually close range (or not closing) must NOT
    stop -- it should still get a substantial turn."""
    c = _contact(own_role="give_way", encounter_type="crossing", risk="critical",
                cpa_distance_m=50.0, tcpa_s=30.0, range_m=CRITICAL_RANGE_M + 500, closing_speed=5.0)
    d = leo_choose_action(_state([c]))
    assert d["action"] != "stop"
    assert d["action"] == "turn_right"


def test_no_contacts_holds_course_or_speeds_up() -> None:
    d_at_speed = leo_choose_action(_state([], own=_own(speed=10.0, target_speed=10.0)))
    assert d_at_speed["action"] == "hold_course"
    assert d_at_speed["rule_applied"] == "none"

    d_below_target = leo_choose_action(_state([], own=_own(speed=5.0, target_speed=10.0)))
    assert d_below_target["action"] == "speed_up"
    assert d_below_target["rule_applied"] == "none"


def test_no_real_risk_follows_goal_course_check_when_off_bearing() -> None:
    """Fase B2 consistency requirement: when no contact poses real risk, the label must
    match what GOAL COURSE CHECK itself recommends (SYSTEM_OOW_AGENT's decision procedure
    step 3), never a fixed hold_course default -- own-ship heading 0 (north), mission
    waypoint due east means a 90 deg-off-course turn is required."""
    own = _own(heading=0.0, x=0.0, y=0.0)
    mission = {"x": 1000.0, "y": 0.0}  # due east -> bearing 90
    d = leo_choose_action(_state([], own=own, mission=mission))
    assert d["action"] == "turn_right"
    assert d["degrees"] == 90.0
    assert d["rule_applied"] == "none"


def test_paused_own_ship_gets_no_manoeuvre_label() -> None:
    d = leo_choose_action(_state([_contact()], own=_own(paused=True)))
    assert d["action"] is None
    assert d["bucket"] == "paused"

    d2 = leo_choose_action(_state([_contact()], own=_own(stopped=True)))
    assert d2["action"] is None
    assert d2["bucket"] == "paused"
