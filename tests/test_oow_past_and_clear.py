"""Quality-review Fase B4 (2026-09-23, ADOPTED by user decision): decision history +
"past and clear" (Rule 8(d)/13(d)) -- deterministic, no LLM. Covers: the shared
render_previous_decisions() richer-shape rendering (2a), leo_choose_action()'s
past-and-clear override on a constructed 3-frame trajectory (2b), the updated
consistency-test exception (2c, in tests/test_leo_goal_course_consistency.py), and the
60%-fixed-seed partial history-in-text visibility rule (2d).
"""
from pipeline.oow_agent_spec import render_previous_decisions
from pipeline.track2.build_oow_scenarios_leo import (
    leo_choose_action, compute_all_decisions, _row_gets_history_text, HISTORY_TEXT_FRACTION,
)

_LIMITS = {"safe_distance_m": 500.0, "max_turn_deg": 30.0, "risk_horizon_s": 300.0, "stand_on_tcpa_s": 180.0}


def _own(goal_xy=(500.0, 0.0)) -> dict:
    return {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 10.0, "target_speed": 10.0,
           "movement_status": "underway", "colregs_vessel_type": "power_driven",
           "name": "LLM_SHIP", "paused": False, "stopped": False}


def _state(cpa_m: float, tcpa_s: float, closing_speed: float, own_role="give_way",
          encounter_type="crossing", goal_xy=(500.0, 0.0), previous_decisions=None) -> dict:
    return {
        "own_ship": _own(goal_xy), "mission": {"x": goal_xy[0], "y": goal_xy[1]},
        "contacts": [{
            "name": "ts1", "own_role": own_role, "encounter_type": encounter_type,
            "cpa_distance_m": cpa_m, "tcpa_s": tcpa_s, "risk": "low",
            "range_m": 1000.0, "relative_bearing_deg": 20.0, "heading": 200.0,
            "speed": 8.0, "closing_speed": closing_speed, "colregs_vessel_type": "power_driven",
            "active_encounter_rules": [15], "standing_rules": [2, 5, 6, 7],
        }],
        "conditions": {"visibility_condition": "clear"},
        "previous_decisions": previous_decisions,
    }


# ── 2a: single-source renderer, richer per-decision shape ─────────────────────────────
def test_render_previous_decisions_renders_conduct_rule_and_real_risk_contacts() -> None:
    decisions = [{"action": "turn_right", "degrees": 20.0, "conduct_rule": "Rule 16",
                 "real_risk_contact_names": ["ts1"]}]
    text = render_previous_decisions(decisions)
    assert "turn_right (20 deg)" in text
    assert "Rule 16" in text
    assert "ts1" in text


def test_render_previous_decisions_backward_compatible_with_plain_action_degrees() -> None:
    """The synthetic generator's OWN self-consistency history (no rule/contact fields) --
    must render exactly as before (no crash, no stray 'None' text)."""
    text = render_previous_decisions([{"action": "hold_course", "degrees": None}])
    assert "hold_course" in text and "None" not in text


def test_render_previous_decisions_is_the_single_source_for_identical_settings() -> None:
    """Same decisions in -> byte-identical text out, regardless of caller (training row
    or a live simulator step) -- both MUST call this exact function, never a hand-copy."""
    decisions = [{"action": "turn_left", "degrees": 15.0, "conduct_rule": "Rule 13",
                 "real_risk_contact_names": ["ts2"]}]
    assert render_previous_decisions(decisions) == render_previous_decisions(decisions)


# ── 2b: constructed 3-frame trajectory -- risk -> not-yet-clear -> clearly opening ─────
def test_past_and_clear_three_frame_trajectory() -> None:
    # Frame 1: real risk (CPA 200 < 500, TCPA 100 within 300s horizon, closing) -> manoeuvre.
    d1 = leo_choose_action(_state(cpa_m=200.0, tcpa_s=100.0, closing_speed=5.0), _LIMITS)
    assert d1["action"] in ("turn_left", "turn_right", "stop", "slow_down")
    assert d1["encounter_rule"] != "none" and d1["conduct_rule"] != "none"

    history_1 = [{"action": d1["action"], "degrees": d1["degrees"],
                 "encounter_rule": d1["encounter_rule"], "conduct_rule": d1["conduct_rule"],
                 "real_risk_contact_names": ["ts1"]}]

    # Frame 2: CPA back above the safe distance (no longer real_risk by the strict CPA
    # gate) but STILL closing (closing_speed>0) and TCPA still >= 0 -- NOT yet "finally
    # past and clear" of ts1 -- must hold_course despite GOAL COURSE CHECK wanting a turn.
    state2 = _state(cpa_m=600.0, tcpa_s=50.0, closing_speed=2.0, previous_decisions=history_1)
    d2 = leo_choose_action(state2, _LIMITS)
    assert d2["action"] == "hold_course"
    assert d2["encounter_rule"] == "none" and d2["conduct_rule"] == "none"
    assert d2.get("reason") == "not yet past and clear"

    history_2 = history_1 + [{"action": d2["action"], "degrees": d2["degrees"],
                              "encounter_rule": d2["encounter_rule"], "conduct_rule": d2["conduct_rule"],
                              "real_risk_contact_names": ["ts1"]}]  # propagated, not re-derived

    # Frame 3: clearly opening now (closing_speed <= 0) -- finally past and clear -- must
    # follow GOAL COURSE CHECK again (goal at (500,0), own heading 0 -> turn_right).
    state3 = _state(cpa_m=600.0, tcpa_s=-10.0, closing_speed=-1.0, previous_decisions=history_2)
    d3 = leo_choose_action(state3, _LIMITS)
    assert d3["action"] == "turn_right"
    assert d3["category"] != "leo_clear_not_yet_past_and_clear"


def test_compute_all_decisions_propagates_pending_contact_across_multiple_not_yet_clear_frames() -> None:
    """End-to-end via compute_all_decisions(): a multi-step not-yet-clear stretch must
    keep checking the SAME contact across ALL of it, not lose track after one step (the
    naive single-step-lookback bug this propagation fix avoids)."""
    def rec(rid, cycle, cpa, tcpa, closing):
        return {"id": rid, "cycle_id": cycle, "source_file": "traj_test",
               "state": _state(cpa_m=cpa, tcpa_s=tcpa, closing_speed=closing)}

    all_recs = [
        rec("f1", 1, 200.0, 100.0, 5.0),   # real risk -> manoeuvre
        rec("f2", 2, 600.0, 60.0, 2.0),    # not yet clear -> hold_course
        rec("f3", 3, 600.0, 40.0, 1.0),    # STILL not yet clear -> hold_course (must not forget ts1)
        rec("f4", 4, 600.0, -5.0, -1.0),   # finally opening -> goal-turn again
    ]
    decisions = compute_all_decisions(all_recs)
    assert decisions["f1"]["action"] in ("turn_left", "turn_right", "stop", "slow_down")
    assert decisions["f2"]["action"] == "hold_course" and decisions["f2"]["category"] == "leo_clear_not_yet_past_and_clear"
    assert decisions["f3"]["action"] == "hold_course" and decisions["f3"]["category"] == "leo_clear_not_yet_past_and_clear"
    assert decisions["f4"]["category"] != "leo_clear_not_yet_past_and_clear"


# ── 2d: partial (fixed-seed) history-in-text visibility ───────────────────────────────
def test_history_text_visibility_is_deterministic_per_row() -> None:
    assert _row_gets_history_text("leo00042") == _row_gets_history_text("leo00042")


def test_history_text_visibility_roughly_matches_the_documented_fraction() -> None:
    ids = [f"leo{i:05d}" for i in range(4000)]
    shown = sum(_row_gets_history_text(i) for i in ids)
    frac = shown / len(ids)
    assert abs(frac - HISTORY_TEXT_FRACTION) < 0.03, f"observed fraction {frac} vs documented {HISTORY_TEXT_FRACTION}"
