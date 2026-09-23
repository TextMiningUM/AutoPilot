"""Quality-review STAP 5 (2026-09-23): MAJOR BUG FIX -- to_unified_action()'s give-way
branch never re-checked real_risk() against the row's own sampled safe_distance_m/
risk_horizon_s (unlike the stand-on/maintain_course branch, which already did). Found
live during the B3 full-population run: a give-way row with tcpa_s=307.5s and a sampled
risk_horizon_s=183.1s was still labelled a real-risk give-way turn.
"""
from pipeline.oow_agent_spec import real_risk
from pipeline.track2.build_oow_scenarios import to_unified_action


def _give_way_rec(cpa_m: float, tcpa_min: float, role: str = "give_way") -> dict:
    target = {
        "_role": role, "_rules": ["Rule 15"], "_cpa_m": cpa_m, "_tcpa_min": tcpa_min,
        "start_xy_m": (300.0, 300.0), "bearing_from_os_deg": 60.0, "heading_deg": 200.0, "speed": 8.0,
    }
    return {"action": "alter_course", "action_params": {}, "own_speed": 10.0,
           "role": role, "rules": ["Rule 15"], "targets": [target]}


def test_give_way_with_tcpa_beyond_a_short_sampled_horizon_is_not_real_risk() -> None:
    """The exact bug found: CPA is tiny (real risk by distance) but TCPA (307.5s) is
    beyond a short sampled risk_horizon_s (183.1s) -- must NOT be treated as real risk."""
    rec = _give_way_rec(cpa_m=0.84, tcpa_min=307.5 / 60.0)
    limits = {"safe_distance_m": 300.0, "max_turn_deg": 35.0, "risk_horizon_s": 183.1}
    assert not real_risk(0.84, 307.5, 300.0, 183.1)
    unified = to_unified_action(rec, limits)
    assert unified["action"] == "hold_course"
    assert unified["encounter_rule"] == "none" and unified["conduct_rule"] == "none"


def test_give_way_within_the_sampled_horizon_is_still_real_risk() -> None:
    """Same geometry, but a LONGER sampled horizon that the TCPA now falls within --
    confirms the fix didn't just always return hold_course."""
    rec = _give_way_rec(cpa_m=0.84, tcpa_min=307.5 / 60.0)
    limits = {"safe_distance_m": 300.0, "max_turn_deg": 35.0, "risk_horizon_s": 400.0}
    assert real_risk(0.84, 307.5, 300.0, 400.0)
    unified = to_unified_action(rec, limits)
    assert unified["action"] == "turn_right"
    assert unified["encounter_rule"] == "Rule 15" and unified["conduct_rule"] == "Rule 16"


def test_give_way_falls_back_to_a_real_risk_stand_on_target_when_present() -> None:
    """No real-risk give-way target, but a co-present stand-on target IS real risk --
    must report that stand-on target's hold_course/Rule 17, not a bare none/none."""
    give_way = {
        "_role": "give_way", "_rules": ["Rule 15"], "_cpa_m": 0.84, "_tcpa_min": 307.5 / 60.0,
        "start_xy_m": (100.0, 100.0), "bearing_from_os_deg": 60.0, "heading_deg": 200.0, "speed": 8.0,
    }
    stand_on = {
        "_role": "stand_on", "_rules": ["Rule 15"], "_cpa_m": 10.0, "_tcpa_min": 50.0 / 60.0,
        "start_xy_m": (-100.0, 100.0), "bearing_from_os_deg": -60.0, "heading_deg": 20.0, "speed": 8.0,
    }
    rec = {"action": "alter_course", "action_params": {}, "own_speed": 10.0,
          "role": "give_way+stand_on", "rules": ["Rule 15"], "targets": [give_way, stand_on]}
    limits = {"safe_distance_m": 300.0, "max_turn_deg": 35.0, "risk_horizon_s": 183.1}
    unified = to_unified_action(rec, limits)
    assert unified["action"] == "hold_course"
    assert unified["encounter_rule"] == "Rule 15" and unified["conduct_rule"] == "Rule 17"
