"""Tests that Basic Simulator/app/agents.py and pipeline/oow_agent_spec.py never drift
apart -- Fase B2 (RAG-rebuild-v2 plan, 2026-09-22): SYSTEM_OOW_AGENT/ACTIONS/the JSON
response schema must be defined EXACTLY ONCE and imported everywhere, never copied."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app.agents as agents  # noqa: E402
from app.missions import Mission, Vessel  # noqa: E402
from app.narrate import narrate  # noqa: E402
from pipeline import oow_agent_spec  # noqa: E402
from pipeline.oow_agent_spec import goal_course_check_line  # noqa: E402


def test_agents_py_shares_the_same_system_prompt_object() -> None:
    """Identity check (not just equal strings) -- catches a future inline copy-paste of
    the prompt back into agents.py, which `==` alone would miss if kept byte-identical."""
    assert agents.SYSTEM_OOW_AGENT is oow_agent_spec.SYSTEM_OOW_AGENT


def test_agents_py_shares_the_same_actions_tuple() -> None:
    assert agents.ACTIONS is oow_agent_spec.ACTIONS


def test_parse_json_action_uses_the_shared_schema() -> None:
    valid = agents._parse_json_action(
        '{"action": "turn_right", "degrees": 15, "encounter_rule": "Rule 15", '
        '"conduct_rule": "Rule 16", "reasoning": "x"}'
    )
    assert valid["action"] == "turn_right" and not valid.get("_parse_error")

    # An action name outside ACTIONS must be rejected (fall back to the parse-error dict),
    # not silently accepted and handed to Simulation.apply_action().
    invalid_action = agents._parse_json_action('{"action": "maintain_course"}')
    assert invalid_action.get("_parse_error") is True

    # degrees required for a turn action.
    missing_degrees = agents._parse_json_action(
        '{"action": "turn_left", "encounter_rule": "none", "conduct_rule": "none", "reasoning": "x"}'
    )
    assert missing_degrees.get("_parse_error") is True

    # degrees must be ABSENT for a non-turn action.
    spurious_degrees = agents._parse_json_action(
        '{"action": "hold_course", "degrees": 10, "encounter_rule": "none", '
        '"conduct_rule": "none", "reasoning": "x"}'
    )
    assert spurious_degrees.get("_parse_error") is True


def test_narrate_uses_the_shared_goal_course_check_function() -> None:
    """For a fixed synthetic mission/own-ship state, narrate()'s GOAL COURSE CHECK line
    must be byte-for-byte what goal_course_check_line() itself returns for the same
    coordinates -- proving narrate.py actually CALLS the shared function rather than a
    second, independently-drifting reimplementation (the exact bug class already found
    and fixed once for mission.targets vs the simulator's live contacts)."""
    own = Vessel(name="own", x=0.0, y=0.0, heading=90.0, speed=5.0)
    mission = Mission(id="t", name="t", rule_refs=[], own_ship_role="none", description="",
                      pass_criteria=[], own_ship=own, goal=(1000.0, 1000.0))
    report = narrate(mission, own, [])
    expected = goal_course_check_line(own.x, own.y, own.heading, *mission.goal)
    goal_check_lines = [ln for ln in report.splitlines() if ln.startswith("GOAL COURSE CHECK")]
    assert len(goal_check_lines) == 1
    assert goal_check_lines[0] == expected


# ── Fase B3 (RAG-rebuild-v2 plan, 2026-09-22): classify_rules() single-source mapping ──
def test_classify_rules_no_real_risk_is_none_none() -> None:
    for role in (None, "none", "cleared"):
        assert oow_agent_spec.classify_rules(role, "hold_course") == ("none", "none")


def test_classify_rules_head_on_turn_is_rule14_rule14() -> None:
    assert oow_agent_spec.classify_rules("mutual", "turn_right") == ("Rule 14", "Rule 14")


def test_classify_rules_give_way_crossing_turn_is_rule15_rule16() -> None:
    assert oow_agent_spec.classify_rules("give_way", "turn_right") == ("Rule 15", "Rule 16")


def test_classify_rules_give_way_crossing_speed_change_is_still_rule16() -> None:
    assert oow_agent_spec.classify_rules("give_way", "slow_down") == ("Rule 15", "Rule 16")


def test_classify_rules_stand_on_hold_course_is_rule15_rule17() -> None:
    assert oow_agent_spec.classify_rules("stand_on", "hold_course") == ("Rule 15", "Rule 17")


def test_classify_rules_stand_on_17b_action_is_still_rule17() -> None:
    assert oow_agent_spec.classify_rules("stand_on", "turn_left") == ("Rule 15", "Rule 17")


def test_classify_rules_overtaking_give_way_turn_is_rule13_rule13() -> None:
    assert oow_agent_spec.classify_rules("overtaking_give_way", "turn_left") == ("Rule 13", "Rule 13")


def test_classify_rules_overtaking_stand_on_is_rule13_rule17() -> None:
    assert oow_agent_spec.classify_rules("overtaking_stand_on", "hold_course") == ("Rule 13", "Rule 17")


def test_classify_rules_give_way_stop_is_rule8_not_rule17() -> None:
    """The bug B3's review caught: 17(b) is exclusively the STAND-ON vessel's provision --
    a give-way vessel's own emergency stop must cite Rule 8, never Rule 17."""
    assert oow_agent_spec.classify_rules("give_way", "stop") == ("Rule 15", "Rule 8")
    assert oow_agent_spec.classify_rules("mutual", "stop") == ("Rule 14", "Rule 8")


def test_classify_rules_stand_on_stop_is_still_rule17() -> None:
    # A stand-on vessel's own emergency action (however drastic) is still under Rule 17.
    assert oow_agent_spec.classify_rules("stand_on", "stop") == ("Rule 15", "Rule 17")


def test_classify_rules_restricted_visibility_forces_rule19() -> None:
    assert oow_agent_spec.classify_rules("give_way", "turn_right", restricted_visibility=True) == (
        "Rule 15", "Rule 19")


def test_validate_action_json_accepts_the_two_rule_field_schema() -> None:
    obj = {"action": "turn_right", "degrees": 20.0, "encounter_rule": "Rule 15",
          "conduct_rule": "Rule 16", "reasoning": "x"}
    assert not oow_agent_spec.validate_action_json(obj)


def test_validate_action_json_rejects_sub_paragraph_rule_numbers() -> None:
    obj = {"action": "hold_course", "degrees": None, "encounter_rule": "Rule 15",
          "conduct_rule": "Rule 17(b)", "reasoning": "x"}
    assert oow_agent_spec.validate_action_json(obj)


def test_validate_action_json_rejects_only_one_field_being_none() -> None:
    obj = {"action": "hold_course", "degrees": None, "encounter_rule": "none",
          "conduct_rule": "Rule 17", "reasoning": "x"}
    errors = oow_agent_spec.validate_action_json(obj)
    assert errors and any("together" in e for e in errors)


def test_validate_action_json_rejects_missing_conduct_rule() -> None:
    obj = {"action": "hold_course", "degrees": None, "encounter_rule": "none", "reasoning": "x"}
    errors = oow_agent_spec.validate_action_json(obj)
    assert any("conduct_rule" in e for e in errors)

