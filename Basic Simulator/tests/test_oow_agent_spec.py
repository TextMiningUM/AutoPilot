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
    """Screening-set-B audit follow-up (2026-09-23): _parse_json_action() now returns
    {"decision", "parse_ok", "schema_errors"} -- parse_ok=True whenever a JSON object with
    the required keys was found at all, regardless of whether its fields are internally
    consistent (schema_errors/decision["_schema_errors"] cover that separately, never
    _parse_error)."""
    valid = agents._parse_json_action(
        '{"action": "turn_right", "degrees": 15, "encounter_rule": "Rule 15", '
        '"conduct_rule": "Rule 16", "reasoning": "x"}'
    )
    assert valid["parse_ok"] is True and not valid["schema_errors"]
    assert valid["decision"]["action"] == "turn_right" and not valid["decision"].get("_parse_error")

    # An action name outside ACTIONS is a SCHEMA error (JSON parsed fine, decision kept
    # with _schema_errors attached), never a parse failure -- Simulation.apply_action()
    # already treats any unrecognised action as a safe no-op by design.
    invalid_action = agents._parse_json_action('{"action": "maintain_course"}')
    assert invalid_action["parse_ok"] is True and invalid_action["schema_errors"]
    assert invalid_action["decision"].get("_parse_error") is not True

    # degrees required for a turn action -- also a schema error, not a parse failure.
    missing_degrees = agents._parse_json_action(
        '{"action": "turn_left", "encounter_rule": "none", "conduct_rule": "none", "reasoning": "x"}'
    )
    assert missing_degrees["parse_ok"] is True and missing_degrees["schema_errors"]

    # degrees must be ABSENT for a non-turn action -- same, schema error not parse failure.
    spurious_degrees = agents._parse_json_action(
        '{"action": "hold_course", "degrees": 10, "encounter_rule": "none", '
        '"conduct_rule": "none", "reasoning": "x"}'
    )
    assert spurious_degrees["parse_ok"] is True and spurious_degrees["schema_errors"]

    # Genuinely unparseable (no JSON object at all) IS a real parse failure.
    unparseable = agents._parse_json_action("not json at all")
    assert unparseable["parse_ok"] is False
    assert unparseable["decision"].get("_parse_error") is True

    # A schema-invalid-but-otherwise-valid encounter_rule='none'+conduct_rule='Rule 17'
    # combination (the exact screening-set-B evidence) must be KEPT, not discarded.
    none_with_conduct = agents._parse_json_action(
        '{"action": "hold_course", "degrees": 0.0, "encounter_rule": "none", '
        '"conduct_rule": "Rule 17", "reasoning": "x"}'
    )
    assert none_with_conduct["parse_ok"] is True
    assert none_with_conduct["decision"]["conduct_rule"] == "Rule 17"
    assert none_with_conduct["decision"].get("_parse_error") is not True
    assert none_with_conduct["schema_errors"]


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
    expected = goal_course_check_line(own.x, own.y, own.heading, *mission.goal, max_turn_deg=30.0)
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


# ── Quality-review STAP 2 (2026-09-23): classify_encounter() single-source move ───────
def test_narrate_shares_the_same_classify_encounter_object() -> None:
    """Identity check -- narrate.py must IMPORT classify_encounter, not keep its own
    second copy (the exact class of drift bug already found/fixed for the system prompt
    and goal_course_check_line)."""
    from app import narrate
    assert narrate.classify_encounter is oow_agent_spec.classify_encounter


def test_classify_encounter_head_on() -> None:
    enc, rules, rel = oow_agent_spec.classify_encounter(0.0, 0.0, 0.0, 0.0, 1000.0, 180.0)
    assert enc == "head_on" and rules == ["Rule 14"]


def test_classify_encounter_crossing_target_on_starboard() -> None:
    enc, rules, _ = oow_agent_spec.classify_encounter(0.0, 0.0, 0.0, 500.0, 500.0, 270.0)
    assert enc == "crossing_target_on_starboard" and rules == ["Rule 15", "Rule 16"]


def test_classify_encounter_crossing_target_on_port() -> None:
    enc, rules, _ = oow_agent_spec.classify_encounter(0.0, 0.0, 0.0, -500.0, 500.0, 90.0)
    assert enc == "crossing_target_on_port" and rules == ["Rule 15", "Rule 17"]


def test_classify_encounter_we_are_overtaking_target() -> None:
    """Own-ship dead astern of a slower target on the SAME course -- the two-perspective
    check (bearing of own-ship AS SEEN FROM the target) must resolve this as overtaking,
    not crossing, even though the closing angle (rel_from_own) is fine, not near-180."""
    enc, rules, _ = oow_agent_spec.classify_encounter(0.0, 0.0, 0.0, 20.0, 300.0, 0.0)
    assert enc == "we_are_overtaking_target" and rules == ["Rule 13"]


def test_classify_encounter_target_is_overtaking_us() -> None:
    enc, rules, _ = oow_agent_spec.classify_encounter(0.0, 300.0, 0.0, -20.0, 0.0, 0.0)
    assert enc == "target_is_overtaking_us" and rules == ["Rule 13"]

