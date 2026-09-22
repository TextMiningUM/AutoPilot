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
        '{"action": "turn_right", "degrees": 15, "rule_applied": "Rule 15", "reasoning": "x"}'
    )
    assert valid["action"] == "turn_right" and not valid.get("_parse_error")

    # An action name outside ACTIONS must be rejected (fall back to the parse-error dict),
    # not silently accepted and handed to Simulation.apply_action().
    invalid_action = agents._parse_json_action('{"action": "maintain_course"}')
    assert invalid_action.get("_parse_error") is True

    # degrees required for a turn action.
    missing_degrees = agents._parse_json_action(
        '{"action": "turn_left", "rule_applied": "none", "reasoning": "x"}'
    )
    assert missing_degrees.get("_parse_error") is True

    # degrees must be ABSENT for a non-turn action.
    spurious_degrees = agents._parse_json_action(
        '{"action": "hold_course", "degrees": 10, "rule_applied": "none", "reasoning": "x"}'
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

