"""Quality-review Fase B4 (2026-09-23): build_oow_prompt()'s optional `previous_decisions`
parameter -- renders via the SAME render_previous_decisions() the training generators use,
prepended inside the "Situation:" block exactly like build_oow_scenarios.py's
user_message_for(), so a training row and a live simulator step given the same history
render byte-identical text. NOTE: Simulation.agent_log is currently declared but never
populated anywhere in the app -- wiring a real per-step history source into the live
simulator is a separate follow-up; this only tests that build_oow_prompt() renders
correctly WHEN a caller supplies history.
"""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app.agents as agents  # noqa: E402
from app.missions import load_mission  # noqa: E402
from pipeline.oow_agent_spec import render_previous_decisions  # noqa: E402

HISTORY = [{"action": "turn_right", "degrees": 20.0, "conduct_rule": "Rule 16",
           "real_risk_contact_names": ["ts1"]}]


def test_build_oow_prompt_with_no_history_is_unaffected() -> None:
    m = load_mission("Imazu01")
    messages, _ = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v0_base")
    assert "Your last" not in messages[1]["content"]


def test_build_oow_prompt_renders_history_via_the_shared_formatter() -> None:
    m = load_mission("Imazu01")
    messages, _ = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v0_base",
                                          previous_decisions=HISTORY)
    user_msg = messages[1]["content"]
    expected_prefix = render_previous_decisions(HISTORY)
    assert expected_prefix in user_msg
    assert f"Situation:\n{expected_prefix}" in user_msg


def test_build_oow_prompt_bare_qwen_never_renders_history() -> None:
    """bare_qwen is the deliberate zero-extra-framing ablation floor -- it must NEVER
    gain a history preamble even if one is supplied."""
    m = load_mission("Imazu01")
    messages, _ = agents.build_oow_prompt(m, m.own_ship, m.targets, config="bare_qwen",
                                          previous_decisions=HISTORY)
    assert "Your last" not in messages[1]["content"]
