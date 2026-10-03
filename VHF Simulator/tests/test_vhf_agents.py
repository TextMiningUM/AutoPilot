"""Tests for VHF Simulator/app/vhf_agents.py -- only the parts that don't need a real
model/GPU (per copilot-instructions.md: tests must run without a GPU or live API key).
Deliberately does NOT call _load_qwen()/_load_retrieval()/ask_vhf_qa()/ask_vhf_comms()
(those load a real model) -- only pure prompt-building/scenario-schema logic."""
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # VHF Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app.vhf_agents as vhf_agents  # noqa: E402
from core.paths import AgentPaths  # noqa: E402

REQUIRED_SCENARIO_FIELDS = {
    "category", "colreg_rules", "region", "own_vessel", "target_vessel",
    "scenario", "vhf_channel",
}


def test_scenario_file_has_required_fields() -> None:
    """Schema test: every row of the held-out scenario bank must have the fields
    `_scenario_brief()`/`ask_vhf_comms()` read directly -- this exact kind of schema-
    assumption test is this repo's most common real bug class (copilot-instructions.md)."""
    paths = AgentPaths.vhf()
    scenarios = json.loads(paths.eval_file("vhf_colreg_scenarios.json").read_text(encoding="utf-8"))
    assert len(scenarios) > 0
    for row in scenarios:
        missing = REQUIRED_SCENARIO_FIELDS - row.keys()
        assert not missing, f"scenario row missing {missing}: {row.get('scenario', '?')[:60]}"
        assert "hailing" in row["vhf_channel"] and "working" in row["vhf_channel"]


def test_scenario_brief_includes_channel_and_rules() -> None:
    scenario = {
        "region": "Houston Ship Channel",
        "own_vessel": "MV Texas Spirit",
        "target_vessel": "MV Gulf Explorer",
        "scenario": "Head-on encounter.",
        "vhf_channel": {"hailing": "16", "working": "71"},
        "colreg_rules": ["Rule 14", "Rule 34"],
    }
    brief = vhf_agents._scenario_brief(scenario)
    assert "16" in brief and "71" in brief
    assert "Rule 14" in brief and "Rule 34" in brief
    assert "MV Texas Spirit" in brief and "MV Gulf Explorer" in brief


def test_ask_vhf_comms_rejects_unknown_mode() -> None:
    scenario = {"region": "?", "own_vessel": "?", "target_vessel": "?", "scenario": "?",
                "vhf_channel": {"hailing": "16", "working": "71"}, "colreg_rules": []}
    try:
        vhf_agents.ask_vhf_comms(scenario, mode="bogus")
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown mode")


def test_build_vhf_prompt_bare_config_needs_no_retrieval() -> None:
    """"bare_qwen"/"v0_base" must not touch _load_retrieval() (no GPU/model-file
    dependency) -- the only 2 configs safe to build a prompt for without a real index."""
    messages, debug = vhf_agents.build_vhf_prompt("What channel for a MAYDAY?", config="bare_qwen")
    assert messages[0]["content"] == vhf_agents.BARE_SYSTEM
    assert debug == {"config": "bare_qwen"}

    messages, debug = vhf_agents.build_vhf_prompt("What channel for a MAYDAY?", config="v0_base")
    assert messages[0]["content"] == vhf_agents.SYSTEM_VHF_QA
    assert "sources" not in debug and "pg_used" not in debug


def test_build_vhf_prompt_rejects_unknown_config() -> None:
    try:
        vhf_agents.build_vhf_prompt("question", config="bogus_config")
    except KeyError as e:
        assert "bogus_config" in str(e)
    else:
        raise AssertionError("expected KeyError for an unknown config")


def test_model_configs_and_config_specs_same_keys() -> None:
    assert set(vhf_agents.MODEL_CONFIGS) == set(vhf_agents._CONFIG_SPECS)
