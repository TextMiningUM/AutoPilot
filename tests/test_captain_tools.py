"""Tests for pipeline/captain_tools.py (design_captain_missions.md Sec 16.3 skeleton) --
the tool registry must stay honest (status reflects real wiring) and in sync with
captain_agent_spec.py's own candidate tool strings. No GPU/API key needed."""
import re
from pathlib import Path

from pipeline.captain_tools import TOOL_REGISTRY

CAPTAIN_AGENT_SPEC = Path(__file__).resolve().parent.parent / "pipeline" / "captain_agent_spec.py"


def test_every_tool_name_is_unique():
    names = [t.name for t in TOOL_REGISTRY]
    assert len(names) == len(set(names))


def test_status_is_one_of_the_two_known_values():
    assert all(t.status in ("wired", "not_implemented") for t in TOOL_REGISTRY)


def test_scope_is_one_of_the_two_known_values():
    assert all(t.scope in ("decision_layer", "future") for t in TOOL_REGISTRY)


def test_future_tools_are_never_marked_wired():
    assert all(t.status == "not_implemented" for t in TOOL_REGISTRY if t.scope == "future")


def test_wired_decision_layer_tools_actually_appear_as_a_captain_action_tool_in_the_spec_file():
    # Grep-level honesty check: every "wired" tool name must literally appear as a
    # CaptainAction(tool="...") call in captain_agent_spec.py -- catches the registry
    # drifting out of sync with the real candidate-generating functions.
    source = CAPTAIN_AGENT_SPEC.read_text(encoding="utf-8")
    wired_names = {t.name for t in TOOL_REGISTRY if t.status == "wired"}
    tool_literals_in_source = set(re.findall(r'tool="([a-z_0-9]+)"', source))
    missing = wired_names - tool_literals_in_source
    assert not missing, f"registry claims these are wired but they never appear in {CAPTAIN_AGENT_SPEC.name}: {missing}"


def test_includes_the_three_documented_future_extension_points():
    future_names = {t.name for t in TOOL_REGISTRY if t.scope == "future"}
    assert {"search_regulations", "weather_lookup", "web_search"} <= future_names
