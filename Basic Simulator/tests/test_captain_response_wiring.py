"""Integration test: every Captain decision handler (design_captain_missions.md Sec
16.2/16.4) now also logs a real 'captain_response' entry alongside the existing
'captain_decision' one. No GPU/API key needed."""
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_skeleton import CaptainSkeleton  # noqa: E402

REPO_ROOT = APP_ROOT.parent
SCENARIOS_DIR = REPO_ROOT / "Data" / "Captain" / "Scenarios"
SCENARIO_PATHS = sorted(SCENARIOS_DIR.glob("*.json"))  # the 4 hand-authored ones only


def _run_to_completion(skeleton: CaptainSkeleton, max_steps: int = 20_000) -> None:
    for _ in range(max_steps):
        if skeleton.sim.reached_destination():
            return
        skeleton.step_mission(1)


@pytest.mark.parametrize("path", SCENARIO_PATHS, ids=lambda p: p.stem)
def test_every_captain_decision_has_a_matching_captain_response(path):
    skeleton = CaptainSkeleton.from_scenario_file(path)
    _run_to_completion(skeleton)
    decisions = [d for d in skeleton.state.event_log if d["field_path"] == "captain_decision"]
    responses = [d for d in skeleton.state.event_log if d["field_path"] == "captain_response"]
    assert len(decisions) == len(responses)
    decision_causes = {d["cause"] for d in decisions}
    response_causes = {r["cause"] for r in responses}
    assert decision_causes == response_causes
    for r in responses:
        assert r["new"]["reasoning"]
        assert "resource_note" in r["new"]["plan"]
        assert "updated_goals" in r["new"]["plan"]
        # The actually-applied tool must be named in its own reasoning text.
        matching_decision = next(d for d in decisions if d["cause"] == r["cause"])
        assert matching_decision["new"]["tool"] in r["new"]["reasoning"]
