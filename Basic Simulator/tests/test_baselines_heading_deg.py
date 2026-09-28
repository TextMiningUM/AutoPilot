"""Regression tests (2026-09-28): every app/baselines/*.py decide() function now includes
an ABSOLUTE `heading_deg` in its decision dict alongside the existing relative
action/degrees -- computed fresh from own-ship's CURRENT actual heading each call, so
repeating the same decision converges instead of stacking (see
Simulation.apply_action()'s docstring). None of these tests call the LLM -- pure
deterministic baseline functions against a real mission's live geometry."""
import math
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.baselines import DECISION_FUNCS  # noqa: E402
from app.missions import load_mission  # noqa: E402
from app.simulation import VesselConstraints  # noqa: E402


@pytest.mark.parametrize("config", list(DECISION_FUNCS))
def test_decision_dict_has_heading_deg_key(config):
    mission = load_mission("Imazu01")
    constraints = VesselConstraints(cruise_speed_mps=mission.own_ship.speed)
    decision, _debug = DECISION_FUNCS[config](mission, mission.own_ship, mission.targets, constraints)
    assert "heading_deg" in decision


@pytest.mark.parametrize("config", list(DECISION_FUNCS))
def test_heading_deg_is_none_only_for_non_turn_actions(config):
    mission = load_mission("Imazu01")
    constraints = VesselConstraints(cruise_speed_mps=mission.own_ship.speed)
    decision, _debug = DECISION_FUNCS[config](mission, mission.own_ship, mission.targets, constraints)
    if decision["action"] in ("turn_left", "turn_right"):
        assert decision["heading_deg"] is not None
        assert 0.0 <= decision["heading_deg"] < 360.0
    else:
        assert decision["heading_deg"] is None


def test_repeated_ruletree_avoidance_decision_converges_not_stacks():
    """The core fix under test: re-deciding "the same" avoidance turn while own-ship's
    ACTUAL heading hasn't caught up yet must compute nearly the SAME absolute heading_deg
    each time (converging), not an ever-larger one (stacking) -- this is what heading_deg
    (derived from own.heading, not a commanded/target heading) guarantees by construction."""
    from app.baselines.ruletree import decide_action

    mission = load_mission("Imazu01")
    constraints = VesselConstraints(cruise_speed_mps=mission.own_ship.speed)
    own = mission.own_ship
    d1 = decide_action(mission, own, mission.targets, constraints)
    d2 = decide_action(mission, own, mission.targets, constraints)  # own.heading unchanged
    assert d1["heading_deg"] == d2["heading_deg"]
