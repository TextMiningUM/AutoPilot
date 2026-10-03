"""Regression/feature test for the Chief Engineer live-condition fallback wired into
Basic Simulator/app/captain_skeleton.py's _handle_engine_failure() -- additive only:
every existing scripted scenario still provides `capped_speed_kn` explicitly and takes
the original path unchanged (covered by test_captain_skeleton.py/_encounter.py, which
stay green). This test exercises the NEW branch taken only when a scenario's
engine_failure event omits `capped_speed_kn`."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.captain_skeleton import CaptainSkeleton  # noqa: E402
from pipeline.captain_types import BrownEnvelopeEvent  # noqa: E402

REPO_ROOT = APP_ROOT.parent
SCENARIO_PATH = REPO_ROOT / "Data" / "Captain" / "Scenarios" / "skeleton_engine_failure_v1.json"


def _fresh_skeleton() -> CaptainSkeleton:
    return CaptainSkeleton.from_scenario_file(SCENARIO_PATH)


def test_engine_failure_without_scripted_speed_cap_falls_back_to_live_condition_model():
    skeleton = _fresh_skeleton()
    original_speed_kn = skeleton.sim.state.current_speed_kn
    event = BrownEnvelopeEvent(event_id="ev_live_condition_test", type="engine_failure",
                                severity="moderate", params={})  # deliberately no capped_speed_kn

    skeleton._handle_engine_failure(event)

    assert skeleton.sim.state.current_speed_kn < original_speed_kn
    decisions = [d for d in skeleton.state.event_log if d["field_path"] == "captain_decision"]
    assert len(decisions) == 1
    assert decisions[0]["cause"] == "ev_live_condition_test"


def test_engine_failure_live_condition_is_deterministic_for_the_same_event_id():
    s1, s2 = _fresh_skeleton(), _fresh_skeleton()
    event = BrownEnvelopeEvent(event_id="ev_repeatable", type="engine_failure",
                                severity="moderate", params={})
    s1._handle_engine_failure(event)
    s2._handle_engine_failure(event)
    assert s1.sim.state.current_speed_kn == s2.sim.state.current_speed_kn


def test_engine_failure_with_explicit_component_selects_that_component():
    skeleton = _fresh_skeleton()
    event = BrownEnvelopeEvent(event_id="ev_component_test", type="engine_failure",
                                severity="moderate",
                                params={"component_id": "turbocharger", "parameter": "exhaust_gas_temp_c"})
    status = skeleton._compute_live_engine_status(event)
    assert status.fault is not None
    assert "turbocharger" in status.fault
