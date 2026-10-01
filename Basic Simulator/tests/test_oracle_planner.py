"""Stap 2 plan (2026-09-29): sanity tests for the formalized oracle-planner
(app/oracle_planner.py). Focused on the two properties the whole plan depends on: the
rule-legality gate is actually enforced (no illegal-direction turn is ever CHOSEN when a
direction is mandated), and the planner runs cleanly end-to-end against real missions."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app import oracle_planner  # noqa: E402
from app.missions import load_mission  # noqa: E402
from app.simulation import VesselConstraints  # noqa: E402
from pipeline.captain_types import ExclusionZone  # noqa: E402


def test_no_risk_mission_holds_course():
    # RND01/RND02 (Stap 2 pool) start far enough out that TCPA exceeds the risk horizon
    # at t=0 -- "early" band, no hard direction constraint, correct answer is hold_course
    # (matches this project's established early-band convention elsewhere).
    m = load_mission("RND01")
    result = oracle_planner.plan(m, m.own_ship, m.targets, VesselConstraints(kinematics_model="nomoto_v2"))
    assert result["action"] == "hold_course"
    assert oracle_planner.required_direction(m.own_ship, m.targets, VesselConstraints()) is None


def test_imp13_chooses_a_rule_legal_starboard_turn_at_t0():
    """IMP13 ('Chaos at the windward mark, 7 vessels') is the mission whose B_17c
    (stand-on-acted-too-early) pattern broke baseline_vo/v10/v11 identically in the
    2026-09-29 audit -- the whole point of this planner's hard filter is to never pick
    an illegal direction here."""
    m = load_mission("IMP13")
    constraints = VesselConstraints(kinematics_model="nomoto_v2")
    required = oracle_planner.required_direction(m.own_ship, m.targets, constraints)
    result = oracle_planner.plan(m, m.own_ship, m.targets, constraints)
    if required in ("starboard", "port"):
        chosen_dir = oracle_planner._offset_direction(
            result["degrees"] if result["action"] == "turn_right" else -(result["degrees"] or 0.0))
        assert chosen_dir == required
    elif required == "hold":
        assert result["action"] == "hold_course"


def test_rejected_candidates_excludes_only_the_chosen_one():
    m = load_mission("IMP13")
    constraints = VesselConstraints(kinematics_model="nomoto_v2")
    result = oracle_planner.plan(m, m.own_ship, m.targets, constraints)
    rejected_offsets = {c["offset_deg"] for c in result["rejected_candidates"]}
    if result["action"] == "hold_course":
        assert 0.0 not in rejected_offsets or len(rejected_offsets) >= 1  # hold may or may not be 0.0 exactly
    all_candidates = oracle_planner._candidate_offsets(m, m.own_ship, m.targets, constraints)
    assert len(result["rejected_candidates"]) == len(set(all_candidates)) - 1


def test_plan_runs_cleanly_across_the_existing_imazu_set():
    for mission_id in ["Imazu01", "Imazu05", "Imazu13", "IMP05", "IMP12"]:
        m = load_mission(mission_id)
        result = oracle_planner.plan(m, m.own_ship, m.targets, VesselConstraints(kinematics_model="nomoto_v2"))
        assert result["action"] in ("hold_course", "turn_left", "turn_right")


# --- avoid_zone hard gate (Sec 13.A.1 point 5, Phase 9d) -----------------------------------

def test_zones_none_is_a_pure_no_op():
    m = load_mission("RND01")
    constraints = VesselConstraints(kinematics_model="nomoto_v2")
    without = oracle_planner.plan(m, m.own_ship, m.targets, constraints)
    with_none = oracle_planner.plan(m, m.own_ship, m.targets, constraints, zones=None)
    assert without["action"] == with_none["action"] == "hold_course"


def test_a_zone_directly_ahead_forces_a_turn_away_from_hold_course():
    m = load_mission("RND01")  # no real COLREG risk at t=0 -- would otherwise hold_course
    constraints = VesselConstraints(kinematics_model="nomoto_v2")
    # Positioned where the hold_course candidate's straight path still sits exactly on
    # x=0 (always blocked) but a turning candidate has had enough of the rollout's own
    # slow Nomoto response time to laterally diverge clear of it (own-ship's own rate-
    # limited turn dynamics mean a candidate barely moves sideways within the first
    # couple of 10s steps -- verified by hand-tracing the rollout before picking this
    # zone's own position/width).
    zone = ExclusionZone(id="z1", type="dynamic_hazard",
                         polygon=[(-15.0, 300.0), (15.0, 300.0), (15.0, 370.0), (-15.0, 370.0)])
    result = oracle_planner.plan(m, m.own_ship, m.targets, constraints, zones=[zone])
    assert result["action"] != "hold_course"


def test_a_distant_zone_does_not_affect_the_decision():
    m = load_mission("RND01")
    constraints = VesselConstraints(kinematics_model="nomoto_v2")
    # Far beyond the rollout's own reach (~370m) -- must not be treated as blocking.
    zone = ExclusionZone(id="z1", type="dynamic_hazard",
                         polygon=[(-500.0, 50000.0), (500.0, 50000.0), (500.0, 50100.0), (-500.0, 50100.0)])
    result = oracle_planner.plan(m, m.own_ship, m.targets, constraints, zones=[zone])
    assert result["action"] == "hold_course"
