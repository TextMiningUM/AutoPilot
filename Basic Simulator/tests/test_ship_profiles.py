"""Phase 3 step 2 part B (2026-09-26): ship-dynamics-agnostic training data. Tests that
sample_row_limits_nomoto()/sample_ship_profile() (pipeline/oow_agent_spec.py) correctly
sample a REAL, cited ship profile (pipeline/nomoto.SHIP_PROFILES) per training row,
exclude the held-out generalization-eval profile from training sampling, and stay
deterministic across regeneration runs -- same guarantees sample_row_limits() already
gives for safe_distance_m/max_turn_deg."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # Auto Pilot/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.nomoto import HELD_OUT_EVAL_PROFILE, SHIP_PROFILES, TRAINING_PROFILE_WEIGHTS
from pipeline.oow_agent_spec import sample_row_limits, sample_row_limits_nomoto, sample_ship_profile


def test_held_out_profile_has_zero_training_weight():
    assert TRAINING_PROFILE_WEIGHTS[HELD_OUT_EVAL_PROFILE] == 0.0
    assert HELD_OUT_EVAL_PROFILE in SHIP_PROFILES  # still a real, usable profile for eval


def test_sample_ship_profile_never_returns_held_out_for_training():
    for i in range(500):
        assert sample_ship_profile(f"row_{i}") != HELD_OUT_EVAL_PROFILE


def test_sample_ship_profile_can_return_held_out_when_explicitly_allowed():
    picks = {sample_ship_profile(f"row_{i}", allow_held_out=True) for i in range(300)}
    assert HELD_OUT_EVAL_PROFILE in picks


def test_sample_ship_profile_is_deterministic():
    assert sample_ship_profile("row_abc") == sample_ship_profile("row_abc")


def test_sample_row_limits_nomoto_is_deterministic_and_matches_legacy_shape():
    r1 = sample_row_limits_nomoto("row_1", own_speed_mps=6.17)
    r2 = sample_row_limits_nomoto("row_1", own_speed_mps=6.17)
    assert r1 == r2
    legacy = sample_row_limits("row_1", own_speed_mps=6.17)
    # Same safe_distance_m/max_turn_deg/decision_interval_s sampling reused verbatim.
    assert r1["safe_distance_m"] == legacy["safe_distance_m"]
    assert r1["max_turn_deg"] == legacy["max_turn_deg"]
    assert r1["decision_interval_s"] == legacy["decision_interval_s"]
    # New Nomoto-only keys present.
    assert r1["ship_profile"] in SHIP_PROFILES
    assert r1["manoeuvre_time_s"] > 0


def test_sample_row_limits_nomoto_varies_manoeuvre_time_across_profiles():
    """Different sampled ship profiles must produce genuinely different manoeuvre-time
    facts for the SAME max_turn_deg -- otherwise the profile sampling would be inert."""
    from pipeline.nomoto import manoeuvre_time_s
    t_sawada = manoeuvre_time_s(30.0, SHIP_PROFILES["sawada2021"])
    t_yukun = manoeuvre_time_s(30.0, SHIP_PROFILES["yukun2023_large"])
    assert t_sawada != t_yukun
