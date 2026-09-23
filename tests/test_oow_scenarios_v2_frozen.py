"""RAG-rebuild-v2 plan point 2 ("GENEREER V2"): oow_colreg_scenarios_v2.json holds the
SAME 325 geometries as v1 (oow_colreg_scenarios_v1.json), re-rendered in the unified
task format (Fase B2) instead of v1's free-prose format. Like v1, it is a FROZEN, held-out
file once written -- this test pins its sha256 the same way test_oow_scenarios_v1_frozen.py
pins v1's, plus verifies the 1:1 v1_id linkage and identical category distribution that
build_v2_eval_records() is required to produce.
"""
import hashlib
import json
from collections import Counter

from core import AgentPaths
from pipeline.oow_agent_spec import ACTIONS

paths = AgentPaths.oow()
V1_FILE = paths.eval_dir / "oow_colreg_scenarios_v1.json"
V2_FILE = paths.eval_dir / "oow_colreg_scenarios_v2.json"
PROBE_300_FILE = paths.eval_dir / "oow_colreg_scenarios_v2_probe_300.json"
PROBE_926_FILE = paths.eval_dir / "oow_colreg_scenarios_v2_probe_926.json"

# Computed 2026-09-23 immediately after `build_oow_scenarios.py --build-v2 --overwrite`
# (RE-computed after quality-review STAP 2 [constraint_line() gains max_turn_deg/
# risk_horizon_s] AND STAP 3b [the _diverging_turn sign-bug fix] -- an intentional,
# approved re-generation per the user's own BLOK II direction, not a drift; the 325
# geometries/v1_id linkage are unchanged, only the rendered situation text and a handful
# of Rule-13/17(b) gold turn DIRECTIONS changed).
V2_SHA256 = "1bf1ed51d3f3c64a5da1bb496ee2ea17d8db9a996f2208e67633f13aec7bd2d6"

# STAP 4 (2026-09-23): the two safe_distance_m probes (ONLY safe_distance_m changed vs
# v2 -- max_turn_deg stays 30, risk horizon stays each scenario's own derived default),
# first-time-frozen the same way v1/v2 are.
PROBE_300_SHA256 = "d25ea39c5b540a45b75d0feab7469d2a5d7782794e8dd616ac140c876d79ae69"
PROBE_926_SHA256 = "e35ee657bf3e05bda1293e8d0ee43519ff7fa76e59d62e0316d2daf94d98042e"


def test_v2_scenarios_file_is_byte_for_byte_frozen() -> None:
    if not V2_FILE.exists():
        return
    actual = hashlib.sha256(V2_FILE.read_bytes()).hexdigest()
    assert actual == V2_SHA256, (
        f"{V2_FILE} has changed (sha256 {actual} != pinned {V2_SHA256}) -- this file is "
        "FROZEN like v1. If this change was truly intentional, that means a NEW version "
        "is needed (v3), not an edit to v2 -- update this pin only after confirming with "
        "the user that invalidating v2-referencing results is intended."
    )


def test_v2_is_325_records_linked_1to1_to_v1_with_matching_category_distribution() -> None:
    if not (V1_FILE.exists() and V2_FILE.exists()):
        return
    v1 = json.loads(V1_FILE.read_text(encoding="utf-8"))
    v2 = json.loads(V2_FILE.read_text(encoding="utf-8"))
    assert len(v2) == len(v1) == 325
    assert Counter(r["category"] for r in v2) == Counter(r["category"] for r in v1)
    v1_ids = [r["id"] for r in v1]
    for i, rec in enumerate(v2):
        assert rec["v1_id"] == v1_ids[i], (
            f"v2 record {rec['id']} links to v1_id={rec['v1_id']!r}, expected "
            f"{v1_ids[i]!r} (positional 1:1 linkage) -- v2 must be regenerated from the "
            "same deterministic geometry as v1, in the same order"
        )
        assert rec["category"] == v1[i]["category"]


# ── STAP 4: the two safe_distance_m probes (frozen the same way v1/v2 are) ────────────
def test_probe_files_are_byte_for_byte_frozen() -> None:
    for probe_file, probe_sha in ((PROBE_300_FILE, PROBE_300_SHA256), (PROBE_926_FILE, PROBE_926_SHA256)):
        if not probe_file.exists():
            continue
        actual = hashlib.sha256(probe_file.read_bytes()).hexdigest()
        assert actual == probe_sha, (
            f"{probe_file} has changed (sha256 {actual} != pinned {probe_sha}) -- frozen "
            "like v1/v2. Update this pin only after confirming an intentional regeneration."
        )


def test_probe_files_isolate_only_safe_distance_m_vs_v2() -> None:
    """probe_300/probe_926 must differ from v2 ONLY in safe_distance_m (and its
    downstream effect on the derived risk horizon, which is DEFINED in terms of
    safe_distance_m -- see derive_risk_horizon_s()) -- max_turn_deg stays 30, and the
    horizon must scale exactly proportionally with safe_distance_m (own_speed/max_turn_deg
    unchanged), never an independently-sampled random multiplier."""
    if not (V2_FILE.exists() and PROBE_300_FILE.exists() and PROBE_926_FILE.exists()):
        return
    v2 = json.loads(V2_FILE.read_text(encoding="utf-8"))
    for probe_file, expected_safe_distance in ((PROBE_300_FILE, 300.0), (PROBE_926_FILE, 926.0)):
        probe = json.loads(probe_file.read_text(encoding="utf-8"))
        assert len(probe) == len(v2) == 325
        for v2_rec, probe_rec in zip(v2, probe):
            assert probe_rec["v1_id"] == v2_rec["v1_id"]
            assert probe_rec["safe_distance_m"] == expected_safe_distance
            assert probe_rec["max_turn_deg"] == v2_rec["max_turn_deg"] == 30.0
            expected_horizon = v2_rec["risk_horizon_s"] * (expected_safe_distance / v2_rec["safe_distance_m"])
            assert abs(probe_rec["risk_horizon_s"] - expected_horizon) < 1e-6, (
                f"{probe_rec['id']}: risk_horizon_s should scale exactly proportionally "
                "with safe_distance_m (both derive from the SAME own_speed/max_turn_deg)"
            )


def test_probe_and_v2_gold_actions_validate_against_the_two_field_schema() -> None:
    for f in (V2_FILE, PROBE_300_FILE, PROBE_926_FILE):
        if not f.exists():
            continue
        for rec in json.loads(f.read_text(encoding="utf-8")):
            gold = rec["gold"]
            assert gold["action"] in ACTIONS
            for field in ("encounter_rule", "conduct_rule"):
                v = gold[field]
                assert v == "none" or (v.startswith("Rule ") and v.split(" ", 1)[-1].isdigit())
            both_none = (gold["encounter_rule"] == "none") == (gold["conduct_rule"] == "none")
            stationary_exception = gold["encounter_rule"] == "none" and gold["conduct_rule"] == "Rule 8"
            assert both_none or stationary_exception, f"{rec['id']}: {gold}"

