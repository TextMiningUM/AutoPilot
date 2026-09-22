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

paths = AgentPaths.oow()
V1_FILE = paths.eval_dir / "oow_colreg_scenarios_v1.json"
V2_FILE = paths.eval_dir / "oow_colreg_scenarios_v2.json"

# Computed 2026-09-22 immediately after `build_oow_scenarios.py --build-v2 --overwrite`.
V2_SHA256 = "cb9de445e8d3c7b427d86ab2134ea5d77cef246350b27946ec480f29a5d663cf"


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
