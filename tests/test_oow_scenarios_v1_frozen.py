"""Fase A/B "BEVRIES V1" (RAG-rebuild-v2 plan, 2026-09-22): oow_colreg_scenarios_v1.json
is the ORIGINAL prose-answer-format Track 2 held-out eval set (325 scenarios, archived
0.714 DirectionCorrect result was measured against this exact file). It must NEVER change
again -- any edit (including a well-intentioned "fix") invalidates every past result that
cites it. This test pins its sha256; a code change that alters the file must fail here
first, loudly, rather than silently drifting old archived numbers out from under them.
"""
import hashlib

from core import AgentPaths

paths = AgentPaths.oow()
V1_FILE = paths.eval_dir / "oow_colreg_scenarios_v1.json"

# Computed 2026-09-22 immediately after renaming oow_colreg_scenarios.json ->
# oow_colreg_scenarios_v1.json (git mv, byte-for-byte, no content change).
V1_SHA256 = "602a80c7e0572559de8872c45afd1eb165ff94b4d67785a08b67ebea81892b2a"


def test_v1_scenarios_file_is_byte_for_byte_frozen() -> None:
    if not V1_FILE.exists():
        return
    actual = hashlib.sha256(V1_FILE.read_bytes()).hexdigest()
    assert actual == V1_SHA256, (
        f"{V1_FILE} has changed (sha256 {actual} != pinned {V1_SHA256}) -- this file is "
        "FROZEN, archived results (e.g. the 0.714 DirectionCorrect number) are only valid "
        "against this exact content. If this change was truly intentional, that means a "
        "NEW version is needed (v3), not an edit to v1 -- update this pin only after "
        "confirming with the user that invalidating v1-referencing results is intended."
    )
