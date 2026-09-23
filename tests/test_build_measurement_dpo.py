"""Fase C2(ii) tests for pipeline/track2/build_measurement_dpo.py -- the deterministic
(no-LLM) "anti-fabricated-risk" DPO pair miner built on top of Fase C0's measurement
retrofit of the archived units_v1 mission checkpoints."""
import json

from pipeline.oow_agent_spec import ACTIONS, validate_action_json
from pipeline.track2.build_measurement_dpo import build_chosen, build_rejected, OUT_PATH


def test_build_chosen_always_validates_and_cites_no_rule() -> None:
    chosen = build_chosen(min_cpa_m=3000.0, safe_distance_m=500.0)
    obj = {**chosen, "reasoning": chosen["reasoning"]}
    assert not validate_action_json(obj)
    assert chosen["action"] == "hold_course"
    assert chosen["encounter_rule"] == "none" and chosen["conduct_rule"] == "none"
    assert "3000" in chosen["reasoning"] and "500" in chosen["reasoning"]


def test_build_rejected_reads_old_rule_applied_as_both_fields() -> None:
    decision = {"action": "turn_right", "degrees": 15.0, "rule_applied": "Rule 15",
               "reasoning": "some wrong reasoning"}
    rejected = build_rejected(decision)
    assert rejected["encounter_rule"] == "Rule 15"
    assert rejected["conduct_rule"] == "Rule 15"
    assert rejected["action"] == "turn_right"
    assert rejected["reasoning"] == "some wrong reasoning"


def test_build_rejected_prefers_new_schema_fields_when_present() -> None:
    decision = {"action": "turn_right", "degrees": 15.0, "encounter_rule": "Rule 14",
               "conduct_rule": "Rule 16", "rule_applied": "Rule 15", "reasoning": "..."}
    rejected = build_rejected(decision)
    assert rejected["encounter_rule"] == "Rule 14"
    assert rejected["conduct_rule"] == "Rule 16"


def test_written_file_rows_have_valid_chosen_and_action_vocabulary() -> None:
    """Integration check against the real written file (no-op if not generated yet)."""
    if not OUT_PATH.exists():
        return
    n = 0
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chosen = json.loads(row["chosen"][0]["content"])
        rejected = json.loads(row["rejected"][0]["content"])
        assert not validate_action_json(chosen)
        assert chosen["action"] in ACTIONS
        assert rejected["action"] in ACTIONS
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1]["role"] == "user"
        n += 1
    assert n > 0, f"{OUT_PATH} exists but is empty"
