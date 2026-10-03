"""Unit tests for pipeline/eval/build_captain_mission_scenarios.py -- pure selection
logic against a temp directory of synthetic mission files, no real corpus needed."""
import json

import pipeline.eval.build_captain_mission_scenarios as m


def _mission(mission_id, held_out, route_template="short", events=None):
    return {
        "mission_id": mission_id, "held_out": held_out, "route_template": route_template,
        "brown_envelopes": events or [{"type": "fog"}],
    }


def test_load_held_out_missions_filters_and_sorts(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "GENERATED_DIR", tmp_path)
    (tmp_path / "MSN-GEN-0100-0002.json").write_text(
        json.dumps(_mission("MSN-GEN-0100-0002", True)), encoding="utf-8")
    (tmp_path / "MSN-GEN-0100-0000.json").write_text(
        json.dumps(_mission("MSN-GEN-0100-0000", False)), encoding="utf-8")
    (tmp_path / "MSN-GEN-0100-0001.json").write_text(
        json.dumps(_mission("MSN-GEN-0100-0001", True)), encoding="utf-8")

    missions = m.load_held_out_missions()
    assert [mm["mission_id"] for mm in missions] == ["MSN-GEN-0100-0001", "MSN-GEN-0100-0002"]


def test_load_held_out_missions_empty_when_none_held_out(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "GENERATED_DIR", tmp_path)
    (tmp_path / "MSN-GEN-0100-0000.json").write_text(
        json.dumps(_mission("MSN-GEN-0100-0000", False)), encoding="utf-8")
    assert m.load_held_out_missions() == []
