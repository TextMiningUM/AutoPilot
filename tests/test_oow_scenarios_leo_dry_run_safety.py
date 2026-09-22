"""Fase B2 dry-run safety tests (RAG-rebuild-v2 plan, 2026-09-22) -- added after a real
incident where `build_oow_scenarios_leo.py --skip-llm` overwrote 5 tracked production
.jsonl files with near-empty content. write_outputs() (both scenario generators use it)
must never touch an existing production file without --overwrite."""
import json
from pathlib import Path

from pipeline.track2.build_oow_scenarios_leo import write_outputs


def test_dry_run_leaves_an_existing_production_file_byte_identical(tmp_path: Path) -> None:
    prod = tmp_path / "oow_scenario_Leo_sft_direct.jsonl"
    original = '{"real": "production data"}\n'
    prod.write_text(original, encoding="utf-8")

    write_outputs({"oow_scenario_Leo_sft_direct.jsonl": [{"fake": "dry run row"}]},
                  tmp_path, overwrite=False)

    assert prod.read_text(encoding="utf-8") == original
    review_file = tmp_path / "_review" / "oow_scenario_Leo_sft_direct.jsonl"
    assert json.loads(review_file.read_text(encoding="utf-8").splitlines()[0]) == {"fake": "dry run row"}


def test_overwrite_flag_allows_writing_to_the_production_path(tmp_path: Path) -> None:
    prod = tmp_path / "oow_scenario_Leo_sft_direct.jsonl"
    prod.write_text('{"old": "data"}\n', encoding="utf-8")

    write_outputs({"oow_scenario_Leo_sft_direct.jsonl": [{"new": "data"}]}, tmp_path, overwrite=True)

    assert json.loads(prod.read_text(encoding="utf-8").splitlines()[0]) == {"new": "data"}


def test_first_ever_run_with_no_overwrite_still_goes_to_review_not_production(tmp_path: Path) -> None:
    """Even when no production file exists yet, a non---overwrite run must still land in
    _review/ -- --overwrite is the only thing that ever selects the production path."""
    write_outputs({"new_file.jsonl": [{"a": 1}]}, tmp_path, overwrite=False)
    assert not (tmp_path / "new_file.jsonl").exists()
    assert (tmp_path / "_review" / "new_file.jsonl").exists()
