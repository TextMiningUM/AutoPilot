"""Tests for Basic Simulator/app/agents.py's v7_super_rag/v8_super_cot_pg/v9_super_all
configs and the v1-v6 archived-ablation-arm cleanup (2026-09-22 redefinition of the
never-run v7_rerank/v8_rerank_cot/v9_fewshot/v10_dpo_contrast/v11_reflect prototype slots
-- confirmed via grep that zero result files under Data/missions/**llm_runs/** ever used
those 5 names, so redefining them in place was safe)."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app.agents as agents  # noqa: E402
from app.llm_runs import parse_run_filename, run_log_path  # noqa: E402
from app.missions import load_mission  # noqa: E402

EXPECTED_CONFIGS = {
    "bare_qwen", "v0_base", "v1_rag", "v2_cot", "v3_rag_cot",
    "v4_pg", "v5_pg_incident", "v6_pg_scenario",
    "v7_super_rag", "v8_super_cot_pg", "v9_super_all",
}


def test_config_specs_exact_set() -> None:
    assert set(agents._CONFIG_SPECS) == EXPECTED_CONFIGS
    assert set(agents.MODEL_CONFIGS) == EXPECTED_CONFIGS


def test_v7_has_no_cot_or_pg_but_has_rag() -> None:
    m = load_mission("Imazu01")
    messages, debug = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v7_super_rag")
    user_msg = messages[1]["content"]
    assert agents.COT_INSTR not in user_msg
    assert "Procedure guidance" not in user_msg
    assert "COLREG reference excerpts" in user_msg
    # v7 must stay fast (no forced thinking) -- see effective_generation_params().
    assert agents.effective_generation_params("v7_super_rag", False, 256) == (False, 256)


def test_v8_has_cot_but_no_rag() -> None:
    m = load_mission("Imazu01")
    messages, debug = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v8_super_cot_pg")
    user_msg = messages[1]["content"]
    assert agents.COT_INSTR in user_msg
    assert "COLREG reference excerpts" not in user_msg
    enable_thinking, max_new_tokens = agents.effective_generation_params("v8_super_cot_pg", False, 256)
    assert enable_thinking is True and max_new_tokens >= 3072


def test_v9_has_both_cot_and_rag() -> None:
    m = load_mission("Imazu01")
    messages, debug = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v9_super_all")
    user_msg = messages[1]["content"]
    assert agents.COT_INSTR in user_msg
    assert "COLREG reference excerpts" in user_msg
    enable_thinking, max_new_tokens = agents.effective_generation_params("v9_super_all", False, 256)
    assert enable_thinking is True and max_new_tokens >= 3072


def test_v8_and_v9_pg_spec_is_scenario_plus_incident() -> None:
    assert agents._CONFIG_SPECS["v8_super_cot_pg"]["pg"] == "scenario+incident"
    assert agents._CONFIG_SPECS["v9_super_all"]["pg"] == "scenario+incident"


def test_dead_ingredient_code_fully_removed() -> None:
    src = Path(agents.__file__).read_text(encoding="utf-8")
    for banned in ("FEWSHOT", "DPO_CONTRAST", "REFLECT_INSTR",
                  "never port", "alter course to STARBOARD"):
        assert banned not in src, f"{banned!r} should no longer appear in agents.py"


def test_run_filename_parser_new_form() -> None:
    p = run_log_path("Imazu04", "v8_super_cot_pg", "W0_base", "units_v2")
    assert p.name == "Imazu04__v8_super_cot_pg__W0_base__units_v2.json"
    assert parse_run_filename(p) == {
        "mission_id": "Imazu04", "config": "v8_super_cot_pg",
        "weights": "W0_base", "tag": "units_v2",
    }


def test_run_filename_parser_old_three_segment_form() -> None:
    # The units_v1 archive's actual naming (e.g. Imazu04__v6_pg_scenario__units_v1.json) --
    # no weights segment, since every run so far has used base Qwen3-8B.
    p = Path("Imazu04__v6_pg_scenario__units_v1.json")
    assert parse_run_filename(p) == {
        "mission_id": "Imazu04", "config": "v6_pg_scenario",
        "weights": "W0_base", "tag": "units_v1",
    }


def test_run_filename_parser_old_two_segment_form() -> None:
    p = Path("Imazu04__v0_base.json")
    assert parse_run_filename(p) == {
        "mission_id": "Imazu04", "config": "v0_base",
        "weights": "W0_base", "tag": "default",
    }
