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
from app.simulation import VesselConstraints  # noqa: E402

EXPECTED_CONFIGS = {
    "bare_qwen", "v0_base", "v1_rag", "v2_cot", "v3_rag_cot",
    "v4_pg", "v5_pg_incident", "v6_pg_scenario",
    "v7_super_rag", "v8_super_cot_pg", "v9_super_all",
    "v10_super_colreg_rag", "v11_super_colreg_rag_cot",
}


def test_config_specs_exact_set() -> None:
    assert set(agents._CONFIG_SPECS) == EXPECTED_CONFIGS
    assert set(agents.MODEL_CONFIGS) == EXPECTED_CONFIGS


def test_kinematics_model_physical_limits_paragraph_unchanged_by_default() -> None:
    """2026-09-26: adding the Nomoto-aware branch must not alter a single byte of the
    default ("kinematics") physical-limits text -- no prompt-hash bump for existing runs."""
    m = load_mission("Imazu01")
    messages, _ = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v0_base",
                                          constraints=VesselConstraints())
    user_msg = messages[1]["content"]
    assert "Own-ship's physical limits: heading changes at 3.0 deg/s." in user_msg
    assert "a 90 deg turn takes about 30s" in user_msg
    assert "responds with lag" not in user_msg


def test_kinematics_model_nomoto_states_real_measured_turn_times() -> None:
    """The Nomoto branch must state REAL simulated manoeuvre_time_s() numbers (not the
    legacy linear formula) -- verified against the SAME function, not a hardcoded number,
    so this test can't silently drift out of sync with the physics."""
    from pipeline.nomoto import NomotoParams, manoeuvre_time_s
    m = load_mission("Imazu01")
    constraints = VesselConstraints(kinematics_model="nomoto")
    messages, _ = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v0_base",
                                          constraints=constraints)
    user_msg = messages[1]["content"]
    assert "responds with lag" in user_msg
    params = NomotoParams(K_per_s=constraints.nomoto_K_per_s, T_s=constraints.nomoto_T_s,
                          T_E_s=constraints.nomoto_T_E_s, rudder_limit_deg=constraints.nomoto_rudder_limit_deg,
                          autopilot_kp=constraints.nomoto_autopilot_kp)
    t60 = manoeuvre_time_s(60.0, params, substep_s=constraints.nomoto_substep_s)
    assert f"a 60 deg turn about {t60:.0f}s" in user_msg
    # A 30deg reference turn (max_rudder_angle_deg default) manoeuvre-time fact must also
    # be present via constraint_line()'s manoeuvre_time_s param (Phase 3 step 1, now wired in).
    assert "degree turn takes about" in user_msg and "to complete." in user_msg


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


_COLREG_ONLY_DOCS = {"colreg_consolidated_2018", "simple_colreg"}


def test_v10_rag_scoped_to_colreg_only_corpus() -> None:
    m = load_mission("Imazu01")
    messages, debug = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v10_super_colreg_rag")
    user_msg = messages[1]["content"]
    assert agents.COT_INSTR not in user_msg
    assert "Procedure guidance" not in user_msg
    assert "COLREG reference excerpts" in user_msg
    assert debug["retrieved_chunk_ids"], "expected at least one retrieved chunk"
    _, _, _, _, _, _, _, _, _, _, chunk_by_id_co = agents._load_retrieval()
    for cid in debug["retrieved_chunk_ids"]:
        assert chunk_by_id_co[cid]["document_id"] in _COLREG_ONLY_DOCS


def test_v11_has_cot_and_colreg_only_rag() -> None:
    m = load_mission("Imazu01")
    messages, debug = agents.build_oow_prompt(m, m.own_ship, m.targets, config="v11_super_colreg_rag_cot")
    user_msg = messages[1]["content"]
    assert agents.COT_INSTR in user_msg
    assert "COLREG reference excerpts" in user_msg
    _, _, _, _, _, _, _, _, _, _, chunk_by_id_co = agents._load_retrieval()
    for cid in debug["retrieved_chunk_ids"]:
        assert chunk_by_id_co[cid]["document_id"] in _COLREG_ONLY_DOCS
    enable_thinking, max_new_tokens = agents.effective_generation_params("v11_super_colreg_rag_cot", False, 256)
    assert enable_thinking is True and max_new_tokens >= 3072
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
