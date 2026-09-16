"""Test that AgentPaths.vhf() points at the real VHF folders on disk."""

from pathlib import Path

from core import AgentPaths


def test_vhf_paths_match_current_layout() -> None:
    paths = AgentPaths.vhf()

    # Repository root — the parent of the ``core`` package.
    assert paths.workspace.exists(), paths.workspace
    assert (paths.workspace / ".git").exists(), "workspace should be the git root"

    # Data folders — must already exist on disk.
    assert paths.data_root.is_dir(),   paths.data_root
    assert paths.source_dir.is_dir(),  paths.source_dir
    assert paths.eval_dir.is_dir(),    paths.eval_dir
    assert paths.json_dir.is_dir(),    paths.json_dir
    assert paths.cache_dir.is_dir(),   paths.cache_dir
    assert paths.gold_file.is_file(),  paths.gold_file

    # Model folders — hf_cache exists (contains Qwen); domain_models_dir was
    # created but is empty until § 14 runs.
    assert paths.models_root.is_dir(),        paths.models_root
    assert paths.hf_cache_dir.is_dir(),       paths.hf_cache_dir
    assert paths.domain_models_dir.is_dir(),  paths.domain_models_dir

    # Names must match the exact strings the pipeline hardcodes today.
    assert paths.source_dir.name == "VHFProtocol"
    assert paths.eval_dir.name   == "VHF_Eval"
    assert paths.json_dir.name   == "VHF_JSON"
    assert paths.cache_dir.name  == "VHF_Agents_Training"
    assert paths.gold_file.name  == "vhf_gold_answers.json"


def test_oow_paths_match_current_layout() -> None:
    paths = AgentPaths.oow()

    assert paths.workspace.exists(), paths.workspace
    assert (paths.workspace / ".git").exists(), "workspace should be the git root"

    # Data folders — must already exist on disk (Data/OfficeroftheWatch was
    # renamed to Data/OOW; subfolders renamed to the OOW_* convention).
    assert paths.data_root.is_dir(),   paths.data_root
    assert paths.source_dir.is_dir(),  paths.source_dir
    assert paths.eval_dir.is_dir(),    paths.eval_dir
    assert paths.json_dir.is_dir(),    paths.json_dir
    assert paths.cache_dir.is_dir(),   paths.cache_dir

    # OOW's committed gold Q&A set keeps its original filename (colreg_qa_500.json)
    # rather than the default oow_gold_answers.json -- load it via eval_file(...).
    assert paths.eval_file("colreg_qa_500.json").is_file(), paths.eval_file("colreg_qa_500.json")

    assert paths.models_root.is_dir(),        paths.models_root
    assert paths.hf_cache_dir.is_dir(),       paths.hf_cache_dir
    assert paths.domain_models_dir.is_dir(),  paths.domain_models_dir

    assert paths.source_dir.name == "OOW_Protocols"
    assert paths.eval_dir.name   == "OOW_Eval"
    assert paths.json_dir.name   == "OOW_JSON"
    assert paths.cache_dir.name  == "OOW_Agents_Training"


def test_new_domain_paths_are_derived() -> None:
    """Constructor for a fresh domain (COLREG) should yield sensible paths."""
    paths = AgentPaths(domain="COLREG", source_dirname="COLREGRules",
                       workspace=Path.cwd())

    assert paths.source_dir.name == "COLREGRules"
    assert paths.eval_dir.name   == "COLREG_Eval"
    assert paths.json_dir.name   == "COLREG_JSON"
    assert paths.cache_dir.name  == "COLREG_Agents_Training"
    assert paths.gold_file.name  == "colreg_gold_answers.json"
    assert paths.domain_models_dir.name == "COLREG"


if __name__ == "__main__":
    test_vhf_paths_match_current_layout()
    test_oow_paths_match_current_layout()
    test_new_domain_paths_are_derived()
    print("OK — AgentPaths matches current VHF/OOW layout and derives new domains correctly.")
    print("\nVHF paths:")
    print(AgentPaths.vhf().describe())
    print("\nOOW paths:")
    print(AgentPaths.oow().describe())
