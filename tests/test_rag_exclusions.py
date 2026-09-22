"""Regression test: the OOW RAG corpus (oow_rag_chunks.json) must never contain a chunk
sourced from an excluded path (eval data, model-output artifacts, derived training JSONL,
meta-literature, or a negative-net-score incident report) -- see
pipeline/ingest/rag_exclusions.py for the full per-category rationale. Run this after
every corpus rebuild (see the RAG-rebuild plan, 2026-09-22).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import pytest

from core import AgentPaths
from pipeline.ingest.rag_exclusions import ExcludedSourceError, is_excluded_chunk, is_excluded_path, raise_if_excluded_source

paths = AgentPaths.oow()
CHUNKS_FILE = paths.cache_dir / "oow_rag_chunks.json"
INCIDENT_SCREENING_FILE = paths.cache_dir / "incident_screening.json"
EVAL_SCENARIOS_FILE = paths.eval_dir / "oow_colreg_scenarios.json"


# ── Unit tests for the exclusion logic itself (synthetic paths, no corpus needed) ──
def test_excludes_eval_dir() -> None:
    assert is_excluded_path("Data/OOW/OOW_Eval/oow_colreg_scenarios.json")


def test_excludes_llm_runs_dir() -> None:
    assert is_excluded_path("Basic Simulator/Data/missions/_llm_runs/Imazu01__v0_base.json")


def test_excludes_literature_review_dir() -> None:
    assert is_excluded_path("Data/OOW/OOW_Literature_Review/some_paper.pdf")


def test_excludes_training_jsonl_filenames() -> None:
    for name in ("oow_sft_direct.jsonl", "oow_incident_sft_cot.jsonl", "oow_dpo_pairs.jsonl",
                "oow_incident_dpo_pairs.jsonl", "oow_reflection.jsonl", "oow_incident_reflection.jsonl",
                "oow_reasoning_traces.jsonl", "oow_multihop.jsonl", "oow_scenario_Leo_checkpoint.jsonl",
                "ablation_summary_full.json", "eval_oow_qwen_full.jsonl", "probe_set.json",
                "ragas_judge_cache.jsonl", "consistency_findings.json"):
        assert is_excluded_path(name), name


def test_does_not_exclude_leo_moos_source() -> None:
    # The one explicit exception: this IS a raw source (Phase 3), not a derived artifact.
    assert not is_excluded_path("Data/OOW/OOW_Scenarios_Leo/moos_temporal_narratives_final.jsonl")


def test_does_not_exclude_legitimate_sources() -> None:
    for name in ("COLREG-Consolidated-2018.pdf", "MAB1605.pdf", "nav_maths_drills.json",
                "free_radar_workbook.pdf", "simple_colreg.json"):
        assert not is_excluded_path(name), name


def test_raise_if_excluded_source_raises() -> None:
    with pytest.raises(ExcludedSourceError):
        raise_if_excluded_source("Data/OOW/OOW_Eval/oow_colreg_scenarios.json")


def test_is_excluded_chunk_checks_both_fields() -> None:
    assert is_excluded_chunk({"source_file": "oow_dpo_pairs.jsonl", "document_id": "x"})
    assert is_excluded_chunk({"source_file": "x", "document_id": "ablation_summary_full"})
    assert not is_excluded_chunk({"source_file": "COLREG-Consolidated-2018.pdf", "document_id": "colreg"})


def _load_chunks() -> list[dict]:
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def test_no_chunk_matches_an_excluded_path() -> None:
    if not CHUNKS_FILE.exists():
        return
    chunks = _load_chunks()
    offenders = [c["chunk_id"] for c in chunks if is_excluded_chunk(c)]
    assert not offenders, (
        f"{len(offenders)} chunk(s) sourced from an excluded path made it into "
        f"oow_rag_chunks.json: {offenders[:10]}"
    )


def test_no_chunk_from_a_negative_net_score_incident() -> None:
    if not CHUNKS_FILE.exists() or not INCIDENT_SCREENING_FILE.exists():
        return
    screening = json.loads(INCIDENT_SCREENING_FILE.read_text(encoding="utf-8"))
    negative_basenames = {Path(r["path"]).name for r in screening if r["net_score"] < 0}
    chunks = _load_chunks()
    offenders = [c["chunk_id"] for c in chunks if c.get("source_file") in negative_basenames]
    assert not offenders, (
        f"{len(offenders)} chunk(s) sourced from a net_score<0 incident report: {offenders[:10]}"
    )


def test_chunk_ids_are_unique_and_match_their_own_text() -> None:
    """A-nawerk-1 (RAG-rebuild plan, 2026-09-22): chunk_id used to be derived from
    section_ids/titles, which repeat across different articles that happen to share a
    title (e.g. CHIRP's per-issue "Initial Report"/"CHIRP Comment" section titles) --
    64 ids silently collided across 196 entries, each carrying DIFFERENT text under the
    SAME id, so chunk_by_id[id]["text"] (used by rerank_hits()/format_context()) could
    silently return a different chunk's text than the one retrieval actually matched.
    chunk_id is now derived from document_id + structural chapter/chunk position + a
    text hash (see build_rag.py's _finalise_chunk()), which is unique by construction --
    this test must stay green after every future corpus rebuild."""
    if not CHUNKS_FILE.exists():
        return
    chunks = _load_chunks()
    ids = [c["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids), (
        f"{len(ids) - len(set(ids))} duplicate chunk_id(s) in oow_rag_chunks.json"
    )
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    for c in chunks:
        assert chunk_by_id[c["chunk_id"]]["text"] == c["text"], (
            f"chunk_by_id[{c['chunk_id']!r}] text mismatch -- id collision"
        )


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _shares_long_run(chunk_words: list[str], eval_words: list[str], min_run: int = 12) -> bool:
    """True if `eval_words` appears in `chunk_words` as a literal, contiguous run of at
    least `min_run` words -- catches verbatim copy-paste of an eval question/gold_answer
    into a chunk, not just topical overlap (which would false-positive on shared COLREG
    terminology that's legitimately common to both)."""
    if len(eval_words) < min_run:
        return False
    haystack = " ".join(chunk_words)
    for start in range(len(eval_words) - min_run + 1):
        run = " ".join(eval_words[start:start + min_run])
        if run in haystack:
            return True
    return False


def test_no_chunk_leaks_eval_scenario_text() -> None:
    if not CHUNKS_FILE.exists() or not EVAL_SCENARIOS_FILE.exists():
        return
    scenarios = json.loads(EVAL_SCENARIOS_FILE.read_text(encoding="utf-8"))
    eval_texts = []
    for s in scenarios:
        for key in ("question", "gold_answer", "situation", "narrative"):
            if isinstance(s.get(key), str):
                eval_texts.append(s[key])
    eval_word_lists = [ws for t in eval_texts if len(ws := _words(t)) >= 12]

    chunks = _load_chunks()
    offenders = []
    for c in chunks:
        # Verbatim COLREG rule text is SUPPOSED to appear in both the corpus (that's the
        # RAG grounding source) and an eval gold_answer (which correctly cites the real
        # rule it's testing) -- that overlap is the intended mechanism, not leakage.
        # Only non-regulation chunks (incidents, CHIRP, moos_case, ...) are checked here;
        # confirmed via a real hit during Phase 1 testing that this exact false-positive
        # class exists (chunk_5b371d03, Rule 14's own text, matched a gold_answer quoting
        # Rule 14 -- correct behaviour, not a leak).
        if c.get("source_type") == "regulation" or "rule" in (c.get("types") or []):
            continue
        chunk_words = _words(c.get("text", ""))
        if any(_shares_long_run(chunk_words, ew) for ew in eval_word_lists):
            offenders.append(c["chunk_id"])
    assert not offenders, (
        f"{len(offenders)} chunk(s) share a >=12-word run with an eval scenario "
        f"(oow_colreg_scenarios.json) -- likely leakage: {offenders[:10]}"
    )


def test_no_moos_case_chunk_overlaps_simulator_missions_or_eval() -> None:
    """Phase 3 (Leo MOOS-narrative canonicalization, see the RAG-rebuild plan) writes
    chunks with source_type == "moos_case" -- deliberately generic so this activates
    automatically once those chunks exist, with no test-file change needed. The actual
    overlap check is implemented alongside Phase 3's ingestion code (needs the same
    canonicalization bucketing to compare against); until then this is a documented no-op,
    not a silent gap."""
    if not CHUNKS_FILE.exists():
        return
    chunks = _load_chunks()
    moos_chunks = [c for c in chunks if c.get("source_type") == "moos_case"]
    if not moos_chunks:
        return
    from pipeline.ingest.build_moos_case_rag import check_no_mission_overlap  # Phase 3
    check_no_mission_overlap(moos_chunks)


KG_FILE = paths.cache_dir / "oow_kg.json"


def test_kg_covers_the_full_corpus() -> None:
    """A-nawerk-2 (RAG-rebuild plan, 2026-09-22): oow_kg.json used to be built once
    against the OLD 678-chunk corpus and never rebuilt, so kg_retrieve()'s concept-boost
    was blind to every chunk added since (chirp_newsletter/moos_case/incident_marginal --
    82% of the corpus at the time). Must stay in sync after every corpus rebuild."""
    if not CHUNKS_FILE.exists() or not KG_FILE.exists():
        return
    chunks = _load_chunks()
    kg = json.loads(KG_FILE.read_text(encoding="utf-8"))
    corpus_ids = {c["chunk_id"] for c in chunks}
    assert set(kg["chunk_meta"]) == corpus_ids, (
        "oow_kg.json's chunk_meta is out of sync with oow_rag_chunks.json -- rerun "
        "pipeline.ingest.build_kg after any corpus rebuild"
    )
    concept_chunk_ids = {cid for cids in kg["concept_chunks"].values() for cid in cids}
    by_type: dict[str, list[str]] = {}
    for c in chunks:
        by_type.setdefault(c.get("source_type"), []).append(c["chunk_id"])
    uncovered_types = [t for t, cids in by_type.items() if not (set(cids) & concept_chunk_ids)]
    assert not uncovered_types, (
        f"source_type(s) with ZERO chunks in any KG concept bucket (never boostable): "
        f"{uncovered_types}"
    )

