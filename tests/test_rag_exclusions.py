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
EVAL_SCENARIOS_FILE = paths.eval_dir / "oow_colreg_scenarios_v1.json"
EVAL_SCENARIOS_V2_FILE = paths.eval_dir / "oow_colreg_scenarios_v2.json"


# ── Unit tests for the exclusion logic itself (synthetic paths, no corpus needed) ──
def test_excludes_eval_dir() -> None:
    assert is_excluded_path("Data/OOW/OOW_Eval/oow_colreg_scenarios_v1.json")


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
        raise_if_excluded_source("Data/OOW/OOW_Eval/oow_colreg_scenarios_v1.json")


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


MIN_LEAK_RUN = 12  # contiguous shared words before we call it a likely leak


def _ngrams(words: list[str], n: int = MIN_LEAK_RUN) -> set[str]:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _regulation_ngrams(chunks: list[dict]) -> set[str]:
    """Every MIN_LEAK_RUN-word run appearing in any source_type=="regulation" chunk --
    i.e. verbatim COLREG treaty text. Both the corpus (the RAG grounding source) and an
    eval gold_answer that correctly cites a real rule are SUPPOSED to contain this same
    text -- that overlap is the intended mechanism, not leakage."""
    ngrams: set[str] = set()
    for c in chunks:
        if c.get("source_type") == "regulation":
            ngrams |= _ngrams(_words(c.get("text", "")))
    return ngrams


def _boilerplate_ngrams(scenarios: list[dict], min_scenarios: int = 2) -> set[str]:
    """N-grams that recur across >= `min_scenarios` DISTINCT scenario entries in
    oow_colreg_scenarios.json itself. Genuinely scenario-specific content (the kind a
    memorizing RAG chunk could actually "leak") appears in exactly one scenario's own
    answer; a phrase independently reused across many different scenarios is, by
    construction, a templated/formulaic legal paraphrase baked into the eval generator
    (e.g. "the give-way vessel to take early and substantial action to keep [well
    clear]", Rule 16's standard textbook paraphrase, confirmed recurring in 15/325
    scenarios here) -- not something a corpus chunk could leak, since it isn't tied to
    any one scenario's answer key."""
    scenario_ngram_sets = []
    for s in scenarios:
        ng: set[str] = set()
        for key in ("question", "gold_answer", "situation", "narrative"):
            if isinstance(s.get(key), str):
                ng |= _ngrams(_words(s[key]))
        scenario_ngram_sets.append(ng)
    counts: dict[str, int] = {}
    for ng in scenario_ngram_sets:
        for g in ng:
            counts[g] = counts.get(g, 0) + 1
    return {g for g, n in counts.items() if n >= min_scenarios}


def _load_eval_scenarios() -> list[dict]:
    """Both v1 and v2 are held-out (RAG-rebuild-v2 plan point 3: 'De lekkage-tests uit
    fase A moeten hun n-grammen uit ZOWEL v1 als v2 trekken (beide zijn held-out).'),
    so leakage checks must scan text from both files, not just v1."""
    scenarios: list[dict] = []
    if EVAL_SCENARIOS_FILE.exists():
        scenarios += json.loads(EVAL_SCENARIOS_FILE.read_text(encoding="utf-8"))
    if EVAL_SCENARIOS_V2_FILE.exists():
        scenarios += json.loads(EVAL_SCENARIOS_V2_FILE.read_text(encoding="utf-8"))
    return scenarios


def _eval_texts(scenarios: list[dict]) -> list[str]:
    texts = []
    for s in scenarios:
        for key in ("question", "gold_answer", "situation", "situation_report",
                    "gold_reasoning", "narrative"):
            if isinstance(s.get(key), str):
                texts.append(s[key])
    return texts


def _find_leaking_chunks(chunks: list[dict], real_leak_ngrams: set[str]) -> list[str]:
    offenders = []
    for c in chunks:
        if c.get("source_type") == "regulation":
            continue
        if _ngrams(_words(c.get("text", ""))) & real_leak_ngrams:
            offenders.append(c["chunk_id"])
    return offenders


def test_no_chunk_leaks_eval_scenario_text() -> None:
    """A-nawerk-4 (RAG-rebuild plan, 2026-09-22): a naive "chunk shares a >=12-word run
    with any eval text" check false-positived on 2 CHIRP chunks. Investigation (see the
    Fase A report) showed the shared runs are NOT verbatim regulation-chunk text as
    first assumed (Rule 16/17's actual treaty wording differs word-for-word -- e.g. "the
    vessel required to keep out of the way" vs the chunks' "the give-way vessel") -- they
    are a standard textbook paraphrase of Rule 16/17 that recurs independently across the
    eval set's OWN generated scenarios (15/325 for the Rule 16 phrasing), i.e. templated
    boilerplate, not scenario-specific leaked content. Fix: build the "real leak" n-gram
    set as (eval scenario n-grams) MINUS (n-grams also in a source_type=="regulation"
    chunk) MINUS (n-grams recurring in >=2 distinct eval scenarios) -- never an id/type
    whitelist, so the next similar case in a non-regulation chunk still gets caught.
    See test_leakage_test_still_catches_a_real_scenario_leak for the counter-test that
    this doesn't just stop catching real leaks."""
    if not CHUNKS_FILE.exists() or not EVAL_SCENARIOS_FILE.exists():
        return
    scenarios = _load_eval_scenarios()
    eval_ngrams: set[str] = set()
    for t in _eval_texts(scenarios):
        eval_ngrams |= _ngrams(_words(t))

    chunks = _load_chunks()
    real_leak_ngrams = eval_ngrams - _regulation_ngrams(chunks) - _boilerplate_ngrams(scenarios)
    offenders = _find_leaking_chunks(chunks, real_leak_ngrams)
    assert not offenders, (
        f"{len(offenders)} chunk(s) share a >={MIN_LEAK_RUN}-word run with an eval scenario "
        f"(oow_colreg_scenarios.json) that is NOT shared regulation treaty text or recurring "
        f"eval-set boilerplate -- likely leakage: {offenders[:10]}"
    )


def test_leakage_test_still_catches_a_real_scenario_leak() -> None:
    """Counter-test for the n-gram-subtraction fix above: a chunk containing an actual
    eval-scenario sentence (one that does NOT also appear in any regulation chunk, and
    isn't recurring boilerplate across many scenarios, so it can't be a false positive
    of either kind) must still be flagged."""
    if not CHUNKS_FILE.exists() or not EVAL_SCENARIOS_FILE.exists():
        return
    scenarios = _load_eval_scenarios()
    chunks = _load_chunks()
    reg_ngrams = _regulation_ngrams(chunks)
    boilerplate_ngrams = _boilerplate_ngrams(scenarios)

    leak_text = None
    for t in _eval_texts(scenarios):
        words = _words(t)
        t_ngrams = _ngrams(words)
        if len(words) >= MIN_LEAK_RUN and not (t_ngrams & reg_ngrams) and not (t_ngrams & boilerplate_ngrams):
            leak_text = t
            break
    assert leak_text is not None, "no usable eval sentence found for this counter-test"

    eval_ngrams = _ngrams(_words(leak_text))
    real_leak_ngrams = eval_ngrams - reg_ngrams - boilerplate_ngrams
    fake_chunk = {"chunk_id": "chunk_test_fake_leak", "source_type": "incident_report",
                 "text": leak_text}
    offenders = _find_leaking_chunks(chunks + [fake_chunk], real_leak_ngrams)
    assert "chunk_test_fake_leak" in offenders, (
        "a chunk containing a real (non-regulation-text, non-boilerplate) eval scenario "
        "sentence was NOT flagged -- the leakage test has gone blind"
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

