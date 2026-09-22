"""Single source of truth for paths/files that must NEVER contribute a chunk to the OOW
RAG corpus (oow_rag_chunks.json) -- eval-set leakage, model-output artifacts, derived
training JSONL, and meta-literature. See the 2026-09-22 RAG-rebuild request for the full
per-category rationale.

Two independent enforcement points:
  1. `raise_if_excluded_source(path)` -- call this from EVERY ingest builder
     (build_oow_json.py, the incident pipeline, and any future CHIRP/Leo builder) before
     it reads a candidate source file, so an excluded file can never even become an
     OOW_JSON/*.json input to build_rag.py in the first place.
  2. `is_excluded_chunk(chunk)` -- used by tests/test_rag_exclusions.py against the FINAL
     oow_rag_chunks.json, matching each chunk's source_file/document_id -- a second,
     independent check in case some future code path bypasses (1).
"""
from __future__ import annotations
import fnmatch
from pathlib import Path

# Directory-level exclusions (relative to the repo workspace root) -- whole trees that
# must never be read as a RAG source. Matched via prefix/substring against a candidate
# file's path string (enforcement point 1) since these are directories, not filenames.
EXCLUDED_DIR_PREFIXES = [
    "Data/OOW/OOW_Eval",                          # held-out eval data -- retrieving it IS the answer sheet
    "Basic Simulator/Data/missions/_llm_runs",    # mission run logs, not source knowledge
    "Data/OOW/OOW_Literature_Review",             # meta-literature about BUILDING agents, not seamanship
]

# Filename-glob exclusions -- matched against just the BASENAME, since a chunk's
# "source_file"/"document_id" field is always a basename, never a full path (see
# build_rag.py's chunk schema). Model-output/eval artifacts and every *_derived_*
# training JSONL (redundant with the source it was generated from; a DPO pair's
# "rejected" half is a DELIBERATELY WRONG answer that must never be retrieved as if it
# were knowledge).
EXCLUDED_FILENAME_GLOBS = [
    "ablation_*", "eval_*", "probe_*",
    "ragas_judge_cache.jsonl", "consistency_findings*.json",
    "oow_sft_*.jsonl", "oow_*_sft_*.jsonl",
    "oow_*dpo_pairs*.jsonl",
    "oow_*reflection*.jsonl",
    "oow_*reasoning_traces*.jsonl",
    "oow_multihop.jsonl",
    "oow_scenario_Leo_checkpoint.jsonl",
]
# EXCEPTION (documented, not a silent gap): moos_temporal_narratives_final.jsonl in
# OOW_Scenarios_Leo is a raw SOURCE (deterministic MOOS trajectory data), not a derived
# training artifact -- it doesn't start with "oow_" so none of the globs above ever
# match it; see Phase 3 of the RAG-rebuild for how it's actually ingested.


class ExcludedSourceError(ValueError):
    """Raised by raise_if_excluded_source() -- an ingest builder tried to read a file
    that is on the RAG-corpus exclusion list."""


def is_excluded_path(path: str | Path) -> bool:
    p = str(path).replace("\\", "/")
    if any(p == prefix or p.startswith(prefix + "/") or f"/{prefix}/" in p for prefix in EXCLUDED_DIR_PREFIXES):
        return True
    name = Path(p).name
    return any(fnmatch.fnmatch(name, g) for g in EXCLUDED_FILENAME_GLOBS)


def raise_if_excluded_source(path: str | Path) -> None:
    if is_excluded_path(path):
        raise ExcludedSourceError(
            f"{path!s} is on the RAG-corpus exclusion list (pipeline/ingest/rag_exclusions.py) "
            "-- never ingest it"
        )


def is_excluded_chunk(chunk: dict) -> bool:
    for field in ("source_file", "document_id"):
        val = chunk.get(field)
        if val and is_excluded_path(val):
            return True
    return False
