"""VHF reranker prototype, step 1 -- mine (query, chunk, label) pairs for a cross-encoder
from VHF's own Track 2 (conversation) reasoning traces.

WHY THIS EXISTS / MINING STRATEGY
----------------------------------
VHF has no code-computed deterministic rule engine the way OOW's Leo MOOS states
(`active_encounter_rules`) do -- flagged as the reason VHF's reranker was skipped in the
first pass of this design (design_vhf_communications.md Sec 12.3/module docstring of
vhf_agents.py). The mining source used here is the same TIER of ground truth OOW's own
reranker already relies on for half its training pairs (`build_reranker_pairs.py`'s own
"Track2 SFT-synthetic scenarios... each trace.channels is a clean list of applicable
Rule-N strings" -- LLM-extracted-but-structured trace fields, not raw free text):

`vhf_conversation_traces.jsonl` (359 rows, `extract_conversation_reasoning.py` output)
carries a `trace.regulations` field -- the real COLREG rule(s) (e.g. "Rule 14", "Rule 34")
genuinely applicable to that collision-avoidance VHF exchange. A query built from
`trace.situation` should retrieve a chunk that actually DISCUSSES that rule over one that
doesn't -- positives are VHF RAG chunks whose own text contains the literal rule-citation
substring (these exist in two places in VHF's corpus: `light_sound_signals.json`'s Part D
rule-text chunks for Rules 32-37, and the case-based `vhf_training_examples.json` chunks,
many of which state "This follows Rule 14, Rule 34" as part of their answer); negatives
are chunks containing NONE of that row's cited rules.

Explicitly EXCLUDED: `Data/VHF/VHF_Eval/vhf_colreg_scenarios.json` (Track 2 held-out eval)
and anything derived from it -- never touched by this script.

OUTPUT
------
Data/VHF/VHF_Agents_Training/vhf_reranker_pairs.jsonl -- rows of
{query, chunk_id, text, label, rule, source, split}. `split` is held out by conversation
TRACE INDEX (every 5th trace -> dev) so dev never reuses a train query's own exact rows.

Safe to run locally -- deterministic (no LLM calls), CPU-only (one embedder-free substring
search over the already-built RAG chunk cache).

Run with: python -m pipeline.ingest.build_vhf_reranker_pairs
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path

from core import AgentPaths
from core.io import load_jsonl, safe_write_jsonl

paths = AgentPaths.vhf()
CACHE = paths.cache_dir
TRACES_FILE = CACHE / "vhf_conversation_traces.jsonl"
CHUNKS_FILE = CACHE / "vhf_rag_chunks.json"
OUT_FILE = CACHE / "vhf_reranker_pairs.jsonl"

MAX_POS_PER_RULE = 3   # cap how many chunks per cited rule become positives (avoid one
                       # heavily-repeated rule dominating the pair set)
MAX_NEG_PER_ROW = 3    # negatives sampled per trace row
DEV_EVERY_N = 5        # every 5th trace (by index) goes to dev, rest to train


def _rule_pattern(rule: str) -> re.Pattern:
    """Matches the literal rule citation as a whole phrase (e.g. "Rule 14") -- not a bare
    number, which would false-positive on channel numbers/distances/degrees elsewhere in
    the same chunk text."""
    return re.compile(re.escape(rule), re.IGNORECASE)


def build_rule_index(chunks: list[dict], rules: set[str]) -> dict[str, list[str]]:
    """rule -> list of chunk_ids whose text contains that literal rule citation."""
    patterns = {r: _rule_pattern(r) for r in rules}
    index: dict[str, list[str]] = {r: [] for r in rules}
    for c in chunks:
        text = c["text"]
        for r, pat in patterns.items():
            if pat.search(text):
                index[r].append(c["chunk_id"])
    return index


def main() -> None:
    random.seed(42)
    traces = list(load_jsonl(TRACES_FILE))
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    all_chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    cited_rules: set[str] = set()
    for t in traces:
        cited_rules.update(t.get("trace", {}).get("regulations") or [])
    rule_index = build_rule_index(chunks, cited_rules)
    print(f"{len(traces)} conversation traces, {len(cited_rules)} distinct cited rules, "
         f"{sum(1 for v in rule_index.values() if v)} rules with >=1 matching chunk")

    rows: list[dict] = []
    for i, t in enumerate(traces):
        trace = t.get("trace", {})
        query = (trace.get("situation") or "").strip()
        regs = trace.get("regulations") or []
        if not query or not regs:
            continue
        split = "dev" if (i % DEV_EVERY_N == 0) else "train"

        pos_ids: set[str] = set()
        for r in regs:
            cand = rule_index.get(r, [])
            pos_ids.update(cand[:MAX_POS_PER_RULE])
        if not pos_ids:
            continue
        for cid in pos_ids:
            rows.append({"query": query, "chunk_id": cid, "text": chunk_by_id[cid]["text"],
                        "label": 1, "rule": ", ".join(regs), "source": t.get("source_file", "?"),
                        "split": split})

        neg_pool = [cid for cid in all_chunk_ids if cid not in pos_ids
                   and not any(rule_index.get(r, []).count(cid) for r in regs)]
        neg_sample = random.sample(neg_pool, min(MAX_NEG_PER_ROW, len(neg_pool)))
        for cid in neg_sample:
            rows.append({"query": query, "chunk_id": cid, "text": chunk_by_id[cid]["text"],
                        "label": 0, "rule": ", ".join(regs), "source": t.get("source_file", "?"),
                        "split": split})

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_dev = sum(1 for r in rows if r["split"] == "dev")
    print(f"Writing {len(rows)} pairs (train={n_train}, dev={n_dev}) -> {OUT_FILE}")
    safe_write_jsonl(rows, OUT_FILE, overwrite=True)


if __name__ == "__main__":
    main()
