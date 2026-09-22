"""OOW reranker prototype, step 1 — mine (query, chunk, label) pairs for a
cross-encoder from ALREADY-EXISTING deterministic rule-number ground truth.

WHY THIS EXISTS
----------------
RAG currently retrieves with dense + KG-concept-boost only (`kg_retrieve` in
build_kg.py) -- no reranking stage. Two sources carry per-contact COLREG rule
numbers that were computed by deterministic code (a simulator's rule engine
or `classify_encounter()`), never guessed by an LLM, so they make trustworthy
supervision for a small COLREG-domain cross-encoder reranker:

  - Leo's raw MOOS states (`Data/OOW/OOW_Scenarios_Leo/
    moos_temporal_narratives_final.jsonl`, 7928 records): each `state.contacts[i]`
    carries `standing_rules` (always-applicable, non-discriminative -- 2/5/6/7/11
    fire on ~100% of records) and `active_encounter_rules` (encounter-specific:
    8/13/14/15/16/17 -- the discriminative signal used here).
  - Track2 SFT-synthetic scenarios, already disjoint from the Track2 eval file
    by construction (`Data/OOW/OOW_Agents_Training/oow_scenario_reasoning_traces.jsonl`,
    390 records): each `trace.channels` is a clean list of the applicable
    Rule-N strings for that scenario.

Explicitly EXCLUDED: `Data/OOW/OOW_Eval/oow_colreg_scenarios.json` (Track 2
held-out eval) and anything derived from it -- never touched by this script.

DEDUPE (per user direction 2026-09-22)
----------------------------------------
Leo's narrative states are highly autocorrelated -- many near-duplicate
consecutive cycles per simulated run (`state["source_file"]`, one of 402
distinct runs). Keeping every record would let a handful of runs dominate
training. Instead: ONE state per (run, dominant discriminative rule) --
first record in which each rule first appears for that run.

Track2 SFT-synthetic scenarios don't have this autocorrelation problem (each
record is already an independently-parameterized instance), so no dedup is
applied there beyond using every record once.

OUTPUT
------
Data/OOW/OOW_Agents_Training/oow_reranker_pairs.jsonl -- rows of
{query, chunk_id, text, label, rule, source, split}. `split` is held out by
RUN (Leo) / by within-category index (Track2) so dev never leaks a
near-duplicate of a train query.
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path

from core import AgentPaths, load_jsonl

paths = AgentPaths.from_env()
_PFX = paths.domain.lower()

CHUNKS_FILE = paths.cache_dir / f"{_PFX}_rag_chunks.json"
LEO_FILE    = paths.data_root / "OOW_Scenarios_Leo" / "moos_temporal_narratives_final.jsonl"
TRACK2_FILE = paths.cache_dir / "oow_scenario_reasoning_traces.jsonl"
OUT_FILE    = paths.cache_dir / "oow_reranker_pairs.jsonl"

# Encounter-specific rules with real discriminative signal (vary by encounter type).
# Excludes the "standing" rules (2/5/6/7/11) that fire on virtually every record and so
# carry no information about WHICH excerpt is relevant.
DISCRIMINATIVE_RULES = {8, 13, 14, 15, 16, 17}
# Priority order for picking ONE "dominant" rule out of a state's rule set (most
# encounter-specific first, the general action-to-avoid-collision rule last).
RULE_PRIORITY = [14, 17, 16, 15, 13, 8]

_RULE_RE = re.compile(r"^Rule\s+(\d+)\b")


def load_rule_chunks() -> dict[int, list[dict]]:
    """{rule_number: [{"chunk_id", "text", "document_id"}, ...]} from the RAG chunks
    whose (standalone, one-section-per-chunk) section title starts with "Rule N"."""
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    out: dict[int, list[dict]] = defaultdict(list)
    for c in chunks:
        titles = c.get("section_titles") or []
        if len(titles) != 1:
            continue
        m = _RULE_RE.match(titles[0])
        if not m:
            continue
        out[int(m.group(1))].append({
            "chunk_id": c["chunk_id"], "text": c["text"], "document_id": c["document_id"],
        })
    return out


def dominant_rule(rules: set[int]) -> int | None:
    cand = rules & DISCRIMINATIVE_RULES
    for r in RULE_PRIORITY:
        if r in cand:
            return r
    return None


def _sample_negative_rules(exclude: set[int], n: int, seed: int) -> list[int]:
    pool = sorted(DISCRIMINATIVE_RULES - exclude)
    if not pool:
        return []
    # deterministic round-robin pick instead of `random` -- keeps the file reproducible
    # byte-for-byte across reruns without needing to thread a seeded RNG through
    return [pool[(seed + i) % len(pool)] for i in range(min(n, len(pool)))]


def _emit(rows: list[dict], query: str, rule_chunks: dict[int, list[dict]],
          pos_rules: set[int], neg_rules: list[int], source: str, split: str) -> None:
    for r in sorted(pos_rules):
        for c in rule_chunks.get(r, []):
            rows.append({"query": query, "chunk_id": c["chunk_id"], "text": c["text"],
                         "label": 1, "rule": r, "source": source, "split": split})
    for r in neg_rules:
        # prefer the consolidated (official-text) variant so pos/neg counts stay balanced
        # (each positive rule contributes ~2 rows -- consolidated + simple_colreg)
        cands = rule_chunks.get(r, [])
        c = next((x for x in cands if x["document_id"] == "colreg_consolidated_2018"), None) \
            or (cands[0] if cands else None)
        if c is not None:
            rows.append({"query": query, "chunk_id": c["chunk_id"], "text": c["text"],
                         "label": 0, "rule": r, "source": source, "split": split})


def mine_leo(rule_chunks: dict[int, list[dict]]) -> list[dict]:
    if not LEO_FILE.exists():
        print(f"  [leo] {LEO_FILE} not found -- skipping", flush=True)
        return []
    seen_keys: set[tuple[str, int]] = set()
    runs = sorted({rec.get("source_file", "") for rec in load_jsonl(LEO_FILE)})
    dev_runs = set(runs[::7])  # ~1 in 7 runs held out for dev, deterministic

    rows: list[dict] = []
    for i, rec in enumerate(load_jsonl(LEO_FILE)):
        run = rec.get("source_file", "")
        state = rec.get("state") or {}
        contacts = state.get("contacts") or []
        rules: set[int] = set()
        for ct in contacts:
            rules.update(ct.get("active_encounter_rules") or [])
        dom = dominant_rule(rules)
        if dom is None:
            continue
        key = (run, dom)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        narrative = rec.get("narrative") or ""
        if not narrative:
            continue
        neg_rules = _sample_negative_rules(rules & DISCRIMINATIVE_RULES, n=2, seed=i)
        split = "dev" if run in dev_runs else "train"
        _emit(rows, narrative, rule_chunks, {dom}, neg_rules, source="leo", split=split)
    return rows


def mine_track2(rule_chunks: dict[int, list[dict]]) -> list[dict]:
    if not TRACK2_FILE.exists():
        print(f"  [track2] {TRACK2_FILE} not found -- skipping", flush=True)
        return []
    by_category: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    for rec in load_jsonl(TRACK2_FILE):
        sf = rec.get("source_file", "")
        category = sf.split("::")[-1] if "::" in sf else sf
        idx = by_category[category]
        by_category[category] += 1

        trace = rec.get("trace") or {}
        situation = trace.get("situation") or ""
        channels = trace.get("channels") or []
        rules = {int(m.group(1)) for m in (_RULE_RE.match(ch) for ch in channels) if m}
        pos_rules = rules & DISCRIMINATIVE_RULES
        if not pos_rules or not situation:
            continue
        neg_rules = _sample_negative_rules(pos_rules, n=2, seed=idx)
        split = "dev" if idx % 7 == 0 else "train"  # matches Leo's ~1-in-7 dev holdout
        _emit(rows, situation, rule_chunks, pos_rules, neg_rules, source="track2_sft", split=split)
    return rows


def main() -> None:
    rule_chunks = load_rule_chunks()
    print(f"Rule -> chunk map: {sorted(rule_chunks)} "
          f"({sum(len(v) for v in rule_chunks.values())} chunks total)", flush=True)

    leo_rows = mine_leo(rule_chunks)
    print(f"Leo: {len(leo_rows)} rows", flush=True)
    t2_rows = mine_track2(rule_chunks)
    print(f"Track2 SFT-synthetic: {len(t2_rows)} rows", flush=True)

    rows = leo_rows + t2_rows
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_train = sum(1 for r in rows if r["split"] == "train")
    print(f"Total: {len(rows)} rows  (pos={n_pos} neg={len(rows) - n_pos}, "
          f"train={n_train} dev={len(rows) - n_train})", flush=True)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
