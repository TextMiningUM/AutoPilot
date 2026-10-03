"""Captain memory (design_captain_missions.md Sec 16.1) -- the RAG/KG/PG retrieval
interface, keyed by brown-envelope event type (the only "query" shape the call site
(Basic Simulator/app/pages/1_Captain_Mission.py) ever passes, since the Captain's decision
layer is organised around the 5 v1 event types, not free-text questions).

Real retrieval (2026-10-03): looks up `captain_reasoning_traces.jsonl` (built by
pipeline/track1/extract_captain_reasoning.py, Sec 16.6 step 3) for traces whose own
mapped_event_type matches the requested event type -- deterministic lookup, no embedding
model needed, since the "query" is always one of the 5 known event-type strings rather
than free text. Falls back to the original hand-written placeholder content (unchanged)
whenever the traces file is missing (a clean checkout before the corpus is built) or the
event type genuinely has no real hits (whale_zone: 0 chunks map to it in the current
corpus -- it is table-determined, not a judgement case, so this is expected, not a bug).

Pure pipeline/ module, no app/ dependency, no GPU/API key needed (CPU-only JSON I/O), safe
to run locally.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from functools import lru_cache

from core import AgentPaths


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved memory item -- the same shape a real RAG/KG/PG hit will have
    (source/text/score); `is_placeholder` distinguishes illustrative fallback content from
    a real, corpus-grounded hit so callers/UI never mistake one for the other."""
    source: str
    text: str
    score: float
    is_placeholder: bool = True


# One illustrative placeholder per v1 brown-envelope event type (Sec 13.A.2) -- hand-written
# text in the STYLE real RAG/KG/PG content will eventually have (a regulatory citation + a
# precedent case), never pulled from any real source. Only engine_failure/distress_call/
# commercial_instruction have entries (the genuine judgement cases, Sec 13.A.4) -- fog/
# whale_zone are table-determined and don't need grounding to justify a decision. Still used
# as the fallback whenever the real corpus has no hits for the requested event type.
_PLACEHOLDER_HITS: dict[str, list[MemoryHit]] = {
    "engine_failure": [
        MemoryHit("ISM Code Art. 7 (placeholder)",
                 "placeholder: shipboard operations -- company procedures for reporting and "
                 "responding to machinery failure.", 0.91),
        MemoryHit("CHIRP MFB (placeholder)",
                 "placeholder: a prior near-miss where continuing at a capped speed avoided a "
                 "missed deadline without breaching the engine limit.", 0.78),
    ],
    "distress_call": [
        MemoryHit("SOLAS Ch. V Reg. 33 (placeholder)",
                 "placeholder: duty to render assistance to persons in distress at sea, subject "
                 "to not seriously endangering one's own ship.", 0.93),
    ],
    "commercial_instruction": [
        MemoryHit("ISM Code Art. 5 (placeholder)",
                 "placeholder: the Master's overriding authority and responsibility to make "
                 "decisions on safety and pollution prevention.", 0.95),
    ],
}
_GENERIC_PLACEHOLDER = [
    MemoryHit("(no corpus hits for this query)",
             "No real RAG/KG/PG hits exist for this specific event type/query -- either the "
             "corpus doesn't cover it yet (see design_captain_missions.md Sec 16.6), or this is "
             "a table-determined event type (e.g. whale_zone) that was never expected to need "
             "grounding. This is illustrative placeholder content only, not retrieved from any "
             "real source.", 0.0),
]

# Rough salience proxy for ranking real hits (no free-text query to score dense similarity
# against) -- a more severe real precedent is more instructive grounding than a minor one.
_SEVERITY_SCORE = {"catastrophic": 0.95, "serious": 0.88, "moderate": 0.80, "minor": 0.72}
_DEFAULT_REAL_SCORE = 0.75


@lru_cache(maxsize=1)
def _load_real_hits_by_event_type() -> dict[str, list[MemoryHit]]:
    """One-time load of captain_reasoning_traces.jsonl, grouped by mapped_event_type.
    Returns {} (never raises) if the file doesn't exist yet -- the "no corpus yet" case."""
    traces_file = AgentPaths.captain().cache_dir / "captain_reasoning_traces.jsonl"
    if not traces_file.exists():
        return {}
    by_type: dict[str, list[tuple[float, MemoryHit]]] = {}
    with traces_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            trace = row.get("trace")
            if not trace:
                continue
            event_type = trace.get("mapped_event_type")
            if not event_type or event_type == "unmapped":
                continue
            citation = trace.get("channels") or trace.get("regulations") or []
            source = (f"{citation[0]} ({row['source_file']})" if citation and citation[0] != row["source_file"]
                      else row["source_file"])
            key_fact = (trace.get("key_facts") or [""])[0]
            text = trace.get("situation", "")
            if key_fact:
                text = f"{text} {key_fact}"
            score = _SEVERITY_SCORE.get(trace.get("severity"), _DEFAULT_REAL_SCORE)
            hit = MemoryHit(source, text.strip(), score, is_placeholder=False)
            by_type.setdefault(event_type, []).append((score, hit))
    # Highest-severity first; ties keep original (document) order (stable sort).
    return {et: [hit for _, hit in sorted(hits, key=lambda sh: -sh[0])]
           for et, hits in by_type.items()}


def retrieve(query: str, k: int = 3) -> list[MemoryHit]:
    """Returns up to `k` real, corpus-grounded MemoryHits for a known brown-envelope event
    type (`query`), ranked by severity; falls back to illustrative placeholder content when
    the real corpus has no hits for this event type (or doesn't exist yet)."""
    real_hits = _load_real_hits_by_event_type().get(query)
    if real_hits:
        return real_hits[:k]
    hits = _PLACEHOLDER_HITS.get(query, _GENERIC_PLACEHOLDER)
    return hits[:k]
