"""Captain memory skeleton (design_captain_missions.md Sec 16.1) -- the RAG/KG/PG retrieval
interface the real pipeline will implement once build_captain_json.py/build_rag.py exist
(Sec 16.6's own build order). Returns illustrative PLACEHOLDER hits only -- never real
retrieval -- so the call-site/UI wiring can be built and tested now, and swapped for the
real implementation later without changing the call site's own shape at all.

Pure pipeline/ module, no app/ dependency, no GPU/API key needed, safe to run locally.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved memory item -- the same shape a real RAG/KG/PG hit will have
    (source/text/score); `is_placeholder` is always True here so callers/UI never mistake
    this skeleton's illustrative content for real retrieval."""
    source: str
    text: str
    score: float
    is_placeholder: bool = True


# One illustrative placeholder per v1 brown-envelope event type (Sec 13.A.2) -- hand-written
# text in the STYLE real RAG/KG/PG content will eventually have (a regulatory citation + a
# precedent case), never pulled from any real source. Only engine_failure/distress_call/
# commercial_instruction have entries (the genuine judgement cases, Sec 13.A.4) -- fog/
# whale_zone are table-determined and don't need grounding to justify a decision.
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
    MemoryHit("(no corpus yet)",
             "No real RAG/KG/PG corpus exists yet -- see design_captain_missions.md Sec 16.6 "
             "(build_captain_json.py has not been written). This is illustrative placeholder "
             "content only, not retrieved from any real source.", 0.0),
]


def retrieve(query: str, k: int = 3) -> list[MemoryHit]:
    """Skeleton retrieval -- looks up a small hand-written placeholder set keyed by `query`
    (matched against known brown-envelope event-type strings), never a real embedding
    search. See the module docstring: this fixes the call-site SHAPE, not the content."""
    hits = _PLACEHOLDER_HITS.get(query, _GENERIC_PLACEHOLDER)
    return hits[:k]
