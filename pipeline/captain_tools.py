"""Captain tool registry skeleton (design_captain_missions.md Sec 16.3/9.1) -- a single,
honest inventory of every tool the Captain decision layer can actually select from today
(mirrors the exact tool strings `pipeline/captain_agent_spec.py`'s candidates_*() functions
produce), plus clearly-marked placeholder entries for future tool-use extensions (Sec
16.3's own deferred-not-rejected agentic tool-calling idea -- e.g. live regulation lookup,
weather routing, web search) that are NOT implemented yet.

Pure pipeline/ module, no app/ dependency, no GPU/API key needed, safe to run locally.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    """One entry in the registry -- `scope` distinguishes the already-wired brown-envelope
    decision layer from a future tool-use extension point; `status` is the honest
    implementation state (never "wired" for something no code path actually calls)."""
    name: str
    scope: str  # "decision_layer" (wired, used by captain_agent_spec.py) | "future" (not implemented)
    status: str  # "wired" | "not_implemented"
    description: str


# The 9 tool strings actually produced by captain_agent_spec.py's candidates_*() functions
# today (grep-verified against every `CaptainAction(tool=...)` call site in that file) --
# this list must stay in sync with that file if a new candidate tool is ever added there.
TOOL_REGISTRY: list[ToolSpec] = [
    ToolSpec("continue_at_capped_speed", "decision_layer", "wired",
            "Continue the voyage at the Chief-Engineer-declared capped speed (engine_failure)."),
    ToolSpec("request_place_of_refuge", "decision_layer", "wired",
            "Divert to a port of refuge for repairs (engine_failure, when one is reachable)."),
    ToolSpec("reduce_to_safe_speed", "decision_layer", "wired",
            "Reduce speed to the Rule 19 safe speed for restricted visibility (fog)."),
    ToolSpec("reduce_to_zone_speed_limit", "decision_layer", "wired",
            "Reduce speed to a charted zone's posted speed limit (whale_zone)."),
    ToolSpec("proceed_to_assist", "decision_layer", "wired",
            "Divert to render assistance to a vessel in distress (distress_call)."),
    ToolSpec("decline_with_logged_reason", "decision_layer", "wired",
            "Decline to assist a distress call, logging a valid safety-margin reason."),
    ToolSpec("comply_with_instruction", "decision_layer", "wired",
            "Comply with a company commercial instruction, e.g. a demanded speed increase."),
    ToolSpec("refuse_citing_ism_art5", "decision_layer", "wired",
            "Refuse a company instruction, citing the Master's ISM Code Art. 5 overriding authority."),
    ToolSpec("hold", "decision_layer", "wired",
            "No action -- maintain the current plan (the output schema's own safe fallback)."),
    # Sec 16.3's own deferred-not-rejected agentic tool-calling idea -- no inference loop or
    # training format exists yet for any of these; listed so the registry is a complete,
    # honest picture of where future tool use would plug in, not just what's wired today.
    ToolSpec("search_regulations", "future", "not_implemented",
            "Look up a specific ISM/SOLAS/STCW/MARPOL article on demand (agentic RAG)."),
    ToolSpec("weather_lookup", "future", "not_implemented",
            "Query a live weather/routing service (e.g. PredictWind, Sec 7) for conditions ahead."),
    ToolSpec("web_search", "future", "not_implemented",
            "General web search for information not covered by the Captain's own corpus."),
]
