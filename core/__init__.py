"""Reusable, domain-agnostic building blocks for the Auto Pilot multi-agent system.

Each maritime agent (VHF, COLREG, Localization/State, Command, …) has its own
notebook and its own subfolder under ``Data/`` and ``_models/``. This ``core``
package holds the code that is *identical* between agents (path conventions,
ingestion, chunking, KG, metrics, training templates), so the per-agent
notebooks stay small and only carry the domain-specific bits (source
classification, concept vocabulary, gold set, system prompts).

See ``Docs/Multi-Agent_Maritime_Navigation_System_Architecture_Plan rev JS v4.docx``
for the full architecture.
"""

from core.paths import AgentPaths
from core.io import load_jsonl, load_env, load_messages_jsonl
from core.prose import clean, cap, decap, steps_sentence
from core.dry_run import add_dry_run_arg, write_stub_output
from core.embedding import (
    EMBEDDER_MODEL, QUERY_PREFIX,
    CONTAM_THRESH, DEDUP_THRESH, CANON_THRESH, MATCH_THRESH, ANCHOR_THRESH,
)

__all__ = [
    "AgentPaths",
    "load_jsonl", "load_env", "load_messages_jsonl",
    "clean", "cap", "decap", "steps_sentence",
    "add_dry_run_arg", "write_stub_output",
    "EMBEDDER_MODEL", "QUERY_PREFIX",
    "CONTAM_THRESH", "DEDUP_THRESH", "CANON_THRESH", "MATCH_THRESH", "ANCHOR_THRESH",
]
