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

__all__ = ["AgentPaths"]
