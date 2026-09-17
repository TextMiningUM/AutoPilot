"""Standard directory layout for a domain-scoped agent in the Auto Pilot repo.

Convention (illustrated for ``domain="VHF"``, ``source_dirname="VHFProtocol"``)::

    workspace/
    ├── Data/
    │   └── VHF/                             ← data_root
    │       ├── VHFProtocol/                 ← source_dir     (training documents)
    │       ├── VHF_Eval/                    ← eval_dir       (held-out evaluation)
    │       │   └── vhf_gold_answers.json    ← gold_file
    │       ├── VHF_JSON/                    ← json_dir       (§ 8 output)
    │       └── VHF_Agents_Training/         ← cache_dir      (RAG/KG/traces/SFT/DPO)
    ├── _models/
    │   ├── hf_cache/                        ← hf_cache_dir   (SHARED, not domain-scoped)
    │   └── VHF/                             ← domain_models_dir  (fine-tuned artefacts)
    └── .env                                 ← env_file       (SHARED, not domain-scoped)

Every notebook and every script for a domain reads paths from an
``AgentPaths`` instance instead of hardcoding — one place to change if the
convention ever evolves, and impossible to have subtly diverging paths
between the notebook, the build scripts, and the training scripts.

For a NEW domain (say COLREG), create the paths object like this::

    paths = AgentPaths(domain="COLREG", source_dirname="COLREGRules")

For the existing VHF agent, use the alias that hardcodes the legacy source
folder name so nothing has to move on disk::

    paths = AgentPaths.vhf()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentPaths:
    """All standard folders for one domain-scoped agent.

    Parameters
    ----------
    domain
        Short domain name, used as the folder segment under ``Data/`` and
        ``_models/`` (e.g. ``"VHF"``, ``"COLREG"``).
    source_dirname
        Name of the training-sources subfolder inside ``Data/<domain>/``.
        VHF uses ``"VHFProtocol"`` for legacy reasons; new domains should
        pick a descriptive name (``"COLREGRules"``, ``"SEAMAPTiles"``, …).
    workspace
        Absolute path to the repository root. Defaults to
        ``Path(__file__).resolve().parent.parent`` — i.e. the parent of the
        ``core/`` package — which is correct when the repo is checked out
        as expected.
    """
    domain: str
    source_dirname: str
    workspace: Path

    # ── constructors ────────────────────────────────────────────────────
    def __init__(
        self,
        domain: str,
        source_dirname: str,
        workspace: Path | str | None = None,
    ) -> None:
        # frozen=True disables normal __setattr__; use object.__setattr__ instead
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "source_dirname", source_dirname)
        if workspace is None:
            workspace = Path(__file__).resolve().parent.parent
        object.__setattr__(self, "workspace", Path(workspace).resolve())

    @classmethod
    def vhf(cls, workspace: Path | str | None = None) -> "AgentPaths":
        """Paths object pre-configured for the existing VHF agent layout."""
        return cls(domain="VHF", source_dirname="VHFProtocol", workspace=workspace)

    @classmethod
    def oow(cls, workspace: Path | str | None = None) -> "AgentPaths":
        """Paths object pre-configured for the Officer of the Watch (OOW) agent layout."""
        return cls(domain="OOW", source_dirname="OOW_Protocols", workspace=workspace)

    @classmethod
    def from_env(cls, workspace: Path | str | None = None) -> "AgentPaths":
        """Paths object selected by the ``AUTOPILOT_DOMAIN`` env var (default ``"VHF"``).

        Lets every pipeline script stay agnostic of which domain it's building
        for: run the exact same ``python -m pipeline.xxx`` command with
        ``AUTOPILOT_DOMAIN=OOW`` in the environment (or ``env=...`` on a
        subprocess call from a notebook) instead of VHF's default, no CLI
        flag threading required. Mirrors the existing ``AUTOPILOT_MODELS_DIR``
        convention.
        """
        import os
        name = os.environ.get("AUTOPILOT_DOMAIN", "VHF").upper()
        factory = {"VHF": cls.vhf, "OOW": cls.oow}.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown AUTOPILOT_DOMAIN={name!r}; add an AgentPaths classmethod for it."
            )
        return factory(workspace=workspace)

    # ── data folders (per domain) ───────────────────────────────────────
    @property
    def data_root(self) -> Path:
        return self.workspace / "Data" / self.domain

    @property
    def source_dir(self) -> Path:
        """Training documents. Auto-discovered by the ingest pipeline."""
        return self.data_root / self.source_dirname

    @property
    def eval_dir(self) -> Path:
        """Held-out evaluation material. NEVER goes into training."""
        return self.data_root / f"{self.domain}_Eval"

    @property
    def json_dir(self) -> Path:
        """One hierarchical JSON per source document (§ 8 output)."""
        return self.data_root / f"{self.domain}_JSON"

    @property
    def incidents_dir(self) -> Path:
        """Raw accident/incident-investigation report PDFs (manually collected,
        not auto-created by mkdirs). Screened for relevance before any of
        their text enters the training pipeline -- see
        pipeline.ingest.screen_incidents."""
        return self.data_root / f"{self.domain}_Incidents"

    @property
    def cache_dir(self) -> Path:
        """RAG chunks, embeddings, KG, traces, SFT/DPO/reflection JSONL."""
        return self.data_root / f"{self.domain}_Agents_Training"

    @property
    def gold_file(self) -> Path:
        """Committed hand-authored gold Q&A file (in eval_dir)."""
        return self.eval_dir / f"{self.domain.lower()}_gold_answers.json"

    def eval_file(self, name: str) -> Path:
        """Any other held-out file in eval_dir, e.g. paths.eval_file("vhf_colreg_scenarios.json")."""
        return self.eval_dir / name

    # ── model folders ───────────────────────────────────────────────────
    @property
    def models_root(self) -> Path:
        """Shared model storage root.

        Honors ``AUTOPILOT_MODELS_DIR`` (set on the cloud pod so multiple
        users/clones share one copy of the multi-GB base weights instead of
        each downloading/merging their own) and falls back to the
        repo-local ``_models/`` folder otherwise (laptop use).
        """
        import os
        override = os.environ.get("AUTOPILOT_MODELS_DIR")
        return Path(override) if override else self.workspace / "_models"

    @property
    def hf_cache_dir(self) -> Path:
        """HuggingFace hub cache. SHARED across all domains, not per-agent."""
        return self.models_root / "hf_cache"

    @property
    def domain_models_dir(self) -> Path:
        """Per-domain fine-tuned artefacts (LoRA adapters, merged models)."""
        return self.models_root / self.domain

    # ── workspace-level ─────────────────────────────────────────────────
    @property
    def env_file(self) -> Path:
        """Shared ``.env`` (OPENAI_API_KEY etc.). Not domain-scoped."""
        return self.workspace / ".env"

    # ── utilities ───────────────────────────────────────────────────────
    def mkdirs(self) -> None:
        """Create every standard directory if missing. Idempotent."""
        for p in (
            self.data_root,
            self.source_dir,
            self.eval_dir,
            self.json_dir,
            self.cache_dir,
            self.models_root,
            self.hf_cache_dir,
            self.domain_models_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        """One-liner per standard path, useful for notebook printouts."""
        rows = [
            ("domain",             self.domain),
            ("workspace",          self.workspace),
            ("source_dir",         self.source_dir),
            ("eval_dir",           self.eval_dir),
            ("gold_file",          self.gold_file),
            ("json_dir",           self.json_dir),
            ("cache_dir",          self.cache_dir),
            ("hf_cache_dir",       self.hf_cache_dir),
            ("domain_models_dir",  self.domain_models_dir),
            ("env_file",           self.env_file),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)
