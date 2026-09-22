"""
================================================================================
pg_guidance.py — turn the Procedural Graph into step-level guidance (Phases B+C)
================================================================================

Shared by:
- prep_ablation.py  (Phase B): the v4_pg config prepends a per-question
  "Procedure guidance" block rendered from the graph — the offline analog of
  the paper's guidance model, deterministic so prompt prep stays LLM-free.
- ragas_metrics.proc_order (Phase B): node matching + reachability.
- evolve_pg.py / the future MOOS loop (Phase C): localize() + neighborhood()
  implement the paper's online Match + N_h(u_t) mechanism.

The graph is loaded once per process; node-label embeddings are cached on
disk next to the pg file (invalidated by node count mismatch).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

# question/answer text → family, mirroring build_pg.FAMILY_RULES
from pipeline.ingest.build_pg import FAMILY_RULES
from core import QUERY_PREFIX, MATCH_THRESH, ANCHOR_THRESH

MAX_HOPS = 3          # reachability horizon for ordering checks / neighborhoods


def load_merged_pg(files: list[Path], embedder) -> "ProceduralGraph":
    """Load and merge multiple already-built PG JSON files into ONE ProceduralGraph --
    e.g. app/agents.py's v8_super_cot_pg/v9_super_all configs, which want guidance drawn
    from BOTH oow_pg_scenario.json and oow_pg_incident.json together. Every per-source PG
    file is built independently with the same "pg_0000", "pg_0001", ... id scheme, so
    node ids WOULD collide on a naive concatenation -- each source's node ids are
    namespaced by its index in `files` (f"{i}:{orig_id}") and every edge's u/v is remapped
    to match, before nodes/edges are merged into one dict. Uses the dict-source
    constructor path (no on-disk embedding cache) -- these per-source graphs are small
    enough that recomputing embeddings each load is cheap."""
    merged_nodes: dict = {}
    merged_edges: list = []
    for i, p in enumerate(files):
        g = json.loads(p.read_text(encoding="utf-8"))
        remap = {nid: f"{i}:{nid}" for nid in g["nodes"]}
        for nid, node in g["nodes"].items():
            merged_nodes[remap[nid]] = node
        for e in g["edges"]:
            merged_edges.append({**e, "u": remap.get(e["u"], e["u"]), "v": remap.get(e["v"], e["v"])})
    return ProceduralGraph({"nodes": merged_nodes, "edges": merged_edges}, embedder)


class ProceduralGraph:
    """In-memory PG with embedding node-matching and hop-limited reachability.

    `source` may be a Path to a pg JSON (embeddings cached on disk next to it)
    or an already-loaded pg dict (embeddings computed in memory — used by the
    self-evolution loop for candidate graphs)."""

    def __init__(self, source: Path | dict, embedder):
        if isinstance(source, dict):
            self.pg = source
            emb_file = None
        else:
            self.pg = json.loads(source.read_text(encoding="utf-8"))
            emb_file = source.with_suffix(".emb.npy")
        self.embedder = embedder
        self.node_ids = list(self.pg["nodes"].keys())
        self.labels = [self.pg["nodes"][n]["label"] for n in self.node_ids]
        self.out_edges: dict[str, list[dict]] = defaultdict(list)
        self.in_edges: dict[str, list[dict]] = defaultdict(list)
        for e in self.pg["edges"]:
            self.out_edges[e["u"]].append(e)
            self.in_edges[e["v"]].append(e)

        emb = None
        if emb_file is not None and emb_file.exists():
            emb = np.load(emb_file)
            expected_dim = embedder.get_sentence_embedding_dimension()
            if emb.shape != (len(self.labels), expected_dim):
                emb = None
        if emb is None:
            emb = embedder.encode(self.labels, normalize_embeddings=True,
                                  batch_size=128, show_progress_bar=False)
            if emb_file is not None:
                np.save(emb_file, emb)
        self.node_emb = emb

    # ── matching / localization (paper's Match step) ─────────────────
    def match(self, text: str, thresh: float = MATCH_THRESH) -> tuple[str | None, float]:
        """Best-matching node id for a free-text step/action, or None."""
        e = self.embedder.encode([QUERY_PREFIX + text], normalize_embeddings=True,
                                 show_progress_bar=False)[0]
        sims = self.node_emb @ e
        i = int(np.argmax(sims))
        return (self.node_ids[i], float(sims[i])) if sims[i] >= thresh else (None, float(sims[i]))

    def match_many(self, texts: list[str], thresh: float = MATCH_THRESH) -> list[str | None]:
        if not texts:
            return []
        embs = self.embedder.encode([QUERY_PREFIX + t for t in texts], normalize_embeddings=True,
                                    batch_size=64, show_progress_bar=False)
        sims = embs @ self.node_emb.T
        out = []
        for row in sims:
            i = int(np.argmax(row))
            out.append(self.node_ids[i] if row[i] >= thresh else None)
        return out

    localize = match  # paper terminology alias for the online loop

    # ── reachability / neighborhood ──────────────────────────────────
    def reachable(self, u: str, v: str, max_hops: int = MAX_HOPS,
                  family: str | None = None) -> bool:
        """Is there a NEXT-path u -> ... -> v within max_hops (optionally family-scoped)?"""
        if u == v:
            return True
        frontier, seen = deque([(u, 0)]), {u}
        while frontier:
            cur, d = frontier.popleft()
            if d >= max_hops:
                continue
            for e in self.out_edges.get(cur, []):
                if family and family not in e["families"]:
                    continue
                if e["v"] == v:
                    return True
                if e["v"] not in seen:
                    seen.add(e["v"])
                    frontier.append((e["v"], d + 1))
        return False

    def neighborhood(self, u: str, hops: int = 2, family: str | None = None) -> list[dict]:
        """Outgoing edges reachable from u within `hops` (paper's N_h(u_t))."""
        out, frontier, seen = [], deque([(u, 0)]), {u}
        while frontier:
            cur, d = frontier.popleft()
            if d >= hops:
                continue
            for e in self.out_edges.get(cur, []):
                if family and family not in e["families"]:
                    continue
                out.append(e)
                if e["v"] not in seen:
                    seen.add(e["v"])
                    frontier.append((e["v"], d + 1))
        return out

    def label(self, node_id: str) -> str:
        return self.pg["nodes"][node_id]["label"]


def classify_text_family(text: str) -> str:
    for fam, pat in FAMILY_RULES:
        if pat.search(text):
            return fam
    return "general"


_STEP_HINT_RE = re.compile(
    r"\b(first|then|next|after|before|finally|step|proceed|switch|transmit|hail|alter|sound|call|tune|press|select|send|report|monitor|acknowledge|confirm)\b",
    re.IGNORECASE)


def split_step_sentences(answer: str) -> list[str]:
    """Sentences of an answer that look like procedure steps (order-bearing)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 8]
    return [s for s in sents if _STEP_HINT_RE.search(s)]


# ── guidance rendering (offline analog of the paper's guidance LLM) ───────
def render_guidance(question: str, graph: ProceduralGraph,
                    max_steps: int = 6, max_pitfalls: int = 3) -> str:
    """Compact 'Procedure guidance' block for one question: the most-supported
    family-scoped step sequence anchored at the question's best-matching node,
    plus the strongest conditions and pitfalls along that path."""
    family = classify_text_family(question)
    anchor, _sim = graph.match(question, thresh=ANCHOR_THRESH)  # anchor is best-effort
    fam = family if family != "general" else None

    # walk: from anchor (or the family's most-attested start) along highest support
    def start_node() -> str | None:
        if anchor is not None:
            return anchor
        cands = [(n["n_starts"] * n["families"].get(family, 0), nid)
                 for nid, n in graph.pg["nodes"].items()]
        cands = [c for c in cands if c[0] > 0]
        return max(cands)[1] if cands else None

    cur = start_node()
    if cur is None:
        return ""
    path, seen = [cur], {cur}
    conditions: list[str] = []
    pitfalls: list[str] = []
    for _ in range(max_steps - 1):
        nxt = [e for e in graph.out_edges.get(cur, [])
               if e["v"] not in seen and (fam is None or fam in e["families"])]
        if not nxt:
            break
        best = max(nxt, key=lambda e: e["support"])
        conditions += best.get("condition", [])[:1]
        pitfalls += best.get("pitfalls", [])[:1]
        cur = best["v"]
        path.append(cur)
        seen.add(cur)

    if len(path) < 2:
        return ""
    lines = [f"Procedure guidance (typical {family.replace('_', ' ')} sequence from the procedure graph):"]
    for i, nid in enumerate(path, 1):
        lines.append(f"  {i}. {graph.label(nid)}")
    conds = list(dict.fromkeys(conditions))[:2]
    if conds:
        lines.append("Applies when: " + "; ".join(conds))
    pits = list(dict.fromkeys(pitfalls))[:max_pitfalls]
    if pits:
        lines.append("Avoid: " + "; ".join(pits))
    lines.append("Follow this order where applicable, but adapt to the specific situation.")
    return "\n".join(lines)
