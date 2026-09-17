"""
================================================================================
build_pg.py — Procedural Graph over the reasoning traces (Lu et al. 2026 style)
================================================================================

Where the KG (build_kg.py) organizes *what-is* knowledge into
(entity, relation, entity) triplets, this builds a Procedural Graph (PG) of
*what-to-do* knowledge: (procedure, NEXT, procedure) triplets mined from the
ordered `procedures` lists in the reasoning traces, with the paper's three
edge attributes mapped from what the traces already contain:

    condition ← trace constraints        (when does this transition apply)
    guidance  ← the target step's "why"  (how/why to proceed)
    pitfalls  ← trace warnings           (what to avoid)

Fully deterministic (no LLM): step actions are canonicalized across traces by
embedding similarity (greedy clustering, cos >= CANON_THRESH after masking
vessel names), consecutive steps become NEXT edges, and identical transitions
from multiple documents merge with a support count — so a step order attested
by many sources outweighs a one-off.

Every trace is tagged with a procedure FAMILY (distress / urgency / safety /
dsc / colreg_encounter / routine_call / general) so paths can be extracted per
procedure type. This is Phase A of the PG plan; Phase B consumes the graph
(prompt-guidance ablation config, step-order training data, ProcOrder metric)
and Phase C adds online guidance + self-evolution once the MOOS loop provides
execution feedback.

Reads:   <cache>/<pfx>_reasoning_traces.jsonl (+ optional extra trace files)
Writes:  <cache>/<pfx>_pg.json

    python -X utf8 -m pipeline.ingest.build_pg                       # VHF default
    AUTOPILOT_DOMAIN=OOW python -X utf8 -m pipeline.ingest.build_pg  # OOW
    python -X utf8 -m pipeline.ingest.build_pg --traces-file a.jsonl b.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths
from core.io import load_jsonl

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

PG_FILE = CACHE / f"{_PFX}_pg.json"

CANON_THRESH = 0.80   # cos-sim above which two step actions are the same node
MAX_ATTR = 3          # attribute strings kept per edge (most frequent first)
MAX_SOURCES = 5

# vessel names/callsigns make identical steps look different — mask them.
# Name tokens must be Capitalized or ALLCAPS words; a lowercase word ends the name,
# so "MV Blue Horizon on channel 16" masks only "MV Blue Horizon".
_VESSEL_RE = re.compile(
    r"\b(?:MV|MY|SV|SS|FV|MS|USS)\s+(?:[A-Z][a-z']+|[A-Z']{2,})(?:[ -](?:[A-Z][a-z']+|[A-Z']{2,})){0,3}(?:'s)?")
_ALLCAPS_NAME_RE = re.compile(r"\b[A-Z][A-Z']{3,25}\b(?:,\s*\b[A-Z][A-Z']{3,25}\b)+")


def normalize_action(action: str) -> str:
    a = _VESSEL_RE.sub("the other vessel", action)
    a = _ALLCAPS_NAME_RE.sub("the other vessel", a)
    a = re.sub(r"\s+", " ", a).strip()
    return _detense(a)


# conversation traces log steps in past tense ("Initiated a call"), reasoning
# traces in imperative ("initiate a call") — normalize to imperative so the
# same step clusters together and downstream prose ("the next step is to …")
# stays grammatical.
_DETENSE_MAP = {
    "initiated": "initiate", "acknowledged": "acknowledge", "hailed": "hail",
    "altered": "alter", "sounded": "sound", "confirmed": "confirm",
    "responded": "respond", "requested": "request", "reduced": "reduce",
    "maintained": "maintain", "reported": "report", "transmitted": "transmit",
    "switched": "switch", "established": "establish", "agreed": "agree",
    "expressed": "express", "stated": "state", "announced": "announce",
    "notified": "notify", "monitored": "monitor", "contacted": "contact",
    "called": "call", "informed": "inform", "communicated": "communicate",
    "coordinated": "coordinate", "proposed": "propose", "suggested": "suggest",
    "declined": "decline", "accepted": "accept", "verified": "verify",
    "replied": "reply", "issued": "issue", "activated": "activate",
    "adjusted": "adjust", "increased": "increase", "slowed": "slow",
    "stopped": "stop", "waited": "wait", "listened": "listen",
    "checked": "check", "identified": "identify", "signaled": "signal",
    "signalled": "signal", "broadcasted": "broadcast", "broadcast": "broadcast",
    "asked": "ask", "answered": "answer", "repeated": "repeat",
}


def _detense(a: str) -> str:
    parts = a.split(" ", 1)
    first = parts[0].lower()
    if first in _DETENSE_MAP:
        rest = f" {parts[1]}" if len(parts) > 1 else ""
        return _DETENSE_MAP[first] + rest
    return a


FAMILY_RULES = [
    # (family, regex over situation+prowords+channels+regulations, first match wins)
    ("distress",        re.compile(r"\bMAYDAY\b", re.IGNORECASE)),
    ("urgency",         re.compile(r"\bPAN[ -]?PAN\b", re.IGNORECASE)),
    ("safety",          re.compile(r"\bSECURIT[EÉ]\b", re.IGNORECASE)),
    ("dsc",             re.compile(r"\bDSC\b|\bchannel 70\b|\bdigital selective\b", re.IGNORECASE)),
    ("colreg_encounter", re.compile(r"\bRule\s+\d+\b|\bgive[- ]way\b|\bstand[- ]on\b|\bcrossing\b|\bhead[- ]on\b|\bovertak", re.IGNORECASE)),
    ("routine_call",    re.compile(r"\bchannel\s+\d+\b|\bworking channel\b|\bhail\b", re.IGNORECASE)),
]


def classify_family(trace: dict) -> str:
    blob = " ".join([
        str(trace.get("situation") or ""),
        " ".join(trace.get("prowords_used") or []),
        " ".join(str(c) for c in (trace.get("channels") or [])),
        " ".join(trace.get("regulations") or []),
    ])
    # channels list like ["16","70"] won't literally say "channel 70"
    blob += " " + " ".join(f"channel {c}" for c in (trace.get("channels") or []))
    for fam, pat in FAMILY_RULES:
        if pat.search(blob):
            return fam
    return "general"


def collect_steps(trace_files: list[Path]) -> list[dict]:
    """One record per usable trace: family, ordered normalized steps, constraints, warnings, source."""
    out = []
    for tf in trace_files:
        if not tf.exists():
            print(f"  [skip] {tf.name} not found")
            continue
        n_used = 0
        for r in load_jsonl(tf):
            t = r.get("trace") or {}
            if r.get("skip") or r.get("error"):
                continue
            procs = t.get("procedures") or []
            steps = []
            for p in sorted(procs, key=lambda p: p.get("step", 0)):
                action = (p.get("action") or "").strip()
                if len(action) < 5:
                    continue
                steps.append({"action": normalize_action(action),
                              "why": (p.get("why") or "").strip()})
            if len(steps) < 2:
                continue
            out.append({
                "family": classify_family(t),
                "steps": steps,
                "constraints": [c for c in (t.get("constraints") or []) if c],
                "warnings": [w for w in (t.get("warnings") or []) if w],
                "source_file": r.get("source_file", tf.name),
            })
            n_used += 1
        print(f"  {tf.name}: {n_used} traces with >=2 ordered steps")
    return out


def canonicalize(actions: list[str], model) -> tuple[list[int], list[str]]:
    """Greedy embedding clustering: action index -> canonical node index.
    Node label = most frequent exact action string in the cluster. Steps whose
    channel numbers or port/starboard directions differ never merge."""
    embs = model.encode(actions, normalize_embeddings=True, batch_size=128,
                        show_progress_bar=False)
    guards = [_merge_guard(a) for a in actions]
    node_of: list[int] = [-1] * len(actions)
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    cluster_guard: list[tuple[frozenset, frozenset]] = []
    for i, e in enumerate(embs):
        assigned = False
        if centroids:
            sims = np.vstack(centroids) @ e
            for j in np.argsort(-sims)[:5]:
                if float(sims[j]) < CANON_THRESH:
                    break
                if cluster_guard[j] != guards[i]:
                    continue  # same wording, different channel/direction -> keep apart
                node_of[i] = int(j)
                members[j].append(i)
                centroids[j] = centroids[j] + (e - centroids[j]) / len(members[j])
                centroids[j] /= np.linalg.norm(centroids[j])
                assigned = True
                break
        if not assigned:
            node_of[i] = len(centroids)
            centroids.append(e.copy())
            members.append([i])
            cluster_guard.append(guards[i])
    labels = []
    for mem in members:
        counts = Counter(actions[i] for i in mem)
        labels.append(counts.most_common(1)[0][0])
    return node_of, labels


def top_strings(counter: Counter, n: int) -> list[str]:
    return [s for s, _ in counter.most_common(n)]


_GUARD_CHANNEL_RE = re.compile(r"\bchannel\s+(\d{1,2})\b|\bch\.?\s*(\d{1,2})\b", re.IGNORECASE)
_GUARD_DIRECTION_RE = re.compile(r"\b(port|starboard)\b", re.IGNORECASE)


def _merge_guard(action: str) -> tuple[frozenset, frozenset]:
    """Safety-critical tokens that must be IDENTICAL for two steps to merge:
    'switch to channel 72' vs 'switch to channel 16' and 'alter to port' vs
    'alter to starboard' embed as near-identical but are operationally opposite."""
    chans = frozenset(g1 or g2 for g1, g2 in _GUARD_CHANNEL_RE.findall(action))
    dirs = frozenset(d.lower() for d in _GUARD_DIRECTION_RE.findall(action))
    return chans, dirs


def build_pg(records: list[dict], model) -> dict:
    all_actions = [s["action"] for r in records for s in r["steps"]]
    print(f"Canonicalizing {len(all_actions)} step actions...")
    node_of, labels = canonicalize(all_actions, model)
    print(f"  -> {len(labels)} canonical procedure nodes")

    node_families: dict[int, Counter] = defaultdict(Counter)
    node_count: Counter = Counter()
    edge_support: Counter = Counter()
    edge_guidance: dict[tuple, Counter] = defaultdict(Counter)
    edge_condition: dict[tuple, Counter] = defaultdict(Counter)
    edge_pitfalls: dict[tuple, Counter] = defaultdict(Counter)
    edge_families: dict[tuple, Counter] = defaultdict(Counter)
    edge_sources: dict[tuple, Counter] = defaultdict(Counter)
    start_count: Counter = Counter()
    end_count: Counter = Counter()

    idx = 0
    for r in records:
        ids = []
        for s in r["steps"]:
            ids.append(node_of[idx])
            idx += 1
        for nid in ids:
            node_families[nid][r["family"]] += 1
            node_count[nid] += 1
        start_count[ids[0]] += 1
        end_count[ids[-1]] += 1
        for k in range(len(ids) - 1):
            u, v = ids[k], ids[k + 1]
            if u == v:
                continue  # canonicalization collapsed two consecutive steps
            key = (u, v)
            edge_support[key] += 1
            why = r["steps"][k + 1]["why"]
            if why:
                edge_guidance[key][why] += 1
            for c in r["constraints"]:
                edge_condition[key][c] += 1
            for w in r["warnings"]:
                edge_pitfalls[key][w] += 1
            edge_families[key][r["family"]] += 1
            edge_sources[key][r["source_file"]] += 1

    nodes = {}
    for nid, label in enumerate(labels):
        if node_count[nid] == 0:
            continue
        nodes[f"pg_{nid:04d}"] = {
            "label": label,
            "n_occurrences": node_count[nid],
            "families": dict(node_families[nid]),
            "n_starts": start_count.get(nid, 0),
            "n_ends": end_count.get(nid, 0),
        }

    edges = []
    for (u, v), sup in sorted(edge_support.items(), key=lambda kv: -kv[1]):
        edges.append({
            "u": f"pg_{u:04d}", "rel": "NEXT", "v": f"pg_{v:04d}",
            "support": sup,
            "condition": top_strings(edge_condition[(u, v)], MAX_ATTR),
            "guidance": top_strings(edge_guidance[(u, v)], MAX_ATTR),
            "pitfalls": top_strings(edge_pitfalls[(u, v)], MAX_ATTR),
            "families": dict(edge_families[(u, v)]),
            "sources": top_strings(edge_sources[(u, v)], MAX_SOURCES),
        })

    fam_counts = Counter(r["family"] for r in records)
    return {
        "nodes": nodes,
        "edges": edges,
        "params": {"canon_thresh": CANON_THRESH, "embedder": "all-MiniLM-L6-v2"},
        "stats": {
            "n_traces": len(records),
            "n_steps": len(all_actions),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "families": dict(fam_counts.most_common()),
        },
    }


def sample_path(pg: dict, family: str, max_len: int = 8) -> list[str]:
    """Greedy highest-support walk within one family, from its most-attested start node."""
    fam_edges = [e for e in pg["edges"] if family in e["families"]]
    if not fam_edges:
        return []
    starts = Counter()
    for nid, n in pg["nodes"].items():
        if family in n["families"] and n["n_starts"] > 0:
            starts[nid] = n["n_starts"] * n["families"][family]
    if not starts:
        return []
    cur = starts.most_common(1)[0][0]
    path, seen = [cur], {cur}
    for _ in range(max_len - 1):
        nxt = [e for e in fam_edges if e["u"] == cur and e["v"] not in seen]
        if not nxt:
            break
        best = max(nxt, key=lambda e: e["support"])
        cur = best["v"]
        path.append(cur)
        seen.add(cur)
    return path


def main() -> None:
    default_traces = [CACHE / f"{_PFX}_reasoning_traces.jsonl"]
    for extra in (f"{_PFX}_conversation_traces.jsonl", f"{_PFX}_incident_reasoning_traces.jsonl"):
        if (CACHE / extra).exists():
            default_traces.append(CACHE / extra)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--traces-file", type=Path, nargs="+", default=default_traces,
                    help="trace JSONL files to mine (default: all trace files in cache_dir)")
    args = ap.parse_args()

    print(f"Building Procedural Graph for domain {paths.domain}")
    records = collect_steps(args.traces_file)
    if not records:
        raise SystemExit("No traces with >=2 ordered procedure steps found.")

    print("Loading embedder...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pg = build_pg(records, model)

    PG_FILE.write_text(json.dumps(pg, ensure_ascii=False, indent=1), encoding="utf-8")
    s = pg["stats"]
    print(f"\nSaved {PG_FILE}")
    print(f"  traces={s['n_traces']}  steps={s['n_steps']}  nodes={s['n_nodes']}  edges={s['n_edges']}")
    print(f"  families: {s['families']}")

    print("\nTop 8 transitions by support:")
    for e in pg["edges"][:8]:
        u, v = pg["nodes"][e["u"]]["label"], pg["nodes"][e["v"]]["label"]
        print(f"  [{e['support']:>3}x] {u[:60]}  ->  {v[:60]}")

    print("\nSample greedy path per family:")
    for fam in s["families"]:
        p = sample_path(pg, fam)
        if p:
            print(f"  {fam}:")
            for nid in p:
                print(f"    -> {pg['nodes'][nid]['label'][:90]}")


if __name__ == "__main__":
    main()
