"""Leo's raw MOOS trajectory narratives -> deduplicated, canonical case-based RAG
chunks (Phase 3 of the OOW RAG rebuild, 2026-09-22).

Source: Data/OOW/OOW_Scenarios_Leo/moos_temporal_narratives_final.jsonl (7928 frames,
402 distinct scenario runs -- see build_reranker_pairs.py's docstring for the same
"402 distinct runs" fact, confirmed independently here). Each frame already carries a
deterministic, template-written "narrative" field plus per-contact
active_encounter_rules -- this is data-derived, not LLM-synthesized, so it's allowed as
a RAG source under the project's "knowledge from data, not from prompts" principle.

STRUCTURE (Phase 3a findings, from direct inspection -- see also
_tmp_inspect_leo.py/_tmp_inspect_leo2.py/_tmp_inspect_leo3.py, one-off, not committed):
  - 7928 frames across 402 runs (source_file), 1-50 frames/run, median 18.
  - 1551/7928 frames have ZERO contacts -- excluded here (no rule to learn from).
  - Of the remaining, only 3062/7928 (39%) have at least one contact with a real
    encounter (active_encounter_rules non-empty AND valid cpa_distance_m/tcpa_s) --
    the rest are "diverging"/"stationary_contact"/"parallel" contacts with no live
    collision-avoidance content, also excluded.
  - Per-contact CPA (m) quartiles (dominant contact only): 94 / 174 / 251 (min 0, max
    699). TCPA (s) quartiles: 45 / 96 / 203 (min 0.1, max 889).

WHY DEDUPLICATION IS NEEDED: consecutive cycles of the SAME run barely change (a vessel
closing at a few m/s moves only a little between cycles), so naive inclusion would let a
handful of runs dominate retrieval with dozens of near-identical frames. Canonicalizing
to ONE representative frame per (encounter_type, 15deg bearing bucket, CPA bucket, TCPA
bucket, own_role, rule-set) bucket collapses this to a few hundred genuinely distinct
situations.

Run with: python -m pipeline.ingest.build_moos_case_rag
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from core import AgentPaths
from pipeline.ingest.build_oow_json import tag_text
from pipeline.ingest.build_reranker_pairs import RULE_PRIORITY
from pipeline.ingest.rag_exclusions import raise_if_excluded_source

paths = AgentPaths.from_env()
LEO_FILE = paths.data_root / "OOW_Scenarios_Leo" / "moos_temporal_narratives_final.jsonl"
JSON_OUT_DIR = paths.json_dir
OUT_FILE = JSON_OUT_DIR / "leo_moos_cases.json"
MISSIONS_DIR = paths.workspace / "Basic Simulator" / "Data" / "missions"
EVAL_SCENARIOS_FILE = paths.eval_dir / "oow_colreg_scenarios_v1.json"
NM_TO_M = 1852.0
KN_TO_MPS = NM_TO_M / 3600.0

BEARING_BUCKET_DEG = 15.0
# Data-driven: the ACTUAL quartiles of the dominant-contact CPA/TCPA distribution (see
# docstring's Phase-3a stats) -- 4 buckets each (roughly one per quartile) rather than a
# finer grid, since combined with 24 bearing buckets x ~7 encounter types x ~4 roles the
# key space is already large; using exactly the observed quartiles keeps each bucket
# similarly POPULATED (by construction) instead of arbitrarily fine-grained.
CPA_BUCKET_EDGES_M = [94.0, 174.0, 251.0]
TCPA_BUCKET_EDGES_S = [45.0, 96.0, 203.0]


def stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.md5(("|".join(parts)).encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def _bucket(value: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def _geom_bucket(bearing_deg: float, cpa_m: float, tcpa_s: float) -> tuple:
    """Geometry-only signature (bearing/CPA/TCPA) -- deliberately coarser than
    canonicalize()'s full bucket key (no encounter_type/own_role/rule-set), since this
    is used ONLY for the cross-source overlap/leakage check below, comparing against
    two other schemas (Basic Simulator missions, oow_colreg_scenarios.json) that don't
    share Leo's encounter-type vocabulary. `% 360.0` normalizes either a signed
    (-180..180) or unsigned (0..360) bearing convention onto the same scale."""
    return (
        _bucket(bearing_deg % 360.0, [BEARING_BUCKET_DEG * i for i in range(1, 24)]),
        _bucket(cpa_m, CPA_BUCKET_EDGES_M),
        _bucket(tcpa_s, TCPA_BUCKET_EDGES_S),
    )


def _cpa_tcpa(ox, oy, ohdg, ospd, tx, ty, thdg, tspd) -> tuple[float, float]:
    """Standalone reimplementation of Basic Simulator/app/narrate.py's cpa_tcpa() --
    kept separate (not imported) so pipeline/ never depends on the Basic Simulator app
    package; same formula, verified against it via the shared test in
    tests/test_rag_exclusions.py."""
    oh, th = math.radians(ohdg), math.radians(thdg)
    vox, voy = ospd * math.sin(oh), ospd * math.cos(oh)
    vtx, vty = tspd * math.sin(th), tspd * math.cos(th)
    dx, dy = tx - ox, ty - oy
    dvx, dvy = vtx - vox, vty - voy
    rel_sq = dvx ** 2 + dvy ** 2
    if rel_sq < 1e-6:
        return math.hypot(dx, dy), 0.0
    t = max(0.0, -(dx * dvx + dy * dvy) / rel_sq)
    return math.hypot(dx + dvx * t, dy + dvy * t), t


def _bearing_deg(ox, oy, ohdg, tx, ty) -> float:
    true_brg = math.degrees(math.atan2(tx - ox, ty - oy)) % 360.0
    return (true_brg - ohdg + 540) % 360 - 180


def _mission_signatures() -> set[tuple]:
    """Geometry bucket for every own-ship/target pair across every Basic Simulator
    mission definition (both the plain x/y/heading/speed schema used by Imazu missions
    and the x_nm/y_nm/heading_deg/speed_kn schema used by UM missions)."""
    sigs = set()
    if not MISSIONS_DIR.exists():
        return sigs
    for p in MISSIONS_DIR.glob("*.json"):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        own, targets = m.get("own_ship"), m.get("targets")
        if not own or not targets:
            continue
        if "x_nm" in own:
            ox, oy = own["x_nm"] * NM_TO_M, own["y_nm"] * NM_TO_M
            ohdg, ospd = own["heading_deg"], own["speed_kn"] * KN_TO_MPS
        else:
            ox, oy, ohdg, ospd = own["x"], own["y"], own["heading"], own["speed"]
        for t in targets:
            if "x_nm" in t:
                tx, ty = t["x_nm"] * NM_TO_M, t["y_nm"] * NM_TO_M
                thdg, tspd = t["heading_deg"], t["speed_kn"] * KN_TO_MPS
            else:
                tx, ty, thdg, tspd = t["x"], t["y"], t["heading"], t["speed"]
            cpa_m, tcpa_s = _cpa_tcpa(ox, oy, ohdg, ospd, tx, ty, thdg, tspd)
            sigs.add(_geom_bucket(_bearing_deg(ox, oy, ohdg, tx, ty), cpa_m, tcpa_s))
    return sigs


def _eval_scenario_signatures() -> set[tuple]:
    sigs = set()
    if not EVAL_SCENARIOS_FILE.exists():
        return sigs
    scenarios = json.loads(EVAL_SCENARIOS_FILE.read_text(encoding="utf-8"))
    for s in scenarios:
        for c in s.get("contacts", []):
            if c.get("cpa_m") is None or c.get("tcpa_min") is None or c.get("bearing_from_os_deg") is None:
                continue
            sigs.add(_geom_bucket(c["bearing_from_os_deg"], c["cpa_m"], c["tcpa_min"] * 60.0))
    return sigs


_NARRATIVE_GEOM_RE = re.compile(
    r"relative bearing (-?[\d.]+) degrees.*?CPA is ([\d.]+) m with TCPA ([\d.]+) seconds",
    re.DOTALL,
)


def check_no_mission_overlap(chunks: list[dict]) -> list[str]:
    """Returns the section/chunk id of every moos_case chunk whose geometry (re-derived
    from its own narrative TEXT via regex -- the exact phrasing build_document() always
    writes, so this works on both the intermediate leo_moos_cases.json chapters and
    final oow_rag_chunks.json chunks) matches a Basic Simulator mission's or an
    oow_colreg_scenarios.json eval scenario's starting geometry bucket. Called from
    main() BEFORE writing the corpus (a real exclusion, not just a post-hoc report) and
    from tests/test_rag_exclusions.py (an independent re-check)."""
    blocked = _mission_signatures() | _eval_scenario_signatures()
    if not blocked:
        return []
    offenders = []
    for c in chunks:
        m = _NARRATIVE_GEOM_RE.search(c.get("text", ""))
        if not m:
            continue
        bearing, cpa, tcpa = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if _geom_bucket(bearing, cpa, tcpa) in blocked:
            offenders.append(c.get("section_id") or c.get("chunk_id") or c.get("title"))
    return offenders


def load_frames() -> list[dict]:
    rows = []
    with LEO_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dominant_real_contact(contacts: list[dict]) -> dict | None:
    """The contact carrying the highest-priority ACTIVE encounter rule (RULE_PRIORITY,
    reused from build_reranker_pairs.py -- same "one dominant rule per state" logic
    already used for the reranker's training pairs), restricted to contacts with a
    valid CPA/TCPA. None if this frame has no real encounter at all."""
    best, best_rank = None, len(RULE_PRIORITY)
    for c in contacts:
        rules = c.get("active_encounter_rules") or []
        if not rules or c.get("cpa_distance_m") is None or c.get("tcpa_s") is None:
            continue
        for rank, r in enumerate(RULE_PRIORITY):
            if r in rules:
                if rank < best_rank:
                    best, best_rank = c, rank
                break
    return best


def canonicalize(frames: list[dict]) -> list[dict]:
    """One representative frame per (encounter_type, bearing bucket, CPA bucket, TCPA
    bucket, own_role, rule-set) bucket -- the frame whose TCPA is closest to that
    bucket's own median TCPA, per the RAG-rebuild plan's explicit instruction."""
    buckets: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)  # key -> [(frame, contact)]
    for frame in frames:
        contacts = frame.get("state", {}).get("contacts", [])
        contact = dominant_real_contact(contacts)
        if contact is None:
            continue
        key = (
            contact["encounter_type"],
            _bucket(contact["relative_bearing_deg"] % 360.0, [BEARING_BUCKET_DEG * i for i in range(1, 24)]),
            _bucket(contact["cpa_distance_m"], CPA_BUCKET_EDGES_M),
            _bucket(contact["tcpa_s"], TCPA_BUCKET_EDGES_S),
            contact.get("own_role"),
            tuple(sorted(contact["active_encounter_rules"])),
        )
        buckets[key].append((frame, contact))

    representatives = []
    for key, members in buckets.items():
        tcpas = [c["tcpa_s"] for _, c in members]
        target = statistics.median(tcpas)
        best_frame, _ = min(members, key=lambda fc: abs(fc[1]["tcpa_s"] - target))
        representatives.append(best_frame)
    return representatives


def build_document(representatives: list[dict]) -> dict:
    raise_if_excluded_source(LEO_FILE.name)  # defense-in-depth, see rag_exclusions.py
    chapters = []
    for frame in representatives:
        contact = dominant_real_contact(frame["state"]["contacts"])
        run_id = frame.get("source_file", "unknown_run")
        t = frame.get("state", {}).get("observed_at_s")
        rules_text = ", ".join(f"Rule {r}" for r in sorted(contact["active_encounter_rules"]))
        role = (contact.get("own_role") or "unknown").replace("_", " ")
        annotation = f"Applicable rules in this situation: {rules_text} (own-ship {role})."
        text = f"{frame['narrative']}\n\n{annotation}"
        title = f"MOOS case: {run_id} @ t={t}"
        concepts, topics = tag_text(text)
        chapters.append({
            "title": title,
            "sections": [{
                "section_id": stable_id("oow", run_id, str(t), frame.get("id", "")),
                "title": title,
                "type": "moos_case",
                "text": text,
                "concepts": concepts, "topics": topics, "pages": [],
            }],
        })
    # Exclude any case whose starting geometry matches a simulator mission or eval
    # scenario BEFORE it's ever written to disk -- a real exclusion, not a post-hoc
    # report (see check_no_mission_overlap's docstring).
    all_sections = [ch["sections"][0] for ch in chapters]
    overlapping_ids = set(check_no_mission_overlap(all_sections))
    if overlapping_ids:
        chapters = [ch for ch in chapters if ch["sections"][0]["section_id"] not in overlapping_ids]
        print(f"  Excluded {len(overlapping_ids)} case(s) overlapping a simulator mission/eval scenario's geometry")
    return {
        "document_id": "leo_moos_cases",
        "source_file": LEO_FILE.name,
        "source_type": "moos_case",
        "chapters": chapters,
    }


def main() -> None:
    frames = load_frames()
    n_no_contacts = sum(1 for f in frames if not f.get("state", {}).get("contacts"))
    n_real_encounter = sum(1 for f in frames if dominant_real_contact(f.get("state", {}).get("contacts", [])) is not None)
    print(f"Loaded {len(frames)} frames ({n_no_contacts} with zero contacts, "
          f"{n_real_encounter} with >=1 real encounter contact)")

    representatives = canonicalize(frames)
    print(f"Canonicalized to {len(representatives)} representative cases "
          f"(dedup ratio {n_real_encounter}/{len(representatives)} = "
          f"{n_real_encounter / len(representatives):.1f}x)" if representatives else "No real-encounter frames found")

    document = build_document(representatives)
    OUT_FILE.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_FILE} ({len(document['chapters'])} chapters)")


if __name__ == "__main__":
    main()
