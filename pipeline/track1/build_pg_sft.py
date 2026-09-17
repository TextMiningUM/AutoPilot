"""
================================================================================
build_pg_sft.py — step-order training data from the Procedural Graph (Phase B)
================================================================================

The regular SFT sets teach WHAT the rules say; nothing yet explicitly teaches
STEP ORDER, even though VHF/COLREG procedures are order-critical (DSC alert
BEFORE voice MAYDAY, hail on 16 BEFORE switching to a working channel). This
builder derives single-turn Q→A rows from the PG (build_pg.py):

  1. next-step:   "After you <u>, what should you do next?"  → <v> + guidance + pitfall
  2. prerequisite:"What should you have done before you <v>?" → <u>
  3. walkthrough: "Walk me through the correct order of steps ..." → numbered
                  family path (only families whose greedy path has >= 3 steps)

Answers are fluent prose (project rule: never telegraphic label:value dumps).
Contamination-filtered against BOTH held-out eval files, like every builder.

Reads:   <cache>/<pfx>_pg.json
Writes:  <cache>/<pfx>_pg_sft.jsonl   (wired into train_sft.SFT_DATASETS)

    python -X utf8 -m pipeline.track1.build_pg_sft \
        --extra-gold-file Data/VHF/VHF_Eval/vhf_colreg_scenarios.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths, decap as _decap
from pipeline.ingest.build_pg import sample_path

paths = AgentPaths.from_env()
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

PG_FILE = CACHE / f"{_PFX}_pg.json"
OUT_FILE = CACHE / f"{_PFX}_pg_sft.jsonl"
GOLD_FILE = paths.gold_file

CONTAM_THRESH = 0.85
MIN_SUPPORT = 1        # every edge came from a real trace step pair
RNG = random.Random(13)

SYSTEM = {
    "VHF": (
        "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
        "Procedures must be carried out in the correct order. Answer accurately, "
        "use correct prowords and channel numbers, and never skip or reorder safety steps."
    ),
    "OOW": (
        "You are the Officer of the Watch, an AI navigation agent responsible for "
        "COLREG-compliant collision avoidance. Actions must be taken in the correct "
        "order. Cite the applicable rule(s) and never skip or reorder safety steps."
    ),
}[paths.domain]

FAMILY_PHRASE = {
    "distress": "a distress (MAYDAY) situation",
    "urgency": "an urgency (PAN PAN) situation",
    "safety": "a safety (SECURITE) broadcast",
    "dsc": "a DSC alert procedure",
    "colreg_encounter": "a collision-avoidance encounter",
    "routine_call": "a routine ship-to-ship or ship-to-shore call",
    "general": "this procedure",
}

NEXT_Q_TEMPLATES = [
    "During {fam}, after you {u}, what should you do next?",
    "In {fam}: I have just completed this step: {u}. What is the next step?",
    "What follows after '{u}' in {fam}?",
]
PREREQ_Q_TEMPLATES = [
    "During {fam}, what should you have done immediately before you {v}?",
    "Which step comes directly before '{v}' in {fam}?",
]


def edge_rows(pg: dict) -> list[dict]:
    rows = []
    for e in pg["edges"]:
        if e["support"] < MIN_SUPPORT:
            continue
        fam = max(e["families"], key=e["families"].get)
        fam_phrase = FAMILY_PHRASE.get(fam, FAMILY_PHRASE["general"])
        u = _decap(pg["nodes"][e["u"]]["label"].rstrip("."))
        v = pg["nodes"][e["v"]]["label"].rstrip(".")
        guidance = (e.get("guidance") or [None])[0]
        pitfall = (e.get("pitfalls") or [None])[0]

        # 1. next-step question (answer must be fluent prose)
        ans = f"The next step is to {_decap(v)}."
        if guidance:
            ans += f" This matters because {_decap(guidance.rstrip('.'))}."
        if pitfall:
            ans += f" Be careful: {_decap(pitfall.rstrip('.'))}."
        q = RNG.choice(NEXT_Q_TEMPLATES).format(fam=fam_phrase, u=u)
        rows.append({"question": q, "answer": ans, "kind": "pg_next_step",
                     "family": fam, "edge": f"{e['u']}->{e['v']}", "support": e["support"]})

        # 2. prerequisite question, only for well-attested transitions
        if e["support"] >= 2:
            q2 = RNG.choice(PREREQ_Q_TEMPLATES).format(fam=fam_phrase, v=_decap(v))
            ans2 = (f"Before that, you should {u}. Carrying out the steps in this "
                    f"order is part of correct procedure.")
            rows.append({"question": q2, "answer": ans2, "kind": "pg_prerequisite",
                         "family": fam, "edge": f"{e['u']}->{e['v']}", "support": e["support"]})
    return rows


def walkthrough_rows(pg: dict) -> list[dict]:
    rows = []
    for fam in pg["stats"]["families"]:
        path = sample_path(pg, fam, max_len=8)
        if len(path) < 3:
            continue
        fam_phrase = FAMILY_PHRASE.get(fam, FAMILY_PHRASE["general"])
        steps = [pg["nodes"][nid]["label"].rstrip(".") for nid in path]
        connectors = ["First", "Then", "Next", "After that", "Subsequently", "Then", "Finally"]
        pieces = []
        for i, s in enumerate(steps):
            c = connectors[i] if i < len(connectors) - 1 else ("Finally" if i == len(steps) - 1 else "Then")
            pieces.append(f"{c}, {_decap(s)}")
        ans = ("; ".join(pieces) + ". Keeping these steps in this order is essential "
               "for correct and safe procedure.")
        rows.append({"question": f"Walk me through the correct order of steps in {fam_phrase}.",
                     "answer": ans, "kind": "pg_walkthrough", "family": fam,
                     "edge": None, "support": min(pg['nodes'][n]['n_occurrences'] for n in path)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--gold-file", type=str, default=str(GOLD_FILE))
    ap.add_argument("--extra-gold-file", type=str, default=None)
    ap.add_argument("--extra-gold-key", type=str, default="question")
    args = ap.parse_args()

    pg = json.loads(PG_FILE.read_text(encoding="utf-8"))
    rows = edge_rows(pg) + walkthrough_rows(pg)
    print(f"Raw PG rows: {len(rows)}")

    # mandatory contamination filter against BOTH held-out eval files
    print("Contamination check...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gold = json.loads(Path(args.gold_file).read_text(encoding="utf-8"))
    gold_questions = [g["question"] for g in gold]
    if args.extra_gold_file:
        extra = json.loads(Path(args.extra_gold_file).read_text(encoding="utf-8"))
        gold_questions += [g[args.extra_gold_key] for g in extra if g.get(args.extra_gold_key)]
    gold_embs = model.encode(gold_questions, normalize_embeddings=True,
                             batch_size=64, show_progress_bar=False)
    q_embs = model.encode([r["question"] for r in rows], normalize_embeddings=True,
                          batch_size=128, show_progress_bar=False)
    kept, dropped = [], 0
    for r, emb in zip(rows, q_embs):
        sim = float(np.max(gold_embs @ emb))
        if sim >= CONTAM_THRESH:
            dropped += 1
            continue
        r["contam_sim"] = round(sim, 3)
        kept.append(r)
    print(f"kept={len(kept)}  contam_dropped={dropped}")

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for i, r in enumerate(kept):
            f.write(json.dumps({
                "id": f"pgsft_{i:05d}",
                "kind": r["kind"], "family": r["family"],
                "edge": r["edge"], "support": r["support"],
                "contam_sim": r["contam_sim"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["answer"]},
                ],
            }, ensure_ascii=False) + "\n")
    kinds = {}
    for r in kept:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"Wrote {OUT_FILE.name}  rows={len(kept)}  per kind: {kinds}")


if __name__ == "__main__":
    main()
