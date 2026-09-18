"""Tutorial 13 § 11 — RLHF (DPO) preference pairs from reasoning traces.

For each trace, build a `chosen` answer (correct, from trace) and one or more `rejected`
answers by applying deterministic perturbations:

  1. wrong_channel   — swap correct VHF channel numbers with a different valid channel
  2. wrong_proword   — MAYDAY <-> PAN PAN, OVER -> OUT (mid-transmission), SECURITE -> MAYDAY
  3. missing_step    — drop one procedure step
  4. drop_regulation — omit the cited regulation
  5. drop_warning    — omit the safety warning

Contamination filter against 540 gold questions.

Output: _cache/vhf_dpo_pairs.jsonl
Format compatible with TRL DPOTrainer: {prompt, chosen, rejected, metadata}
"""
from __future__ import annotations
import json, random, argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths, load_jsonl, clean as _clean, cap as _cap, decap as _decap, EMBEDDER_MODEL, CONTAM_THRESH

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

TRACES_FILE = CACHE / f"{_PFX}_reasoning_traces.jsonl"
GOLD_FILE   = paths.gold_file
OUT_FILE    = CACHE / "vhf_dpo_pairs.jsonl"

RNG = random.Random(7)

SYSTEM = {
    "VHF": (
        "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
        "Answer accurately, use correct prowords (MAYDAY, PAN PAN, SECURITE, OVER, OUT, THIS IS), "
        "cite VHF channel numbers, follow ITU/IMO/GMDSS regulations, and never omit safety steps."
    ),
    "OOW": (
        "You are the Officer of the Watch, an AI navigation agent responsible for COLREG-compliant "
        "collision avoidance. Answer accurately, cite the correct COLREG rule number(s), respect "
        "give-way/stand-on obligations, and never omit a safety-critical step."
    ),
}[paths.domain]

# Alternate channels for perturbation (avoid pairing 16<->16)
CHANNEL_POOL = ["06", "08", "09", "13", "16", "22A", "67", "68", "70", "72", "77"]

PROWORD_SWAPS = {
    "MAYDAY":   "PAN PAN",
    "PAN PAN":  "MAYDAY",
    "SECURITE": "PAN PAN",
    "OVER":     "OUT",
    "OUT":      "OVER",
    "ROGER":    "WILCO",
    "WILCO":    "ROGER",
}


def build_chosen_answer(
    trace: dict,
    *,
    procs: list | None = None,
    channels: list | None = None,
    prowords: list | None = None,
    regulations: list | None = None,
    warnings: list | None = None,
) -> str:
    """Fluent, natural-language answer assembled from trace fields.

    Every perturbation below calls this SAME formatter with exactly one field
    swapped out, so chosen/rejected only ever differ in substance -- never in
    style -- which is what DPO should actually be learning to prefer.
    """
    procs       = trace.get("procedures")    if procs       is None else procs
    channels    = trace.get("channels")      if channels    is None else channels
    prowords    = trace.get("prowords_used") if prowords    is None else prowords
    regulations = trace.get("regulations")   if regulations is None else regulations
    warnings    = trace.get("warnings")      if warnings    is None else warnings
    outcomes    = trace.get("outcomes") or []

    sentences: list[str] = []

    situation = _clean(trace.get("situation"))
    if situation:
        sentences.append(_cap(situation) + ".")

    procs = procs or []
    steps = [_clean(p.get("action", "")) for p in procs]
    steps = [s for s in steps if s]
    if steps:
        if len(steps) == 1:
            sentences.append(_cap(steps[0]) + ".")
        else:
            connectors = ["First", "Then", "Next", "After that", "Finally"]
            pieces = [f"{connectors[i] if i < len(connectors) else 'Then'}, {_decap(s)}"
                      for i, s in enumerate(steps)]
            sentences.append("; ".join(pieces) + ".")

    channels = [str(c) for c in (channels or [])]
    if channels:
        chs = [c if c.lower().startswith("channel") else f"Channel {c}" for c in channels]
        joined = chs[0] if len(chs) == 1 else ", ".join(chs[:-1]) + f" and {chs[-1]}"
        sentences.append(f"Use {joined}.")

    prowords = prowords or []
    if prowords:
        sentences.append(f"Use the prowords {', '.join(prowords)} as appropriate.")

    regulations = regulations or []
    if regulations:
        sentences.append(f"This follows {', '.join(regulations)}.")

    warnings = warnings or []
    if warnings:
        sentences.append(f"Be careful: {_decap(_clean(warnings[0]))}.")

    if outcomes:
        sentences.append(f"Done correctly, {_decap(_clean(outcomes[0]))}.")

    return " ".join(sentences).strip()


def perturb_wrong_channel(trace: dict) -> str | None:
    """Build a 'rejected' DPO answer with one channel number swapped for a wrong one."""
    chans = [str(c).strip() for c in (trace.get("channels") or []) if str(c).strip()]
    if not chans: return None
    idx = RNG.randrange(len(chans))
    alts = [x for x in CHANNEL_POOL if x != chans[idx] and x != chans[idx].zfill(2)]
    if not alts: return None
    wrong = list(chans)
    wrong[idx] = RNG.choice(alts)
    return build_chosen_answer(trace, channels=wrong)


def perturb_wrong_proword(trace: dict) -> str | None:
    """Build a 'rejected' DPO answer with one proword swapped for a confusable wrong one."""
    pws = trace.get("prowords_used") or []
    for idx, pw in enumerate(pws):
        swap = PROWORD_SWAPS.get(pw.upper())
        if swap:
            wrong = list(pws)
            wrong[idx] = swap
            return build_chosen_answer(trace, prowords=wrong)
    return None


def perturb_missing_step(trace: dict) -> str | None:
    procs = trace.get("procedures") or []
    if len(procs) < 2: return None
    drop_idx = RNG.randint(0, len(procs) - 1)
    kept = [p for i, p in enumerate(procs) if i != drop_idx]
    return build_chosen_answer(trace, procs=kept)


def perturb_drop_regulation(trace: dict) -> str | None:
    if not (trace.get("regulations") or []): return None
    return build_chosen_answer(trace, regulations=[])


def perturb_drop_warning(trace: dict) -> str | None:
    if not (trace.get("warnings") or []): return None
    return build_chosen_answer(trace, warnings=[])


def perturb_swap_steps(trace: dict) -> str | None:
    """Swap two adjacent procedure steps -- procedures are order-critical
    (DSC alert BEFORE voice MAYDAY, hail on 16 BEFORE switching), so a
    swapped order with otherwise identical content must be dispreferred."""
    procs = trace.get("procedures") or []
    if len(procs) < 2: return None
    i = RNG.randint(0, len(procs) - 2)
    swapped = list(procs)
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    return build_chosen_answer(trace, procs=swapped)


PERTURBATIONS = [
    ("wrong_channel",   perturb_wrong_channel),
    ("wrong_proword",   perturb_wrong_proword),
    ("missing_step",    perturb_missing_step),
    ("drop_regulation", perturb_drop_regulation),
    ("drop_warning",    perturb_drop_warning),
    ("swap_step_order", perturb_swap_steps),
]


def main() -> None:
    """CLI entry point: build chosen/rejected DPO pairs by perturbing reasoning-trace answers."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-file", type=str, default=str(TRACES_FILE))
    ap.add_argument("--out-file", type=str, default=str(OUT_FILE))
    ap.add_argument("--gold-file", type=str, default=str(GOLD_FILE),
                    help="primary held-out gold file to filter against (default: paths.gold_file)")
    ap.add_argument("--extra-gold-file", type=str, default=None,
                    help="optional 2nd held-out file to also filter against (e.g. vhf_colreg_scenarios.json)")
    ap.add_argument("--extra-gold-key", type=str, default="question")
    args = ap.parse_args()

    traces_file = Path(args.traces_file)
    out_file    = Path(args.out_file)

    traces = [t for t in load_jsonl(traces_file)
              if t.get("trace") and not t.get("skip") and not t.get("error")]
    print(f"Usable traces: {len(traces)}")

    print("Loading embedder + gold-Q embeddings...")
    model = SentenceTransformer(EMBEDDER_MODEL)
    gold = json.loads(Path(args.gold_file).read_text(encoding="utf-8"))
    gold_questions = [g["question"] for g in gold]
    if args.extra_gold_file:
        extra = json.loads(Path(args.extra_gold_file).read_text(encoding="utf-8"))
        gold_questions += [g[args.extra_gold_key] for g in extra if g.get(args.extra_gold_key)]
        print(f"  + {len(extra)} extra held-out questions from {args.extra_gold_file}")
    gold_embs = model.encode(gold_questions,
                             normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False)

    candidates: list[dict] = []
    per_kind = {k: 0 for k, _ in PERTURBATIONS}
    for rec in traces:
        t = rec["trace"]
        seeds = t.get("question_seeds") or []
        if not seeds: continue
        chosen = build_chosen_answer(t)
        if len(chosen) < 40: continue
        seed = seeds[0]
        q = (seed.get("text") or "").strip()
        if len(q) < 8: continue

        for kind, fn in PERTURBATIONS:
            rej = fn(t)
            if rej and rej != chosen and len(rej) >= 20:
                candidates.append({
                    "source_chunk_id": rec["chunk_id"],
                    "source_file":     rec["source_file"],
                    "chapter_title":   rec["chapter_title"],
                    "question":        q,
                    "chosen":          chosen,
                    "rejected":        rej,
                    "perturbation":    kind,
                    "expected_points": t.get("key_facts", []),
                })
                per_kind[kind] += 1
    print(f"Raw candidates: {len(candidates)}   per perturbation: {per_kind}")

    # Contamination filter
    print("Contamination check...")
    q_embs = model.encode([c["question"] for c in candidates],
                          normalize_embeddings=True, batch_size=128,
                          show_progress_bar=True)
    kept = []
    dropped = 0
    for c, emb in zip(candidates, q_embs):
        sim = float(np.max(gold_embs @ emb))
        if sim >= CONTAM_THRESH:
            dropped += 1
            continue
        c["contam_sim"] = round(sim, 3)
        kept.append(c)
    print(f"kept={len(kept)}  contam_dropped={dropped}")

    with out_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(kept):
            row = {
                "id":           f"dpo_{i:05d}",
                "source_chunk_id": r["source_chunk_id"],
                "source_file":     r["source_file"],
                "chapter_title":   r["chapter_title"],
                "perturbation":    r["perturbation"],
                "expected_points": r["expected_points"],
                "contam_sim":      r["contam_sim"],
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": r["question"]},
                ],
                "chosen":   [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out_file.name}  ({out_file.stat().st_size/1024:.1f} KB)  rows={len(kept)}")


if __name__ == "__main__":
    main()
