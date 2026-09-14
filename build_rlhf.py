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
import json, random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"

TRACES_FILE = CACHE / "vhf_reasoning_traces.jsonl"
GOLD_FILE   = W / "Data" / "VHF" / "VHF_Eval" / "vhf_gold_answers.json"
OUT_FILE    = CACHE / "vhf_dpo_pairs.jsonl"

CONTAM_THRESH = 0.85
RNG = random.Random(7)

SYSTEM = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Answer accurately, use correct prowords (MAYDAY, PAN PAN, SECURITE, OVER, OUT, THIS IS), "
    "cite VHF channel numbers, follow ITU/IMO/GMDSS regulations, and never omit safety steps."
)

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


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue


def build_chosen_answer(trace: dict) -> str:
    """Full correct answer with steps + channels + prowords + warnings + outcome."""
    parts: list[str] = []
    if trace.get("situation"): parts.append(trace["situation"].rstrip("."))
    procs = trace.get("procedures") or []
    if procs:
        steps = "; ".join(f"{p.get('step','?')}. {p.get('action','').rstrip('.')}" for p in procs)
        parts.append(f"Procedure: {steps}.")
    if trace.get("channels"):
        parts.append(f"Channels: {', '.join(str(c) for c in trace['channels'])}.")
    if trace.get("prowords_used"):
        parts.append(f"Prowords: {', '.join(trace['prowords_used'])}.")
    if trace.get("regulations"):
        parts.append(f"Regulations: {', '.join(trace['regulations'])}.")
    warnings = trace.get("warnings") or []
    if warnings:
        parts.append(f"Warning: {warnings[0].rstrip('.')}.")
    if trace.get("outcomes"):
        parts.append(f"Outcome: {trace['outcomes'][0].rstrip('.')}.")
    return " ".join(parts).strip()


def perturb_wrong_channel(chosen: str, trace: dict) -> str | None:
    chans = [str(c).strip() for c in (trace.get("channels") or []) if str(c).strip()]
    if not chans: return None
    replaced = chosen
    changed = False
    for c in chans:
        alts = [x for x in CHANNEL_POOL if x != c and x != c.zfill(2)]
        wrong = RNG.choice(alts)
        for pat in (f"Channel {c}", f"channel {c}", f" {c},", f" {c}."):
            if pat in replaced:
                replaced = replaced.replace(pat, pat.replace(c, wrong), 1)
                changed = True
                break
    return replaced if changed else None


def perturb_wrong_proword(chosen: str, trace: dict) -> str | None:
    pws = trace.get("prowords_used") or []
    if not pws: return None
    for pw in pws:
        swap = PROWORD_SWAPS.get(pw.upper())
        if swap and pw.upper() in chosen.upper():
            idx = chosen.upper().find(pw.upper())
            return chosen[:idx] + swap + chosen[idx + len(pw):]
    return None


def perturb_missing_step(chosen: str, trace: dict) -> str | None:
    procs = trace.get("procedures") or []
    if len(procs) < 2: return None
    drop_idx = RNG.randint(0, len(procs) - 1)
    kept = [p for i, p in enumerate(procs) if i != drop_idx]
    parts: list[str] = []
    if trace.get("situation"): parts.append(trace["situation"].rstrip("."))
    steps = "; ".join(f"{i+1}. {p.get('action','').rstrip('.')}" for i, p in enumerate(kept))
    parts.append(f"Procedure: {steps}.")
    if trace.get("channels"):
        parts.append(f"Channels: {', '.join(str(c) for c in trace['channels'])}.")
    if trace.get("prowords_used"):
        parts.append(f"Prowords: {', '.join(trace['prowords_used'])}.")
    return " ".join(parts).strip()


def perturb_drop_regulation(chosen: str, trace: dict) -> str | None:
    regs = trace.get("regulations") or []
    if not regs: return None
    for r in regs:
        needle = f"Regulations: {', '.join(regs)}."
        if needle in chosen:
            return chosen.replace(needle, "", 1).strip()
        if r in chosen:
            return chosen.replace(r, "", 1).strip()
    return None


def perturb_drop_warning(chosen: str, trace: dict) -> str | None:
    warnings = trace.get("warnings") or []
    if not warnings: return None
    needle = f"Warning: {warnings[0].rstrip('.')}."
    if needle in chosen:
        return chosen.replace(needle, "", 1).strip()
    return None


PERTURBATIONS = [
    ("wrong_channel",   perturb_wrong_channel),
    ("wrong_proword",   perturb_wrong_proword),
    ("missing_step",    perturb_missing_step),
    ("drop_regulation", perturb_drop_regulation),
    ("drop_warning",    perturb_drop_warning),
]


def main():
    traces = [t for t in load_jsonl(TRACES_FILE)
              if t.get("trace") and not t.get("skip") and not t.get("error")]
    print(f"Usable traces: {len(traces)}")

    print("Loading embedder + gold-Q embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    gold_embs = model.encode([g["question"] for g in gold],
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
            rej = fn(chosen, t)
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

    with OUT_FILE.open("w", encoding="utf-8") as f:
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
    print(f"Wrote {OUT_FILE.name}  ({OUT_FILE.stat().st_size/1024:.1f} KB)  rows={len(kept)}")


if __name__ == "__main__":
    main()
