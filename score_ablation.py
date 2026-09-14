"""
================================================================================
score_ablation.py — score the outputs from run_ablation.py across all configs
================================================================================

Reads _cache/ablation_answers.jsonl and produces:
  _cache/ablation_scored.jsonl        (per-answer metrics)
  _cache/ablation_summary.json        (per-config means)
Prints a comparison table so you can see V0 (base) vs V1 (RAG) vs V2 (CoT) vs
V3 (RAG+CoT) on the same held-out questions.

Uses the same metric definitions as eval_finetuned.py.
"""
from __future__ import annotations
import os, json, math, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from eval_finetuned import (
    load_env, semsim, cover, num_hit, lit_hit,
    judge_faith, judge_correct, composite,
)

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"

ANSWERS_FILE = CACHE / "ablation_answers.jsonl"
SCORED_FILE  = CACHE / "ablation_scored.jsonl"
SUMMARY_FILE = CACHE / "ablation_summary.json"

METRIC_KEYS = ["SemSim","AnsRel","Faith","Correct","Cover","NumHit","LitHit","Composite"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-judge", action="store_true",
                    help="skip Faith+Correct (OpenAI) to iterate faster")
    args = ap.parse_args()

    load_env(W / ".env")
    if not args.skip_judge and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing. Use --skip-judge to bypass.")
    judge = None if args.skip_judge else OpenAI()

    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with ANSWERS_FILE.open("r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Records to score: {len(records)}")

    with SCORED_FILE.open("w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            m = {
                "SemSim":  round(semsim(embedder, r["gold_answer"], r["answer"]), 3),
                "AnsRel":  round(semsim(embedder, r["question"],    r["answer"]), 3),
                "Cover":   round(cover(embedder, r["answer"], r["expected_points"]), 3)
                                if r.get("expected_points") else None,
                "NumHit":  round(num_hit(r["gold_answer"], r["answer"]), 3),
                "LitHit":  round(lit_hit(r["gold_answer"], r["answer"]), 3),
                "Faith":   (judge_faith(judge, r["question"], r["gold_answer"], r["answer"])
                            if judge else None),
                "Correct": (judge_correct(judge, r["question"], r["gold_answer"],
                                          r["expected_points"], r["answer"])
                            if judge else None),
            }
            m["Composite"] = round(composite(m), 3)
            f.write(json.dumps({**r, "metrics": m}, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(records):
                print(f"  scored {i}/{len(records)}", flush=True)

    # ── Aggregate per config ─────────────────────────────────────────────
    per_cfg: dict[str, dict[str, list[float]]] = defaultdict(lambda: {k: [] for k in METRIC_KEYS})
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cfg = r["config"]
            for k in METRIC_KEYS:
                v = r["metrics"].get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)): continue
                per_cfg[cfg][k].append(v)

    summary = {}
    for cfg, mets in per_cfg.items():
        summary[cfg] = {k: round(float(np.mean(v)), 3) if v else None for k, v in mets.items()}
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    # ── Print comparison table ──────────────────────────────────────────
    configs = sorted(per_cfg.keys())
    header = f"{'Metric':<10}" + "".join(f"{c:>14}" for c in configs)
    print("\n" + "=" * (10 + 14 * len(configs)))
    print("Ablation comparison (higher is better)")
    print("=" * (10 + 14 * len(configs)))
    print(header)
    print("-" * len(header))
    for k in METRIC_KEYS:
        row = f"{k:<10}"
        base = summary[configs[0]][k] if configs else None
        for c in configs:
            v = summary[c][k]
            s = f"{v:.3f}" if v is not None else "  NA "
            if base is not None and v is not None and c != configs[0]:
                delta = v - base
                s += f" ({delta:+.3f})"
            row += f"{s:>14}"
        print(row)

    print(f"\nDetails: {SCORED_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
