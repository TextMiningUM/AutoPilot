"""
================================================================================
score_ablation.py — score the outputs from run_ablation.py across all configs
================================================================================

Reads _cache/ablation_answers.jsonl and produces:
  _cache/ablation_scored.jsonl        (per-answer metrics)
  _cache/ablation_summary.json        (per-config means + paired-bootstrap CIs)
Prints a comparison table so you can see V0 (base) vs V1 (RAG) vs V2 (CoT) vs
V3 (RAG+CoT) on the same held-out questions.

Default: the claim-level RAGAS suite (pipeline.eval.ragas_metrics) -- full
RAGAS metrics (Faithfulness/ContextPrecision/ContextRecall) for the RAG
configs, CorpusGrounded for the closed-book ones, AnswerCorrectness/
AnswerRelevancy/NumericF1/LitHit/Cover everywhere, fixed per-config composite
weights, and paired-bootstrap 95% CIs of each config's Composite vs the first
config. Retrieved contexts are recovered from ablation_prompts.json
(retrieved_chunk_ids -> RAG chunk texts); gold_claims from the Claude-enriched
gold file (pipeline.eval.enrich_gold_claims).

--legacy scores with the old v1 metric suite (eval_finetuned.py functions).
"""
from __future__ import annotations
import os, json, math, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from pipeline.eval.eval_finetuned import (
    load_env, semsim, cover, num_hit, lit_hit,
    judge_faith, judge_correct, composite,
)

from core import AgentPaths

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir

ANSWERS_FILE = CACHE / "ablation_answers.jsonl"
PROMPTS_FILE = CACHE / "ablation_prompts.json"
SCORED_FILE  = CACHE / "ablation_scored.jsonl"
SUMMARY_FILE = CACHE / "ablation_summary.json"

LEGACY_KEYS = ["SemSim","AnsRel","Faith","Correct","Cover","NumHit","LitHit","Composite"]


def _load_contexts_by_qid() -> dict[str, list[str]]:
    """{q_id -> retrieved chunk texts} recovered from ablation_prompts.json."""
    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    chunks_file = CACHE / f"{paths.domain.lower()}_rag_chunks.json"
    by_id = {c["chunk_id"]: c.get("text_with_context", c["text"])
             for c in json.loads(chunks_file.read_text(encoding="utf-8"))}
    out = {}
    for r in prompts["records"]:
        out[str(r["q_id"])] = [by_id[cid] for cid in r.get("retrieved_chunk_ids", [])
                               if cid in by_id]
    return out


def main() -> None:
    """CLI entry point: score generated ablation answers with the shared metric suite and write a summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true",
                    help="score with the old v1 metric suite instead of the RAGAS suite")
    ap.add_argument("--skip-judge", action="store_true",
                    help="(legacy only) skip Faith+Correct to iterate faster")
    ap.add_argument("--gold-file", type=Path, default=None,
                    help="gold file whose _claims enrichment to use (default: paths.gold_file; "
                         "OOW uses Data/OOW/OOW_Eval/colreg_qa_500_normalised.json)")
    args = ap.parse_args()

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY") and not (args.legacy and args.skip_judge):
        raise SystemExit("OPENAI_API_KEY missing. Use --legacy --skip-judge to bypass.")
    judge = None if (args.legacy and args.skip_judge) else OpenAI()

    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with ANSWERS_FILE.open("r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Records to score: {len(records)}")

    if args.legacy:
        metric_keys = LEGACY_KEYS
        suite_stamp = {"metric_suite": "legacy_v1"}
    else:
        from pipeline.eval.ragas_metrics import (
            RagasScorer, load_gold_claims, summary_stamp, paired_bootstrap,
            RAG_METRICS, CLOSED_METRICS,
        )
        gold_file = args.gold_file or paths.gold_file
        claims_by_id = load_gold_claims(gold_file)
        if not claims_by_id:
            raise SystemExit(f"No gold_claims next to {gold_file} -- run "
                             "pipeline.eval.enrich_gold_claims first, or pass --legacy.")
        contexts_by_qid = _load_contexts_by_qid()
        scorer = RagasScorer(judge, embedder, paths)
        metric_keys = sorted(set(RAG_METRICS + CLOSED_METRICS)) + ["Composite"]
        suite_stamp = summary_stamp()

    n_no_claims = 0
    with SCORED_FILE.open("w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            if args.legacy:
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
            else:
                claims = claims_by_id.get(str(r["q_id"]))
                if claims is None:
                    n_no_claims += 1
                    m = {"no_gold_claims": True}
                else:
                    is_rag = "rag" in r["config"]
                    ctx = contexts_by_qid.get(str(r["q_id"])) if is_rag else None
                    m = scorer.score_row(r["question"], r["answer"], r["gold_answer"],
                                         r.get("expected_points"), claims,
                                         contexts=ctx or None)
            f.write(json.dumps({**r, "metrics": m}, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(records):
                print(f"  scored {i}/{len(records)}", flush=True)
    if n_no_claims:
        print(f"  [warn] {n_no_claims} rows had no gold_claims yet (enrichment incomplete)")

    # ── Aggregate per config ─────────────────────────────────────────────
    per_cfg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_cfg_latency: dict[str, list[float]] = defaultdict(list)
    composite_by_cfg_qid: dict[str, dict[str, float]] = defaultdict(dict)
    with SCORED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cfg = r["config"]
            for k in metric_keys:
                v = r["metrics"].get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)): continue
                per_cfg[cfg][k].append(v)
            comp = r["metrics"].get("Composite")
            if comp is not None and not (isinstance(comp, float) and math.isnan(comp)):
                composite_by_cfg_qid[cfg][str(r["q_id"])] = comp
            if r.get("latency_s"):
                per_cfg_latency[cfg].append(r["latency_s"])

    summary = {"_meta": {**suite_stamp}}
    for cfg, mets in per_cfg.items():
        lat = sorted(per_cfg_latency.get(cfg, []))
        n = len(lat)
        summary[cfg] = {
            **{k: round(float(np.mean(v)), 3) if v else None for k, v in mets.items()},
            "latency_mean_s": round(sum(lat) / n, 3) if n else None,
            "latency_p50_s":  round(lat[n // 2], 3) if n else None,
        }

    # ── Paired-bootstrap CI of Composite vs the first config ─────────────
    configs = sorted(c for c in summary if c != "_meta")
    if not args.legacy and len(configs) > 1:
        base_cfg = configs[0]
        base_map = composite_by_cfg_qid[base_cfg]
        for cfg in configs[1:]:
            other = composite_by_cfg_qid[cfg]
            shared = sorted(set(base_map) & set(other))
            ci = paired_bootstrap([base_map[q] for q in shared],
                                  [other[q] for q in shared])
            summary[cfg][f"Composite_vs_{base_cfg}"] = ci
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    # ── Print comparison table ──────────────────────────────────────────
    header = f"{'Metric':<18}" + "".join(f"{c:>14}" for c in configs)
    print("\n" + "=" * (18 + 14 * len(configs)))
    print(f"Ablation comparison (higher is better) [{suite_stamp.get('metric_suite')}]")
    print("=" * (18 + 14 * len(configs)))
    print(header)
    print("-" * len(header))
    for k in metric_keys:
        row = f"{k:<18}"
        base = summary[configs[0]].get(k) if configs else None
        for c in configs:
            v = summary[c].get(k)
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
