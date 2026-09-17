"""
================================================================================
evolve_pg.py — offline self-evolution of the Procedural Graph (Phase C)
================================================================================

Implements the paper's evolution loop (Lu et al. 2026, §3.3) adapted to this
project's single-turn setting:

  Round k:
   1. DIAGNOSTIC ROLLOUT  — answer a training batch of gold questions with
      v4_pg-style guidance rendered from the retained graph; score each row
      with AnswerCorrectness (claim-level, needs gold_claims) + ProcOrder.
   2. FEEDBACK-DRIVEN MUTATION — an LLM refiner (gpt-4o-mini) contrasts the
      worst rows with the best ones and the family subgraph, and proposes a
      structured edit set (add/delete/update edges with condition/guidance/
      pitfalls attributes).
   3. VALIDATION GATING — the candidate graph is used to answer a held-out
      validation batch; the edit set is committed only if the mean score does
      not decrease (paper's preserve-or-improve rule), otherwise rolled back.
   4. REJECTION MEMORY — rejected edit sets are summarized back into the next
      round's refiner prompt to discourage repeated unsuccessful proposals.

EVALUATION HYGIENE: the gold questions used here (train + validation batches)
are recorded in the evolved graph's meta block. Any later v4_pg evaluation
must EXCLUDE those q_ids — tuning the graph on questions you then evaluate on
would leak. prep_ablation is unaffected as long as its stratified sample is
checked against pg meta (printed at the end).

GPU-bound (model inference per round): run on the cloud pod, or locally with
--force-4bit and the small default batch sizes.

    python -X utf8 -m pipeline.train.evolve_pg --model Qwen/Qwen2.5-7B-Instruct --force-4bit
Writes: <cache>/<pfx>_pg_evolved.json + <cache>/pg_evolution_log.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from core import AgentPaths, load_env

paths = AgentPaths.from_env()
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

PG_FILE = CACHE / f"{_PFX}_pg.json"
EVOLVED_FILE = CACHE / f"{_PFX}_pg_evolved.json"
LOG_FILE = CACHE / "pg_evolution_log.json"

SEED = 41
REFINER_MODEL = "gpt-4o-mini"

REFINER_PROMPT = """You maintain a Procedural Graph for {domain} procedures: nodes are procedure \
steps, directed NEXT edges mean "step v is admissible after step u", with textual attributes \
condition (when), guidance (how/why) and pitfalls (what to avoid).

Below are (A) evaluation rows where an agent answered WITH guidance rendered from the current \
graph — the worst-scoring rows first, then some high-scoring ones for contrast — and (B) the \
edges of the relevant part of the graph, and (C) previously REJECTED edit sets you must not repeat.

Propose at most {max_edits} edits that would make the rendered guidance more correct and more \
complete for questions like the failing ones. Only propose knowledge you can ground in the \
failing questions' gold answers shown below — never invent procedures.

Return JSON: {{"edits": [
  {{"op": "add" | "delete" | "update",
    "u_label": "<step text>", "v_label": "<step text>",
    "condition": "<when, or empty>", "guidance": "<how/why, or empty>",
    "pitfalls": "<what to avoid, or empty>", "family": "<one of: {families}>",
    "reason": "<one line>"}}, ...]}}

(A) EVALUATION ROWS:
{rows}

(B) CURRENT GRAPH EDGES (family-relevant, "u -> v [support]"):
{edges}

(C) PREVIOUSLY REJECTED EDIT SETS (do not repeat):
{rejected}
"""


# ── rollout: answer questions with guidance from a given graph ────────────
@torch.inference_mode()
def answer_with_guidance(tok, model, system: str, question: str, guidance: str) -> str:
    user = f"{guidance}\n\nQuestion: {question}" if guidance else question
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    out = model.generate(**inp, max_new_tokens=400, do_sample=False,
                         temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def score_batch(tok, model, system, graph, scorer, claims_by_id, batch) -> list[dict]:
    from pipeline.ingest.pg_guidance import render_guidance
    rows = []
    for g in batch:
        guidance = render_guidance(g["question"], graph) or ""
        ans = answer_with_guidance(tok, model, system, g["question"], guidance)
        ac = scorer.answer_correctness(g["question"], ans, g["gold_answer"],
                                       claims_by_id[str(g["id"])])
        po = scorer.proc_order(ans)
        score = ac if math.isnan(po) else 0.7 * ac + 0.3 * po
        rows.append({"q_id": g["id"], "question": g["question"],
                     "gold_answer": g["gold_answer"], "answer": ans,
                     "guidance": guidance, "AnswerCorrectness": ac,
                     "ProcOrder": po, "score": score})
    return rows


def mean_score(rows: list[dict]) -> float:
    vals = [r["score"] for r in rows if not math.isnan(r["score"])]
    return float(np.mean(vals)) if vals else float("nan")


# ── mutation: refiner proposes an edit set; apply with structural checks ──
def propose_edits(client, rows: list[dict], graph, rejected_log: list[str],
                  max_edits: int) -> list[dict]:
    from pipeline.ingest.build_pg import FAMILY_RULES
    families = [f for f, _ in FAMILY_RULES] + ["general"]
    rows_sorted = sorted(rows, key=lambda r: r["score"])
    shown = rows_sorted[:4] + rows_sorted[-2:]
    rows_txt = "\n".join(
        f"- score={r['score']:.2f} Q: {r['question'][:150]}\n"
        f"  GOLD: {r['gold_answer'][:250]}\n"
        f"  ANSWER: {r['answer'][:250]}\n"
        f"  GUIDANCE SHOWN: {(r['guidance'] or '(none)')[:200]}"
        for r in shown)
    edge_txt = "\n".join(
        f"- {graph.label(e['u'])[:70]} -> {graph.label(e['v'])[:70]} [{e['support']}]"
        for e in graph.pg["edges"][:80])
    rej_txt = "\n".join(rejected_log[-5:]) or "(none)"
    prompt = REFINER_PROMPT.format(domain=paths.domain, max_edits=max_edits,
                                   families=", ".join(families), rows=rows_txt,
                                   edges=edge_txt, rejected=rej_txt)
    r = client.chat.completions.create(
        model=REFINER_MODEL, temperature=0.3,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    try:
        edits = json.loads(r.choices[0].message.content).get("edits", [])
    except json.JSONDecodeError:
        return []
    return [e for e in edits if isinstance(e, dict) and e.get("op") in ("add", "delete", "update")][:max_edits]


def apply_edits(pg_dict: dict, edits: list[dict], graph) -> dict | None:
    """Apply an edit set to a COPY; returns None if nothing structurally valid remains."""
    cand = copy.deepcopy(pg_dict)
    applied = 0

    def node_for(label: str, create: bool) -> str | None:
        nid, _sim = graph.match(label, thresh=0.85)
        if nid is not None:
            return nid
        if not create:
            return None
        new_id = f"pg_ev_{len(cand['nodes']):04d}"
        cand["nodes"][new_id] = {"label": label.strip(), "n_occurrences": 1,
                                 "families": {}, "n_starts": 0, "n_ends": 0}
        return new_id

    for e in edits:
        u_lab, v_lab = (e.get("u_label") or "").strip(), (e.get("v_label") or "").strip()
        if not u_lab or not v_lab or u_lab == v_lab:
            continue
        fam = e.get("family") or "general"
        if e["op"] == "delete":
            u, v = node_for(u_lab, False), node_for(v_lab, False)
            if u is None or v is None:
                continue
            before = len(cand["edges"])
            cand["edges"] = [ed for ed in cand["edges"] if not (ed["u"] == u and ed["v"] == v)]
            applied += int(len(cand["edges"]) < before)
        else:  # add / update = delete-then-add (paper's edit interface)
            u, v = node_for(u_lab, True), node_for(v_lab, True)
            if u == v or u is None or v is None:
                continue
            cand["edges"] = [ed for ed in cand["edges"] if not (ed["u"] == u and ed["v"] == v)]
            cand["nodes"][u]["families"][fam] = cand["nodes"][u]["families"].get(fam, 0) + 1
            cand["nodes"][v]["families"][fam] = cand["nodes"][v]["families"].get(fam, 0) + 1
            cand["edges"].append({
                "u": u, "rel": "NEXT", "v": v, "support": 1,
                "condition": [e["condition"]] if e.get("condition") else [],
                "guidance": [e["guidance"]] if e.get("guidance") else [],
                "pitfalls": [e["pitfalls"]] if e.get("pitfalls") else [],
                "families": {fam: 1}, "sources": ["pg_evolution"],
            })
            applied += 1
    return cand if applied else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--train-n", type=int, default=12)
    ap.add_argument("--val-n", type=int, default=16)
    ap.add_argument("--max-edits", type=int, default=6)
    ap.add_argument("--gold-file", type=Path, default=paths.gold_file)
    ap.add_argument("--force-4bit", action="store_true")
    args = ap.parse_args()

    load_env(paths.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing (refiner + AnswerCorrectness judge)")

    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    from pipeline.eval.eval_finetuned import load_lm, SYSTEM_PLAIN
    from pipeline.eval.ragas_metrics import RagasScorer, load_gold_claims
    from pipeline.ingest.pg_guidance import ProceduralGraph

    claims_by_id = load_gold_claims(args.gold_file)
    if not claims_by_id:
        raise SystemExit(f"No gold_claims next to {args.gold_file} -- run enrich_gold_claims first.")

    gold = [g for g in json.loads(args.gold_file.read_text(encoding="utf-8"))
            if str(g["id"]) in claims_by_id]
    rng = random.Random(SEED)
    rng.shuffle(gold)
    need = args.rounds * args.train_n + args.val_n
    if len(gold) < need:
        raise SystemExit(f"Need {need} enriched gold rows, have {len(gold)}")
    val_batch = gold[:args.val_n]
    train_pool = gold[args.val_n:]

    client = OpenAI()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    scorer = RagasScorer(client, embedder, paths)
    tok, model = load_lm(args.model, force_4bit=args.force_4bit)

    pg_dict = json.loads(PG_FILE.read_text(encoding="utf-8"))
    graph = ProceduralGraph(pg_dict, embedder)

    print("Baseline validation rollout...")
    val_rows = score_batch(tok, model, SYSTEM_PLAIN, graph, scorer, claims_by_id, val_batch)
    retained_score = mean_score(val_rows)
    print(f"  baseline val score = {retained_score:.3f}")

    log = {"model": args.model, "gold_file": str(args.gold_file), "seed": SEED,
           "baseline_val_score": round(retained_score, 4), "rounds": []}
    rejected_log: list[str] = []

    for k in range(1, args.rounds + 1):
        t0 = time.time()
        batch = train_pool[(k - 1) * args.train_n: k * args.train_n]
        print(f"\n=== Round {k}: rollout on {len(batch)} training questions ===")
        rows = score_batch(tok, model, SYSTEM_PLAIN, graph, scorer, claims_by_id, batch)
        print(f"  train score = {mean_score(rows):.3f}")

        edits = propose_edits(client, rows, graph, rejected_log, args.max_edits)
        print(f"  refiner proposed {len(edits)} edits")
        if not edits:
            log["rounds"].append({"round": k, "edits": [], "outcome": "no_proposal"})
            continue

        cand_dict = apply_edits(pg_dict, edits, graph)
        if cand_dict is None:
            log["rounds"].append({"round": k, "edits": edits, "outcome": "structurally_invalid"})
            continue
        cand_graph = ProceduralGraph(cand_dict, embedder)

        print("  validation rollout with candidate graph...")
        cand_val = score_batch(tok, model, SYSTEM_PLAIN, cand_graph, scorer, claims_by_id, val_batch)
        cand_score = mean_score(cand_val)
        committed = cand_score >= retained_score  # preserve-or-improve
        print(f"  candidate val score = {cand_score:.3f} vs retained {retained_score:.3f} "
              f"-> {'COMMIT' if committed else 'ROLLBACK'}  ({(time.time()-t0)/60:.1f} min)")
        if committed:
            pg_dict, graph, retained_score = cand_dict, cand_graph, cand_score
        else:
            rejected_log.append(json.dumps(
                [{k2: e.get(k2) for k2 in ("op", "u_label", "v_label")} for e in edits]))
        log["rounds"].append({"round": k, "edits": edits,
                              "candidate_val_score": round(cand_score, 4),
                              "outcome": "committed" if committed else "rolled_back"})

    pg_dict.setdefault("meta", {})["evolution"] = {
        "final_val_score": round(retained_score, 4),
        "evolved_from": PG_FILE.name,
        # q_ids seen during evolution -- EXCLUDE these from any later v4_pg eval
        "tuned_on_q_ids": sorted({str(g["id"]) for g in val_batch}
                                 | {str(g["id"]) for g in train_pool[:args.rounds * args.train_n]}),
    }
    EVOLVED_FILE.write_text(json.dumps(pg_dict, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG_FILE.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    n_committed = sum(1 for r in log["rounds"] if r.get("outcome") == "committed")
    print(f"\nDone: {n_committed}/{args.rounds} rounds committed, "
          f"val {log['baseline_val_score']:.3f} -> {retained_score:.3f}")
    print(f"Wrote {EVOLVED_FILE}\n      {LOG_FILE}")
    print(f"NOTE: {len(pg_dict['meta']['evolution']['tuned_on_q_ids'])} gold q_ids were used for "
          "graph tuning -- exclude them from any later v4_pg evaluation.")


if __name__ == "__main__":
    main()
