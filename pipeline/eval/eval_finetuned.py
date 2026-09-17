"""
================================================================================
eval_finetuned.py — evaluate the merged VHF-QWEN on 540 held-out gold questions
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Runs Qwen (either the base model or the merged VHF-QWEN) on the 540 gold
questions with NO prompt tricks — no RAG context, no CoT instruction, no
reflection scaffold — just the plain question.

This isolates what the fine-tune itself contributes.  Comparing:
  * base + prompt tricks (RAG/CoT)  — what the run_ablation.py measured
  * VHF-QWEN, no prompt tricks      — what THIS script measures

tells us whether the fine-tune actually internalized the knowledge, or only
learned to use prompt-engineered context.

METRICS COMPUTED
----------------
SemSim   : cosine(embed(gold), embed(pred))            — semantic similarity
AnsRel   : cosine(embed(question), embed(pred))         — on-topic-ness
Faith    : gpt-4o-mini judge: is pred grounded in gold? (0/1)
Correct  : gpt-4o-mini judge: is pred correct given gold + expected points? (0-1)
Cover    : fraction of expected_points covered (embedding threshold)
NumHit   : channels & numeric values from gold that appear in pred (rule)
LitHit   : proword literal-match rate                                    (rule)
Composite: weighted mean (SemSim 0.15, AnsRel 0.1, Faith 0.15,
                          Correct 0.25, Cover 0.15, NumHit 0.1, LitHit 0.1)

Weights follow the mix from Tutorial 13 (Scholtes, § 7.9), tilted toward
Correct because for safety-critical VHF a plausible-sounding wrong answer
is worse than an awkward correct one.

USAGE
-----
    python eval_finetuned.py --model _models/VHF-QWEN
    python eval_finetuned.py --model Qwen/Qwen2.5-7B-Instruct --tag base
    python eval_finetuned.py --n 50    # smoke test with 50 questions

OUTPUT
------
_cache/eval_{tag}.jsonl     one line per question with answer + all metrics
_cache/eval_{tag}_summary.json   aggregate metric means
"""
from __future__ import annotations
import os, json, argparse, math, re, time, gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from core import AgentPaths, load_env

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
MODELS = paths.models_root
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))

GOLD_FILE = paths.gold_file

SYSTEM_PLAIN = {
    "VHF": (
        "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
        "Answer accurately, use correct prowords, cite channel numbers, "
        "follow ITU/IMO/GMDSS regulations. Be concise."
    ),
    "OOW": (
        "You are the Officer of the Watch, an AI navigation agent responsible for COLREG-compliant "
        "collision avoidance. Answer accurately, cite the correct COLREG rule number(s), and respect "
        "give-way/stand-on obligations. Be concise."
    ),
}[paths.domain]


# ── Qwen inference (loads only the LM) ───────────────────────────────────
def load_lm(model_dir: str, force_4bit: bool = False):
    """Loads a fine-tuned or base causal LM. Uses bf16 if it fits without any
    CPU offload; falls back to 4-bit NF4 for the 7B on 8 GB VRAM -- either on
    request (force_4bit) or automatically if bf16 "succeeds" but silently
    offloads part of the model to CPU (device_map="auto" doesn't raise in that
    case, it just makes generation extremely slow). AWQ-quantized checkpoints
    (own quantization_config baked into config.json) are loaded as-is, since
    forcing bf16/BitsAndBytes on top of an existing AWQ config conflicts.
    """
    from transformers import BitsAndBytesConfig
    print(f"Loading LM from {model_dir}...")

    is_awq = False
    cfg_path = Path(model_dir) / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            is_awq = cfg.get("quantization_config", {}).get("quant_method") == "awq"
        except (json.JSONDecodeError, OSError):
            pass

    def _load_4bit():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        # Force everything onto the single GPU rather than device_map="auto" -- "auto"
        # estimates free VRAM at load time and will offload a few layers to CPU/disk if
        # it judges the margin too tight (e.g. another process still holding a few hundred
        # MB), which then makes bitsandbytes refuse to proceed at all (ValueError: "Some
        # modules are dispatched on the CPU or the disk"). A 7B model in 4-bit NF4 fits
        # comfortably in 8 GB on its own, so there's no reason to let "auto" hedge here.
        m = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb, device_map={"": 0},
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        print("  loaded in 4-bit NF4")
        return m

    if is_awq:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        print("  loaded (AWQ int4)")
    elif force_4bit:
        model = _load_4bit()
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="sdpa",
            )
            offloaded = any(str(d) in ("cpu", "disk") for d in getattr(model, "hf_device_map", {}).values())
            if offloaded:
                print("  bf16 loaded but offloaded part of the model to CPU/disk (would be very slow) -- retrying in 4-bit NF4")
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = _load_4bit()
                still_offloaded = any(str(d) in ("cpu", "disk") for d in getattr(model, "hf_device_map", {}).values())
                if still_offloaded:
                    raise RuntimeError(
                        "4-bit NF4 load still offloaded part of the model to CPU/disk -- the failed "
                        "bf16 attempt's VRAM likely wasn't fully released before this retry. Pass "
                        "--force-4bit to skip the bf16 attempt entirely (recommended on an 8 GB GPU), "
                        "or restart the process to clear stale GPU memory."
                    )
            else:
                print("  loaded in bf16")
        except Exception as e:
            print(f"  bf16 failed ({type(e).__name__}: {e}) — retrying in 4-bit NF4")
            model = _load_4bit()
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return tok, model


@torch.inference_mode()
def generate(tok, model, question: str, max_new_tokens: int = 512) -> tuple[str, float]:
    """Greedy-decode one plain-prompt answer; returns (answer_text, latency_seconds)."""
    messages = [
        {"role": "system", "content": SYSTEM_PLAIN},
        {"role": "user",   "content": question},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    t0 = time.time()
    out = model.generate(
        **inp, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=1.0, top_p=1.0,
        pad_token_id=tok.eos_token_id,
    )
    dt = time.time() - t0
    ans = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ans, dt


# ── Metrics (Tutorial 13 § 7.9) ──────────────────────────────────────────
NUMBER_RE  = re.compile(r"\b\d+[A-Za-z]?\b")
PROWORDS   = ["MAYDAY", "PAN PAN", "PAN-PAN", "SECURITE", "SECURITÉ",
              "THIS IS", "OVER", "OUT", "ROGER", "WILCO", "AFFIRMATIVE",
              "NEGATIVE", "SAY AGAIN", "STAND BY", "OUT"]


def semsim(embedder, a: str, b: str) -> float:
    """Cosine similarity between the embeddings of `a` and `b`."""
    ea, eb = embedder.encode([a, b], normalize_embeddings=True)
    return float(np.dot(ea, eb))


def cover(embedder, pred: str, expected_points: list[str], thresh: float = 0.55) -> float:
    """Fraction of `expected_points` that are semantically present (>= thresh) in `pred`.

    Compares each expected_point against every individual SENTENCE of `pred`,
    not the whole-paragraph embedding: a long multi-topic answer's overall
    embedding is diluted across everything it talks about, so a single-topic
    expected_point rarely clears a fixed threshold against it even when the
    point genuinely is covered by one sentence within pred. Taking the max
    over sentences fixes that -- confirmed via the first full run, where
    Cover was ~0 for 71% of rows across every model tag (i.e. not
    discriminating between models at all, a sign of a broken metric rather
    than genuinely uncovered content).
    """
    if not expected_points: return math.nan
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', pred) if len(s.strip()) > 3]
    if not sentences:
        sentences = [pred]
    se = embedder.encode(sentences, normalize_embeddings=True)
    epe = embedder.encode(expected_points, normalize_embeddings=True)
    sims = epe @ se.T  # (n_points, n_sentences)
    best_per_point = sims.max(axis=1)
    hits = best_per_point >= thresh
    return float(np.mean(hits))


def num_hit(gold: str, pred: str) -> float:
    gold_nums = set(NUMBER_RE.findall(gold))
    if not gold_nums: return math.nan
    pred_nums = set(NUMBER_RE.findall(pred))
    return len(gold_nums & pred_nums) / len(gold_nums)


def lit_hit(gold: str, pred: str) -> float:
    g_up, p_up = gold.upper(), pred.upper()
    gold_hits = [w for w in PROWORDS if w in g_up]
    if not gold_hits: return math.nan
    return sum(1 for w in gold_hits if w in p_up) / len(gold_hits)


# ── OpenAI judge (Faith + Correct) ───────────────────────────────────────
JUDGE_MODEL = "gpt-4o-mini"

FAITH_PROMPT = """You are grading whether a candidate answer is faithful to a reference (gold) answer for a VHF marine radio question. Return JSON: {{"faith": 0 or 1, "reason": "..."}}.
1 = the candidate's SPECIFIC, CHECKABLE claims (channel numbers, prowords, procedure steps, regulations, named facts) do not contradict the gold reference. Minor additional plausible elaboration, rephrasing, or general context that is not explicitly contradicted by gold should NOT count against faithfulness -- gold answers are terse examples, not an exhaustive list of every true statement.
0 = the candidate contradicts the gold reference, or states a SPECIFIC checkable fact (wrong channel number, wrong proword, wrong procedure step, wrong regulation) that conflicts with or is unsupported by gold.
Question: {q}
Gold: {g}
Candidate: {p}"""

CORRECT_PROMPT = """You are grading correctness for a VHF marine radio question. Return JSON: {{"correct": <float in [0,1]>, "reason": "..."}}. Consider whether the candidate addresses the question, matches the gold's essential facts, and covers the listed expected points. A wrong channel number, wrong proword, or missing safety-critical step should lower the score.
Question: {q}
Gold: {g}
Expected points: {ep}
Candidate: {p}"""

_JUDGE_ERR_LOGGED = {"faith": 0, "correct": 0}  # rate-limit stderr noise


def judge_faith(client: OpenAI, q: str, g: str, p: str) -> float:
    """LLM-judge: 1.0 if `p` doesn't contradict/invent facts beyond gold answer `g`, else 0.0 (nan on judge failure)."""
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": FAITH_PROMPT.format(q=q, g=g, p=p)}],
        )
        return float(json.loads(r.choices[0].message.content).get("faith", math.nan))
    except Exception as e:
        if _JUDGE_ERR_LOGGED["faith"] < 3:
            print(f"  [judge_faith error] {type(e).__name__}: {e}", flush=True)
            _JUDGE_ERR_LOGGED["faith"] += 1
        return math.nan


def judge_correct(client: OpenAI, q: str, g: str, ep: list[str], p: str) -> float:
    """LLM-judge correctness score in [0,1] for prediction `p` against gold `g` and expected points `ep` (nan on judge failure)."""
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": CORRECT_PROMPT.format(q=q, g=g, ep=ep, p=p)}],
        )
        return float(json.loads(r.choices[0].message.content).get("correct", math.nan))
    except Exception as e:
        if _JUDGE_ERR_LOGGED["correct"] < 3:
            print(f"  [judge_correct error] {type(e).__name__}: {e}", flush=True)
            _JUDGE_ERR_LOGGED["correct"] += 1
        return math.nan


def composite(m: dict) -> float:
    """Weighted average of the per-metric scores in `m`, skipping any nan/missing metric."""
    weights = {"SemSim":0.15,"AnsRel":0.1,"Faith":0.15,
               "Correct":0.25,"Cover":0.15,"NumHit":0.1,"LitHit":0.1}
    num, den = 0.0, 0.0
    for k, w in weights.items():
        v = m.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)): continue
        num += w * v; den += w
    return num / den if den > 0 else math.nan


# ── Latency helper ────────────────────────────────────────────────────────
def _latency_stats(sorted_vals: list[float]) -> dict:
    """Mean/p50/p95 latency in seconds; None fields if no data."""
    if not sorted_vals:
        return {"mean_s": None, "p50_s": None, "p95_s": None}
    n = len(sorted_vals)
    return {
        "mean_s": round(sum(sorted_vals) / n, 3),
        "p50_s":  round(sorted_vals[n // 2], 3),
        "p95_s":  round(sorted_vals[min(n - 1, int(n * 0.95))], 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point: generate + score answers to the 540 gold Q&A for one model and write results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(paths.domain_models_dir / f"{paths.domain}-QWEN"),
                    help="path to model dir (merged) or HF id")
    ap.add_argument("--tag", type=str, default=None, help="tag for output filename")
    ap.add_argument("--n", type=int, default=None, help="how many gold Q's (default: all 540)")
    ap.add_argument("--gold-file", type=str, default=str(GOLD_FILE),
                    help="held-out gold Q&A file (default: paths.gold_file)")
    ap.add_argument("--force-4bit", action="store_true",
                    help="skip the bf16 attempt and load directly in 4-bit NF4 -- "
                         "recommended on an 8GB laptop GPU, where bf16 silently "
                         "offloads part of the model to CPU and generation crawls")
    ap.add_argument("--legacy", action="store_true",
                    help="score with the old (pre-RAGAS) v1 metric suite instead of "
                         "the claim-level suite in pipeline.eval.ragas_metrics")
    args = ap.parse_args()

    tag = args.tag or Path(args.model).name.replace("/", "_")
    out_file    = CACHE / f"eval_{tag}.jsonl"
    summary_out = CACHE / f"eval_{tag}_summary.json"

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing (needed for Faith/Correct)")
    judge = OpenAI()

    gold = json.loads(Path(args.gold_file).read_text(encoding="utf-8"))
    if args.n: gold = gold[:args.n]
    print(f"Evaluating {len(gold)} questions")

    # --- 1) Generation phase (only LM in VRAM) ---
    log_every = 1 if len(gold) <= 20 else 20
    tok, model = load_lm(args.model, force_4bit=args.force_4bit)
    gen_records = []
    for i, g in enumerate(gold, 1):
        print(f"  [gen {i}/{len(gold)}] {g['id']}: {g['question'][:80]!r}", flush=True)
        try:
            ans, dt = generate(tok, model, g["question"])
        except Exception as e:
            ans, dt = f"[ERROR:{e}]", 0.0
        gen_records.append({**g, "answer": ans, "latency_s": round(dt, 2)})
        if i % log_every == 0 or i == len(gold):
            print(f"  gen {i}/{len(gold)} done  last {dt:.1f}s", flush=True)

    # Free VRAM before loading embedder + calling judge
    del model
    torch.cuda.empty_cache()

    # --- 2) Metric phase (embedder on CPU/GPU; API calls to judge) ---
    print("\nLoading embedder for metrics...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if args.legacy:
        suite_stamp = {"metric_suite": "legacy_v1", "judge_model": JUDGE_MODEL}
        keys = ["SemSim","AnsRel","Faith","Correct","Cover","NumHit","LitHit","Composite"]
    else:
        from pipeline.eval.ragas_metrics import (
            RagasScorer, load_gold_claims, summary_stamp, CLOSED_METRICS, DETAIL_METRICS,
        )
        claims_by_id = load_gold_claims(Path(args.gold_file))
        if not claims_by_id:
            raise SystemExit(
                f"No gold_claims found next to {args.gold_file} -- run "
                "pipeline.eval.enrich_gold_claims first, or pass --legacy.")
        scorer = RagasScorer(judge, embedder, paths)
        suite_stamp = summary_stamp()
        keys = CLOSED_METRICS + DETAIL_METRICS + ["Composite"]

    print("Scoring...")
    n_no_claims = 0
    with out_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(gen_records, 1):
            if args.legacy:
                m = {
                    "SemSim":  round(semsim(embedder, r["gold_answer"], r["answer"]), 3),
                    "AnsRel":  round(semsim(embedder, r["question"],    r["answer"]), 3),
                    "Cover":   round(cover(embedder, r["answer"], r["expected_points"]), 3)
                                    if r.get("expected_points") else None,
                    "NumHit":  round(num_hit(r["gold_answer"], r["answer"]), 3),
                    "LitHit":  round(lit_hit(r["gold_answer"], r["answer"]), 3),
                    "Faith":   judge_faith(judge, r["question"], r["gold_answer"], r["answer"]),
                    "Correct": judge_correct(judge, r["question"], r["gold_answer"],
                                             r["expected_points"], r["answer"]),
                }
                m["Composite"] = round(composite(m), 3)
            else:
                claims = claims_by_id.get(str(r["id"]))
                if claims is None:
                    n_no_claims += 1
                    m = {"no_gold_claims": True}
                else:
                    m = scorer.score_row(r["question"], r["answer"], r["gold_answer"],
                                         r.get("expected_points"), claims, contexts=None)
            row = {**r, "metrics": m}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % log_every == 0 or i == len(gen_records):
                print(f"  scored {i}/{len(gen_records)}  Composite={m.get('Composite')}", flush=True)
    if n_no_claims:
        print(f"  [warn] {n_no_claims} rows skipped: no gold_claims yet (enrichment incomplete)")

    # --- 3) Summarize ---
    sums, cnts = {k: 0.0 for k in keys}, {k: 0 for k in keys}
    with out_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)["metrics"]
            for k in keys:
                v = m.get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)): continue
                sums[k] += v; cnts[k] += 1
    means = {k: round(sums[k]/cnts[k], 3) if cnts[k] else None for k in keys}

    latencies = sorted(r["latency_s"] for r in gen_records if r.get("latency_s"))
    lat_stats = _latency_stats(latencies)

    summary = {"model": args.model, "tag": tag, "n": len(gen_records),
               **suite_stamp,
               "means": means, "counts": cnts, "latency": lat_stats}
    summary_out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"Evaluation summary — {tag} (n={len(gen_records)})")
    print("=" * 60)
    for k in keys:
        v = means[k]
        v_str = f"{v:.3f}" if v is not None else " NA "
        print(f"  {k:<10} {v_str}    (n={cnts[k]})")
    print(f"\nDetails: {out_file}")
    print(f"Summary: {summary_out}")


if __name__ == "__main__":
    main()
