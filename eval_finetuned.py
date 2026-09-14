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
import os, json, argparse, math, re, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from openai import OpenAI

W = Path(__file__).resolve().parent
CACHE = W / "_cache"
MODELS = W / "_models"
os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))

GOLD_FILE = W / "vhf_gold_answers.json"

SYSTEM_PLAIN = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Answer accurately, use correct prowords, cite channel numbers, "
    "follow ITU/IMO/GMDSS regulations. Be concise."
)


# ── env loader (avoid python-dotenv dependency) ──────────────────────────
def load_env(path: Path):
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── Qwen inference (loads only the LM) ───────────────────────────────────
def load_lm(model_dir: str):
    """Loads a fine-tuned or base causal LM. Uses bf16 if the model fits;
    falls back to 4-bit NF4 for the 7B on 8 GB VRAM.
    """
    from transformers import BitsAndBytesConfig
    print(f"Loading LM from {model_dir}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa",
        )
        print("  loaded in bf16")
    except Exception:
        print("  bf16 failed — retrying in 4-bit NF4")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return tok, model


@torch.inference_mode()
def generate(tok, model, question: str, max_new_tokens: int = 512) -> tuple[str, float]:
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
    ea, eb = embedder.encode([a, b], normalize_embeddings=True)
    return float(np.dot(ea, eb))


def cover(embedder, pred: str, expected_points: list[str], thresh: float = 0.55) -> float:
    if not expected_points: return math.nan
    pe = embedder.encode([pred], normalize_embeddings=True)[0]
    epe = embedder.encode(expected_points, normalize_embeddings=True)
    hits = (epe @ pe) >= thresh
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

FAITH_PROMPT = """You are grading whether a candidate answer is faithful to a reference (gold) answer for a VHF marine radio question. Return JSON: {"faith": 0 or 1, "reason": "..."}. 1 = candidate does not contradict gold and its factual claims can be supported by gold. 0 = contradicts or invents facts not in gold.
Question: {q}
Gold: {g}
Candidate: {p}"""

CORRECT_PROMPT = """You are grading correctness for a VHF marine radio question. Return JSON: {"correct": <float in [0,1]>, "reason": "..."}. Consider whether the candidate addresses the question, matches the gold's essential facts, and covers the listed expected points. A wrong channel number, wrong proword, or missing safety-critical step should lower the score.
Question: {q}
Gold: {g}
Expected points: {ep}
Candidate: {p}"""


def judge_faith(client, q, g, p) -> float | None:
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": FAITH_PROMPT.format(q=q, g=g, p=p)}],
        )
        return float(json.loads(r.choices[0].message.content).get("faith", math.nan))
    except Exception:
        return math.nan


def judge_correct(client, q, g, ep, p) -> float | None:
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": CORRECT_PROMPT.format(q=q, g=g, ep=ep, p=p)}],
        )
        return float(json.loads(r.choices[0].message.content).get("correct", math.nan))
    except Exception:
        return math.nan


def composite(m: dict) -> float:
    weights = {"SemSim":0.15,"AnsRel":0.1,"Faith":0.15,
               "Correct":0.25,"Cover":0.15,"NumHit":0.1,"LitHit":0.1}
    num, den = 0.0, 0.0
    for k, w in weights.items():
        v = m.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)): continue
        num += w * v; den += w
    return num / den if den > 0 else math.nan


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(MODELS / "VHF-QWEN"),
                    help="path to model dir (merged) or HF id")
    ap.add_argument("--tag", type=str, default=None, help="tag for output filename")
    ap.add_argument("--n", type=int, default=None, help="how many gold Q's (default: all 540)")
    args = ap.parse_args()

    tag = args.tag or Path(args.model).name.replace("/", "_")
    out_file    = CACHE / f"eval_{tag}.jsonl"
    summary_out = CACHE / f"eval_{tag}_summary.json"

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing (needed for Faith/Correct)")
    judge = OpenAI()

    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    if args.n: gold = gold[:args.n]
    print(f"Evaluating {len(gold)} questions")

    # --- 1) Generation phase (only LM in VRAM) ---
    tok, model = load_lm(args.model)
    gen_records = []
    for i, g in enumerate(gold, 1):
        try:
            ans, dt = generate(tok, model, g["question"])
        except Exception as e:
            ans, dt = f"[ERROR:{e}]", 0.0
        gen_records.append({**g, "answer": ans, "latency_s": round(dt, 2)})
        if i % 20 == 0 or i == len(gold):
            print(f"  gen {i}/{len(gold)}  last {dt:.1f}s", flush=True)

    # Free VRAM before loading embedder + calling judge
    del model
    torch.cuda.empty_cache()

    # --- 2) Metric phase (embedder on CPU/GPU; API calls to judge) ---
    print("\nLoading embedder for metrics...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Scoring...")
    with out_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(gen_records, 1):
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
            row = {**r, "metrics": m}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(gen_records):
                print(f"  scored {i}/{len(gen_records)}  Composite={m['Composite']}", flush=True)

    # --- 3) Summarize ---
    keys = ["SemSim","AnsRel","Faith","Correct","Cover","NumHit","LitHit","Composite"]
    sums, cnts = {k: 0.0 for k in keys}, {k: 0 for k in keys}
    with out_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)["metrics"]
            for k in keys:
                v = m.get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)): continue
                sums[k] += v; cnts[k] += 1
    means = {k: round(sums[k]/cnts[k], 3) if cnts[k] else None for k in keys}
    summary = {"model": args.model, "tag": tag, "n": len(gen_records),
               "means": means, "counts": cnts}
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
