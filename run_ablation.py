"""Ablation step 2 — run Qwen2.5-7B-Instruct in 4-bit NF4 on precomputed prompts.

Loads ONLY Qwen. The embedder / KG / sentence-transformer are NOT loaded here
(they already ran in prep_ablation.py). This keeps VRAM below 8 GB on RTX 4070.

Resume-safe: skips (config, q_id) pairs already present in the output file.

Writes _cache/ablation_answers.jsonl (one line per (config, q_id)).
"""
from __future__ import annotations
import json, os, time, argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

W = Path(__file__).resolve().parent
CACHE = W / "_cache"
os.environ.setdefault("HF_HOME", str(W / "_models" / "hf_cache"))

PROMPTS_FILE = CACHE / "ablation_prompts.json"
ANSWERS_FILE = CACHE / "ablation_answers.jsonl"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def load_qwen():
    print(f"Loading {MODEL_ID} in 4-bit NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    mdl.eval()
    print(f"  VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return tok, mdl


@torch.inference_mode()
def generate(tok, mdl, messages: list[dict], max_new_tokens: int = 512):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
    t0 = time.time()
    out = mdl.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tok.eos_token_id,
    )
    dt = time.time() - t0
    new_tokens = out[0, inp["input_ids"].shape[1]:]
    ans = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return ans, dt


def load_done() -> set[tuple[str, str]]:
    if not ANSWERS_FILE.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with ANSWERS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["config"], r["q_id"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["v0_base", "v1_rag", "v2_cot", "v3_rag_cot"])
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    if not PROMPTS_FILE.exists():
        raise SystemExit(f"Missing {PROMPTS_FILE}. Run prep_ablation.py first.")
    data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    records = data["records"]
    print(f"Prompts loaded: {len(records)} Q's x {len(args.configs)} configs = "
          f"{len(records)*len(args.configs)} generations")

    done = load_done()
    print(f"Already done: {len(done)}")

    tok, mdl = load_qwen()

    total = len(records) * len(args.configs)
    idx = 0
    t_start = time.time()
    with ANSWERS_FILE.open("a", encoding="utf-8") as f:
        for cfg in args.configs:
            for rec in records:
                idx += 1
                key = (cfg, rec["q_id"])
                if key in done:
                    continue
                messages = rec["prompts"][cfg]
                try:
                    ans, dt = generate(tok, mdl, messages, args.max_new_tokens)
                except Exception as e:
                    ans, dt = f"[GEN_ERROR: {e}]", 0.0
                row = {
                    "config":  cfg,
                    "q_id":    rec["q_id"],
                    "section_id":    rec["section_id"],
                    "section_title": rec["section_title"],
                    "type":          rec["type"],
                    "question":      rec["question"],
                    "gold_answer":   rec["gold_answer"],
                    "expected_points": rec["expected_points"],
                    "answer":        ans,
                    "latency_s":     round(dt, 2),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if idx % 5 == 0 or idx == total:
                    rate = idx / (time.time() - t_start + 1e-9)
                    remain = (total - idx) / rate if rate > 0 else 0
                    print(f"  {idx}/{total}  cfg={cfg}  q={rec['q_id']}  "
                          f"lat={dt:.1f}s  ETA={remain/60:.1f}min", flush=True)

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min")
    print(f"Output: {ANSWERS_FILE}  ({ANSWERS_FILE.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
