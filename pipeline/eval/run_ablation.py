"""Ablation step 2 — run Qwen3-8B in 4-bit NF4 on precomputed prompts.

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

from core import AgentPaths

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))

PROMPTS_FILE = CACHE / "ablation_prompts.json"
ANSWERS_FILE = CACHE / "ablation_answers.jsonl"

MODEL_ID = "Qwen/Qwen3-8B"


def load_qwen():
    """Load the base Qwen3-8B model in 4-bit NF4 for ablation inference."""
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
        attn_implementation="sdpa",
    )
    mdl.eval()
    print(f"  VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return tok, mdl


@torch.inference_mode()
def generate(tok, mdl, messages: list[dict], max_new_tokens: int = 512) -> tuple[str, float]:
    """Greedy-decode one answer from a chat-formatted message list; returns (answer_text, latency_seconds)."""
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
    """Read already-completed (config, q_id) pairs from ANSWERS_FILE, for resume support."""
    if not ANSWERS_FILE.exists():
        return set()
    done: set[tuple[str, str]] = set()
    skipped = 0
    with ANSWERS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["config"], r["q_id"]))
            except (json.JSONDecodeError, KeyError):
                skipped += 1
    if skipped:
        print(f"  [load_done] skipped {skipped} malformed line(s) in {ANSWERS_FILE.name}")
    return done


def main() -> None:
    """CLI entry point: generate base-Qwen answers for each ablation config, with resume support."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None,
                    help="default: the configs list stored in ablation_prompts.json")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--track2", action="store_true",
                    help="read/write the Track 2 (scenario) ablation files instead of Track 1's")
    ap.add_argument("--tag", type=str, default="",
                    help="must match prep_ablation.py's --tag -- keeps smoke-test answers in "
                         "their own file, never appended to a full run's ablation_answers.jsonl")
    args = ap.parse_args()
    global PROMPTS_FILE, ANSWERS_FILE
    prompts_stem = "ablation_prompts_track2" if args.track2 else "ablation_prompts"
    answers_stem = "ablation_answers_track2" if args.track2 else "ablation_answers"
    if args.track2:
        PROMPTS_FILE = CACHE / f"{prompts_stem}.json"
        ANSWERS_FILE = CACHE / f"{answers_stem}.jsonl"
    if args.tag:
        PROMPTS_FILE = CACHE / f"{prompts_stem}_{args.tag}.json"
        ANSWERS_FILE = CACHE / f"{answers_stem}_{args.tag}.jsonl"

    if not PROMPTS_FILE.exists():
        raise SystemExit(f"Missing {PROMPTS_FILE}. Run prep_ablation.py first.")
    data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    records = data["records"]
    if args.configs is None:
        args.configs = data.get("configs", ["v0_base", "v1_rag", "v2_cot", "v3_rag_cot"])
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
                # Print BEFORE generating (not just after) -- otherwise a genuinely stuck/hung
                # generation looks identical to a slow-but-working one: no output at all until
                # the NEXT item completes. This line is what tells you which (config, q_id) is
                # currently in progress if it seems stuck.
                print(f"  [{idx}/{total}] starting cfg={cfg} q={rec['q_id']}...", end="", flush=True)
                # v4_pg falls back to the plain prompt for questions where no
                # guidance path could be rendered (keeps the paired sample complete)
                messages = rec["prompts"].get(cfg) or rec["prompts"]["v0_base"]
                try:
                    ans, dt = generate(tok, mdl, messages, args.max_new_tokens)
                except Exception as e:
                    ans, dt = f"[GEN_ERROR: {e}]", 0.0
                row = {
                    "config":  cfg,
                    "q_id":    rec["q_id"],
                    "question":      rec["question"],
                    "gold_answer":   rec["gold_answer"],
                    "expected_points": rec["expected_points"],
                    "answer":        ans,
                    "latency_s":     round(dt, 2),
                }
                if args.track2:
                    row["colreg_rules"] = rec.get("colreg_rules", [])
                    row["category"] = rec.get("category", "")
                    if "scenario" in rec:
                        row["scenario"] = rec["scenario"]
                        row["own_vessel"] = rec.get("own_vessel", "")
                    if "situation_report" in rec:
                        row["situation_report"] = rec["situation_report"]
                        row["correct_action"] = rec.get("correct_action")
                        row["correct_action_params"] = rec.get("correct_action_params", {})
                else:
                    row["section_id"] = rec["section_id"]
                    row["section_title"] = rec["section_title"]
                    row["type"] = rec["type"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f" done in {dt:.1f}s", flush=True)

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min")
    print(f"Output: {ANSWERS_FILE}  ({ANSWERS_FILE.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
