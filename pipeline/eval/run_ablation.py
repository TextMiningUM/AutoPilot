"""Ablation step 2 — run Qwen3-8B in 4-bit NF4 on precomputed prompts.

Loads ONLY Qwen. The embedder / KG / sentence-transformer are NOT loaded here
(they already ran in prep_ablation.py). This keeps VRAM below 8 GB on RTX 4070.

Resume-safe: skips (config, q_id) pairs already present in the output file.

Two speed-ups on top of the naive one-prompt-at-a-time loop:
  1. Prompt dedup: v4_pg/v5_pg_incident/v6_pg_scenario fall back to the exact
     v0_base prompt for questions with no renderable PG guidance path (see
     prep_ablation.py). A fallback prompt is byte-identical to v0_base's, so
     greedy-decoding it again always reproduces the same answer -- we cache
     answers by prompt content and reuse them instead of re-generating.
  2. Batched generation: prompts for a config are grouped into batches (left
     padding) and generated in one model.generate() call instead of one at a
     time -- autoregressive decoding is memory-bandwidth-bound, not compute-
     bound, so batching gives close to linear throughput scaling until VRAM
     runs out.

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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Left-padding is required for batched decoder-only generation: every
    # sequence's real content must end at the same index so new tokens are
    # appended in the same position across the batch.
    tok.padding_side = "left"
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
def generate_batch(tok, mdl, messages_list: list[list[dict]],
                   max_new_tokens: int = 512) -> tuple[list[str], float]:
    """Greedy-decode a BATCH of chat-formatted message lists in one model.generate()
    call; returns (answer_texts, total_batch_latency_seconds)."""
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    inp = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=4096).to(mdl.device)
    t0 = time.time()
    out = mdl.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tok.pad_token_id,
    )
    dt = time.time() - t0
    prompt_len = inp["input_ids"].shape[1]
    answers = [tok.decode(out[i, prompt_len:], skip_special_tokens=True).strip()
              for i in range(len(messages_list))]
    return answers, dt


def load_done() -> dict[tuple[str, str], dict]:
    """Read already-completed (config, q_id) -> row from ANSWERS_FILE, for resume support."""
    if not ANSWERS_FILE.exists():
        return {}
    done: dict[tuple[str, str], dict] = {}
    skipped = 0
    with ANSWERS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done[(r["config"], r["q_id"])] = r
            except (json.JSONDecodeError, KeyError):
                skipped += 1
    if skipped:
        print(f"  [load_done] skipped {skipped} malformed line(s) in {ANSWERS_FILE.name}")
    return done


def _resolve_messages(rec: dict, cfg: str) -> list[dict]:
    """Same fallback rule prep_ablation.py's PG configs rely on: use cfg's own
    prompt if it was rendered, else fall back to the plain v0_base prompt."""
    return rec["prompts"].get(cfg) or rec["prompts"]["v0_base"]


def _prompt_key(messages: list[dict]) -> str:
    """Content-based cache key -- two (config, q_id) pairs with byte-identical
    prompts always produce the same greedy-decoded answer, so their generation
    can be skipped in favor of a cached reuse."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


def _row_for(cfg: str, rec: dict, ans: str, dt: float, track2: bool,
            reused: bool = False) -> dict:
    row = {
        "config":  cfg,
        "q_id":    rec["q_id"],
        "question":      rec["question"],
        "gold_answer":   rec["gold_answer"],
        "expected_points": rec["expected_points"],
        "answer":        ans,
        "latency_s":     round(dt, 2),
    }
    if reused:
        row["reused_prompt"] = True  # dedup hit -- no fresh generation happened
    if track2:
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
    return row


def main() -> None:
    """CLI entry point: generate base-Qwen answers for each ablation config, with resume support."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None,
                    help="default: the configs list stored in ablation_prompts.json")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts per model.generate() call -- autoregressive decoding is "
                         "memory-bandwidth-bound, not compute-bound, so batching gives near-"
                         "linear throughput until VRAM runs out. Lower this if you hit OOM.")
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
    total = len(records) * len(args.configs)
    print(f"Prompts loaded: {len(records)} Q's x {len(args.configs)} configs = {total} generations")

    done = load_done()
    print(f"Already done: {len(done)}")

    # Pre-populate the prompt-dedup cache from anything already on disk (resume case).
    records_by_qid = {r["q_id"]: r for r in records}
    prompt_cache: dict[str, tuple[str, float]] = {}
    for (cfg, qid), row in done.items():
        rec = records_by_qid.get(qid)
        if rec is None:
            continue
        key = _prompt_key(_resolve_messages(rec, cfg))
        prompt_cache.setdefault(key, (row["answer"], row.get("latency_s", 0.0)))

    tok, mdl = load_qwen()

    idx = 0
    n_reused = 0
    t_start = time.time()
    with ANSWERS_FILE.open("a", encoding="utf-8") as f:
        for cfg in args.configs:
            # Pass 1: resolve everything already done or dedup-cacheable for this
            # config without touching the model at all.
            pending: list[tuple[dict, list[dict], str]] = []
            for rec in records:
                key = (cfg, rec["q_id"])
                if key in done:
                    continue
                messages = _resolve_messages(rec, cfg)
                msg_key = _prompt_key(messages)
                cached = prompt_cache.get(msg_key)
                idx += 1
                if cached is not None:
                    ans, dt = cached
                    f.write(json.dumps(_row_for(cfg, rec, ans, dt, args.track2, reused=True),
                                       ensure_ascii=False) + "\n")
                    f.flush()
                    done[key] = {"config": cfg, "q_id": rec["q_id"], "answer": ans, "latency_s": dt}
                    n_reused += 1
                    print(f"  [{idx}/{total}] cfg={cfg} q={rec['q_id']} -- reused cached answer "
                         f"(identical prompt already generated)")
                    continue
                pending.append((rec, messages, msg_key))

            # Pass 2: batch-generate whatever's left for this config.
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                recs = [b[0] for b in batch]
                msgs_list = [b[1] for b in batch]
                msg_keys = [b[2] for b in batch]
                qids = ", ".join(str(r["q_id"]) for r in recs)
                print(f"  [{idx - len(pending) + start + 1}-{idx - len(pending) + start + len(batch)}"
                     f"/{total}] cfg={cfg} batch of {len(batch)} (q={qids})...", end="", flush=True)
                try:
                    answers, dt_total = generate_batch(tok, mdl, msgs_list, args.max_new_tokens)
                except Exception as e:
                    answers = [f"[GEN_ERROR: {e}]"] * len(batch)
                    dt_total = 0.0
                dt_each = dt_total / len(batch) if batch else 0.0
                for rec, ans, msg_key in zip(recs, answers, msg_keys):
                    prompt_cache[msg_key] = (ans, dt_each)
                    f.write(json.dumps(_row_for(cfg, rec, ans, dt_each, args.track2),
                                       ensure_ascii=False) + "\n")
                    done[(cfg, rec["q_id"])] = {"config": cfg, "q_id": rec["q_id"],
                                               "answer": ans, "latency_s": dt_each}
                f.flush()
                print(f" done in {dt_total:.1f}s ({dt_each:.1f}s/item avg)", flush=True)

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min  ({n_reused} reused via prompt dedup)")
    print(f"Output: {ANSWERS_FILE}  ({ANSWERS_FILE.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()

