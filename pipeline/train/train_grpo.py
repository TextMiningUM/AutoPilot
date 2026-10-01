"""pipeline/train/train_grpo.py -- Stap 2 Step 9: GRPO (Group Relative Policy
Optimization, RLVR-style -- reward is a verifiable Nomoto-physics rollout + hard COLREG
legality gate, never an LLM judge) fine-tune for the OOW agent.

Starts from a PREVIOUS stage's merged SFT+DPO(+DAgger) checkpoint (--base-dir, e.g.
_models/OOW/OOW-QWEN_v3_sftdpo or a later V4 once the next DAgger round lands) and trains
a fresh LoRA on top using TRL's GRPOTrainer -- samples --k completions per prompt
(group), scores each via pipeline.train.reward_grpo's 3 reward functions, and updates
toward higher-reward completions using TRL's own internal group-relative-advantage loss
(no separate value/critic network, unlike PPO).

ALWAYS run --pilot first (small dataset subset + few steps, no adapter saved) and
inspect the logged per-reward-function means before committing to a full run -- same
established convention as pipeline/track2/build_rft_filter.py's own --pilot flag, for
the same reason: an untested reward/rollout wiring bug (e.g. a state-reconstruction
mismatch) could silently train against a degenerate or inverted signal for hours before
anyone notices, if only measured after the fact.

USAGE
-----
    # 1. Build the dataset once (see build_grpo_dataset.py):
    python -m pipeline.train.build_grpo_dataset

    # 2. Pilot run -- no adapter saved, just inspect logged reward means:
    python -m pipeline.train.train_grpo --base-dir _models/OOW/OOW-QWEN_v3_sftdpo --pilot

    # 3. Full run:
    python -m pipeline.train.train_grpo --base-dir _models/OOW/OOW-QWEN_v3_sftdpo \\
        --out-dir _models/OOW/oow_qwen_grpo_lora_v1 --force
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from core import AgentPaths
from pipeline.train.reward_grpo import REWARD_FUNCS, REWARD_WEIGHTS

paths = AgentPaths.oow()
CACHE = paths.cache_dir
MODELS = paths.domain_models_dir
MODEL_ID = "Qwen/Qwen3-8B"
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))


def load_dataset(path: Path, limit: int | None = None) -> Dataset:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
    if limit is not None:
        rows = rows[:limit]
    print(f"Loaded {len(rows)} GRPO prompts from {path.name}")
    return Dataset.from_list(rows)


def load_base_plus_checkpoint(base_dir: str):
    """Load `base_dir` (a MERGED, already-quantization-matched checkpoint from a
    previous stage -- NOT a bare LoRA adapter dir, unlike train_dpo.py's SFT_ADAPTER)
    in 4-bit NF4, ready for a fresh LoRA on top. Matches train_dpo.py's own
    prepare_model_for_kbit_training() call, same rationale (gradient checkpointing
    needs the kbit-training wrapper applied before PEFT attaches new adapters)."""
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    print(f"Loading {base_dir} in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(base_dir, quantization_config=bnb, device_map="auto",
                                                 torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def new_grpo_lora_config() -> LoraConfig:
    """Same target modules/rank as train_dpo.py's own DPO-stage LoRA -- GRPO shifts are
    similarly subtle (reward-shaped, not a full new task), same overfitting-avoidance
    rationale for keeping rank low."""
    return LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dataset-file", type=str, default=str(CACHE / "oow_grpo_dataset.jsonl"))
    ap.add_argument("--base-dir", type=str, required=True,
                    help="merged checkpoint dir to start from, e.g. _models/OOW/OOW-QWEN_v3_sftdpo")
    ap.add_argument("--out-dir", type=str, default=str(MODELS / "oow_qwen_grpo_lora_v1"))
    ap.add_argument("--k", type=int, default=8, help="GRPOConfig.num_generations -- completions sampled per prompt")
    # 1024 was the original default and is TOO SMALL -- same facts-only-prompt <think>
    # trace this agent produces elsewhere commonly runs 1500-2800 tokens (see build_rft_
    # filter.py's own MAX_NEW_TOKENS=3584 fix for the identical truncation bug). A real
    # pilot run on OOW-QWEN_v4_sftdpo at 1024 showed 87.5% of completions clipped before
    # finishing <think>, which `reward_schema()`/`reward_legality()` correctly scored as
    # unparseable (schema/legality means landed exactly on the math predicted by that
    # clip fraction) -- NOT a real model-capability failure.
    ap.add_argument("--max-completion-length", type=int, default=3584)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)  # Qwen3 loops without this, see repo memory
    ap.add_argument("--beta", type=float, default=0.0, help="KL penalty vs. the (frozen) base -- 0.0 = TRL default")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pilot", type=int, default=None,
                    help="cap the dataset to this many prompts + run only a few steps, "
                        "no adapter saved -- inspect logged per-reward-function means "
                        "before committing to a full run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.force and args.pilot is None:
        raise SystemExit(f"{out_dir} already exists -- pass --force to overwrite.")

    ds = load_dataset(Path(args.dataset_file), limit=args.pilot)
    model = load_base_plus_checkpoint(args.base_dir)
    tok = AutoTokenizer.from_pretrained(args.base_dir)

    cfg = GRPOConfig(
        output_dir=str(out_dir) if args.pilot is None else str(out_dir) + "_pilot",
        num_generations=args.k,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        reward_weights=REWARD_WEIGHTS,
        num_train_epochs=args.epochs if args.pilot is None else 1.0,
        max_steps=10 if args.pilot is not None else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=True,
        logging_steps=1,
        save_strategy="no" if args.pilot is not None else "steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        seed=42,
    )

    trainer = GRPOTrainer(model=model, reward_funcs=REWARD_FUNCS, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=new_grpo_lora_config())
    trainer.train()

    if args.pilot is not None:
        print("\nThis was a PILOT run -- no adapter saved. Inspect the per-step logged "
             "rewards/reward_schema/reward_legality/reward_quality means above before "
             "committing to a full run (see module docstring).")
        return

    trainer.save_model(str(out_dir))
    print(f"Saved GRPO LoRA adapter to {out_dir}")


if __name__ == "__main__":
    main()
