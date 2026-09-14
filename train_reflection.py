"""
================================================================================
train_reflection.py — Stage 3 of the VHF-QWEN fine-tune pipeline (self-critique)
================================================================================

WHAT THIS SCRIPT DOES
---------------------
A short SFT pass on the reflection dataset (295 rows).  Each assistant reply
has the schema:

    Draft:    <plausible but incomplete answer>
    Critique: <names the missing element: step, channel, warning, proword>
    Refined: <complete answer>

This teaches the model to check its own output against safety-critical
requirements before finalizing — analogous to Self-Refine / Reflexion but
learned rather than prompted.

WHY A SEPARATE STAGE?
---------------------
Blending reflection data into Stage-1 SFT dilutes its effect: only 6% of
the SFT mixture would have this format.  A dedicated small pass (1 epoch,
lower LR) preserves the SFT + DPO gains while installing the reflection
behavior.

We start from the (base + SFT LoRA + DPO LoRA) stack and add a third small
LoRA adapter.  At inference we'll merge all three into VHF-QWEN.

USAGE
-----
    .venv\\Scripts\\python.exe train_reflection.py

OUTPUT
------
LoRA adapter saved to _models/vhf_qwen_reflect_lora/
Merge all three (SFT + DPO + reflection) into VHF-QWEN with merge_adapter.py.
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

W = Path(__file__).resolve().parent
CACHE = W / "_cache"
MODELS = W / "_models"
os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))

MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"
SFT_ADAPTER  = MODELS / "vhf_qwen_sft_lora"
DPO_ADAPTER  = MODELS / "vhf_qwen_dpo_lora"
OUTPUT_DIR   = MODELS / "vhf_qwen_reflect_lora"
REFL_FILE    = CACHE / "vhf_reflection.jsonl"


def load_reflection() -> Dataset:
    rows = []
    with REFL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"messages": r["messages"]})
    ds = Dataset.from_list(rows).shuffle(seed=91)
    print(f"Reflection rows: {len(ds)}")
    return ds


def load_stacked_model():
    """Base + SFT LoRA + DPO LoRA + frozen; new reflect LoRA will train on top."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    print(f"Attaching SFT adapter from {SFT_ADAPTER}...")
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER), adapter_name="sft")
    print(f"Attaching DPO adapter from {DPO_ADAPTER}...")
    model.load_adapter(str(DPO_ADAPTER), adapter_name="dpo")
    # PEFT active adapter = which one the LoRA delta uses.  We want BOTH
    # SFT and DPO active as the frozen starting point, then add a fresh
    # trainable "reflect" adapter on top.
    model.set_adapter(["sft", "dpo"])
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def reflect_lora_config() -> LoraConfig:
    """Small rank because the dataset is small (295) — larger rank overfits."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=8)
    args = ap.parse_args()

    print("=" * 70)
    print("VHF-QWEN — Stage 3 Reflection tuning (Draft/Critique/Refined)")
    print("=" * 70)

    if not SFT_ADAPTER.exists() or not DPO_ADAPTER.exists():
        raise SystemExit("Need both SFT and DPO adapters before this stage.")

    train_ds = load_reflection()
    tok = AutoTokenizer.from_pretrained(str(SFT_ADAPTER))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_stacked_model()

    sft_cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        max_seq_length=2048,
        packing=False,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to=[],
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        peft_config=reflect_lora_config(),
        processing_class=tok,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nTrainable params (reflect LoRA only): {trainable:,}")

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tok.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved reflection adapter to {OUTPUT_DIR}")
    print("Next step: python merge_adapter.py")


if __name__ == "__main__":
    main()
