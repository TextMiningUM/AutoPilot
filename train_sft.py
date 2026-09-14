"""
================================================================================
train_sft.py — Stage 1 of the VHF-QWEN fine-tune pipeline
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Supervised Fine-Tuning (SFT) of Qwen2.5-7B-Instruct on the VHF training corpus.
This is the "domain-adaptation" stage: we teach the base model what VHF radio
procedures, prowords, channels, GMDSS/ITU regulations look like, using
carefully constructed instruction-response pairs.

We use QLoRA:
  * Base model is loaded in 4-bit NF4 quantization (frozen).
  * A small LoRA adapter (~50 MB, ~0.7% of parameters) is trained on top.
  * Result fits in ~7 GB VRAM on an 8 GB laptop GPU (RTX 4070) with:
      - batch_size = 1
      - gradient_accumulation_steps = 16  (effective batch 16)
      - gradient checkpointing
      - bf16 compute
      - paged_adamw_8bit optimizer

TRAINING DATA
-------------
Merges four datasets built from the 30 VHF protocol JSONs:
  vhf_sft_direct.jsonl   (1226 rows)  concise Q -> A
  vhf_sft_cot.jsonl      (1226 rows)  Q -> chain-of-thought -> A
  vhf_sft_rag.jsonl      (1226 rows)  (Q + retrieved context) -> A
  vhf_multihop.jsonl     ( 560 rows)  compound Q spanning two sources
                        --------
                         4238 rows total

The 540 gold questions in vhf_gold_answers.json are HELD OUT for evaluation
and never touched.  Contamination was filtered during dataset construction.

OUTPUT
------
LoRA adapter saved to _models/vhf_qwen_sft_lora/
  Contains adapter_model.safetensors + adapter_config.json + tokenizer files.
Merge it into the base weights with merge_adapter.py to produce VHF-QWEN.

USAGE
-----
    .venv\\Scripts\\python.exe train_sft.py                  # full 3-epoch run
    .venv\\Scripts\\python.exe train_sft.py --epochs 1       # quick smoke test
    .venv\\Scripts\\python.exe train_sft.py --resume         # continue from checkpoint

WHY QLoRA?
----------
Full fine-tuning of a 7B model needs ~60 GB VRAM.  QLoRA freezes the base
weights in 4-bit and only trains ~40 M LoRA parameters, which fits in 8 GB.
Empirically QLoRA reaches within 1-2% of full fine-tune quality for domain
adaptation tasks — see the QLoRA paper (Dettmers et al., 2023).
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Paths ────────────────────────────────────────────────────────────────
W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"
MODELS = W / "_models"
VHF_MODELS = MODELS / "VHF"
VHF_MODELS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Keep the HF hub cache local to this project so we don't fill %USERPROFILE%.
os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = VHF_MODELS / "vhf_qwen_sft_lora"

# The four SFT datasets we built from the 30 protocol JSONs.
# NOTE: gold eval data (vhf_gold_answers.json) is deliberately NOT included.
SFT_DATASETS = [
    CACHE / "vhf_sft_direct.jsonl",
    CACHE / "vhf_sft_cot.jsonl",
    CACHE / "vhf_sft_rag.jsonl",
    CACHE / "vhf_multihop.jsonl",
]


# ── Data loading ─────────────────────────────────────────────────────────
def load_jsonl_dataset(path: Path) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset, keeping only the `messages` field.

    TRL's SFTTrainer knows how to consume `messages` directly: it applies the
    tokenizer's chat template automatically and computes loss only on the
    assistant tokens (not on the system/user prompt).
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"messages": r["messages"]})
    return Dataset.from_list(rows)


def load_all_sft() -> Dataset:
    parts = []
    for p in SFT_DATASETS:
        if not p.exists():
            raise FileNotFoundError(f"Missing training file: {p}")
        d = load_jsonl_dataset(p)
        print(f"  {p.name:<30} {len(d):>5} rows")
        parts.append(d)
    combined = concatenate_datasets(parts)
    # Shuffle so the model doesn't see all direct-rows then all CoT-rows in order.
    combined = combined.shuffle(seed=42)
    print(f"  {'TOTAL SFT (shuffled)':<30} {len(combined):>5} rows")
    return combined


# ── Model + LoRA config ──────────────────────────────────────────────────
def load_base_model_4bit():
    """Load Qwen2.5-7B in 4-bit NF4. Frozen; only LoRA on top is trained.

    NF4 = "NormalFloat 4-bit", the QLoRA-recommended quantization scheme.
    Double quantization saves another ~0.4 bits/param on average.
    Compute in bf16 for stability; RTX 4070 supports bf16 natively.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL_ID} in 4-bit NF4...")
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # PyTorch scaled_dot_product_attention; memory-efficient
    )
    # This helper turns off dropout on the frozen base, enables input-grad on
    # the embeddings (needed for gradient checkpointing to propagate to LoRA),
    # and casts LayerNorms to fp32 for numerical stability.
    mdl = prepare_model_for_kbit_training(mdl, use_gradient_checkpointing=True)
    return mdl


def lora_config() -> LoraConfig:
    """LoRA hyperparameters chosen for a 7B model on 8 GB VRAM.

    r=16    : rank of the LoRA update. 16 is a good default for 7B; higher
              rank = more capacity but more memory and risk of overfitting.
    alpha=32: scaling factor. Rule of thumb alpha = 2 * r.
    dropout=0.05: mild regularization on the LoRA input.
    target_modules: all attention projections + all MLP projections.
                    Adapting only q_proj/v_proj (like the original LoRA paper)
                    trains fewer params but performs worse on domain adaptation.
    modules_to_save=[]: we do NOT unfreeze the LM head or embeddings (memory).
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            # Attention projections
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP projections (Qwen2 uses SwiGLU: gate + up -> down)
            "gate_proj", "up_proj", "down_proj",
        ],
    )


# ── Training config ──────────────────────────────────────────────────────
def make_sft_config(args) -> SFTConfig:
    """SFT training hyperparameters.

    Effective batch size = per_device * grad_accum = 1 * 16 = 16
    Learning rate 2e-4 is the QLoRA-paper default for r=16.
    Cosine schedule with warmup smoothly decays LR.
    max_seq_length=2048 covers the longest RAG-augmented prompts (~1500 tok).
    """
    return SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",  # bitsandbytes 8-bit optimizer, paged to CPU RAM
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        # SFT-specific: length + chat template
        max_seq_length=args.max_seq_length,
        packing=False,  # keep examples separate; packing hurts small datasets
        # I/O
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],       # no wandb/tensorboard for offline runs
        seed=42,
        data_seed=42,
        dataloader_num_workers=2,
    )


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--epochs", type=int, default=3, help="training epochs (3 is standard for SFT)")
    ap.add_argument("--batch_size", type=int, default=1, help="per-device train batch (keep 1 on 8 GB VRAM)")
    ap.add_argument("--grad_accum", type=int, default=16, help="gradient accumulation (effective batch = batch * this)")
    ap.add_argument("--lr", type=float, default=2e-4, help="peak learning rate (QLoRA paper default)")
    ap.add_argument("--max_seq_length", type=int, default=2048, help="truncation length; covers RAG prompts")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--resume", action="store_true", help="continue from latest checkpoint in output_dir")
    args = ap.parse_args()

    print("=" * 70)
    print("VHF-QWEN — Stage 1 SFT (Supervised Fine-Tuning with QLoRA)")
    print("=" * 70)

    # --- Data ---
    print("\nLoading SFT datasets...")
    train_ds = load_all_sft()

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Qwen tokenizer already has proper eos/pad handling for the chat template.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Base model ---
    mdl = load_base_model_4bit()

    # --- SFT trainer ---
    trainer = SFTTrainer(
        model=mdl,
        args=make_sft_config(args),
        train_dataset=train_ds,
        peft_config=lora_config(),
        processing_class=tok,
    )

    # --- Sanity: how many trainable params? ---
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.model.parameters())
    print(f"\nTrainable params: {trainable:,}  ({trainable/total*100:.2f}% of {total:,} total)")
    print(f"VRAM allocated after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # --- Train ---
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # --- Save final adapter ---
    trainer.save_model(str(OUTPUT_DIR))
    tok.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved LoRA adapter to {OUTPUT_DIR}")
    print("Next step: python train_dpo.py")


if __name__ == "__main__":
    main()
