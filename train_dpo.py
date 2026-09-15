"""
================================================================================
train_dpo.py — Stage 2 of the VHF-QWEN fine-tune pipeline (preference learning)
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Direct Preference Optimization (DPO) on the SFT'd VHF-QWEN model.

After SFT the model can generate fluent VHF responses.  But it may still:
  * quote a wrong channel number ("Channel 097" instead of "67")
  * omit a step from a distress procedure
  * confuse MAYDAY (grave danger) with PAN PAN (urgent, not life-threatening)
  * drop a required safety warning

DPO teaches the model to prefer the "chosen" (correct) answer over the
"rejected" (perturbed) answer for each prompt.  Unlike RLHF-PPO it needs no
separate reward model — the model IS the implicit reward.

TRAINING DATA
-------------
_cache/vhf_dpo_pairs.jsonl (848 preference pairs) built from reasoning traces:
  * 235 missing_step        — one procedure step deleted
  * 217 drop_warning        — safety warning removed
  * 151 drop_regulation     — regulatory citation removed
  * 134 wrong_channel       — VHF channel number swapped
  * 121 wrong_proword       — MAYDAY <-> PAN PAN, OVER -> OUT, etc.

WHY START FROM THE SFT ADAPTER?
-------------------------------
DPO on a base model rarely works — the model needs to already generate
sensible domain answers so preference gradients are meaningful.
We load base Qwen + SFT LoRA, then either:
  (a) merge SFT LoRA into base and train a fresh DPO LoRA on top   [option A]
  (b) continue training the same SFT LoRA with DPO loss            [option B]

We take option (a): cleaner separation, smaller reference model.

REFERENCE MODEL
---------------
DPO requires a "reference" model whose distribution we compare against.
Convention: use the pre-DPO SFT model as reference.  With PEFT, TRL can
share the base 4-bit backbone and just disable the adapters to get the
reference forward pass — this saves memory.  We set model_ref=None and
peft_config so DPOTrainer knows to use the "adapter-disabled" pass.

OUTPUT
------
LoRA adapter saved to _models/vhf_qwen_dpo_lora/
Merge into base with merge_adapter.py --add-dpo.

USAGE
-----
    .venv\\Scripts\\python.exe train_dpo.py                 # 1 epoch, beta=0.1
    .venv\\Scripts\\python.exe train_dpo.py --beta 0.05     # more aggressive shift

KEY HYPERPARAMETER — beta
-------------------------
beta controls how strongly the model is pushed away from the reference.
  * beta = 0.1  (default): moderate; safe start
  * beta = 0.5  : very aggressive; can destroy fluency
  * beta = 0.01 : minimal shift; useful if reference is already good
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"
MODELS = W / "_models"
VHF_MODELS = MODELS / "VHF"
VHF_MODELS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))

MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
SFT_ADAPTER = VHF_MODELS / "vhf_qwen_sft_lora"
OUTPUT_DIR  = VHF_MODELS / "vhf_qwen_dpo_lora"
DPO_FILE    = CACHE / "vhf_dpo_pairs.jsonl"


# ── Data ─────────────────────────────────────────────────────────────────
def load_dpo() -> Dataset:
    """Load preference pairs into TRL's expected DPO format.

    TRL DPOTrainer accepts either:
      * conversational: {"prompt": [msgs], "chosen": [msgs], "rejected": [msgs]}
      * text-only:      {"prompt": str,    "chosen": str,    "rejected": str}

    Our JSONL is already conversational (list of messages), which is what
    build_rlhf.py emitted, so we can pass through unchanged.
    """
    rows = []
    with DPO_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "prompt":   r["prompt"],
                "chosen":   r["chosen"],
                "rejected": r["rejected"],
            })
    ds = Dataset.from_list(rows).shuffle(seed=17)
    print(f"DPO pairs loaded: {len(ds)}")
    return ds


# ── Model ────────────────────────────────────────────────────────────────
def load_model_with_sft_merged():
    """Load base Qwen in 4-bit, load SFT LoRA on top, merge into a new 4-bit
    checkpoint so the DPO stage starts from the SFT-trained distribution.

    Note: PEFT's merge_and_unload() dequantizes the 4-bit weights, merges the
    LoRA delta in fp16, then re-quantizes when we save.  This costs a brief
    memory spike but the result is a clean starting point for stage 2.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading base {MODEL_ID} in 4-bit + SFT adapter from {SFT_ADAPTER}...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER))
    # For DPO we don't merge (keeps memory low); TRL uses adapter-disable trick
    # to compute the reference logprobs, which is memory-optimal.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def new_dpo_lora_config() -> LoraConfig:
    """Same target modules as SFT, slightly lower rank because DPO shifts are
    subtle — big rank invites overfitting on 848 pairs.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )


# ── Config ───────────────────────────────────────────────────────────────
def make_dpo_config(args) -> DPOConfig:
    return DPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=args.lr,             # ~1/4 of SFT lr -- DPO shifts are subtle
        lr_scheduler_type="cosine",
        warmup_steps=10,                    # TRL >=1.13 dropped warmup_ratio; ~10% of ~80 steps
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        # DPO-specific
        beta=args.beta,
        max_length=2048,                    # TRL >=1.13: single max_length for prompt+response
        loss_type="sigmoid",  # the original DPO loss; others: ipo, hinge, kto_pair
        # I/O
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to=[],
        seed=42,
    )


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO temperature; higher = larger shift from reference")
    args = ap.parse_args()

    print("=" * 70)
    print("VHF-QWEN — Stage 2 DPO (preference tuning on channel/proword/step perturbations)")
    print("=" * 70)

    if not SFT_ADAPTER.exists():
        raise SystemExit(f"SFT adapter not found at {SFT_ADAPTER}. Run train_sft.py first.")

    train_ds = load_dpo()

    tok = AutoTokenizer.from_pretrained(str(SFT_ADAPTER))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_model_with_sft_merged()

    trainer = DPOTrainer(
        model=model,
        # ref_model=None: TRL will disable adapters to compute reference logprobs
        # — this is why we keep the SFT adapter attached rather than merging.
        ref_model=None,
        args=make_dpo_config(args),
        train_dataset=train_ds,
        peft_config=new_dpo_lora_config(),
        processing_class=tok,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nTrainable params (DPO LoRA only): {trainable:,}")
    print(f"VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("\nTraining DPO...")
    trainer.train()

    trainer.save_model(str(OUTPUT_DIR))
    tok.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved DPO adapter to {OUTPUT_DIR}")
    print("Next step: python train_reflection.py")


if __name__ == "__main__":
    main()
