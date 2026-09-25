"""
================================================================================
train_reflection.py — Stage 3 of the VHF-QWEN fine-tune pipeline (self-critique)
================================================================================

WHAT THIS SCRIPT DOES
---------------------
A short SFT pass on the reflection dataset (295 rows).  Each assistant reply
has the schema:

    <think>
    Draft:    <plausible but incomplete answer>
    <names the missing element: step, channel, warning, proword>
    </think>

    <complete answer>

Draft/critique reasoning lives INSIDE the <think> block (2026-09-25 fix) so the
VISIBLE completion is always just the plain refined answer -- the live inference
task never asks for a Draft/Critique/Refined scaffold, only bare JSON/prose, and
training on that raw scaffold as the visible output caused 100% parse failures at
inference (see repo memory). This teaches the model to check its own output against
safety-critical requirements before finalizing — analogous to Self-Refine / Reflexion
but learned rather than prompted.

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

from core import AgentPaths, add_dry_run_arg, write_stub_output, load_jsonl_rows_capped

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
MODELS = paths.models_root
VHF_MODELS = paths.domain_models_dir
VHF_MODELS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))

MODEL_ID     = "Qwen/Qwen3-8B"
_PFX = paths.domain.lower()
SFT_ADAPTER  = VHF_MODELS / f"{_PFX}_qwen_sft_lora"
DPO_ADAPTER  = VHF_MODELS / f"{_PFX}_qwen_dpo_lora"
OUTPUT_DIR   = VHF_MODELS / f"{_PFX}_qwen_reflect_lora"
# Track 1 (protocol-derived) + Track 2 (mined from the 360 training
# conversations -- see notebook § 12.6) reflection triples, trained together.
REFL_FILES   = {
    "VHF": [
        CACHE / "vhf_reflection.jsonl",
        CACHE / "vhf_colreg_reflection.jsonl",
    ],
    "OOW": [
        CACHE / "oow_reflection.jsonl",
        # Real accident-report FULL TEXT reflection triples (Fase C1 -- build_reflection.py
        # run against oow_incident_reasoning_traces_fulltext.jsonl, see
        # extract_incident_reasoning.py --full-text). Supersedes the earlier excerpt-based
        # extraction (same filename).
        CACHE / "oow_incident_reflection.jsonl",
        # CHIRP collision-relevant newsletter article reflection triples (Fase C1 --
        # extract_chirp_reasoning.py -> build_reflection.py).
        CACHE / "oow_chirp_reflection.jsonl",
        # Fase C3 "C1 pairs": REAL incident/CHIRP reflection triples -- critique is the
        # investigators' OWN avoidance_summary analysis, never a synthetic "too vague"
        # template (build_incident_reflection_real.py, listed twice for the same
        # "higher weight" rationale as build_incident_dpo_real.py in train_dpo.py).
        CACHE / "oow_incident_reflection_real.jsonl",
        CACHE / "oow_incident_reflection_real.jsonl",
        # Fase C3 (continued): Track-2 reflection redesign -- 3-check-question critique
        # (does a real risk exist? does direction match the rule? is the manoeuvre
        # physically achievable?) grounded in REAL model mistakes from the archived
        # units_v1 missions (build_measurement_reflection.py against app/measurement.py's
        # Check A/B/C hits), replacing the old "too vague, add parameters" pattern.
        CACHE / "oow_measurement_reflection.jsonl",
        # RUN-VERDICT-DRIVEN reflection triples (build_outcome_reflection.py) -- same
        # collision/CPA-violation/goal-not-reached mining as train_dpo.py's
        # oow_outcome_dpo_pairs.jsonl, reused here as draft/critique/refined triples.
        CACHE / "oow_outcome_reflection.jsonl",
        # Track 2 -- applied helm/engine-order decisions (build_reflection.py run against
        # the deterministic MOOS-scenario reasoning traces, notebook § 6.5).
        CACHE / "oow_scenario_reflection.jsonl",
        # Track 2 (continued) -- Leo MOOS-trajectory sample (notebook § 6.5.1),
        # kept separate from the plain oow_scenario_* reflection above for traceability.
        # Fase B5: capped at LEO_REFLECTION_CAP below, same rationale as train_sft.py's LEO_SFT_CAP.
        CACHE / "oow_scenario_Leo_reflection.jsonl",
    ],
}[paths.domain]

LEO_REFLECTION_CAP = 500


def load_reflection() -> Dataset:
    """Load and shuffle the reflection JSONL files (Track 1 + Track 2) into one Dataset.
    Leo's file is deterministically subsampled to LEO_REFLECTION_CAP rows (Fase B5)."""
    rows = []
    for path in REFL_FILES:
        if not path.exists():
            print(f"  (skipping {path.name} -- not found)")
            continue
        raw = load_jsonl_rows_capped(path, LEO_REFLECTION_CAP, seed=42) if "_Leo_" in path.name \
            else list(json.loads(line) for line in path.open("r", encoding="utf-8"))
        for r in raw:
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
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER))
    # Merge SFT before attaching DPO: this PEFT version's set_adapter() only
    # accepts a single adapter name, not a list of simultaneously-active ones.
    model = model.merge_and_unload()
    print(f"Attaching DPO adapter from {DPO_ADAPTER}...")
    model = PeftModel.from_pretrained(model, str(DPO_ADAPTER))
    # Merge both frozen adapters into the base: newer TRL versions reject
    # passing an existing PeftModel together with a fresh peft_config.
    model = model.merge_and_unload()
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


def main() -> None:
    """CLI entry point: run reflection LoRA training on top of the merged SFT+DPO model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--max_steps", type=int, default=-1, help="-1 = unlimited")
    ap.add_argument("--force", action="store_true", help="retrain even if an adapter already exists in OUTPUT_DIR")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="override OUTPUT_DIR (e.g. a _smoke-suffixed path for pipeline sanity checks)")
    ap.add_argument("--sft-dir", type=str, default=None, help="override SFT_ADAPTER")
    ap.add_argument("--dpo-dir", type=str, default=None, help="override DPO_ADAPTER")
    add_dry_run_arg(ap)
    args = ap.parse_args()

    global OUTPUT_DIR, SFT_ADAPTER, DPO_ADAPTER
    if args.out_dir:
        OUTPUT_DIR = Path(args.out_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.sft_dir:
        SFT_ADAPTER = Path(args.sft_dir)
    if args.dpo_dir:
        DPO_ADAPTER = Path(args.dpo_dir)

    print("=" * 70)
    print("VHF-QWEN — Stage 3 Reflection tuning (Draft/Critique/Refined)")
    print("=" * 70)

    if (OUTPUT_DIR / "adapter_model.safetensors").exists() and not args.force:
        print(f"Reflection adapter already exists at {OUTPUT_DIR} -- skipping (use --force to retrain).")
        return

    if not SFT_ADAPTER.exists() or not DPO_ADAPTER.exists():
        raise SystemExit("Need both SFT and DPO adapters before this stage.")

    train_ds = load_reflection()

    if args.dry_run:
        print(f"[dry-run] {len(train_ds)} reflection rows loaded and validated OK.")
        write_stub_output(OUTPUT_DIR)
        return

    tok = AutoTokenizer.from_pretrained(str(SFT_ADAPTER))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_stacked_model()

    sft_cfg_kwargs = dict(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        max_length=args.max_length,
        packing=False,
        # See train_sft.py's identical fix: TRL defaults to whole-sequence loss
        # for conversational "messages" data unless this is set.
        assistant_only_loss=True,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        seed=42,
    )
    if args.max_steps and args.max_steps > 0:
        sft_cfg_kwargs["max_steps"] = args.max_steps
    sft_cfg = SFTConfig(**sft_cfg_kwargs)

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
