"""
================================================================================
merge_adapter.py — merge SFT + DPO + Reflection LoRA adapters into VHF-QWEN
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Takes the three LoRA adapters produced by the three training stages, applies
each one to the base Qwen2.5-7B-Instruct weights, and saves the result as a
single self-contained "VHF-QWEN" model directory.

The output is a normal HuggingFace model directory — no PEFT/LoRA dependency
needed at inference.  It can be loaded with:

    AutoModelForCausalLM.from_pretrained("_models/VHF-QWEN")

WHY MERGE?
----------
At inference, PEFT's runtime adapter application costs ~5-10% latency and
requires the peft library.  Merging bakes the adapter delta into the base
weights so the deployment stack is minimal.

The trade-off: after merging you can't easily undo the adapter or swap it
for a different one.  Keep the adapter directories around if you might.

MEMORY NOTE
-----------
Merging dequantizes the base weights to fp16 briefly.  Qwen2.5-7B in fp16
is about 14 GB.  This will spill into shared GPU/CPU memory on an 8 GB VRAM
laptop; it works but is slow.  For faster merging, do it on a bigger GPU or
in the cloud, then copy the merged directory to your laptop.

USAGE
-----
    python merge_adapter.py
    python merge_adapter.py --sft-only        # merge only Stage 1
    python merge_adapter.py --sft --dpo       # skip reflection
    python merge_adapter.py --output my_dir   # custom output name
"""
from __future__ import annotations
import os, argparse, shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

W = Path(__file__).resolve().parent
MODELS = W / "_models"
os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))

MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
SFT_ADAPTER = MODELS / "vhf_qwen_sft_lora"
DPO_ADAPTER = MODELS / "vhf_qwen_dpo_lora"
REFL_ADAPTER = MODELS / "vhf_qwen_reflect_lora"
DEFAULT_OUT  = MODELS / "VHF-QWEN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", action="store_true", help="apply SFT adapter (default: yes)")
    ap.add_argument("--dpo", action="store_true", help="apply DPO adapter (default: yes)")
    ap.add_argument("--reflect", action="store_true", help="apply reflection adapter (default: yes)")
    ap.add_argument("--sft-only", action="store_true", help="only SFT (skip DPO+reflect)")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    # If no explicit stage flags, apply all three by default.
    if not (args.sft or args.dpo or args.reflect):
        args.sft = args.dpo = args.reflect = True
    if args.sft_only:
        args.dpo = args.reflect = False

    out = Path(args.output)
    if out.exists():
        print(f"WARN: {out} exists. Removing.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # --- Load base in fp16 (needed for merging; 4-bit doesn't merge cleanly) ---
    print(f"Loading base {MODEL_ID} in bf16 (this uses ~14 GB — spill into shared memory OK)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # --- Apply adapters in training order ---
    if args.sft:
        print(f"Applying SFT adapter: {SFT_ADAPTER}")
        model = PeftModel.from_pretrained(model, str(SFT_ADAPTER))
        model = model.merge_and_unload()
        print("  merged.")
    if args.dpo:
        print(f"Applying DPO adapter: {DPO_ADAPTER}")
        model = PeftModel.from_pretrained(model, str(DPO_ADAPTER))
        model = model.merge_and_unload()
        print("  merged.")
    if args.reflect:
        print(f"Applying reflection adapter: {REFL_ADAPTER}")
        model = PeftModel.from_pretrained(model, str(REFL_ADAPTER))
        model = model.merge_and_unload()
        print("  merged.")

    # --- Save ---
    print(f"\nSaving merged model to {out}...")
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(str(out))
    # Drop a manifest so we remember what went into this checkpoint.
    (out / "vhf_merge_manifest.txt").write_text(
        f"base   : {MODEL_ID}\n"
        f"sft    : {args.sft}\n"
        f"dpo    : {args.dpo}\n"
        f"reflect: {args.reflect}\n"
    )
    print("Done. Test with:")
    print(f'  from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'  m = AutoModelForCausalLM.from_pretrained(r"{out}", torch_dtype="auto", device_map="auto")')


if __name__ == "__main__":
    main()
