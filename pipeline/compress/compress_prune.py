"""
================================================================================
compress_prune.py — Structured layer pruning for VHF-QWEN
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Removes the least-important transformer blocks from VHF-QWEN.  Result: fewer
layers, less VRAM, faster inference — at the cost of some quality that the
distillation stage will partly recover.

WHY LAYER PRUNING (not weight pruning)?
---------------------------------------
On modern GPUs, unstructured weight sparsity (setting individual weights to 0)
gives almost no speedup because dense matmul kernels don't skip zeros.
Structured pruning — dropping WHOLE transformer blocks — literally removes
computation and gives proportional speedup.

Empirically (see ShortGPT paper, Men et al. 2024) transformers have layers
whose input ≈ output; removing them barely hurts.  We detect them by
Block Influence:

    BI(layer_i) = 1 - mean_over_calib( cos(input_i, output_i) )

Low BI ⇒ layer barely transforms its input ⇒ safe to drop.

Qwen3-8B has 36 transformer blocks.  Pruning 4-6 of them typically loses
<3% quality and gains ~10-15% inference speed.

IMPORTANT: after pruning you MUST run distillation to recover quality.
This script only produces the pruned checkpoint; run compress_distill.py next.

USAGE
-----
    python compress_prune.py --n-prune 4
    python compress_prune.py --n-prune 6 --measure-only    # just report BI scores

OUTPUT
------
_models/VHF-QWEN-pruned/           model with N layers removed
_cache/vhf_qwen_prune_report.json  BI scores + which layers were dropped
"""
from __future__ import annotations
import os, json, argparse, random
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from core import AgentPaths

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
MODELS = paths.models_root
VHF_MODELS = paths.domain_models_dir
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))

DEFAULT_IN  = VHF_MODELS / f"{paths.domain}-QWEN"
DEFAULT_OUT = VHF_MODELS / f"{paths.domain}-QWEN-pruned"
SFT_RAG     = CACHE / "vhf_sft_rag.jsonl"
REPORT_FILE = CACHE / "vhf_qwen_prune_report.json"


def load_model(path: Path):
    """Load a causal LM in bf16 with hidden states enabled, for block-influence measurement."""
    print(f"Loading {path} in bf16 for BI measurement...")
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", output_hidden_states=True,
    )
    model.eval()
    return tok, model


def calibration_prompts(n: int) -> list[str]:
    """Sample `n` user-turn prompts (200-2500 chars) from the SFT RAG data for BI calibration."""
    rng = random.Random(0)
    lines = SFT_RAG.read_text(encoding="utf-8").splitlines()
    rng.shuffle(lines)
    prompts = []
    for line in lines:
        r = json.loads(line)
        user = next((m["content"] for m in r["messages"] if m["role"] == "user"), None)
        if user and 200 <= len(user) <= 2500:
            prompts.append(user)
        if len(prompts) >= n: break
    return prompts


@torch.inference_mode()
def measure_block_influence(model, tok, prompts: list[str]) -> list[float]:
    """Block Influence: 1 - cos(input, output) averaged over layers and prompts."""
    n_layers = model.config.num_hidden_layers
    bi_sum = [0.0] * n_layers
    n_samples = 0
    for prompt in prompts:
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        out = model(**inp, output_hidden_states=True)
        # hidden_states is a tuple of length (n_layers + 1)
        # hidden_states[0] = embedding output; hidden_states[i] = after block i-1
        hs = out.hidden_states
        for i in range(n_layers):
            x_in  = hs[i]     .float().reshape(-1, hs[i].shape[-1])
            x_out = hs[i + 1] .float().reshape(-1, hs[i].shape[-1])
            cs = F.cosine_similarity(x_in, x_out, dim=-1).mean().item()
            bi_sum[i] += 1.0 - cs
        n_samples += 1
    return [b / n_samples for b in bi_sum]


def prune_layers(model, layer_indices: list[int]):
    """Physically drop transformer blocks from the ModuleList.
    Works for Qwen2 (model.model.layers is a ModuleList).
    """
    layers = model.model.layers
    keep = [i for i in range(len(layers)) if i not in layer_indices]
    new_layers = torch.nn.ModuleList([layers[i] for i in keep])
    model.model.layers = new_layers
    # Config update so save/load consistency holds.
    model.config.num_hidden_layers = len(new_layers)
    # Newer Qwen2.5 configs carry a per-layer `layer_types` list (e.g. full vs
    # sliding-window attention) that HF validates against num_hidden_layers on save.
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = [model.config.layer_types[i] for i in keep]
    return model


def main() -> None:
    """CLI entry point: measure block influence, prune the lowest-influence layers, and save the result."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=str, default=str(DEFAULT_IN))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--n-prune", type=int, default=4, help="how many layers to drop")
    ap.add_argument("--n-calib", type=int, default=32, help="calibration prompts")
    ap.add_argument("--measure-only", action="store_true",
                    help="print BI scores and exit — no pruning, no save")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.exists():
        raise SystemExit(f"Input model not found: {in_dir}")

    tok, model = load_model(in_dir)
    prompts = calibration_prompts(args.n_calib)
    print(f"Calibration prompts: {len(prompts)}")

    print("Measuring Block Influence for each transformer block...")
    bi = measure_block_influence(model, tok, prompts)

    ranked = sorted(range(len(bi)), key=lambda i: bi[i])  # lowest BI first
    print("\nBI per layer (lowest = safest to drop):")
    for r, i in enumerate(ranked):
        marker = " <-- drop" if r < args.n_prune else ""
        print(f"  layer {i:>2}: BI={bi[i]:.4f}{marker}")

    to_drop = sorted(ranked[:args.n_prune])
    report = {
        "n_layers_before": len(bi),
        "bi_per_layer":    [round(x, 6) for x in bi],
        "dropped_layers":  to_drop,
        "n_calibration":   len(prompts),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(f"\nSaved BI report: {REPORT_FILE}")

    if args.measure_only:
        print("--measure-only set; not saving pruned model.")
        return

    print(f"\nDropping layers {to_drop}...")
    model = prune_layers(model, to_drop)
    print(f"  new num_hidden_layers: {model.config.num_hidden_layers}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving pruned model to {out_dir}...")
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(str(out_dir))
    (out_dir / "prune_manifest.txt").write_text(
        f"source           : {in_dir}\n"
        f"n_dropped_layers : {len(to_drop)}\n"
        f"dropped_indices  : {to_drop}\n"
        f"method           : block-influence (ShortGPT-style)\n"
    )
    print("\nRecommended next step: python compress_distill.py --student", out_dir,
          "\n  (to distill back the quality lost by pruning)")


if __name__ == "__main__":
    main()
