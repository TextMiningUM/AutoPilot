"""Model-variant registry: single source of truth mapping a short, filesystem-safe
"model variant" ID to the actual `--weights` string app.agents._load_qwen() understands,
plus a human-readable label for dashboards/the simulator.

WHY THIS EXISTS
---------------
`--weights` values are either "W0_base", a "+"-joined LoRA adapter-directory chain (e.g.
"oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2"), or "MERGED:<dir>" for a standalone merged
model directory. All three are fine for *loading* a model, but none of them are fine as a
run-log FILENAME segment: the "+" and especially the ":" (MERGED:<dir>) are illegal in
Windows filenames (NTFS reads ":" as an Alternate-Data-Stream separator -- confirmed the
hard way: 3 real run logs scp'd to a Windows laptop silently became 0-byte files with
their content lost inside a hidden ADS). A short registry key (no "+"/"/"/":") is always
safe cross-platform and is also far more readable in a dashboard/dropdown than the raw
weights string.

This module has ZERO heavy imports (no torch/transformers/peft) so it's safe to import
from import-light, read-only tools (sweep_dashboard.py, streamlit_app.py's run-picker)
that must never pull in the model-loading stack, as well as from app/agents.py and the
run_llm_scenario.py/sweep_llm_params.py CLIs that do.

ADDING A NEW MODEL VARIANT: add one entry below (key = short id, no "+"/":"/"/") -- every
consumer (CLI --model flag, dashboard, simulator) picks it up automatically, nothing else
to wire up.
"""
from __future__ import annotations

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "qwen_base": {
        "label": "QWEN (base)",
        "weights": "W0_base",
        "description": "Untuned Qwen/Qwen3-8B, 4-bit NF4 -- no fine-tuning.",
    },
    "qwen_sftdpo": {
        "label": "QWEN-SFT-DPO",
        "weights": "MERGED:OOW-QWEN_v2_sftdpo_fix",
        "description": "SFT+DPO merged checkpoint (2026-09-25 quantization-matched merge "
                       "fix -- merged onto the SAME 4-bit NF4 base the adapters were "
                       "trained against, see pipeline/train/merge_adapter.py).",
    },
    # Add "qwen_sftdpo_reflect" here once a reflection adapter that doesn't regress
    # mission quality exists (see repo memory: the 2026-09-25 reflection retrain fixed the
    # Draft/Critique/Refined format-leak bug but still underperforms qwen_sftdpo on real
    # missions -- not registered as a variant until that's resolved).
}

DEFAULT_VARIANT = "qwen_base"

# weights-string -> variant id, for resolving an ad-hoc `--weights` value (or an OLD run
# log's filename/params, written before this registry existed) back to a known variant.
_WEIGHTS_TO_VARIANT: dict[str, str] = {v["weights"]: k for k, v in MODEL_VARIANTS.items()}


def resolve_weights(model: str) -> str:
    """--model <variant-id> -> the --weights string to actually load. Raises with the
    valid options listed if `model` isn't a registered variant."""
    if model not in MODEL_VARIANTS:
        raise KeyError(f"Unknown model variant {model!r} -- valid options: "
                       f"{sorted(MODEL_VARIANTS)}")
    return MODEL_VARIANTS[model]["weights"]


def variant_for_weights(weights: str) -> str:
    """Reverse lookup: a --weights string -> its registered variant id, or a sanitized
    fallback (safe as a filename segment) for an ad-hoc weights string with no registry
    entry -- e.g. one-off adapter combos being tried out before promoting them to a named
    variant here."""
    if weights in _WEIGHTS_TO_VARIANT:
        return _WEIGHTS_TO_VARIANT[weights]
    return (weights.replace(":", "-").replace("+", "-").replace("/", "-")
           .replace("\\", "-"))


def variant_label(variant_or_weights: str) -> str:
    """Human-readable label for a variant id OR a raw weights string (old run logs, or an
    ad-hoc combo with no registry entry) -- always returns SOMETHING displayable, never
    raises, so dashboard/simulator code can call this unconditionally."""
    if variant_or_weights in MODEL_VARIANTS:
        return MODEL_VARIANTS[variant_or_weights]["label"]
    if variant_or_weights in _WEIGHTS_TO_VARIANT:
        return MODEL_VARIANTS[_WEIGHTS_TO_VARIANT[variant_or_weights]]["label"]
    return variant_or_weights  # unregistered ad-hoc weights/variant -- show as-is
