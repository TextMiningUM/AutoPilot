"""Model-variant registry for the VHF Communications page -- same shape as
`Basic Simulator/app/model_variants.py` (short key -> {label, weights, description}) so
both apps share one convention, but with exactly ONE entry for now: VHF's own SFT/DPO/
compression attempts did not work out and were dropped (see
design_vhf_communications.md Sec 9.3, decided 2026-10-03) -- the interface runs on plain
base Qwen3-8B. Adding a future better VHF checkpoint is a one-line dict entry here, no
interface-code change needed.
"""
from __future__ import annotations

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "qwen_base": {
        "label": "QWEN (base)",
        "weights": "W0_base",
        "description": "Untuned Qwen/Qwen3-8B, 4-bit NF4 -- no VHF fine-tuning "
                       "(design_vhf_communications.md Sec 9.3: SFT/DPO/compression "
                       "dropped for VHF, base is good enough for the MVP interface).",
    },
}

DEFAULT_VARIANT = "qwen_base"


def resolve_weights(model: str) -> str:
    """--model <variant-id> -> the --weights string core.qwen_loader.load_qwen() understands."""
    if model not in MODEL_VARIANTS:
        raise KeyError(f"Unknown model variant {model!r} -- valid options: "
                       f"{sorted(MODEL_VARIANTS)}")
    return MODEL_VARIANTS[model]["weights"]


def variant_label(variant: str) -> str:
    """Human-readable label for a registered variant id; returns the id itself if unregistered."""
    return MODEL_VARIANTS.get(variant, {}).get("label", variant)
