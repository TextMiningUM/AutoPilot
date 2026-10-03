"""Shared Qwen3-8B (4-bit NF4) loader, usable by any domain's Streamlit app.

Lifted out of `Basic Simulator/app/agents.py::_load_qwen()` (2026-10-03) so a second
domain app (VHF Simulator) doesn't duplicate this logic -- the two copies would drift the
moment one side fixed a bug the other didn't get (see copilot-instructions.md "Reuse
before you rebuild"). Callers wrap this in their own `st.cache_resource`-decorated
function (caching must stay per-app since each app has its own Streamlit session/cache).

Safe to run on an 8 GB laptop GPU (confirmed repeatedly for OOW's Basic Simulator) --
this is INFERENCE only, never training (never use this to fine-tune).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from core.paths import AgentPaths

MODEL_ID = "Qwen/Qwen3-8B"


def load_qwen(weights: str, paths: AgentPaths):
    """`weights="W0_base"` loads bare Qwen3-8B. `weights="MERGED:<dir>"` loads a standalone
    already-merged model directory (produced by pipeline/train/merge_adapter.py) under
    `paths.domain_models_dir` directly as the base -- no adapters applied. Any other value
    is a "+"-joined chain of LoRA adapter directory names (also under
    `paths.domain_models_dir`) applied in order via PEFT, each merged into the base before
    the next is applied. `weights="MERGED:<dir>+<adapter>[+<adapter2>...]"` combines both:
    starts from the merged dir as the base, then stacks the given adapter(s) on top via
    PEFT WITHOUT merging/re-saving (sidesteps a transformers NotImplementedError hit when
    re-merging+saving onto an already-merged-and-quantized directory a second time).
    `paths` selects which domain's `_models/<domain>/` folder adapter/merged-dir names are
    resolved against (e.g. `AgentPaths.oow()` vs `AgentPaths.vhf()`) -- callers must still
    do their own `st.cache_resource` caching keyed on `(weights, paths.domain)`."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    extra_adapter_names: list[str] = []
    if weights.startswith("MERGED:"):
        rest = weights[len("MERGED:"):]
        merged_name, *extra_adapter_names = rest.split("+")
        merged_dir = paths.domain_models_dir / merged_name
        if not merged_dir.exists():
            raise FileNotFoundError(f"No merged model directory at {merged_dir} for weights={weights!r}")
        model_source = str(merged_dir)
    else:
        model_source = MODEL_ID
    tok = AutoTokenizer.from_pretrained(model_source)
    # Pin the whole (4-bit) model onto the single GPU instead of device_map="auto": accelerate's
    # auto-placement can decide to offload a few layers to CPU/disk when it under-estimates free
    # VRAM, and bitsandbytes 4-bit refuses that combination outright unless
    # llm_int8_enable_fp32_cpu_offload=True is set -- but Qwen3-8B fits comfortably in ~5-6 GB on
    # an 8 GB card, so there's no need for CPU offload at all.
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_source, quantization_config=bnb, device_map=device_map,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if weights != "W0_base" and not weights.startswith("MERGED:"):
        adapter_names = weights.split("+")
        models_dir = paths.domain_models_dir
        for i, name in enumerate(adapter_names):
            adapter_dir = models_dir / name
            if not adapter_dir.exists():
                raise FileNotFoundError(f"No adapter directory at {adapter_dir} for weights={weights!r}")
            mdl = PeftModel.from_pretrained(mdl, str(adapter_dir))
            if i < len(adapter_names) - 1:
                mdl = mdl.merge_and_unload()  # fold in before the NEXT adapter trains/applies on top
    elif extra_adapter_names:
        models_dir = paths.domain_models_dir
        for i, name in enumerate(extra_adapter_names):
            adapter_dir = models_dir / name
            if not adapter_dir.exists():
                raise FileNotFoundError(f"No adapter directory at {adapter_dir} for weights={weights!r}")
            mdl = PeftModel.from_pretrained(mdl, str(adapter_dir))
            if i < len(extra_adapter_names) - 1:
                mdl = mdl.merge_and_unload()
    mdl.eval()
    return tok, mdl
