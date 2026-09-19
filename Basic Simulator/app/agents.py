"""ask_oow(): the OOW navigation agent.

Runs base Qwen3-8B (the fine-tuned OOW-QWEN checkpoint isn't trained yet --
see repo memory notes) under one of 8 selectable prompt configurations that
mirror the notebook's §11 prompt ablation study (prep_ablation.py's V0-V6),
plus an extra 'bare' baseline with no COLREG framing at all:

  bare_qwen      -- minimal system prompt, no COLREG framing, no RAG/CoT/PG
  v0_base        -- OOW framing (rules-of-thumb + JSON contract), no extras
  v1_rag         -- + retrieved COLREG excerpts
  v2_cot         -- + chain-of-thought instruction
  v3_rag_cot     -- both combined
  v4_pg          -- + procedural-graph guidance (merged, all sources)
  v5_pg_incident -- + procedural-graph guidance (real incidents only)
  v6_pg_scenario -- + procedural-graph guidance (Track 2 scenarios only)

As in prep_ablation.py's Track 2 design, the response-format JSON contract
never changes across v0-v6 -- only the USER turn gains CoT/RAG/PG content --
so every config stays directly comparable and always produces a parseable
decision. 'bare_qwen' is the one exception: a genuinely different, minimal
system prompt, to measure how much the OOW framing itself is worth.

SYSTEM_OOW_AGENT (the v0-v6 system prompt) is editable at runtime from the
sidebar's "System prompt" popover in streamlit_app.py; call preload() once
at app startup (see the splash screen) to warm both st.cache_resource caches
(retrieval index + Qwen3-8B on GPU) up front, so later ask_oow() calls are
as fast as possible instead of paying the load cost on the first click.
"""
from __future__ import annotations
import gc
import json
import os
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent               # Basic Simulator/
REPO_ROOT = ROOT.parent             # Auto Pilot/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must be set BEFORE importing any pipeline module -- several read
# AUTOPILOT_DOMAIN at import time (AgentPaths.from_env() at module level).
os.environ.setdefault("AUTOPILOT_DOMAIN", "OOW")

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core import AgentPaths, EMBEDDER_MODEL
from pipeline.ingest.build_kg import kg_retrieve
from pipeline.ingest.pg_guidance import ProceduralGraph, render_guidance
from pipeline.eval.prep_ablation import format_context

from app.missions import Mission, Vessel
from app.narrate import narrate

MODEL_ID = "Qwen/Qwen3-8B"

# Ordered so a UI selectbox lists them exactly like the notebook's §11 table (plus bare_qwen first).
MODEL_CONFIGS: dict[str, str] = {
    "bare_qwen":      "Bare Qwen3-8B -- no COLREG framing, no RAG/CoT/PG",
    "v0_base":        "v0 base -- OOW framing only",
    "v1_rag":         "v1 RAG -- + retrieved COLREG excerpts",
    "v2_cot":         "v2 CoT -- + chain-of-thought instruction",
    "v3_rag_cot":     "v3 RAG+CoT -- both combined",
    "v4_pg":          "v4 PG (merged) -- + procedure guidance, all sources",
    "v5_pg_incident": "v5 PG (incidents) -- + guidance from real incidents only",
    "v6_pg_scenario": "v6 PG (scenarios) -- + guidance from Track 2 scenarios only",
}

_CONFIG_SPECS = {
    "bare_qwen":      dict(bare=True,  rag=False, cot=False, pg=None),
    "v0_base":        dict(bare=False, rag=False, cot=False, pg=None),
    "v1_rag":         dict(bare=False, rag=True,  cot=False, pg=None),
    "v2_cot":         dict(bare=False, rag=False, cot=True,  pg=None),
    "v3_rag_cot":     dict(bare=False, rag=True,  cot=True,  pg=None),
    "v4_pg":          dict(bare=False, rag=False, cot=False, pg="merged"),
    "v5_pg_incident": dict(bare=False, rag=False, cot=False, pg="incident"),
    "v6_pg_scenario": dict(bare=False, rag=False, cot=False, pg="scenario"),
}

# Truly bare -- no COLREG rules-of-thumb, just told to answer in the required JSON shape.
BARE_SYSTEM = """You are an AI assistant helping a ship's navigation system decide on a heading/speed \
change. Reply with ONLY a JSON object, no other text:
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": <float, only for turn_left/turn_right>,
 "rule_applied": "<e.g. Rule 15, or 'none' if no rule applies>",
 "reasoning": "<one or two sentences>"}"""

# Fixed response-format contract for v0-v6, byte-identical across all of them -- only the
# USER turn varies (CoT instruction / RAG context / PG guidance), same pattern as
# prep_ablation.py's build_prompts_track2(). Editable at runtime from the sidebar's
# "System prompt" popover (streamlit_app.py) -- the override is passed in as build_oow_prompt's
# `system_prompt` arg and replaces this default for every config EXCEPT bare_qwen.
SYSTEM_OOW_AGENT = """You are the navigator on a large commercial vessel. You must navigate \
according to COLREG (the International Regulations for Preventing Collisions at Sea) at all \
times, avoiding collisions while making safe and efficient progress. Avoiding collisions comes \
first; once safe, reach the goal via the shortest direct track, with as few manoeuvres as \
possible and no zigzagging. Ground your reasoning in the provided COLREG excerpts and procedure \
guidance where given. If no target poses a real risk of collision, recommend holding course -- \
but if your current speed is below the nominal/rated speed given for this mission, consider \
speeding up instead of just holding your current pace: reaching the goal sooner (when safe) is \
part of efficient progress too, not just the shortest path. What order do you give to the helm? \
Reply with ONLY a JSON object, no other text:
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": <float, only for turn_left/turn_right>,
 "rule_applied": "<e.g. Rule 15, or 'none' if no rule applies>",
 "reasoning": "<one or two sentences>"}"""

COT_INSTR = ("Think step by step through the encounter, the applicable COLREG rule(s), and the "
            "give-way/stand-on obligations BEFORE giving your final answer.")


@st.cache_resource(show_spinner="Loading OOW retrieval index (RAG + Procedural Graphs)...")
def _load_retrieval():
    paths = AgentPaths.oow()
    cache = paths.cache_dir
    pfx = paths.domain.lower()
    chunks = json.loads((cache / f"{pfx}_rag_chunks.json").read_text(encoding="utf-8"))
    embs = np.load(cache / f"{pfx}_rag_embeddings.npy")
    ids = json.loads((cache / f"{pfx}_rag_chunk_ids.json").read_text(encoding="utf-8"))
    kg = json.loads((cache / f"{pfx}_kg.json").read_text(encoding="utf-8"))
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    # CPU embedder -- it's tiny (~100M params) and keeps the full 8 GB of VRAM free for Qwen.
    embedder = SentenceTransformer(EMBEDDER_MODEL, device="cpu")
    pg_graphs: dict[str, ProceduralGraph | None] = {}
    for key, suffix in (("merged", ""), ("incident", "_incident"), ("scenario", "_scenario")):
        pg_file = cache / f"{pfx}_pg{suffix}.json"
        pg_graphs[key] = ProceduralGraph(pg_file, embedder) if pg_file.exists() else None
    return embedder, embs, ids, kg, chunk_by_id, pg_graphs


@st.cache_resource(show_spinner="Loading Qwen3-8B (4-bit NF4) -- first call only, ~1-2 min...")
def _load_qwen():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Pin the whole (4-bit) model onto the single GPU instead of device_map="auto": accelerate's
    # auto-placement can decide to offload a few layers to CPU/disk when it under-estimates free
    # VRAM, and bitsandbytes 4-bit refuses that combination outright ("Some modules are dispatched
    # on the CPU or the disk...") unless llm_int8_enable_fp32_cpu_offload=True is set -- but the
    # model fits comfortably in ~5-6 GB on this 8 GB card, so there's no need for CPU offload at all.
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map=device_map,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    mdl.eval()
    return tok, mdl


def preload(status_cb=None) -> None:
    """Warms both st.cache_resource caches (retrieval index + Qwen3-8B onto GPU) so the
    first real 'Ask OOW agent' click is immediate instead of paying the load cost then.
    Called once from the splash screen at app startup; `status_cb`, if given, is called
    with a short human-readable status string before each stage."""
    if status_cb:
        status_cb("Loading retrieval index (RAG chunks + KG + Procedural Graphs)...")
    _load_retrieval()
    if status_cb:
        status_cb("Loading Qwen3-8B (4-bit NF4) onto GPU...")
    _load_qwen()


def unload() -> None:
    """Drops both st.cache_resource caches (retrieval index + Qwen3-8B) and releases their
    GPU/CPU memory -- used by the sidebar's 'Shut down & free GPU' button. Clearing the
    cache only drops OUR references; torch.cuda.empty_cache() then lets CUDA actually hand
    the freed VRAM back to the OS-visible free pool instead of holding it in its allocator."""
    _load_qwen.clear()
    _load_retrieval.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_json_action(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action": "hold_course", "rule_applied": "none",
                "reasoning": f"[parse error -- raw model output] {text[:300]}", "_parse_error": True}


def build_oow_prompt(mission: Mission, own: Vessel, config: str = "v3_rag_cot",
                     system_prompt: str | None = None,
                     k: int = 6, dense_n: int = 40) -> tuple[list[dict], dict]:
    """Returns (messages, debug_info) for the selected MODEL_CONFIGS key.
    `system_prompt`, if given, overrides SYSTEM_OOW_AGENT for every config except
    bare_qwen (which always keeps its own separate, deliberately minimal prompt as the
    true ablation baseline). debug_info exposes what was retrieved, for the Agent panel
    so a user can see WHY the agent decided what it decided."""
    if config not in _CONFIG_SPECS:
        raise ValueError(f"Unknown model config {config!r}; choose one of {list(MODEL_CONFIGS)}")
    spec = _CONFIG_SPECS[config]
    situation = narrate(mission, own)

    if spec["bare"]:
        user_msg = f"Situation:\n{situation}\n\nRecommend exactly ONE manoeuvre as the specified JSON object."
        messages = [{"role": "system", "content": BARE_SYSTEM},
                    {"role": "user", "content": user_msg}]
        debug = {"situation": situation, "config": config, "retrieved_chunk_ids": [],
                 "query_concepts": [], "expanded_concepts": [], "pg_guidance": None,
                 "user_msg_chars": len(user_msg)}
        return messages, debug

    embedder, embs, ids, kg, chunk_by_id, pg_graphs = _load_retrieval()

    hits, q_cons, expanded, ctx = [], [], [], None
    if spec["rag"]:
        hits, q_cons, expanded = kg_retrieve(situation, embedder, embs, ids, kg, k=k, dense_n=dense_n)
        ctx = format_context(hits, chunk_by_id)

    pg_text = None
    if spec["pg"]:
        graph = pg_graphs.get(spec["pg"])
        if graph is not None:
            pg_text = render_guidance(situation, graph)

    user_parts = []
    if spec["cot"]:
        user_parts.append(COT_INSTR)
    if pg_text:
        user_parts.append(f"Procedure guidance:\n{pg_text}")
    if ctx is not None:
        user_parts.append(f"COLREG reference excerpts:\n\n{ctx}")
    user_parts.append(f"Situation:\n{situation}")
    user_parts.append("Recommend exactly ONE manoeuvre as the specified JSON object.")
    user_msg = "\n\n".join(user_parts)

    messages = [{"role": "system", "content": system_prompt or SYSTEM_OOW_AGENT},
                {"role": "user", "content": user_msg}]
    debug = {
        "situation": situation, "config": config,
        "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
        "query_concepts": q_cons, "expanded_concepts": expanded,
        "pg_guidance": pg_text, "user_msg_chars": len(user_msg),
    }
    return messages, debug


def rag_context_preview(mission: Mission, own: Vessel, k: int = 6, dense_n: int = 40) -> dict:
    """CPU-only preview of what v1_rag/v3_rag_cot would inject for the given k, without
    touching the GPU/model -- lets the Agent panel show a live "context size" readout as
    the user adjusts the RAG on/off switch and chunk-count slider."""
    if k <= 0:
        return {"chars": 0, "chunks": 0}
    embedder, embs, ids, kg, chunk_by_id, _ = _load_retrieval()
    situation = narrate(mission, own)
    hits, _, _ = kg_retrieve(situation, embedder, embs, ids, kg, k=k, dense_n=dense_n)
    ctx = format_context(hits, chunk_by_id)
    return {"chars": len(ctx), "chunks": len(hits)}


@torch.inference_mode()
def _generate(tok, mdl, messages: list[dict], max_new_tokens: int = 256,
             enable_thinking: bool = False) -> str:
    # enable_thinking=False (default) turns off Qwen3's native hidden <think>...</think>
    # reasoning channel -- for this JSON-only decision task that hidden reasoning was the
    # single biggest latency cost (it can silently burn most of max_new_tokens before ever
    # reaching the visible JSON answer) while adding nothing visible/useful here. Both
    # params are exposed up through ask_oow() so the Agent panel can adjust them live.
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
    out = mdl.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id)
    result = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    # Free this call's KV-cache/activation buffers back to the free-VRAM pool immediately
    # instead of letting PyTorch's caching allocator hold them as "reserved". On an 8GB
    # card with only a few hundred MB of headroom once the 4-bit model is loaded, a
    # request with a differently-shaped prompt/response than the previous one can
    # otherwise be forced to spill into Windows' slow shared/system-RAM GPU memory
    # (WDDM paging) instead of the fast dedicated VRAM -- the likely cause of "first
    # request fast, second request much slower".
    del out, inp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result



def ask_oow(mission: Mission, own: Vessel, config: str = "v3_rag_cot",
           system_prompt: str | None = None, max_new_tokens: int = 256,
           enable_thinking: bool = False, k: int = 6) -> tuple[dict, dict]:
    """Returns (decision_json, debug_info). `config` is one of MODEL_CONFIGS's keys;
    `system_prompt`, if given, overrides SYSTEM_OOW_AGENT (see build_oow_prompt).
    `max_new_tokens`/`enable_thinking` control generation speed (see _generate); `k`
    controls how many RAG chunks get injected for v1_rag/v3_rag_cot -- each retrieved
    chunk adds ~500 tokens to the PROMPT (not the response), so this is the main knob
    for why those two configs are slower to first-token than the others: a longer
    prompt costs more prefill time even though max_new_tokens/generation is unchanged."""
    # CoT configs (v2_cot/v3_rag_cot) instruct the model to "think step by step... BEFORE
    # giving your final answer", but the JSON schema's "reasoning" field is capped at 1-2
    # sentences -- with enable_thinking=False (the default, since Qwen3's native <think>
    # channel is otherwise the single biggest latency cost) there was literally nowhere for
    # that step-by-step reasoning to go. CoT configs therefore always force thinking on
    # (ignoring the caller's enable_thinking) and get extra token budget so the reasoning
    # doesn't crowd out the final JSON -- regardless of latency settings elsewhere.
    # NOTE: 512 was NOT enough -- verified on s01-s12 (busy, multi-contact scenarios) that
    # the <think> block routinely ran past 512 tokens without ever closing, truncating
    # before the JSON and falling back to the hold_course parse-error default every single
    # time (100% collision rate, not a real quality signal). 2048 gives real headroom.
    if _CONFIG_SPECS.get(config, {}).get("cot"):
        enable_thinking = True
        max_new_tokens = max(max_new_tokens, 2048)
    messages, debug = build_oow_prompt(mission, own, config=config, system_prompt=system_prompt, k=k)
    tok, mdl = _load_qwen()
    raw = _generate(tok, mdl, messages, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
    decision = _parse_json_action(raw)
    debug["raw_response"] = raw
    return decision, debug
