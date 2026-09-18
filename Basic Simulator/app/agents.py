"""ask_oow(): the OOW navigation agent.

Runs base Qwen3-8B (the fine-tuned OOW-QWEN checkpoint isn't trained yet --
see repo memory notes) with ONE combined prompt drawing on every context
source the ablation study (§11) found promising: RAG-retrieved COLREG
excerpts + a chain-of-thought instruction + Procedural-Graph guidance from
the merged oow_pg.json. This deliberately differs from the ablation STUDY
(which isolates each source to measure its individual contribution) -- for
the deployed agent there's no reason to leave useful context out.
"""
from __future__ import annotations
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

SYSTEM_OOW_AGENT = """You are the Officer of the Watch, an AI navigation agent responsible for \
COLREG-compliant collision avoidance. Think step by step through the encounter, the applicable \
COLREG rule(s), and the give-way/stand-on obligations, grounding your reasoning in the provided \
COLREG excerpts and procedure guidance where given. Rules of thumb: give-way vessels alter early \
and substantially, normally to starboard; stand-on vessels hold course/speed unless the other \
vessel clearly isn't keeping clear; head-on situations mean both vessels alter to starboard; never \
recommend altering to port toward a vessel that is itself on your port side while avoiding collision; \
if no target poses a real risk of collision, recommend holding course. \
Reply with ONLY a JSON object, no other text:
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": <float, only for turn_left/turn_right>,
 "rule_applied": "<e.g. Rule 15, or 'none' if no rule applies>",
 "reasoning": "<one or two sentences>"}"""


@st.cache_resource(show_spinner="Loading OOW retrieval index (RAG + Procedural Graph)...")
def _load_retrieval():
    paths = AgentPaths.oow()
    cache = paths.cache_dir
    pfx = paths.domain.lower()
    chunks = json.loads((cache / f"{pfx}_rag_chunks.json").read_text(encoding="utf-8"))
    embs = np.load(cache / f"{pfx}_rag_embeddings.npy")
    ids = json.loads((cache / f"{pfx}_rag_chunk_ids.json").read_text(encoding="utf-8"))
    kg = json.loads((cache / f"{pfx}_kg.json").read_text(encoding="utf-8"))
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    pg_file = cache / f"{pfx}_pg.json"
    graph = ProceduralGraph(pg_file, embedder) if pg_file.exists() else None
    return embedder, embs, ids, kg, chunk_by_id, graph


@st.cache_resource(show_spinner="Loading Qwen3-8B (4-bit NF4) -- first call only, ~1-2 min...")
def _load_qwen():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    mdl.eval()
    return tok, mdl


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


def build_oow_prompt(mission: Mission, own: Vessel, k: int = 6, dense_n: int = 40) -> tuple[list[dict], dict]:
    """Returns (messages, debug_info). debug_info exposes what was retrieved,
    for the Agents tab so a user can see WHY the agent decided what it decided."""
    embedder, embs, ids, kg, chunk_by_id, graph = _load_retrieval()
    situation = narrate(mission, own)

    hits, q_cons, expanded = kg_retrieve(situation, embedder, embs, ids, kg, k=k, dense_n=dense_n)
    ctx = format_context(hits, chunk_by_id)
    pg_text = render_guidance(situation, graph) if graph is not None else None

    user_parts = []
    if pg_text:
        user_parts.append(f"Procedure guidance:\n{pg_text}")
    user_parts.append(f"COLREG reference excerpts:\n\n{ctx}")
    user_parts.append(f"Situation:\n{situation}")
    user_parts.append("Recommend exactly ONE manoeuvre as the specified JSON object.")
    user_msg = "\n\n".join(user_parts)

    messages = [{"role": "system", "content": SYSTEM_OOW_AGENT},
                {"role": "user", "content": user_msg}]
    debug = {
        "situation": situation,
        "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
        "query_concepts": q_cons, "expanded_concepts": expanded,
        "pg_guidance": pg_text,
    }
    return messages, debug


@torch.inference_mode()
def _generate(tok, mdl, messages: list[dict], max_new_tokens: int = 400) -> str:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
    out = mdl.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def ask_oow(mission: Mission, own: Vessel) -> tuple[dict, dict]:
    """Returns (decision_json, debug_info)."""
    messages, debug = build_oow_prompt(mission, own)
    tok, mdl = _load_qwen()
    raw = _generate(tok, mdl, messages)
    decision = _parse_json_action(raw)
    debug["raw_response"] = raw
    return decision, debug
