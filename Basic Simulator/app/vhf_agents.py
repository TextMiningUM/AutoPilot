"""VHF Communications agents: ask_vhf_qa() (Track 1 -- rules/knowledge Q&A, with a
Basic-Simulator-style prompt-ablation config matrix) and ask_vhf_comms() (Track 2 --
transmit/receive radio-call simulation).

Runs plain base Qwen3-8B, 4-bit NF4 (design_vhf_communications.md Sec 9.3 -- VHF's own
SFT/DPO/compression attempts did not work out and are dropped; better VHF models are
future work, not a prerequisite for this interface). Safe to run locally (inference
only, confirmed on an 8 GB laptop GPU for the same base model in Basic Simulator).

RETRIEVAL STACK (2026-10-03 rebuild, design_vhf_communications.md Sec 12): VHF's RAG/KG/PG
were rebuilt from scratch on the current embedder (BAAI/bge-large-en-v1.5, matching
`core.embedding.EMBEDDER_MODEL` -- VHF's corpus had drifted onto an older bge-base
(768d) embedding and was never migrated when OOW's was). The RAG corpus now also
includes a case-based "vhf_training_examples" document (see
pipeline/ingest/build_vhf_training_case_rag.py) derived from VHF's own now-unused
SFT-multihop/DPO-chosen training data -- repurposed as retrievable content per user
direction ("add the SFT and DPO and other text data we have as prompt ablation") since
actual fine-tuning on it was dropped. Three Procedural Graphs exist: `vhf_pg.json`
(merged, rule text + Track 2 conversations), `vhf_pg_rule.json` (rule text only),
`vhf_pg_conversation.json` (Track 2 conversations only) -- mirrors OOW's
merged/incident/scenario PG split.

PROMPT-ABLATION CONFIGS mirror Basic Simulator/app/agents.py's MODEL_CONFIGS/
_CONFIG_SPECS pattern (same architecture, per user direction "make everything for VHF
similar to what we did for OOW"), with one deliberate difference: NO chain-of-thought
configs ("except for the reasoning", user's explicit 2026-10-03 direction) -- no CoT
instruction anywhere in this module.

PHASE 2 (2026-10-03, design_vhf_communications.md Sec 12.7's deferred-phase list, now
partially built):
  - Flags (ICS) + light/sound signal channels: `flag_signals.json` (26 single-flag
    meanings, hand-transcribed from Wikipedia's CC BY-SA 'International maritime signal
    flags' article) and `light_sound_signals.json` (COLREG Part D Rules 32-37, copied
    from OOW's own `simple_colreg.json` -- same text, own VHF-scoped section_ids, per
    the "copy rather than build cross-domain retrieval" resolution) are now part of the
    RAG index -- Ask VHF (v1_rag/v5_rag_pg) answers flag/light-sound questions directly,
    no new ask_* function needed for this.
  - A VHF reranker (`v6_rag_rerank` config) now exists, mined from `vhf_conversation_
    traces.jsonl`'s own `regulations` field against chunks that literally cite that rule
    (`pipeline/ingest/build_vhf_reranker_pairs.py`) -- same tier of ground truth as half
    of OOW's own reranker pairs (its Track2-synthetic-scenario source), not OOW's Leo-MOOS
    code-computed half, which has no VHF equivalent.
  - OOW->Comms wiring (Sec 8.1): `ask_vhf_from_oow_decision()` below.
  - Crypto easter egg (Sec 12.7): deliberately NOT in this module -- see `app/vhf_crypto.py`
    (kept structurally separate, per the design doc's "never mistaken for real content" rule).

2026-10-03: merged into Basic Simulator's own `app` package (was a separate `VHF
Simulator/` Streamlit process) as the "Communications" page -- see
app/pages/3_Communications.py. The shared retrieval helpers imported below
(kg_retrieve/render_guidance/format_context) take paths/kg data as explicit function
parameters rather than relying on frozen import-time globals, so sharing one process
with app/agents.py (OOW) is safe.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent                # Basic Simulator/
REPO_ROOT = ROOT.parent              # Auto Pilot/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Harmless no-op if app/agents.py (OOW) already set this in the same process -- the
# shared retrieval helpers imported below don't depend on it at call time (see above).
os.environ.setdefault("AUTOPILOT_DOMAIN", "VHF")

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from core import EMBEDDER_MODEL
from core.paths import AgentPaths
from core.qwen_loader import load_qwen
from pipeline.ingest.build_kg import kg_retrieve, rerank_hits
from pipeline.ingest.pg_guidance import ProceduralGraph, render_guidance
from pipeline.eval.prep_ablation import format_context

from app.vhf_model_variants import resolve_weights, DEFAULT_VARIANT

# The RAG corpus includes a large case-based "vhf_training_examples" document (~2770
# sections derived from VHF's own now-unused SFT/DPO training data) alongside ~1100 real
# rule-text chunks -- capping how many of the k retrieved slots a single document may
# occupy prevents it from dominating retrieval regardless of real relevance, the exact
# same fix OOW needed for its own case-based "leo_moos_cases" source (see Basic
# Simulator/app/agents.py's RAG_MAX_PER_DOCUMENT docstring for the measured evidence
# this is necessary).
RAG_MAX_PER_DOCUMENT = 1

SYSTEM_VHF_QA = (
    "You are a VHF marine radio expert assisting a vessel's Auto Pilot. "
    "Answer accurately, use correct prowords, cite channel numbers, "
    "follow ITU/IMO/GMDSS regulations. Be concise."
)

# Truly bare -- no VHF framing at all, to measure how much the framing itself is worth
# (mirrors Basic Simulator/app/agents.py's BARE_SYSTEM / "bare_qwen" config).
BARE_SYSTEM = "You are a helpful assistant. Answer the question concisely."

SYSTEM_VHF_TRANSMIT = (
    "You are a VHF/GMDSS radio instructor grading a trainee operator's transmission for a "
    "given collision-avoidance scenario. Check: (1) did they hail on the correct channel "
    "and propose the correct working channel; (2) correct call format (hailing the OTHER "
    "vessel by name first, not self-hailing); (3) correct SMCP phraseology; (4) does the "
    "transmission correctly reflect the cited COLREG rule(s) WITHOUT treating VHF agreement "
    "as a substitute for the COLREG-required action. Give brief feedback, then a corrected "
    "model transmission."
)

SYSTEM_VHF_RECEIVE = (
    "You are playing the OTHER vessel/station's radio operator in a collision-avoidance "
    "VHF exchange, following correct SMCP/GMDSS procedure. Reply with ONLY the radio "
    "transmission text (no narration, no stage directions) for your side of the exchange, "
    "reacting naturally to what the trainee operator (the other party) just said."
)

# Ordered so a UI selectbox lists them in a sensible "add one ingredient at a time" order.
MODEL_CONFIGS: dict[str, str] = {
    "bare_qwen":          "Bare Qwen3-8B -- no VHF framing, no RAG/PG",
    "v0_base":            "v0 base -- VHF framing only, no retrieval",
    "v1_rag":             "v1 RAG -- + retrieved VHF corpus excerpts (dense + KG concept-boost)",
    "v2_pg_rule":         "v2 PG (rule text only) -- procedure guidance from rule-text traces",
    "v3_pg_conversation": "v3 PG (Track 2 conversations only) -- guidance from VHF dialogue traces",
    "v4_pg_merged":       "v4 PG (merged) -- guidance from rule text + conversations combined",
    "v5_rag_pg":          "v5 RAG + PG (merged) combined",
    "v6_rag_rerank":      "v6 RAG + reranker -- dense+KG pool re-scored by a fine-tuned cross-encoder (falls back to v1_rag if not trained yet)",
}

# Every entry carries the SAME keys so build_vhf_prompt() never has to .get() with a
# default -- a missing key would silently no-op that ingredient instead of raising.
_CONFIG_SPECS: dict[str, dict] = {
    "bare_qwen":          dict(bare=True,  rag=False, pg=None, rerank=False),
    "v0_base":            dict(bare=False, rag=False, pg=None, rerank=False),
    "v1_rag":             dict(bare=False, rag=True,  pg=None, rerank=False),
    "v2_pg_rule":         dict(bare=False, rag=False, pg="rule", rerank=False),
    "v3_pg_conversation": dict(bare=False, rag=False, pg="conversation", rerank=False),
    "v4_pg_merged":       dict(bare=False, rag=False, pg="merged", rerank=False),
    "v5_rag_pg":          dict(bare=False, rag=True,  pg="merged", rerank=False),
    "v6_rag_rerank":      dict(bare=False, rag=True,  pg=None, rerank=True),
}

# v6_rag_rerank: retrieve a wider pool at this size, then rerank_hits() narrows back to
# the caller's real k -- mirrors Basic Simulator/app/agents.py's v7_super_rag pattern.
RERANK_POOL_SIZE = 20


@st.cache_resource(show_spinner="Loading VHF retrieval index (RAG + KG + Procedural Graphs)...")
def _load_retrieval():
    paths = AgentPaths.vhf()
    cache = paths.cache_dir
    chunks = json.loads((cache / "vhf_rag_chunks.json").read_text(encoding="utf-8"))
    embs = np.load(cache / "vhf_rag_embeddings.npy")
    ids = json.loads((cache / "vhf_rag_chunk_ids.json").read_text(encoding="utf-8"))
    kg = json.loads((cache / "vhf_kg.json").read_text(encoding="utf-8"))
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    # CPU embedder -- it's a few hundred MB and keeps the full GPU memory free for Qwen.
    embedder = SentenceTransformer(EMBEDDER_MODEL, device="cpu")
    pg_graphs: dict[str, ProceduralGraph | None] = {}
    for key, suffix in (("merged", ""), ("rule", "_rule"), ("conversation", "_conversation")):
        pg_file = cache / f"vhf_pg{suffix}.json"
        pg_graphs[key] = ProceduralGraph(pg_file, embedder) if pg_file.exists() else None
    # v6_rag_rerank only -- see pipeline/train/train_reranker.py + build_vhf_reranker_pairs.py.
    # Also tiny (~22M params), CPU-only. None if not yet trained, so v6_rag_rerank gracefully
    # falls back to plain v1_rag-style retrieval (no reranking) rather than erroring.
    reranker_dir = paths.domain_models_dir / "vhf_reranker"
    reranker = CrossEncoder(str(reranker_dir), device="cpu") if reranker_dir.exists() else None
    return embedder, embs, ids, kg, chunk_by_id, pg_graphs, reranker


@st.cache_resource(show_spinner="Loading Qwen3-8B (4-bit NF4) onto GPU...")
def _load_qwen(weights: str = "W0_base"):
    """Thin `st.cache_resource`-wrapped call into the shared `core.qwen_loader.load_qwen()`
    -- see that module's docstring for the full `weights` syntax."""
    return load_qwen(weights, AgentPaths.vhf())


def preload(status_cb=None) -> None:
    """Warms both st.cache_resource caches so the first real click is immediate instead of
    paying the load cost then. Called once from the app's splash/startup."""
    if status_cb:
        status_cb("Loading VHF retrieval index (RAG chunks + KG + Procedural Graphs)...")
    _load_retrieval()
    if status_cb:
        status_cb("Loading Qwen3-8B (4-bit NF4) onto GPU...")
    _load_qwen(resolve_weights(DEFAULT_VARIANT))


def unload() -> None:
    """Drops both st.cache_resource caches and releases their GPU/CPU memory."""
    import gc
    _load_qwen.clear()
    _load_retrieval.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def _generate(tok, mdl, messages: list[dict], max_new_tokens: int = 400) -> str:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
    out = mdl.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id,
                       repetition_penalty=1.15)
    result = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    del out, inp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result.strip()


def build_vhf_prompt(question: str, config: str = "v1_rag", k: int = 6, dense_n: int = 40,
                     system_prompt: str | None = None) -> tuple[list[dict], dict]:
    """Builds the (system, user) messages for `config` (one of MODEL_CONFIGS's keys) --
    see module docstring for what each config adds. Returns (messages, debug_info);
    debug_info carries "sources" (retrieved chunk_ids, if rag) and "pg_used" (if pg)."""
    if config not in _CONFIG_SPECS:
        raise KeyError(f"Unknown config {config!r} -- valid options: {sorted(_CONFIG_SPECS)}")
    spec = _CONFIG_SPECS[config]
    debug: dict = {"config": config}

    if spec["bare"]:
        return [{"role": "system", "content": BARE_SYSTEM},
                {"role": "user", "content": question}], debug

    system = system_prompt or SYSTEM_VHF_QA
    parts: list[str] = []

    if spec["rag"] or spec["pg"]:
        embedder, embs, ids, kg, chunk_by_id, pg_graphs, reranker = _load_retrieval()
        if spec["rag"]:
            if spec["rerank"] and reranker is not None:
                pool, _, _ = kg_retrieve(question, embedder, embs, ids, kg, k=RERANK_POOL_SIZE,
                                         dense_n=max(dense_n, RERANK_POOL_SIZE * 2),
                                         max_per_document=RAG_MAX_PER_DOCUMENT)
                hits = rerank_hits(question, pool, chunk_by_id, reranker, k=k)
                debug["reranked"] = True
            else:
                hits, _, _ = kg_retrieve(question, embedder, embs, ids, kg, k=k, dense_n=dense_n,
                                         max_per_document=RAG_MAX_PER_DOCUMENT)
                debug["reranked"] = False
                if spec["rerank"]:
                    debug["rerank_unavailable"] = True  # requested but vhf_reranker not trained yet
            parts.append(f"Reference excerpts:\n{format_context(hits, chunk_by_id)}")
            debug["sources"] = [h["chunk_id"] for h in hits]
        if spec["pg"]:
            graph = pg_graphs.get(spec["pg"])
            if graph is not None:
                guidance = render_guidance(question, graph)
                if guidance:
                    parts.append(guidance)
                debug["pg_used"] = spec["pg"]
            else:
                debug["pg_used"] = None  # graph file not built yet on this machine

    parts.append(f"Question: {question}")
    messages = [{"role": "system", "content": system},
               {"role": "user", "content": "\n\n".join(parts)}]
    return messages, debug


def ask_vhf_qa(question: str, config: str = "v1_rag", model: str = DEFAULT_VARIANT,
              k: int = 6, dense_n: int = 40, max_new_tokens: int = 400) -> dict:
    """Track 1: answer `question` under prompt-ablation `config` (see MODEL_CONFIGS).
    Returns {"answer", "config", "sources": [chunk_id, ...] | None, "pg_used": str | None}."""
    messages, debug = build_vhf_prompt(question, config=config, k=k, dense_n=dense_n)
    tok, mdl = _load_qwen(resolve_weights(model))
    answer = _generate(tok, mdl, messages, max_new_tokens=max_new_tokens)
    return {"answer": answer, **debug}


def load_scenarios() -> list[dict]:
    """The held-out Track 2 scenario bank (`vhf_colreg_scenarios.json`), reused here purely
    as a scenario SOURCE for the interactive demo -- nothing is trained on these rows, they
    stay held-out for eval exactly as before (design_vhf_communications.md Sec 12.3)."""
    paths = AgentPaths.vhf()
    return json.loads(paths.eval_file("vhf_colreg_scenarios.json").read_text(encoding="utf-8"))


def _scenario_brief(scenario: dict) -> str:
    ch = scenario.get("vhf_channel", {})
    rules = ", ".join(scenario.get("colreg_rules", []))
    return (
        f"Region: {scenario.get('region', '?')}\n"
        f"Own vessel: {scenario.get('own_vessel', '?')}\n"
        f"Target vessel: {scenario.get('target_vessel', '?')}\n"
        f"Situation: {scenario.get('scenario', '?')}\n"
        f"Correct hailing channel: {ch.get('hailing', '?')}, working channel: {ch.get('working', '?')}\n"
        f"Applicable COLREG rule(s): {rules}"
    )


def scenario_playback(scenario: dict) -> dict:
    """Read-only narrative preview of one scenario (design_vhf_communications.md
    Sec 12.11) -- region/vessels/situation/rules/channel plus the held-out gold
    reference transmission, for a user to review what's in a scenario BEFORE engaging
    in Transmit/Receive mode. Pure formatting, no model call -- same "fine to use
    held-out eval data as a DEMO source" precedent as load_scenarios()/_scenario_brief()."""
    ch = scenario.get("vhf_channel", {})
    return {
        "region": scenario.get("region", "?"),
        "category": scenario.get("category", "?"),
        "own_vessel": scenario.get("own_vessel", "?"),
        "target_vessel": scenario.get("target_vessel", "?"),
        "situation": scenario.get("scenario", "?"),
        "colreg_rules": scenario.get("colreg_rules", []),
        "channel_hailing": ch.get("hailing", "?"),
        "channel_working": ch.get("working", "?"),
        "channel_note": ch.get("note", ""),
        "channel_settings": ch.get("settings", ""),
        "reference_transmission": scenario.get("gold_answer"),
        "expected_points": scenario.get("expected_points", []),
    }


def ask_vhf_comms(scenario: dict, mode: str, user_text: str | None = None,
                 history: list[dict] | None = None, model: str = DEFAULT_VARIANT,
                 max_new_tokens: int = 300) -> dict:
    """Track 2: the "zenden/ontvangen" (transmit/receive) radio simulation.

    mode="transmit": grades the operator's own drafted call (`user_text`, required)
    against the scenario's channel/COLREG facts, returns {"feedback"}.

    mode="receive": the model plays the OTHER station. First call (history=None/empty)
    returns its opening hail; each subsequent call appends `user_text` (the operator's
    reply) to `history` and returns the model's next in-character turn. Returns
    {"message", "history"} -- `history` is a list of {"role", "content"} dicts to pass
    back in on the next turn (the same list this call's transmission-log panel renders).
    """
    if mode not in ("transmit", "receive"):
        raise ValueError(f"Unknown mode {mode!r} -- must be 'transmit' or 'receive'")
    tok, mdl = _load_qwen(resolve_weights(model))
    brief = _scenario_brief(scenario)

    if mode == "transmit":
        if not user_text:
            raise ValueError("mode='transmit' requires user_text (the operator's drafted call)")
        messages = [
            {"role": "system", "content": SYSTEM_VHF_TRANSMIT},
            {"role": "user", "content": f"Scenario:\n{brief}\n\nThe operator transmitted:\n\"{user_text}\""},
        ]
        feedback = _generate(tok, mdl, messages, max_new_tokens=max_new_tokens)
        return {"feedback": feedback}

    if mode == "receive":
        history = list(history or [])
        if not history:
            messages = [
                {"role": "system", "content": SYSTEM_VHF_RECEIVE},
                {"role": "user", "content": f"Scenario:\n{brief}\n\nIssue your opening VHF hail."},
            ]
            reply = _generate(tok, mdl, messages, max_new_tokens=max_new_tokens)
            new_history = [{"role": "assistant", "content": reply}]
            return {"message": reply, "history": new_history}
        if not user_text:
            raise ValueError("mode='receive' with existing history requires user_text (the operator's reply)")
        messages = ([{"role": "system", "content": f"{SYSTEM_VHF_RECEIVE}\n\nScenario:\n{brief}"}]
                   + history + [{"role": "user", "content": user_text}])
        reply = _generate(tok, mdl, messages, max_new_tokens=max_new_tokens)
        new_history = history + [{"role": "user", "content": user_text},
                                 {"role": "assistant", "content": reply}]
        return {"message": reply, "history": new_history}


SYSTEM_VHF_OOW_COMMS = (
    "You are the VHF/Comms operator confirming an already-decided manoeuvre from the "
    "Officer of the Watch (OOW). Draft the radio call for this manoeuvre. You NEVER decide "
    "the manoeuvre yourself and you NEVER imply the manoeuvre depends on the other vessel's "
    "agreement -- it has already been decided for COLREG compliance; your job is only to "
    "communicate it clearly using correct SMCP phraseology and the correct channel. Reply "
    "with ONLY the radio transmission text."
)


def ask_vhf_from_oow_decision(situation: str, oow_decision: dict, model: str = DEFAULT_VARIANT,
                              max_new_tokens: int = 300) -> dict:
    """OOW -> Comms protocol (design_vhf_communications.md Sec 8.1): VHF never decides the
    manoeuvre, only drafts the radio call confirming an OOW decision already made.
    Promotes the `ask_vhf(oow_decision)` prototype from `Basic Simulator/Brain Storming/
    pilot_agents.ipynb` (which called the Anthropic API) to the local base/fine-tuned model.

    `situation` is a short plain-text description of the encounter (own vessel/target
    vessel/geometry -- the same kind of text `narrate()` produces in Basic Simulator),
    passed in here as plain text rather than importing app.narrate directly -- the OOW/
    Comms boundary stays a plain data contract (situation text + decision dict) even
    though both modules now live in the same `app` package, since VHF should never need
    to decide/compute anything from raw mission state itself.
    `oow_decision` is the OOW agent's own decision JSON shape (pipeline/oow_agent_spec.py):
    {"action", "degrees", "encounter_rule", "conduct_rule", "reasoning"}.
    Returns {"transmission": str}.
    """
    tok, mdl = _load_qwen(resolve_weights(model))
    messages = [
        {"role": "system", "content": SYSTEM_VHF_OOW_COMMS},
        {"role": "user", "content": (
            f"Situation:\n{situation}\n\nOOW decision: {json.dumps(oow_decision)}\n\n"
            "Draft the radio call for this manoeuvre."
        )},
    ]
    transmission = _generate(tok, mdl, messages, max_new_tokens=max_new_tokens)
    return {"transmission": transmission}
