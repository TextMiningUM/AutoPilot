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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

from core import AgentPaths, EMBEDDER_MODEL
from pipeline.ingest.build_kg import kg_retrieve
from pipeline.ingest.pg_guidance import ProceduralGraph, render_guidance
from pipeline.eval.prep_ablation import format_context

from app.missions import Mission, Vessel
from app.narrate import contact_line, narrate
from app.simulation import VesselConstraints

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
SYSTEM_OOW_AGENT = """You are the navigator on a large commercial vessel. Your objective is to \
get the ship to the mission goal (position/bearing/distance given in the situation report) as \
directly and efficiently as possible -- that is the task. The one hard constraint on every \
decision, with no exceptions: never put the ship on a collision course with another vessel. \
When a real risk of collision exists, COLREG (the International Regulations for Preventing \
Collisions at Sea) governs which vessel gives way and how -- satisfy that constraint first, \
then resume progress toward the goal. All positions/bearings/headings given below are in \
metres and degrees, heading 0=north, clockwise, matching compass bearings -- take this as \
given, don't re-derive or second-guess it. rel.bearing is signed: positive = target is to \
starboard (right), negative = to port (left), 0=dead ahead, ~180/-180=dead astern. CPA is the \
closest distance the contact will EVER come to you at current headings/speeds; TCPA is the \
seconds until that closest point. TCPA=0 does NOT always mean a collision is imminent -- it \
also occurs when the vessels are already moving apart (closest point already passed); the \
situation report says so explicitly when that's the case. Judge real risk from the CPA \
distance itself, not from TCPA alone. DEFAULT PROCEDURE, apply this every single time: the \
situation report has a line starting "GOAL COURSE CHECK:" -- use ONLY that line to decide the \
goal-correction action and degrees, never a contact's rel.bearing (a contact's rel.bearing is \
about THAT CONTACT, not the goal, even if the numbers look similar). If "GOAL COURSE CHECK" \
says you're already on the goal bearing, hold_course (for the goal, at least). Otherwise it \
gives you the exact action ("turn_left"/"turn_right") and degrees to use -- copy those values \
directly into your answer, don't recompute them and don't substitute a different number from \
elsewhere in the report. Your track to the goal must look like a smooth curve or a straight \
line, NEVER a zigzag -- do not answer turn_right and then turn_left (or vice versa) on \
consecutive decisions just to chase a small residual mismatch; "GOAL COURSE CHECK" already \
has a 10-degree deadband built in for this. Only override any of this if a target poses a REAL \
risk of collision, in which case the COLREG-required give-way/stand-on manoeuvre takes \
precedence instead. Do not just describe the mismatch in your reasoning and then answer \
hold_course anyway when "GOAL COURSE CHECK" calls for a turn -- the action MUST match what \
that line says, never hold_course when it names a turn. Ground your reasoning in the provided \
COLREG excerpts and procedure guidance where given. If you are already on the goal bearing \
(within about 10 degrees) and your current speed is below the nominal/rated speed given for \
this mission, speed_up instead: reaching the goal sooner (when safe) is part of efficient \
progress too. What order do you give to the helm? Reply with ONLY a JSON object, no other \
text:
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": <float, only for turn_left/turn_right>,
 "rule_applied": "<e.g. Rule 15, or 'none' if no rule applies>",
 "reasoning": "<one or two sentences>"}"""

COT_INSTR = ("Think step by step through the encounter, the applicable COLREG rule(s), and the "
            "give-way/stand-on obligations BEFORE giving your final answer. Keep this reasoning "
            "BRIEF -- 3 to 5 short sentences covering only: the encounter type, the applicable "
            "rule (if any), and why the chosen action resolves it. Take the given coordinates/ "
            "bearings/heading convention as fact -- do not re-derive or second-guess basic "
            "geometry already stated in the situation report.")


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


def _extract_json_objects(text: str) -> list[str]:
    """Balanced-brace scan for every top-level {...} object in `text`, in order of
    appearance -- unlike a single greedy `\\{.*\\}` regex, this doesn't stitch multiple
    separate objects (and whatever sits between them) into one invalid blob."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None
    return objs


def _parse_json_action(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # Prefer the LAST complete {...} object that actually parses as an action dict -- Qwen3
    # sometimes echoes the JSON once before </think> closes and once again after (a
    # duplicate-answer pattern seen in the sweep logs), which the previous single greedy
    # first-{-to-last-} match stitched into one invalid blob spanning both copies.
    for candidate in reversed(_extract_json_objects(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed
    return {"action": "hold_course", "rule_applied": "none",
            "reasoning": f"[parse error -- raw model output] {text[:300]}", "_parse_error": True}


def _pg_match_query(mission: Mission, own: Vessel) -> str:
    """Short COLREG-encounter phrase used ONLY to retrieve procedural-graph guidance --
    never shown to the model in place of the real situation report. render_guidance()'s
    node/family matching is embedding-similarity based against a graph built from
    incident-report/Track2 reasoning traces ("crossing situation", "give-way vessel",
    "Rule 15", ...); narrate()'s situation report is deliberately numeric/geometric
    (headings, CPA/TCPA) with NO rule/encounter words at all, so as not to leak the
    answer to the model -- but that same omission also made every PG anchor/family
    match fail and fall back to one generic default (verified: identical pg_guidance
    text for all 14 missions before this fix). classify_encounter()/contact_line()
    already compute the real encounter type + rule(s) per contact for the Evaluation
    panel; reusing that here (not shown to the model) is exactly the retrieval-query
    role RAG's own query already plays."""
    if not mission.targets:
        return "routine passage, no other traffic, no close-quarters encounter"
    contacts = [contact_line(own, t) for t in mission.targets]
    live = [c for c in contacts if not c["quiet"]]
    if not live:
        return "routine passage, no close-quarters encounter, maintain course and speed"
    pick = min(live, key=lambda c: c["cpa_m"])
    enc = pick["encounter"].replace("_", " ")
    rules = " and ".join(pick["rules"]) if pick["rules"] else "none"
    return f"{enc} encounter, applicable {rules}, give-way/stand-on obligations"


def build_oow_prompt(mission: Mission, own: Vessel, config: str = "v3_rag_cot",
                     system_prompt: str | None = None,
                     k: int = 6, dense_n: int = 40,
                     constraints: VesselConstraints | None = None) -> tuple[list[dict], dict]:
    """Returns (messages, debug_info) for the selected MODEL_CONFIGS key.
    `system_prompt`, if given, overrides SYSTEM_OOW_AGENT for every config except
    bare_qwen (which always keeps its own separate, deliberately minimal prompt as the
    true ablation baseline). debug_info exposes what was retrieved, for the Agent panel
    so a user can see WHY the agent decided what it decided.
    `constraints`, if given (the live simulator's VesselConstraints -- see
    app/simulation.py), tells the agent own-ship's ACTUAL physical envelope so it doesn't
    recommend something the kinematics layer can't deliver: turn_rate_deg_s and
    max_rudder_angle_deg (both hard-enforced -- a single turn command beyond the latter is
    silently capped), max_speed_mps (a hard ceiling -- speed_up has no effect once already there) and
    max_acceleration_mps2/max_deceleration_mps2 (speed changes gradually, not instantly),
    and min_cpa_m (this mission's configured safe-passing distance -- without this the
    model has NO numeric anchor for what counts as a real collision risk; observed
    v0_base/v1_rag calling a 17m CPA "safe" and colliding as a direct result).
    cruise_speed_mps is threaded into narrate()'s nominal/rated-speed reference instead of
    a separate line here (see app/narrate.py). Never added for bare_qwen -- that config is
    the deliberate zero-extra-framing ablation floor."""
    if config not in _CONFIG_SPECS:
        raise ValueError(f"Unknown model config {config!r}; choose one of {list(MODEL_CONFIGS)}")
    spec = _CONFIG_SPECS[config]
    situation = narrate(mission, own, cruise_speed_mps=constraints.cruise_speed_mps if constraints else None)

    if spec["bare"]:
        user_msg = f"Situation:\n{situation}\n\nRecommend exactly ONE manoeuvre as the specified JSON object."
        messages = [{"role": "system", "content": BARE_SYSTEM},
                    {"role": "user", "content": user_msg}]
        debug = {"situation": situation, "config": config, "retrieved_chunk_ids": [],
                 "query_concepts": [], "expanded_concepts": [], "pg_guidance": None,
                 "user_msg_chars": len(user_msg), "user_msg": user_msg}
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
            pg_text = render_guidance(_pg_match_query(mission, own), graph)

    user_parts = []
    if spec["cot"]:
        user_parts.append(COT_INSTR)
    if constraints is not None:
        per_step = constraints.turn_rate_deg_s * constraints.time_step_s
        user_parts.append(
            f"Own-ship's physical limits: heading changes at most {constraints.turn_rate_deg_s:.1f} "
            f"deg/s (~{per_step:.0f} deg per {constraints.time_step_s:.0f}s step). A single "
            f"turn_left/turn_right command can request AT MOST {constraints.max_rudder_angle_deg:.0f} "
            "degrees -- a larger request will be silently capped, so a course change bigger than that "
            "needs several separate turn commands across multiple steps, not one big one. "
            f"Speed is capped at {constraints.max_speed_mps:.1f} m/s, changing gradually "
            f"({constraints.max_acceleration_mps2:.2f} m/s\u00b2 up / {constraints.max_deceleration_mps2:.2f} "
            "m/s\u00b2 down) -- speed_up/slow_down are not instant. This mission's safe passing distance is "
            f"{constraints.min_cpa_m:.0f}m: CPA below that is a real collision risk, CPA well above it is "
            "safe regardless of how small it looks."
        )
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
        "pg_guidance": pg_text, "user_msg_chars": len(user_msg), "user_msg": user_msg,
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
    # Stop as soon as the JSON object closes instead of always burning the full
    # max_new_tokens budget -- the schema always ends with `"reasoning": "...text"}`, so
    # the literal `"}` tail is a safe, specific stop signal (unlike a bare "}", which could
    # falsely trigger on a stray brace inside a CoT config's free-form <think> reasoning).
    # Previously EVERY call ran to max_new_tokens even when the model finished in a
    # fraction of that, which is most of them (CoT configs get 2048 tokens of headroom
    # for the rare long <think> block, but the median response is much shorter -- see
    # basic_simulator.md's raw_len_chars stats).
    # repetition_penalty/no_repeat_ngram_size: greedy decoding (do_sample=False) has no
    # built-in defense against looping -- found CoT/PG configs repeating the EXACT same
    # "Rule 15 says..." sentence 20+ times verbatim (never self-correcting, never closing
    # the JSON) until max_new_tokens ran out, causing most of the parse-error/hold_course
    # fallbacks seen in the sweep. no_repeat_ngram_size=4 hard-blocks any 4-token sequence
    # from repeating at all -- enough to break a whole-sentence loop like that one, too
    # short to block legitimate short repeats (e.g. saying "Rule 15" twice in one reply).
    # AUTOPILOT_STREAM=1 prints tokens to stdout live as they're generated (via
    # transformers' TextStreamer) -- opt-in only, for watching a slow/long-running CLI
    # sweep (tail -f the log) to see the actual <think> reasoning as it happens instead
    # of waiting minutes for the whole response with no visibility into what it's doing.
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) \
        if os.environ.get("AUTOPILOT_STREAM") else None
    out = mdl.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id,
                       stop_strings="\"}", tokenizer=tok,
                       repetition_penalty=1.15, no_repeat_ngram_size=4,
                       streamer=streamer)
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



def effective_generation_params(config: str, enable_thinking: bool, max_new_tokens: int) -> tuple[bool, int]:
    """CoT configs (v2_cot/v3_rag_cot) AND PG configs (v4_pg/v5_pg_incident/v6_pg_scenario)
    force enable_thinking=True + a larger token budget inside ask_oow() regardless of what's
    passed in -- PG guidance is a dense, structured procedure block the model needs the same
    reasoning room to weigh (accept/adapt/ignore) as CoT's own step-by-step instruction; with
    thinking off it was just being ignored outright (verified: v4/v5/v6_pg decisions were
    near-identical to v0_base's, same failures). Callers that log/display `params`
    (e.g. run_llm_scenario.py) should call this instead of logging their own raw pre-override
    values, which previously showed "Thinking: off" even on runs where it was actually forced on."""
    spec = _CONFIG_SPECS.get(config, {})
    if spec.get("cot") or spec.get("pg"):
        return True, max(max_new_tokens, 3072)
    return enable_thinking, max_new_tokens


def ask_oow(mission: Mission, own: Vessel, config: str = "v3_rag_cot",
           system_prompt: str | None = None, max_new_tokens: int = 256,
           enable_thinking: bool = False, k: int = 6,
           constraints: VesselConstraints | None = None) -> tuple[dict, dict]:
    """Returns (decision_json, debug_info). `config` is one of MODEL_CONFIGS's keys;
    `system_prompt`, if given, overrides SYSTEM_OOW_AGENT (see build_oow_prompt).
    `max_new_tokens`/`enable_thinking` control generation speed (see _generate); `k`
    controls how many RAG chunks get injected for v1_rag/v3_rag_cot -- each retrieved
    chunk adds ~500 tokens to the PROMPT (not the response), so this is the main knob
    for why those two configs are slower to first-token than the others: a longer
    prompt costs more prefill time even though max_new_tokens/generation is unchanged.
    `constraints`, if given, is forwarded to build_oow_prompt() -- see its docstring."""
    # CoT configs (v2_cot/v3_rag_cot) instruct the model to "think step by step... BEFORE
    # giving your final answer", but the JSON schema's "reasoning" field is capped at 1-2
    # sentences -- with enable_thinking=False (the default, since Qwen3's native <think>
    # channel is otherwise the single biggest latency cost) there was literally nowhere for
    # that step-by-step reasoning to go. CoT AND PG configs therefore always force thinking on
    # (ignoring the caller's enable_thinking) and get extra token budget so the reasoning
    # doesn't crowd out the final JSON -- regardless of latency settings elsewhere.
    # NOTE: 512 was NOT enough -- verified on s01-s12 (busy, multi-contact scenarios) that
    # the <think> block routinely ran past 512 tokens without ever closing, truncating
    # before the JSON and falling back to the hold_course parse-error default every single
    # time (100% collision rate, not a real quality signal). 2048 gave real headroom until
    # the physical-limits paragraph (see build_oow_prompt) gave the model more to reason
    # about and pushed some generations past 2048 too (same parse-error/hold_course failure
    # mode, reproduced on s01_head_on/v2_cot) -- bumped to 3072.
    enable_thinking, max_new_tokens = effective_generation_params(config, enable_thinking, max_new_tokens)
    messages, debug = build_oow_prompt(mission, own, config=config, system_prompt=system_prompt, k=k,
                                       constraints=constraints)
    tok, mdl = _load_qwen()
    raw = _generate(tok, mdl, messages, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
    decision = _parse_json_action(raw)
    debug["raw_response"] = raw
    return decision, debug
