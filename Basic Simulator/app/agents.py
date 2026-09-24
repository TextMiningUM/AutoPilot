"""ask_oow(): the OOW navigation agent.

Runs base Qwen3-8B (the merged SFT+DPO+Reflection OOW-QWEN checkpoint exists and has been
evaluated -- see eval_oow_qwen_full_summary.json -- but currently scores WORSE than base on
Track 1 gold Q&A, Composite 0.375 vs 0.551; not yet wired into this app pending that
regression being understood, see repo memory notes) under one of the selectable prompt
configurations that mirror the notebook's §11 prompt ablation study (prep_ablation.py's
V0-V6), plus an extra 'bare' baseline and 3 "super" configs (v7-v9) that combine the full
retrieval stack and CoT+PG guidance into the experiment matrix's actual prompt columns:

  bare_qwen      -- minimal system prompt, no COLREG framing, no RAG/CoT/PG
  v0_base        -- OOW framing (rules-of-thumb + JSON contract), no extras
  v1_rag         -- + retrieved COLREG excerpts
  v2_cot         -- + chain-of-thought instruction
  v3_rag_cot     -- both combined
  v4_pg          -- + procedural-graph guidance (merged, all sources)
  v5_pg_incident -- + procedural-graph guidance (real incidents only)
  v6_pg_scenario -- + procedural-graph guidance (Track 2 scenarios only)
  v1_rag..v6_pg_scenario are an archived ablation arm (147 runs depend on them staying
  reproducible) -- no longer in the default sweep, never redefine.

  v7_super_rag     -- the full retrieval stack on corpus v2: dense RAG + reranker + KG,
                      no CoT, no PG (kept fast, v0_base-like latency)
  v8_super_cot_pg  -- chain-of-thought + procedure guidance from BOTH the rebuilt
                      scenario and incident procedural graphs, no retrieval
  v9_super_all     -- v7_super_rag + v8_super_cot_pg combined
These three are the actual experiment-matrix prompt columns (P1/P2/P3); the earlier
v7_rerank/v8_rerank_cot/v9_fewshot/v10_dpo_contrast/v11_reflect prototype slots were
never run (no result files ever used those names) and were redefined in place
2026-09-22 rather than left as dead placeholders.

As in prep_ablation.py's Track 2 design, the response-format JSON contract
never changes across v0-v9 -- only the USER turn gains CoT/RAG/PG content --
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

from sentence_transformers import CrossEncoder

from core import AgentPaths, EMBEDDER_MODEL
from pipeline.ingest.build_kg import kg_retrieve, rerank_hits
from pipeline.ingest.pg_guidance import ProceduralGraph, render_guidance, load_merged_pg
from pipeline.eval.prep_ablation import format_context
from pipeline.oow_agent_spec import (
    SYSTEM_OOW_AGENT, ACTIONS, validate_action_json, derive_risk_horizon_s, constraint_line,
    render_previous_decisions,
)

from app.missions import Mission, Vessel
from app.narrate import contact_line, narrate
from app.simulation import VesselConstraints
from app.units import mps_to_kn

MODEL_ID = "Qwen/Qwen3-8B"

# `leo_moos_cases` (case-based RAG, numeric/geometric MOOS-narrative style -- same
# vocabulary as a live situation-report query) otherwise dominates dense retrieval
# regardless of real relevance; see kg_retrieve()'s max_per_document docstring
# (pipeline/ingest/build_kg.py) for the measured evidence. Caps pool COMPOSITION only,
# never the query text (that would fight the reranker's own training distribution).
# MUST be < the smallest `k` actually used in production (sweep_llm_params.py's own
# --k default is 2) -- a cap >= k never engages at all, since it only limits EXCESS
# beyond the cap, not which candidates win the top-k positions by raw score (moos_cases
# chunks still legitimately score highest by raw cosine similarity; the cap's only job
# is to guarantee at least (k - max_per_document) slots for something else). Verified
# empirically: max_per_document=3 with the real production k=2 changed nothing (moos
# chunks still filled both slots); max_per_document=1 is the correct, effective value.
RAG_MAX_PER_DOCUMENT = 1

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
    "v7_super_rag":    "v7 super RAG -- full retrieval stack on corpus v2 (dense + reranker + KG), no CoT/PG",
    "v8_super_cot_pg": "v8 super CoT+PG -- CoT + procedure guidance from scenario+incident graphs, no retrieval",
    "v9_super_all":    "v9 super all -- v7_super_rag + v8_super_cot_pg combined",
}

# Every entry carries the SAME keys so build_oow_prompt() never has to .get() with a
# default -- a missing key would silently no-op that ingredient instead of raising.
_CONFIG_SPECS = {
    "bare_qwen":      dict(bare=True,  rag=False, rerank=False, cot=False, pg=None),
    "v0_base":        dict(bare=False, rag=False, rerank=False, cot=False, pg=None),
    # v1_rag..v6_pg_scenario: archived ablation arm, no longer in the standard sweep;
    # 147 archived runs depend on these exact definitions -- never redefine.
    "v1_rag":         dict(bare=False, rag=True,  rerank=False, cot=False, pg=None),
    "v2_cot":         dict(bare=False, rag=False, rerank=False, cot=True,  pg=None),
    "v3_rag_cot":     dict(bare=False, rag=True,  rerank=False, cot=True,  pg=None),
    "v4_pg":          dict(bare=False, rag=False, rerank=False, cot=False, pg="merged"),
    "v5_pg_incident": dict(bare=False, rag=False, rerank=False, cot=False, pg="incident"),
    "v6_pg_scenario": dict(bare=False, rag=False, rerank=False, cot=False, pg="scenario"),
    # The actual experiment-matrix prompt columns (2026-09-22, redefined in place over the
    # never-run v7_rerank/v8_rerank_cot/v9_fewshot/v10_dpo_contrast/v11_reflect prototype
    # slots -- confirmed via grep that zero result files ever used those 5 names).
    "v7_super_rag":    dict(bare=False, rag=True,  rerank=True,  cot=False, pg=None),
    "v8_super_cot_pg": dict(bare=False, rag=False, rerank=False, cot=True,  pg="scenario+incident"),
    "v9_super_all":    dict(bare=False, rag=True,  rerank=True,  cot=True,  pg="scenario+incident"),
}

# Truly bare -- no COLREG rules-of-thumb, just told to answer in the required JSON shape.
BARE_SYSTEM = """You are an AI assistant helping a ship's navigation system decide on a heading/speed \
change. Reply with ONLY a JSON object, no other text:
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": <float, only for turn_left/turn_right>,
 "encounter_rule": "<'Rule <N>' citing the applicable COLREG rule, or 'none' if no contact poses real risk>",
 "conduct_rule": "<'Rule <N>' citing the applicable COLREG rule, or 'none' if no real risk>",
 "reasoning": "<one or two sentences>"}"""

# SYSTEM_OOW_AGENT (the v0-v9 system prompt) lives in pipeline/oow_agent_spec.py -- the
# single source of truth shared with the Track-2 training-data generators (Fase B2, RAG-
# rebuild-v2 plan), so train and eval can never silently drift onto different task
# formats. Editable at runtime from the sidebar's "System prompt" popover
# (streamlit_app.py) -- the override is passed in as build_oow_prompt's `system_prompt`
# arg and replaces this default for every config EXCEPT bare_qwen.

COT_INSTR = (
    "Before answering, write your reasoning as EXACTLY these 4 steps, one short sentence each -- "
    "no more steps, no re-deriving bearings/CPA/TCPA (they are already given -- just quote them):\n"
    "1. Contacts: name each contact and say whether its CPA is below or above the safe passing "
    "distance.\n"
    "2. Rule: for any contact below it, name the applicable COLREG rule and the action it requires.\n"
    "3. Goal: if no contact is below the safe passing distance, state what GOAL COURSE CHECK says.\n"
    "4. Decision: state the one action you will take and why.\n"
    "Then give the final JSON answer."
)


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
    # v8_super_cot_pg/v9_super_all's pg="scenario+incident" -- guidance from BOTH graphs
    # together, deliberately excluding oow_pg_rule.json (a separate, optional test arm).
    scenario_file, incident_file = cache / f"{pfx}_pg_scenario.json", cache / f"{pfx}_pg_incident.json"
    pg_graphs["scenario+incident"] = (
        load_merged_pg([scenario_file, incident_file], embedder)
        if scenario_file.exists() and incident_file.exists() else None
    )
    # v7_super_rag/v9_super_all only -- see pipeline/train/train_reranker.py. Also tiny (~22M
    # params), CPU-only, same rationale as `embedder` above. None if not yet trained, so
    # every OTHER config keeps working even before this prototype exists on a given machine.
    reranker_dir = paths.domain_models_dir / "oow_reranker"
    reranker = CrossEncoder(str(reranker_dir), device="cpu") if reranker_dir.exists() else None
    return embedder, embs, ids, kg, chunk_by_id, pg_graphs, reranker


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
    """Returns {"decision": dict, "parse_ok": bool, "schema_errors": list[str]}.

    Screening-set-B audit follow-up (2026-09-23): PARSE and SCHEMA are two different
    failure modes, previously conflated. `parse_ok=True` means a JSON object with the
    required keys was found at all (regardless of whether its FIELD VALUES are
    internally consistent) -- schema violations (e.g. encounter_rule='none' combined with
    conduct_rule='Rule 17', which validate_action_json() correctly rejects) are surfaced
    via `schema_errors` / `decision["_schema_errors"]`, NEVER folded into `_parse_error`.
    Evidence this mattered: v7/screening_standard_cloud set B showed 69.8% "parse-fail"
    on genuinely valid JSON like {"action":"hold_course","degrees":0.0,"encounter_rule":
    "none","conduct_rule":"Rule 17",...} -- the OLD code discarded the model's real
    answer entirely and replaced it with the hold_course/_parse_error fallback just
    because ONE field combination failed schema validation, destroying the actual
    decision-quality signal (a genuine but different error -- see
    E_rule_matrix_none_with_conduct in _analysis/audit_runs.py) a real parse failure is
    not."""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # Prefer the LAST complete {...} object that actually parses as an action dict -- Qwen3
    # sometimes echoes the JSON once before </think> closes and once again after (a
    # duplicate-answer pattern seen in the sweep logs), which the previous single greedy
    # first-{-to-last-} match stitched into one invalid blob spanning both copies.
    # validate_action_json() (pipeline/oow_agent_spec.py) is the SAME schema check the
    # Track-2 training-data generators run their own written assistant answers against.
    schema_invalid_candidate, schema_invalid_errors = None, None
    for candidate in reversed(_extract_json_objects(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not (isinstance(parsed, dict) and "action" in parsed):
            continue
        errors = validate_action_json(parsed)
        if not errors:
            return {"decision": parsed, "parse_ok": True, "schema_errors": []}
        if schema_invalid_candidate is None:  # keep the LAST (most recent) one seen
            schema_invalid_candidate, schema_invalid_errors = parsed, errors
    if schema_invalid_candidate is not None:
        schema_invalid_candidate["_schema_errors"] = schema_invalid_errors
        return {"decision": schema_invalid_candidate, "parse_ok": True,
               "schema_errors": schema_invalid_errors}
    return {"decision": {"action": "hold_course", "encounter_rule": "none", "conduct_rule": "none",
                        "reasoning": f"[parse error -- raw model output] {text[:300]}",
                        "_parse_error": True},
           "parse_ok": False, "schema_errors": []}


def _pg_match_query(own: Vessel, targets: list[Vessel], safe_distance_m: float = 500.0,
                    max_turn_deg: float = 30.0) -> str:
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
    role RAG's own query already plays.
    `targets` MUST be the simulation's live, currently-moving contact list (Simulation.
    targets) -- NEVER `mission.targets` (bug fixed 2026-09-21: this used to take `mission`
    and read mission.targets directly, so the encounter classification driving THIS
    retrieval was computed against each contact's frozen t=0 position forever; verified
    concretely on s01_head_on__v4_pg that this flipped a real close-quarters Rule 15/16
    crossing at t=100s, CPA 285m, into a false 'routine passage' match, handing the model
    generic lookout guidance instead of give-way guidance at the moment it mattered most)."""
    if not targets:
        return "routine passage, no other traffic, no close-quarters encounter"
    contacts = [contact_line(own, t, safe_distance_m, max_turn_deg) for t in targets]
    live = [c for c in contacts if not c["quiet"]]
    if not live:
        return "routine passage, no close-quarters encounter, maintain course and speed"
    pick = min(live, key=lambda c: c["cpa_m"])
    enc = pick["encounter"].replace("_", " ")
    rules = " and ".join(pick["rules"]) if pick["rules"] else "none"
    return f"{enc} encounter, applicable {rules}, give-way/stand-on obligations"


def build_oow_prompt(mission: Mission, own: Vessel, targets: list[Vessel], config: str = "v3_rag_cot",
                     system_prompt: str | None = None,
                     k: int = 6, dense_n: int = 40,
                     constraints: VesselConstraints | None = None,
                     previous_decisions: list[dict] | None = None) -> tuple[list[dict], dict]:
    """Returns (messages, debug_info) for the selected MODEL_CONFIGS key.
    `targets` MUST be the simulation's live, currently-moving contact list (Simulation.
    targets), passed straight through to narrate()/_pg_match_query() -- see their
    docstrings for the bug this fixes (mission.targets is frozen at t=0).
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
    # Compliance-rebuild STAP 1 (2026-09-23): the SAME mission/constraints values used below for
    # the constraint-line text and risk_horizon_s, never a hardcoded/independently-drifting
    # default once real constraints are known -- see narrate.contact_line()'s docstring.
    safe_distance_m = constraints.min_cpa_m if constraints else 500.0
    max_turn_deg = constraints.max_rudder_angle_deg if constraints else 30.0
    situation = narrate(mission, own, targets, cruise_speed_mps=constraints.cruise_speed_mps if constraints else None,
                        safe_distance_m=safe_distance_m, max_turn_deg=max_turn_deg)

    if spec["bare"]:
        user_msg = f"Situation:\n{situation}\n\nRecommend exactly ONE manoeuvre as the specified JSON object."
        messages = [{"role": "system", "content": BARE_SYSTEM},
                    {"role": "user", "content": user_msg}]
        debug = {"situation": situation, "config": config, "retrieved_chunk_ids": [],
                 "query_concepts": [], "expanded_concepts": [], "pg_guidance": None,
                 "user_msg_chars": len(user_msg), "user_msg": user_msg}
        return messages, debug

    embedder, embs, ids, kg, chunk_by_id, pg_graphs, reranker = _load_retrieval()

    hits, q_cons, expanded, ctx = [], [], [], None
    if spec["rag"]:
        # rerank configs retrieve a WIDER pool (dense_n instead of k) so the cross-encoder has
        # real candidates to promote/demote -- reranking a k-sized pool can only reshuffle what
        # kg_retrieve already decided to keep, never recover a chunk it dropped.
        pool_k = dense_n if (spec["rerank"] and reranker is not None) else k
        hits, q_cons, expanded = kg_retrieve(situation, embedder, embs, ids, kg, k=pool_k, dense_n=dense_n,
                                            max_per_document=RAG_MAX_PER_DOCUMENT)
        if spec["rerank"] and reranker is not None:
            hits = rerank_hits(situation, hits, chunk_by_id, reranker, k=k)
        ctx = format_context(hits, chunk_by_id)

    pg_text = None
    if spec["pg"]:
        graph = pg_graphs.get(spec["pg"])
        if graph is not None:
            pg_text = render_guidance(_pg_match_query(own, targets, safe_distance_m, max_turn_deg), graph)

    user_parts = []
    if spec["cot"] or spec["pg"]:
        # PG configs also force native thinking on (see effective_generation_params) but had NO
        # reasoning structure at all before this -- same unstructured-rambling risk as CoT, so they
        # get the same step template.
        user_parts.append(COT_INSTR)
    if constraints is not None:
        per_step = constraints.turn_rate_deg_s * constraints.time_step_s
        # Risk horizon (quality-review STAP 2, 2026-09-23): geometry-derived from THIS
        # mission's own safe distance/max turn/own-ship speed via the SAME
        # derive_risk_horizon_s() the training generators sample around -- the live
        # simulator always uses the derived default (no random multiplier, that variation
        # is a training-only device). The safe-distance/max-turn/horizon sentence itself
        # is rendered via the SAME constraint_line() the training generators call (not a
        # hand-duplicated copy), so it is guaranteed byte-identical whenever settings
        # coincide (see the parity test) -- fixes a real bug found at STOP-1/2-
        # verification: the old hand-duplicated text said "CPA below that is a real
        # collision risk" unconditionally, which is the ALREADY-FIXED STAP-1 CPA-alone
        # bug's own definition, not the CPA-AND-TCPA conjunction real_risk() actually uses.
        risk_horizon_s = derive_risk_horizon_s(constraints.min_cpa_m, constraints.max_rudder_angle_deg,
                                               own.speed)
        accel_kt_per_min = mps_to_kn(constraints.max_acceleration_mps2 * 60.0)
        decel_kt_per_min = mps_to_kn(constraints.max_deceleration_mps2 * 60.0)
        user_parts.append(
            f"Own-ship's physical limits: heading changes at most {constraints.turn_rate_deg_s:.1f} "
            f"deg/s (~{per_step:.0f} deg per {constraints.time_step_s:.0f}s step) -- a larger turn "
            "request will be silently capped, so a course change bigger than the per-command max "
            "needs several separate turn commands across multiple steps, not one big one. "
            f"Speed is capped at {mps_to_kn(constraints.max_speed_mps):.1f} kt, changing gradually "
            f"(~{accel_kt_per_min:.2f} kt/min up / ~{decel_kt_per_min:.2f} kt/min down) -- "
            "speed_up/slow_down are not instant. "
            + constraint_line(constraints.min_cpa_m, constraints.max_rudder_angle_deg, risk_horizon_s)
        )
    if pg_text:
        user_parts.append(f"Procedure guidance:\n{pg_text}")
    if ctx is not None:
        user_parts.append(f"COLREG reference excerpts:\n\n{ctx}")
    # Fase B4 (quality-review, 2026-09-23): SAME render_previous_decisions() the training
    # generators use, prepended INSIDE the "Situation:" block exactly like
    # build_oow_scenarios.py's user_message_for() -- guarantees byte-identical text for
    # the same history, given the same settings. `previous_decisions` is None by default
    # here (Simulation.agent_log is currently declared but never populated -- wiring a
    # real per-step history source into the live simulator is a separate, not-yet-done
    # follow-up; this parameter exists so the prompt CAN render one the moment a caller
    # supplies it, without any further changes to this function).
    history_prefix = render_previous_decisions(previous_decisions)
    user_parts.append(f"Situation:\n{history_prefix}{situation}")
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


def rag_context_preview(mission: Mission, own: Vessel, targets: list[Vessel],
                        k: int = 6, dense_n: int = 40) -> dict:
    """CPU-only preview of what v1_rag/v3_rag_cot would inject for the given k, without
    touching the GPU/model -- lets the Agent panel show a live "context size" readout as
    the user adjusts the RAG on/off switch and chunk-count slider. `targets` MUST be the
    simulation's live contact list (Simulation.targets), not mission.targets -- see
    narrate()'s docstring."""
    if k <= 0:
        return {"chars": 0, "chunks": 0}
    embedder, embs, ids, kg, chunk_by_id, _, _ = _load_retrieval()
    situation = narrate(mission, own, targets)
    hits, _, _ = kg_retrieve(situation, embedder, embs, ids, kg, k=k, dense_n=dense_n,
                             max_per_document=RAG_MAX_PER_DOCUMENT)
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
    # repetition_penalty: greedy decoding (do_sample=False) has no built-in defense against
    # looping -- found CoT/PG configs repeating the EXACT same "Rule 15 says..." sentence 20+
    # times verbatim (never self-correcting, never closing the JSON) until max_new_tokens ran
    # out, causing most of the parse-error/hold_course fallbacks seen in the sweep. A mild
    # repetition_penalty discourages that without banning any exact token sequence outright.
    # Tried no_repeat_ngram_size=4 as well -- REVERTED: observed on a cloud A30 re-test that
    # hard-blocking every repeated 4-gram forces the model off a CORRECT number (e.g. a given
    # distance) once it needs to restate it a second time, since repeating it verbatim is now
    # forbidden -- produced garbled/hallucinated distances and even stray CJK characters
    # instead. repetition_penalty alone still stops the sentence-level loop without this
    # side effect.
    # AUTOPILOT_STREAM=1 prints tokens to stdout live as they're generated (via
    # transformers' TextStreamer) -- opt-in only, for watching a slow/long-running CLI
    # sweep (tail -f the log) to see the actual <think> reasoning as it happens instead
    # of waiting minutes for the whole response with no visibility into what it's doing.
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) \
        if os.environ.get("AUTOPILOT_STREAM") else None
    out = mdl.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id,
                       stop_strings="\"}", tokenizer=tok,
                       repetition_penalty=1.15,
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


def ask_oow(mission: Mission, own: Vessel, targets: list[Vessel], config: str = "v3_rag_cot",
           system_prompt: str | None = None, max_new_tokens: int = 256,
           enable_thinking: bool = False, k: int = 6,
           constraints: VesselConstraints | None = None) -> tuple[dict, dict]:
    """Returns (decision_json, debug_info). `targets` MUST be the simulation's live,
    currently-moving contact list (Simulation.targets) -- NEVER mission.targets, see
    narrate()'s docstring for the bug this fixes. `config` is one of MODEL_CONFIGS's keys;
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
    messages, debug = build_oow_prompt(mission, own, targets, config=config, system_prompt=system_prompt, k=k,
                                       constraints=constraints)
    tok, mdl = _load_qwen()
    raw = _generate(tok, mdl, messages, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
    parsed = _parse_json_action(raw)
    decision = parsed["decision"]
    debug["raw_response"] = raw
    debug["parse_ok"] = parsed["parse_ok"]
    return decision, debug
