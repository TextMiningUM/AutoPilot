"""
================================================================================
build_oow_scenarios.py — OOW Track 2: applied helm/engine-order decisions
================================================================================

WHY THIS EXISTS (see notebook § 6.5)
-------------------------------------
OOW's Track 1 (COLREG rules & knowledge, `colreg_qa_500.json`) tests whether
the model KNOWS the rules. Track 2 tests whether it can actually DECIDE and
STATE a correct action from a fused situation report, the way the real
Navigation Agent will have to.

THREE-LAYER DESIGN (per user direction 2026-09-17)
-----------------------------------------------------
The Navigation Agent (the LLM being trained/evaluated) must be trained and
evaluated purely on TEXT -- MOOS telemetry numbers never enter its
input/output directly. Three separate layers, only the middle one is the
model this repo trains:

  1. Numbers -> narrative (THIS SCRIPT, deterministic, no LLM): own-ship
     position/heading/speed/waypoint + per-contact range/bearing/CPA/TCPA/
     risk/role, rendered as a fixed MOOS-style narrative block (see
     `render_situation_narrative()`). Zero hallucination risk since it's a
     pure Python template over already-computed geometry.
  2. Narrative -> decision (the Navigation Agent, i.e. what gets trained):
     reads the narrative + the allowed-action menu, must pick ONE action
     (maintain_course / alter_course / set_speed / resume_cruising_speed / stop) with
     parameters, and state a COLREG justification in prose. In THIS
     script's training/eval data, the action+parameters are computed
     deterministically (`choose_action()`) and Claude renders only the
     justification prose around them (see BUILD vs VALIDATE below) --
     never invents a different action.
  3. Decision -> MOOS API call (future work, not this script): a separate
     deterministic parser turns the agent's chosen action+parameters into
     the actual MOOS variable posts (DESIRED_HEADING/DESIRED_SPEED or
     equivalent) -- exactly mirroring how `colreg_llm_bridge.py` already
     asks the live model for a small fixed JSON action shape today.

Geometry, per-contact role/rule classification, risk level, and the chosen
action are ALL computed deterministically (this module) from bearing/range/
speed via CPA/TCPA and the same COLREG encounter-classification logic
already implemented (and used live) in `Data/OOW/OOW_MOOS_Integration/
colreg_llm_bridge.py`'s `classify_encounter()`/`cpa_tcpa()`.

BUILD vs VALIDATE (per project convention)
-------------------------------------------
- BUILD (this script): Anthropic (claude-sonnet-4-5) writes ONLY the
  `gold_answer` justification prose around an ALREADY-DECIDED action --
  same reasoning as `enrich_gold_claims.py` (keeps the GPT-4o-mini judge
  used at eval time from ever grading its own model family's output).
  `situation_report` and `question` are 100% deterministic, no LLM call.
- VALIDATE: `eval_oow_scenarios.py` (Track 2 eval, mirrors
  `eval_colreg_scenarios.py`) scores model answers with the RAGAS suite v2
  claim-level metrics via OpenAI (gpt-4o-mini judge) plus rule-based
  RoleCorrect/ActionCorrect/RuleCite checks computed straight from this
  script's stored ground-truth fields -- no judge needed for those three.

OUTPUTS
-------
- Data/OOW/OOW_Eval/oow_colreg_scenarios_v1.json
      FROZEN held-out eval set, original free-prose task format (NEVER used for
      training, never regenerated -- see tests/test_oow_scenarios_v1_frozen.py).
- Data/OOW/OOW_Eval/oow_colreg_scenarios_v2.json
      SAME 325 v1 geometries (1:1 linked via `v1_id`), re-rendered in the UNIFIED
      task format (Fase B2, RAG-rebuild-v2 plan) that training now uses -- see
      `--build-v2` / `build_v2_eval_records()`. Also frozen once written (see
      tests/test_oow_scenarios_v2_frozen.py).
- Data/OOW/OOW_Agents_Training/oow_scenario_sft_direct.jsonl / _cot.jsonl
      Training SFT rows in the UNIFIED format: system=SYSTEM_OOW_AGENT,
      user=render_scenario_situation() (the SAME renderer v2's eval records use),
      assistant=JSON {action, degrees, encounter_rule, conduct_rule, reasoning} (reasoning
      comes from Fase B3's gated, geometry-derived reasoning generation -- see
      pipeline/track2/b3_reasoning_gates.py). Written DIRECTLY,
      NOT via build_sft.py -- see write_scenario_sft_files()'s docstring for why
      that generic builder doesn't fit this fixed-question data shape.
- Data/OOW/OOW_Agents_Training/oow_scenario_dpo_pairs.jsonl
      DPO pairs; the rejected side is a DETERMINISTIC field swap on the unified
      JSON action (see wrong_action_variant()) -- no second Claude pass.
- Data/OOW/OOW_Agents_Training/oow_scenario_reflection.jsonl
      Draft/Critique/Refined triples (draft = vague action, no parameters/rule).
- Data/OOW/OOW_Agents_Training/oow_scenario_reasoning_traces.jsonl
      Reasoning-trace-schema copy of the training scenarios (still OLD prose
      situation_report -- this is a separate downstream artifact consumed by
      build_pg.py and build_reranker_pairs.py, not by SFT/DPO/reflection).

Train/eval scenarios are geometrically disjoint by construction (first
N_EVAL_PER_CATEGORY instances of every category are reserved for eval,
never used to build a training instance). Training `gold_answer` text is
ALSO cosine-filtered at 0.85 against Track 1's `colreg_qa_500.json`
`question` field -- NOT against this file's own new eval half (a self-
comparison there false-positives almost everything, since the deterministic
narrative/action text is intentionally formulaic across categories; see
`filter_contamination()`'s docstring for the measured numbers).

NOT YET MODELED (documented, not silently assumed) -- next-pass follow-ups:
  - Restricted visibility (Rule 19), narrow-channel keep-to-starboard
    (Rule 9), and special-status vessels (NUC/restricted-manoeuvre/fishing/
    sailing overriding the normal power-driven hierarchy, Rule 18) are NOT
    generated here -- add as a v2 category set once this first pass is
    validated.
  - `choose_action()`'s degrees/lookahead/TCPA-escalation thresholds are a
    simple, documented heuristic, not a full COLREG-compliant manoeuvre
    planner -- treat the ROLE and RULE fields as ground truth with high
    confidence, the exact degrees/lookahead numbers as plausible-but-
    illustrative.
  - Position/speed magnitudes are illustrative MOOS-local-mission scale
    (metres, bare speed numbers, no unit conversion), matching the
    convention already established in `generate_moos_scenarios.py` --
    not tied to a specific real MOOS bridge's exact unit wiring yet.

USAGE
-----
    python -m pipeline.track2.build_oow_scenarios --skip-llm      # geometry+narrative only, free, no API calls
    python -m pipeline.track2.build_oow_scenarios --smoke         # ~12 eval + ~12 train, real LLM calls (gold_answer only)
    python -m pipeline.track2.build_oow_scenarios                # full run (defaults below)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from core import AgentPaths, load_env, review_path, safe_write_jsonl, EMBEDDER_MODEL, CONTAM_THRESH
from pipeline.oow_agent_spec import (
    SYSTEM_OOW_AGENT, ACTIONS, validate_action_json, classify_rules,
    bearing_and_range, relative_bearing, goal_course_action, goal_course_check_line,
    render_previous_decisions, real_risk,
)

paths = AgentPaths.oow()
W = paths.workspace
CACHE = paths.cache_dir
# FROZEN (2026-09-22, RAG-rebuild-v2 plan point 1) -- this is the ORIGINAL prose/free-
# text-answer task format's held-out eval set (325 scenarios, "Applicable COLREG rules"/
# encounter-type/role text IN the prompt). Byte-for-byte pinned by
# tests/test_oow_scenarios_v1_frozen.py's sha256 check -- NEVER regenerate this file.
# See EVAL_OUT_V2 below for the same 325 geometries re-rendered in the unified task
# format (pipeline/oow_agent_spec.py) that training now uses.
EVAL_OUT = paths.eval_dir / "oow_colreg_scenarios_v1.json"
EVAL_OUT_V2 = paths.eval_dir / "oow_colreg_scenarios_v2.json"
TRACES_OUT = CACHE / "oow_scenario_reasoning_traces.jsonl"

MODEL_DEFAULT = "claude-sonnet-4-5"
BATCH_SIZE_DEFAULT = 15
MAX_TOKENS_DEFAULT = 6000

CONTACT_NAME_POOL = ["RANDOM_TS1", "RANDOM_TS2", "RANDOM_TS3"]
BASE_RULES = ["Rule 2", "Rule 5", "Rule 6", "Rule 7", "Rule 8"]  # always-applicable seamanship rules
NM_TO_M = 1852.0
SYSTEM_OOW = (
    "You are the Navigation Agent aboard an autonomous surface vessel. You receive a fused "
    "situation report (own-ship state, tracked contacts with bearing/range/CPA/TCPA/risk, "
    "applicable COLREG rules, and the allowed actions this cycle with their parameters). "
    "Reply by stating the ONE action you take this cycle (maintain_course, alter_course with "
    "a degree figure and direction, set_speed, stop, or resume_cruising_speed), then justify it citing the "
    "COLREG rule(s) that apply. Never invent an action outside the allowed list."
)  # kept byte-identical to eval_oow_scenarios.py's SYSTEM_OOW so train/eval framing matches
KN_TO_MS = 0.514444  # only used to keep bearing/range/speed SAMPLING ranges realistic; the
                     # narrative itself presents bare "speed"/metre numbers, no unit label,
                     # matching the MOOS-local-mission convention (see module docstring).


# ── Geometry (heading 0 = north, clockwise, local x/y metres) ─────────────────────
def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def bearing_range_to_xy(bearing_deg: float, rng_m: float) -> tuple[float, float]:
    b = deg2rad(bearing_deg)
    return rng_m * math.sin(b), rng_m * math.cos(b)


def solve_intercept(ts_start: tuple[float, float], v_ts: float, v_os: float) -> tuple[float, float]:
    """Smallest positive time T at which a target ship starting at `ts_start`, moving
    in a straight line at speed v_ts, could reach own-ship's position (own-ship starts
    at origin, heading 0/north, speed v_os) -- i.e. a genuine collision course if
    neither vessel avoids. Returns (T_seconds, target_heading_deg). Raises ValueError
    if no real positive solution exists for the given speeds/geometry."""
    xs, ys = ts_start
    a = v_os ** 2 - v_ts ** 2
    b = -2.0 * ys * v_os
    c = xs ** 2 + ys ** 2
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            raise ValueError("degenerate geometry, no intercept solution")
        T = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            raise ValueError("no real intercept solution for given speeds/geometry")
        sq = math.sqrt(disc)
        cands = [t for t in ((-b + sq) / (2 * a), (-b - sq) / (2 * a)) if t > 1.0]
        if not cands:
            raise ValueError("no positive intercept time")
        T = min(cands)
    os_x, os_y = 0.0, v_os * T
    dx, dy = os_x - xs, os_y - ys
    heading_ts = math.degrees(math.atan2(dx, dy)) % 360.0
    return T, heading_ts


def compute_target(bearing_from_os_deg: float, range_nm: float, v_ts_kn: float, v_os_kn: float) -> dict:
    """Given own-ship at the origin heading 0/north at v_os_kn, and a desired initial
    relative bearing/range for a target moving at v_ts_kn, solve for the target's start
    position/heading such that this is a genuine collision course. Sampling is done in
    nm/knots for convenient category authoring; the returned dict carries metres/bare-
    speed for the narrative (see module docstring on units)."""
    range_m = range_nm * NM_TO_M
    v_ts, v_os = v_ts_kn * KN_TO_MS, v_os_kn * KN_TO_MS
    ts_start = bearing_range_to_xy(bearing_from_os_deg, range_m)
    T, heading_ts = solve_intercept(ts_start, v_ts, v_os)
    return {
        "start_xy_m": ts_start, "heading_deg": round(heading_ts, 1),
        "speed": v_ts_kn, "bearing_from_os_deg": bearing_from_os_deg,
        "t_collision_min": round(T / 60.0, 1),
    }


def same_line_target(range_ahead_nm: float, v_ts_kn: float, v_os_kn: float, astern: bool) -> dict:
    """Overtaking-family construction: target on own-ship's own track, dead ahead
    (own-ship overtakes, astern=False) or dead astern (own-ship is overtaken,
    astern=True)."""
    brg = 0.0 if not astern else 180.0
    x, y = bearing_range_to_xy(brg, range_ahead_nm * NM_TO_M)
    v_ts, v_os = v_ts_kn * KN_TO_MS, v_os_kn * KN_TO_MS
    closing = (v_ts - v_os) if astern else (v_os - v_ts)
    t_coll = (range_ahead_nm * NM_TO_M) / closing if closing > 0 else math.inf
    return {
        "start_xy_m": (x, y), "heading_deg": 0.0, "speed": v_ts_kn,
        "bearing_from_os_deg": brg,
        "t_collision_min": round(t_coll / 60.0, 1) if math.isfinite(t_coll) else None,
    }


def cpa_tcpa_m(v_os_kn: float, target: dict) -> tuple[float, float]:
    """CPA (metres) and TCPA (minutes) from own-ship (origin, heading 0, v_os_kn) and a
    target dict produced by compute_target/same_line_target, assuming both hold course/speed."""
    th = deg2rad(target["heading_deg"])
    v_os, v_ts = v_os_kn * KN_TO_MS, target["speed"] * KN_TO_MS
    vox, voy = 0.0, v_os
    vtx, vty = v_ts * math.sin(th), v_ts * math.cos(th)
    xs, ys = target["start_xy_m"]
    dvx, dvy = vtx - vox, vty - voy
    rel_speed_sq = dvx ** 2 + dvy ** 2
    if rel_speed_sq < 1e-6:
        return math.hypot(xs, ys), 0.0
    t_cpa = max(0.0, -(xs * dvx + ys * dvy) / rel_speed_sq)
    cx, cy = xs + dvx * t_cpa, ys + dvy * t_cpa
    return math.hypot(cx, cy), t_cpa / 60.0


# relative_bearing() is imported from pipeline.oow_agent_spec (see imports above) -- the
# SAME function narrate.py and build_oow_scenarios_leo.py use, not a local re-definition.


def advance_past_cpa(target: dict, v_os: float, extra_frac: float = 0.6) -> dict:
    """Given a target dict from compute_target()/same_line_target() (a genuine collision-
    course geometry at t=0), analytically re-parameterise it to a LATER moment -- `extra_frac`
    beyond its own CPA time -- so the returned dict describes the SAME encounter already
    resolved: range now opening, TCPA clamped to 0. Used to build risk_cleared_resume
    instances without a separate, error-prone hand-authored geometry. Own-ship is always
    rendered at the origin heading 0/north (narrative convention), so both vessels' motion
    since t=0 is folded into the target's new relative start_xy/bearing; heading/speed are
    unchanged (both hold course/speed -- that's what makes CPA/TCPA well-defined here)."""
    cpa_m, t_cpa_min = cpa_tcpa_m(v_os, target)
    dt_s = t_cpa_min * 60.0 * (1.0 + extra_frac)
    th = deg2rad(target["heading_deg"])
    vtx, vty = target["speed"] * math.sin(th), target["speed"] * math.cos(th)
    xs, ys = target["start_xy_m"]
    new_x, new_y = xs + vtx * dt_s, ys + (vty - v_os) * dt_s
    new_bearing = math.degrees(math.atan2(new_x, new_y)) % 360.0
    return {**target, "start_xy_m": (new_x, new_y), "bearing_from_os_deg": new_bearing,
            "t_collision_min": None, "_orig_cpa_m": cpa_m, "_orig_tcpa_min": t_cpa_min}


def classify_encounter(rel_brg: float, own_hdg: float, tgt_hdg: float) -> tuple[str, list[str]]:
    """Same COLREG encounter classification used live in colreg_llm_bridge.py's
    classify_encounter() -- kept in sync deliberately so training/eval data reflects
    the exact same geometry-to-rule mapping the live bridge will use."""
    course_diff = (tgt_hdg - own_hdg + 540) % 360 - 180
    if abs(rel_brg) <= 6 and abs(abs(course_diff) - 180) <= 20:
        return "head_on", ["Rule 14"]
    if abs(rel_brg) > 112.5:
        return "overtaking_geometry", ["Rule 13"]
    if rel_brg > 0:
        return "crossing_target_on_starboard", ["Rule 15", "Rule 16"]
    return "crossing_target_on_port", ["Rule 15", "Rule 17"]


def bow_phrase(rel_brg: float) -> str:
    a = abs(rel_brg)
    side = "starboard" if rel_brg >= 0 else "port"
    if a <= 5:
        return "dead ahead"
    if a >= 175:
        return "dead astern"
    if a <= 45:
        return f"fine on our {side} bow"
    if a <= 90:
        return f"on our {side} bow"
    if a <= 135:
        return f"on our {side} quarter"
    return f"broad on our {side} quarter"


def closing_rate(own_speed: float, target: dict) -> float:
    """Rate of range closure (same units as speed) at t=0 -- positive = closing,
    negative = opening. NOT the same as the raw speed difference: two vessels can
    have similar speeds but a high closing rate if converging, or a large speed
    difference but near-zero closing rate if running nearly parallel."""
    th = deg2rad(target["heading_deg"])
    vtx, vty = target["speed"] * math.sin(th), target["speed"] * math.cos(th)
    dvx, dvy = vtx - 0.0, vty - own_speed
    xs, ys = target["start_xy_m"]
    rng = math.hypot(xs, ys)
    if rng < 1e-6:
        return 0.0
    return -((dvx * xs + dvy * ys) / rng)


def risk_level(cpa_m: float, tcpa_min: float, closing: bool = True) -> str:
    """Simple, documented CPA/TCPA-threshold heuristic (see module docstring) -- NOT a
    full COLREG risk-of-collision model. TCPA=0 is ambiguous on its own (it occurs both
    for 'collision right now' and 'closest point already passed') -- `closing=False`
    (range opening) always resolves to 'low' regardless of the CPA/TCPA magnitude, same
    disambiguation the live Basic Simulator agent's narrate.py uses."""
    if not closing:
        return "low"
    if cpa_m < 200 and tcpa_min < 6:
        return "high"
    if cpa_m < 500 and tcpa_min < 12:
        return "medium"
    return "low"


def role_sentence(role: str) -> str:
    return {
        "mutual": "We are meeting head-on. Both vessels are give-way.",
        "give_way": "She is crossing from our starboard side. We are the give-way vessel.",
        "stand_on": "She is crossing from our port side. We are the stand-on vessel.",
        "overtaking_give_way": "We are overtaking her. We are the give-way vessel.",
        "overtaking_stand_on": "She is overtaking us. We are the stand-on vessel.",
        "cleared": "That vessel is now well clear of us and the range is increasing; the "
                   "earlier close-quarters situation has been resolved.",
    }.get(role, "Encounter role under assessment.")


def choose_action(contact_roles: list[str], worst_tcpa_min: float) -> tuple[str, dict, str]:
    """Deterministic mapping from the aggregated encounter (all contacts' roles + the
    most urgent TCPA) to ONE of the 5 allowed MOOS actions this cycle. Simple, documented
    heuristic (see module docstring) -- NOT a full COLREG-compliant manoeuvre planner.
    NOTE: this is the ORIGINAL (v1-era) action taxonomy/heuristic, kept byte-for-byte
    unchanged so oow_colreg_scenarios_v1.json stays reproducible from the same code that
    built it. See to_unified_action() below for the Fase B2 unified-task-format mapping
    used by training rows and oow_colreg_scenarios_v2.json -- NEVER change this function
    to "fix" or "improve" it; add to the unified path instead."""
    give_way_present = any(r in ("mutual", "give_way", "overtaking_give_way") for r in contact_roles)
    if not give_way_present:
        return "maintain_course", {}, "maintain_course"
    if worst_tcpa_min < 1.5:
        return "stop", {}, "stop"
    degrees, lookahead_m = 30, 200
    return ("alter_course", {"degrees": degrees, "lookahead_distance_m": lookahead_m},
            f"alter_course (+{degrees} degrees to starboard, lookahead distance {lookahead_m} m)")


# ══════════════════════════════════════════════════════════════════════════════════
# UNIFIED TASK FORMAT (Fase B2, RAG-rebuild-v2 plan, 2026-09-22)
# ══════════════════════════════════════════════════════════════════════════════════
# Everything below derives NEW fields (action/degrees/encounter_rule/conduct_rule in
# pipeline.oow_agent_spec.ACTIONS' vocabulary, plus a unified situation-report renderer)
# from the SAME geometry/CATEGORIES/generate_*_instance() machinery above -- none of that
# machinery is touched, so oow_colreg_scenarios_v1.json stays exactly reproducible.
# Training rows (SFT/DPO/reflection/traces) and oow_colreg_scenarios_v2.json both use
# ONLY the functions in this section, sharing the SAME render_scenario_situation() (see
# tests/test_oow_scenarios_shared_renderer.py's identity test).

SAFE_CPA_M = 500.0        # matches Basic Simulator VesselConstraints' default min_cpa_m
CRITICAL_RANGE_M = 200.0  # Rule 17(b): manoeuvre alone no longer suffices -- range/closing
                          # criterion, never a TCPA cutoff (same as build_oow_scenarios_leo.py)
MIN_TURN_DEG = 15.0       # Rule 16's "early and substantial" rules out a token gesture
MAX_TURN_DEG = 30.0       # matches Basic Simulator VesselConstraints' max_rudder_angle_deg


def _turn_degrees(cpa_m: float) -> float:
    shortfall = min(1.0, max(0.0, (SAFE_CPA_M - cpa_m) / SAFE_CPA_M))
    return round(MIN_TURN_DEG + shortfall * (MAX_TURN_DEG - MIN_TURN_DEG), 1)


def _diverging_turn(rel_bearing_deg: float) -> str:
    return "turn_left" if rel_bearing_deg >= 0 else "turn_right"


def _should_stop(cpa_m: float, range_m: float, closing_speed: float) -> bool:
    return cpa_m < CRITICAL_RANGE_M and range_m < CRITICAL_RANGE_M and closing_speed > 0


def to_unified_action(rec: dict) -> dict:
    """Maps this generator's OLD action taxonomy (maintain_course/alter_course/stop/
    resume_cruising_speed, always a fixed +/-30 degrees) onto the unified simulator
    vocabulary with geometry-scaled degrees -- for training rows / v2 eval only, never
    touches rec["action"]/rec["action_params"] (v1's own fields). encounter_rule/
    conduct_rule come from pipeline.oow_agent_spec.classify_rules() -- the ONE shared
    mapping table, so this generator, build_oow_scenarios_leo.py, and the B3 reasoning
    cross-check can never silently disagree about which rule pair a role/action implies.
    `decisive_contact_index` (index into rec["targets"], or None when no real risk) is
    the contact that actually drove this decision -- fed to the B3 teacher prompt as
    answer-side context (never leaked to render_scenario_situation()'s user-facing text)
    and checked by the B3 acceptance gates' contact-consistency check.

    NO stationary-contact branch here (unlike build_oow_scenarios_leo.py's quality-review
    STAP 1 extension, 2026-09-23): verified -- this generator's CATEGORIES/_classify_target()
    only ever produce moving-vessel encounter_type values (head_on/crossing_target_on_*/
    overtaking_geometry/same_line ahead/astern); no "stationary_contact"/"not_applicable"
    role is ever generated, so there is nothing for a stationary branch to catch here."""
    old_action = rec["action"]
    roles = rec["role"].split("+")
    if old_action == "resume_cruising_speed":
        encounter_rule, conduct_rule = classify_rules("none", "speed_up")
        return {"action": "speed_up", "degrees": None, "encounter_rule": encounter_rule,
               "conduct_rule": conduct_rule, "decisive_contact_index": None}
    if old_action == "maintain_course":
        # A genuine stand-on situation (real risk, Rule 17 correctly says hold course) is
        # NOT the same as no encounter at all ("none"/"none") -- never conflate them.
        # Gated by the ONE shared real_risk() (quality-review STAP 1) -- same function
        # leo_choose_action() and measurement.py's Check A use, so a role label alone
        # (assigned upstream from bearing geometry only) can never stand in for an actual
        # CPA/TCPA risk judgement here either.
        stand_on_targets = [t for t in rec["targets"] if t["_role"] in ("stand_on", "overtaking_stand_on")
                           and real_risk(t["_cpa_m"], t["_tcpa_min"] * 60.0, SAFE_CPA_M)]
        if stand_on_targets:
            worst = min(stand_on_targets, key=lambda t: t["_cpa_m"])
            encounter_rule, conduct_rule = classify_rules(worst["_role"], "hold_course")
            return {"action": "hold_course", "degrees": None, "encounter_rule": encounter_rule,
                   "conduct_rule": conduct_rule,
                   "decisive_contact_index": rec["targets"].index(worst)}
        encounter_rule, conduct_rule = classify_rules("none", "hold_course")
        return {"action": "hold_course", "degrees": None, "encounter_rule": encounter_rule,
               "conduct_rule": conduct_rule, "decisive_contact_index": None}
    # Every remaining old_action (alter_course/stop) is driven by the give-way contact(s)
    # specifically, not necessarily whichever target has the smallest CPA overall (a
    # multi-target instance can mix give-way and stand-on contacts).
    give_way_targets = [t for t in rec["targets"] if t["_role"] in ("mutual", "give_way", "overtaking_give_way")]
    worst = min(give_way_targets or rec["targets"], key=lambda t: t["_cpa_m"])
    worst_index = rec["targets"].index(worst)
    range_m = math.hypot(*worst["start_xy_m"])
    closing = closing_rate(rec["own_speed"], worst)
    role = worst["_role"]
    if old_action == "stop" or _should_stop(worst["_cpa_m"], range_m, closing):
        encounter_rule, conduct_rule = classify_rules(role, "stop")
        return {"action": "stop", "degrees": None, "encounter_rule": encounter_rule,
               "conduct_rule": conduct_rule, "decisive_contact_index": worst_index}
    degrees = _turn_degrees(worst["_cpa_m"])
    if role == "overtaking_give_way":
        action = _diverging_turn(relative_bearing(0.0, worst["bearing_from_os_deg"]))
    else:
        action = "turn_right"  # Rule 14 (mutual/head-on) / Rule 15+16 (give_way, crossing)
    encounter_rule, conduct_rule = classify_rules(role, action)
    return {"action": action, "degrees": degrees, "encounter_rule": encounter_rule,
           "conduct_rule": conduct_rule, "decisive_contact_index": worst_index}


def render_scenario_situation(rec: dict) -> str:
    """Unified house-style narrative -- shared byte-for-byte between training rows and
    oow_colreg_scenarios_v2.json (see the identity test). Deliberately parallel to
    build_oow_scenarios_leo.py's render_leo_narrative() and Basic Simulator's narrate():
    no own_role/encounter-classification/rule-number mentions (exactly what the model
    must derive itself), GOAL COURSE CHECK + a safe-passing-distance fact always
    included. Own-ship is always at the origin, heading 0 (north); the mission waypoint
    sits straight ahead by construction (see generate_single_target_instance()), so GOAL
    COURSE CHECK always reports "already on bearing" here -- still rendered (never hand-
    waved away) so this matches the other two callers' structure exactly."""
    own_speed = rec["own_speed"]
    cruise = own_speed + 7.0
    wx, wy = 0.0, 2000.0
    n = len(rec["targets"])
    lines = [
        f"Own-ship is underway at (0.0, 0.0), heading 0.0 degrees, speed {own_speed:.2f}. "
        f"Target cruise speed is {cruise:.1f}.",
        f"Mission waypoint is at ({wx:.1f}, {wy:.1f}).",
        goal_course_check_line(0.0, 0.0, 0.0, wx, wy),
        f"This mission's safe passing distance is {SAFE_CPA_M:.0f}m: CPA below that is a real "
        "collision risk, CPA well above it is safe regardless of how small it looks.",
        f"{n} other ship{'s' if n != 1 else ''}:" if n else "No other ships tracked.",
    ]
    for i, t in enumerate(rec["targets"]):
        name = CONTACT_NAME_POOL[i] if i < len(CONTACT_NAME_POOL) else f"RANDOM_TS{i + 1}"
        rel_brg = relative_bearing(0.0, t["bearing_from_os_deg"])
        range_m = math.hypot(*t["start_xy_m"])
        closing = closing_rate(own_speed, t)
        if closing > 0:
            cpa_txt = f", CPA {t['_cpa_m']:.0f} m, TCPA {t['_tcpa_min']:.1f} min"
        elif "_orig_cpa_m" in t:
            cpa_txt = (f" (already past closest point, ranges now increasing; closest "
                      f"approach was {t['_orig_cpa_m']:.0f} m, {t['_orig_tcpa_min']:.1f} min ago)")
        else:
            cpa_txt = f", CPA {t['_cpa_m']:.0f} m (already past closest point, ranges now increasing)"
        lines.append(
            f'  - Ship named "{name}": range {range_m:.0f} m, rel.bearing {rel_brg:.1f} deg, '
            f"heading {t['heading_deg']:.1f}, speed {t['speed']:.2f}, closing speed {closing:.2f}"
            f"{cpa_txt}."
        )
    lines.append("Conditions: visibility is clear, assessed visibility range is 10000 m, "
                 "narrow-channel context is not indicated, traffic-separation context is not indicated.")
    return "\n".join(lines)


FIXED_QUESTION_UNIFIED = "Recommend exactly ONE manoeuvre as the specified JSON object."


def build_v2_eval_records() -> list[dict]:
    """RAG-rebuild-v2 plan point 2: regenerates the SAME 325 held-out geometries as v1
    (generate_population(seed=0) + split_eval_train with the SAME defaults v1 was built
    with -- verified byte-for-byte identical, see tests/test_oow_scenarios_v2_eval.py)
    and re-renders them in the unified task format instead of v1's free-prose format.
    NEVER regenerates v1 itself, and NEVER runs as part of training-data regeneration --
    only from main()'s explicit --build-v2 flag. `gold_reasoning` reuses v1's own already-
    LLM-authored gold_answer text as a non-scored reference -- no new [LLM] call needed,
    since that text already exists and was never the field any scoring reads for v2."""
    if not EVAL_OUT.exists():
        raise FileNotFoundError(f"{EVAL_OUT} (v1) not found -- v2 is built FROM v1's exact "
                                "geometries, it cannot be built standalone")
    v1_records = json.loads(EVAL_OUT.read_text(encoding="utf-8"))
    pop = generate_population(N_EVAL_PER_CATEGORY_DEFAULT + N_TRAIN_PER_CATEGORY_DEFAULT, seed=0)
    eval_recs, _ = split_eval_train(pop, N_EVAL_PER_CATEGORY_DEFAULT)
    if len(eval_recs) != len(v1_records):
        raise RuntimeError(f"regenerated eval pool ({len(eval_recs)}) != v1 record count "
                          f"({len(v1_records)}) -- generation code has drifted from v1, "
                          "refusing to build v2 with a broken v1_id linkage")
    out = []
    for i, (rec, v1_rec) in enumerate(zip(eval_recs, v1_records)):
        if rec["category"] != v1_rec["category"]:
            raise RuntimeError(f"record {i}: category mismatch ({rec['category']!r} vs "
                              f"v1's {v1_rec['category']!r}) -- geometry has drifted from v1")
        unified = to_unified_action(rec)
        gold = {k: v for k, v in unified.items() if k != "decisive_contact_index"}
        out.append({
            "id": f"oowcol_v2_{i + 1:05d}", "v1_id": v1_rec["id"], "category": rec["category"],
            "situation": render_scenario_situation(rec),
            "question": FIXED_QUESTION_UNIFIED,
            "gold": gold,
            "degrees_tolerance": 10.0 if unified["degrees"] is not None else None,
            "gold_reasoning": v1_rec.get("gold_answer", ""),
            "expected_points": rec["pass_criteria"],
        })
    return out


# ── Category taxonomy (bearing/range/speed families; role/rules are re-derived from
#    geometry via classify_encounter, NOT trusted from this table, except for the
#    overtaking give-way/stand-on distinction which needs the speed comparison below) ──
CATEGORIES: list[dict] = [
    {"name": "head_on", "bearing_range": (-6, 6),
     "pass_criteria": ["Own-ship alters course to starboard (not port).",
                       "Minimum CPA stays above the safe-distance threshold.",
                       "Own-ship returns to her original track once past and clear."]},
    {"name": "crossing_give_way_fine", "bearing_range": (10, 30),
     "pass_criteria": ["Own-ship is give-way: alters course and/or speed early.",
                       "Own-ship avoids crossing ahead of the target.",
                       "Minimum CPA stays above the safe-distance threshold."]},
    {"name": "crossing_give_way_broad", "bearing_range": (45, 80),
     "pass_criteria": ["Own-ship is give-way: early, substantial action to keep well clear.",
                       "Minimum CPA stays above the safe-distance threshold."]},
    {"name": "crossing_give_way_abeam", "bearing_range": (85, 100),
     "pass_criteria": ["Own-ship is give-way and passes astern of the target.",
                       "Minimum CPA stays above the safe-distance threshold."]},
    {"name": "crossing_stand_on_fine", "bearing_range": (-30, -10),
     "pass_criteria": ["Own-ship holds course and speed initially.",
                       "Own-ship does not alter to port toward the target if she must eventually act.",
                       "Minimum CPA stays above the safe-distance threshold."]},
    {"name": "crossing_stand_on_broad", "bearing_range": (-80, -45),
     "pass_criteria": ["Own-ship holds course and speed for a meaningful initial period.",
                       "If the target never gives way, own-ship takes late avoiding action under Rule 17(b).",
                       "Minimum CPA stays above the safe-distance threshold."]},
    {"name": "overtaking_give_way", "bearing_range": (0, 0), "same_line": "ahead",
     "pass_criteria": ["Own-ship's approach is from more than 22.5deg abaft the target's beam.",
                       "Own-ship keeps clear of the target throughout the pass (does not cut in front).",
                       "Minimum CPA stays above the safe-distance threshold."]},
    {"name": "overtaking_stand_on", "bearing_range": (180, 180), "same_line": "astern",
     "pass_criteria": ["Own-ship holds course and speed while being overtaken.",
                       "Minimum CPA stays above the safe-distance threshold."]},
    # Not a fresh encounter geometry -- generate_resume_instance() re-parameterises a
    # head_on-family instance to a later moment where the CPA has already passed (see
    # advance_past_cpa()). Added because NO existing category ever has "resume" as the
    # ground-truth action -- every other category only teaches taking avoiding action,
    # never standing back down from it once the risk has cleared.
    {"name": "risk_cleared_resume",
     "pass_criteria": ["Own-ship resumes cruise speed/course now that the earlier "
                       "close-quarters situation is resolved.",
                       "Own-ship does not needlessly remain at reduced speed or off track "
                       "once safely clear of the contact."]},
]

# Two/three-target composite categories -- reuse fixed bearings from
# generate_moos_scenarios.py's s09-s12 (their geometry is already verified),
# only range/speed are jittered per instance.
MULTI_TARGET_TEMPLATES: list[dict] = [
    {"name": "double_crossing_squeeze", "bearings": [45, 315],
     "pass_criteria": ["The manoeuvre for TS1 (starboard give-way) does not create a new "
                       "close-quarters situation with TS2.",
                       "Minimum CPA to BOTH targets stays above the safe-distance threshold."]},
    {"name": "headon_plus_crossing", "bearings": [0, 60],
     "pass_criteria": ["Own-ship alters to starboard, satisfying both encounters simultaneously.",
                       "Minimum CPA to BOTH targets stays above the safe-distance threshold."]},
    {"name": "overtake_plus_crossing", "bearings": [0, 50],
     "pass_criteria": ["Own-ship keeps clear of the overtaken target while also giving way to the crosser.",
                       "Minimum CPA to BOTH targets stays above the safe-distance threshold."]},
    {"name": "converging_cluster", "bearings": [40, 320, 0],
     "pass_criteria": ["Own-ship finds a single trajectory keeping a safe CPA from all three targets.",
                       "Own-ship does not alter to port toward the port-side target while manoeuvring.",
                       "Minimum CPA to ALL THREE targets stays above the safe-distance threshold."]},
]

N_EVAL_PER_CATEGORY_DEFAULT = 25   # 9 single + 4 multi categories x 25 ~= 325 held-out
N_TRAIN_PER_CATEGORY_DEFAULT = 30  # ~= 360 training scenarios before contamination filtering


def _classify_target(t: dict, v_os: float, same_line: str | None) -> tuple[str, list[str]]:
    """Per-contact role + rules. Uses classify_encounter() for crossing/head-on geometry;
    the give-way/stand-on split for overtaking needs a speed comparison classify_encounter
    can't do from bearing alone, so that case is handled directly from `same_line`."""
    if same_line == "ahead":
        return "overtaking_give_way", ["Rule 13"]
    if same_line == "astern":
        return "overtaking_stand_on", ["Rule 13", "Rule 17"]
    rel_brg = relative_bearing(0.0, t["bearing_from_os_deg"])
    enc, rules = classify_encounter(rel_brg, 0.0, t["heading_deg"])
    role = {"head_on": "mutual", "crossing_target_on_starboard": "give_way",
            "crossing_target_on_port": "stand_on", "overtaking_geometry": "give_way"}[enc]
    return role, rules


def generate_resume_instance(rnd: random.Random, v_os: float = 10.0) -> dict:
    """A head_on-family encounter, re-parameterised (via advance_past_cpa()) to a moment
    after the CPA has already passed -- range now opening. Ground-truth action is
    'resume_cruising_speed' (return to cruise speed/course), unlike every other category
    which only ever teaches taking avoiding action."""
    for _ in range(50):
        bearing = round(rnd.uniform(-6, 6), 1)
        rng_nm = round(rnd.uniform(0.4, 1.0), 2)
        v_ts = round(rnd.uniform(6, 14), 1)
        try:
            target = compute_target(bearing, rng_nm, v_ts, v_os)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("no valid geometry found for category 'risk_cleared_resume' after 50 tries")
    _, orig_rules = classify_encounter(relative_bearing(0.0, target["bearing_from_os_deg"]),
                                       0.0, target["heading_deg"])
    target = advance_past_cpa(target, v_os)
    cpa_m, tcpa_min = cpa_tcpa_m(v_os, target)
    target.update(_role="cleared", _rules=orig_rules, _cpa_m=cpa_m, _tcpa_min=tcpa_min)
    return {
        "category": "risk_cleared_resume",
        "pass_criteria": next(c for c in CATEGORIES if c["name"] == "risk_cleared_resume")["pass_criteria"],
        "own_speed": v_os, "targets": [target],
        "action": "resume_cruising_speed", "action_params": {},
        "action_params_text": "resume_cruising_speed (return to cruising speed)",
        "role": "cleared", "rules": orig_rules,
    }


def generate_single_target_instance(cat: dict, rnd: random.Random, v_os: float = 10.0) -> dict:
    if cat["name"] == "risk_cleared_resume":
        return generate_resume_instance(rnd, v_os)
    same_line = cat.get("same_line")
    if same_line:
        rng_ahead = round(rnd.uniform(0.3, 0.6), 2)
        if same_line == "ahead":
            v_ts = round(rnd.uniform(4, v_os - 2), 1)  # target must be slower to be overtaken
            target = same_line_target(rng_ahead, v_ts, v_os, astern=False)
        else:
            v_ts = round(rnd.uniform(v_os + 2, v_os + 8), 1)  # target must be faster to overtake
            target = same_line_target(rng_ahead, v_ts, v_os, astern=True)
    else:
        # Not every (bearing, range, speed) triple has a real intercept solution
        # (analogous to a pursuit-curve problem where the target is too slow for
        # that geometry) -- resample until one does.
        for _ in range(50):
            bearing = round(rnd.uniform(*cat["bearing_range"]), 1)
            rng_nm = round(rnd.uniform(0.4, 1.0), 2)
            v_ts = round(rnd.uniform(6, 14), 1)
            try:
                target = compute_target(bearing, rng_nm, v_ts, v_os)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"no valid geometry found for category {cat['name']!r} after 50 tries")
    role, rules = _classify_target(target, v_os, same_line)
    cpa_m, tcpa_min = cpa_tcpa_m(v_os, target)
    target.update(_role=role, _rules=rules, _cpa_m=cpa_m, _tcpa_min=tcpa_min)
    action, params, params_text = choose_action([role], tcpa_min)
    return {
        "category": cat["name"], "pass_criteria": cat["pass_criteria"],
        "own_speed": v_os, "targets": [target],
        "action": action, "action_params": params, "action_params_text": params_text,
        "role": role, "rules": rules,
    }


def generate_multi_target_instance(tpl: dict, rnd: random.Random, v_os: float = 10.0) -> dict:
    targets = []
    for bearing in tpl["bearings"]:
        for _ in range(50):
            v_ts = round(rnd.uniform(6, 12), 1)
            rng_nm = round(rnd.uniform(0.4, 0.9), 2)
            try:
                t = compute_target(bearing, rng_nm, v_ts, v_os)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"no valid geometry found for template {tpl['name']!r} after 50 tries")
        role, rules = _classify_target(t, v_os, None)
        cpa_m, tcpa_min = cpa_tcpa_m(v_os, t)
        t.update(_role=role, _rules=rules, _cpa_m=cpa_m, _tcpa_min=tcpa_min)
        targets.append(t)
    worst_tcpa = min(t["_tcpa_min"] for t in targets)
    all_rules = sorted({r for t in targets for r in t["_rules"]},
                       key=lambda r: int("".join(ch for ch in r if ch.isdigit()) or 0))
    all_roles = [t["_role"] for t in targets]
    action, params, params_text = choose_action(all_roles, worst_tcpa)
    return {
        "category": tpl["name"], "pass_criteria": tpl["pass_criteria"],
        "own_speed": v_os, "targets": targets,
        "action": action, "action_params": params, "action_params_text": params_text,
        "role": "+".join(all_roles), "rules": all_rules,
    }


def generate_population(n_per_category: int, seed: int = 0, only_category: str | None = None,
                        return_rng: bool = False):
    """Deterministic (seeded) population of scenario-fact dicts, no prose yet.
    Same seed -> same population every time (reproducibility for train/eval split).
    `only_category` restricts generation to a single named category -- used to add ONE
    new category's data on top of already-committed files without touching/regenerating
    the other (already reviewed) categories. `return_rng=True` additionally returns the
    still-live random.Random so a caller can keep drawing MORE instances continuing the
    exact same stream (used by resample_train_away_from_held_out()) -- default False
    keeps every existing caller's return shape unchanged."""
    rnd = _rng_for(seed)
    pop: list[dict] = []
    for cat in CATEGORIES:
        if only_category and cat["name"] != only_category:
            continue
        for _ in range(n_per_category):
            pop.append(generate_single_target_instance(cat, rnd))
    for tpl in MULTI_TARGET_TEMPLATES:
        if only_category and tpl["name"] != only_category:
            continue
        for _ in range(n_per_category):
            pop.append(generate_multi_target_instance(tpl, rnd))
    if return_rng:
        return pop, rnd
    return pop


def _rng_for(seed: int) -> random.Random:
    return random.Random(seed)


def split_eval_train(pop: list[dict], n_eval_per_category: int) -> tuple[list[dict], list[dict]]:
    """First `n_eval_per_category` instances of each category -> eval (held out forever);
    the rest -> train pool. Geometrically disjoint by construction."""
    by_cat: dict[str, list[dict]] = {}
    for rec in pop:
        by_cat.setdefault(rec["category"], []).append(rec)
    eval_recs, train_recs = [], []
    for cat, recs in by_cat.items():
        eval_recs += recs[:n_eval_per_category]
        train_recs += recs[n_eval_per_category:]
    return eval_recs, train_recs


# ══════════════════════════════════════════════════════════════════════════════════
# HELD-OUT CONTAMINATION AVOIDANCE (generation-time invariant, per user decision
# 2026-09-22 on the 2 overtaking_give_way geometry duplicates found by
# tests/test_oow_scenarios_v2_contamination.py): v1/v2 stay byte-for-byte frozen --
# instead, any TRAINING geometry that lands within tolerance of a held-out (v1/v2)
# geometry is rejected and redrawn at generation time, never a post-hoc filter/dedupe.
# ══════════════════════════════════════════════════════════════════════════════════

# The 2 v1/v2 overtaking_give_way eval records (found once, BEFORE this fix landed) whose
# geometry is byte-for-byte identical to a v1-era training record -- v1/v2 stay frozen (a
# model already trained on the old, unfixed training set may have seen these exact
# geometries), so archived/new evaluation results for oow_qwen_full (trained before this
# fix) should be reported for overtaking_give_way BOTH with and without these 2 ids (see
# eval_oow_scenarios.py's --schema v2 "means_by_category_excl_known_dupes"). A model
# trained AFTER this fix landed has no such overlap and this exclusion is a no-op for it.
KNOWN_CONTAMINATED_V1_IDS = {"oowcol_00160", "oowcol_00168"}

BEARING_COLLISION_TOL_DEG = 0.5
SPEED_COLLISION_TOL_KN = 0.1
HEADING_COLLISION_TOL_DEG = 0.5
TCPA_COLLISION_TOL_MIN = 0.5
MAX_REDRAW_ATTEMPTS = 200

_V2_CONTACT_RE = re.compile(
    r'range (?P<range>[\d.]+) m, rel\.bearing (?P<bearing>-?[\d.]+) deg, '
    r'heading (?P<heading>[\d.]+), speed (?P<speed>[\d.]+), closing speed (?P<closing>-?[\d.]+), '
    r'CPA (?P<cpa>[\d.]+) m, TCPA (?P<tcpa>[\d.]+) min'
)


def load_held_out_signatures() -> list[tuple[float, float, float, float]]:
    """(bearing_from_os_deg, speed, heading_deg, tcpa_min) for every contact in the
    FROZEN oow_colreg_scenarios_v1.json and v2.json files on disk -- the ground truth
    held-out set, loaded fresh every call rather than re-derived from
    generate_population(), so this stays correct even if the generator code/seed/
    defaults ever change. tcpa_min stands in for the raw start_xy_m/range the user asked
    for: v1's frozen schema stores bearing/speed/heading/cpa_m/tcpa_min per contact but
    NOT the raw range/start_xy_m, so bearing+speed+heading+tcpa_min (tcpa_min is itself a
    function of range and closing speed) is the closest available proxy for "same
    physical geometry" without reopening v1's frozen schema. v2 is parsed from its
    rendered `situation` text (it stores no raw contacts field) since it currently
    reuses v1's exact geometries 1:1 via `v1_id` -- reading both is still done (rather
    than assuming v2 adds nothing new) so a future v2 revision with its own new
    geometries would still be covered without this function needing to change."""
    sigs: list[tuple[float, float, float, float]] = []
    if EVAL_OUT.exists():
        for rec in json.loads(EVAL_OUT.read_text(encoding="utf-8")):
            for c in rec["contacts"]:
                sigs.append((c["bearing_from_os_deg"], c["speed"], c["heading_deg"], c["tcpa_min"]))
    if EVAL_OUT_V2.exists():
        for rec in json.loads(EVAL_OUT_V2.read_text(encoding="utf-8")):
            for m in _V2_CONTACT_RE.finditer(rec["situation"]):
                sigs.append((float(m["bearing"]), float(m["speed"]), float(m["heading"]), float(m["tcpa"])))
    return sigs


def _target_collides(target: dict, held_out: list[tuple[float, float, float, float]]) -> bool:
    b, s, h = target["bearing_from_os_deg"], target["speed"], target["heading_deg"]
    tcpa = target["_tcpa_min"]
    for hb, hs, hh, htcpa in held_out:
        if (abs(b - hb) <= BEARING_COLLISION_TOL_DEG and abs(s - hs) <= SPEED_COLLISION_TOL_KN
                and abs(h - hh) <= HEADING_COLLISION_TOL_DEG and abs(tcpa - htcpa) <= TCPA_COLLISION_TOL_MIN):
            return True
    return False


def _rec_collides(rec: dict, held_out: list[tuple[float, float, float, float]]) -> bool:
    return any(_target_collides(t, held_out) for t in rec["targets"])


def resample_train_away_from_held_out(train_recs: list[dict], rnd: random.Random,
                                      held_out: list[tuple[float, float, float, float]],
                                      max_attempts: int = MAX_REDRAW_ATTEMPTS) -> list[dict]:
    """Generation-time invariant: every TRAINING record must be geometrically disjoint
    (within tolerance) from the held-out v1/v2 set. A colliding record is replaced by a
    freshly-drawn instance of the SAME category, continuing the SAME rnd stream (never
    reset), up to `max_attempts` tries. Never touches eval records -- those must stay
    byte-for-byte reproducible as v1/v2. If max_attempts is exhausted, fails LOUDLY
    (the draw space for that category is too narrow -- see the Fase-B4 distinct-geometry
    report) rather than silently letting a duplicate through."""
    cat_by_name = {c["name"]: c for c in CATEGORIES}
    tpl_by_name = {t["name"]: t for t in MULTI_TARGET_TEMPLATES}
    out = []
    for rec in train_recs:
        candidate = rec
        attempts = 0
        while _rec_collides(candidate, held_out):
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"category {rec['category']!r}: could not draw a training geometry "
                    f"disjoint from the held-out v1/v2 set after {max_attempts} attempts -- "
                    "the random draw space for this category is too narrow (see the "
                    "distinct-geometry report); widen its training-side ranges before "
                    "regenerating, do not raise max_attempts to paper over it."
                )
            if rec["category"] in cat_by_name:
                candidate = generate_single_target_instance(cat_by_name[rec["category"]], rnd)
            elif rec["category"] in tpl_by_name:
                candidate = generate_multi_target_instance(tpl_by_name[rec["category"]], rnd)
            else:
                raise RuntimeError(f"unknown category {rec['category']!r}, cannot redraw")
        out.append(candidate)
    return out


FIXED_QUESTION = "Given the situation above, what action do you take this cycle, and why?"


def render_situation_narrative(rec: dict) -> str:
    """Deterministic MOOS-style narrative -- no LLM, exact format agreed with the user.
    Own-ship/contact phrasing ("Own-ship", 'Ship named "X"', "N other ship(s):") matches
    the Basic Simulator's narrate.py convention on purpose -- proven to work well there,
    and avoids a train/inference mismatch if the fine-tuned model is ever used in it."""
    own_speed = rec["own_speed"]
    cruise = own_speed + 7.0  # illustrative "still accelerating toward cruise" framing
    wx, wy = 0.0, 2000.0
    n = len(rec["targets"])
    lines = [
        f"Own-ship is underway at (0.0, 0.0), heading 0.0 degrees, speed {own_speed:.2f}. "
        f"Target cruise speed is {cruise:.1f} (not yet reached). "
        f"Next waypoint / mission objective is at ({wx:.1f}, {wy:.1f}).",
        f"{n} other ship{'s' if n != 1 else ''}:" if n else "No other ships tracked.",
    ]
    all_rules = set(BASE_RULES)
    for i, t in enumerate(rec["targets"]):
        name = CONTACT_NAME_POOL[i] if i < len(CONTACT_NAME_POOL) else f"RANDOM_TS{i + 1}"
        rel_brg = relative_bearing(0.0, t["bearing_from_os_deg"])
        x, y = t["start_xy_m"]
        range_m = math.hypot(x, y)
        closing = closing_rate(own_speed, t)
        all_rules |= set(t["_rules"])
        is_closing = closing > 0
        risk_txt = risk_level(t["_cpa_m"], t["_tcpa_min"], closing=is_closing)
        if is_closing:
            cpa_phrase = f"Projected CPA: {t['_cpa_m']:.0f} m in {t['_tcpa_min']:.1f} min."
        elif "_orig_cpa_m" in t:
            # Already past CPA (advance_past_cpa()'s output) -- the recomputed "CPA" from
            # here on is just the current, growing range, not the true closest approach,
            # so state the ACTUAL historical CPA/TCPA instead of the now-meaningless number.
            cpa_phrase = (f"Closest point of approach was {t['_orig_cpa_m']:.0f} m, "
                         f"{t['_orig_tcpa_min']:.1f} min ago; range is now increasing.")
        else:
            cpa_phrase = (f"Projected CPA: {t['_cpa_m']:.0f} m "
                         f"(already past closest point of approach; range now increasing).")
        lines.append(
            f'  - Ship named "{name}": range {range_m:.0f} m, {bow_phrase(rel_brg)} (relative bearing '
            f"{rel_brg:.1f} deg). She is on heading {t['heading_deg']:.1f}, speed {t['speed']:.2f}, "
            f"closing speed {closing:.2f}. {cpa_phrase} "
            f"Risk assessment: {risk_txt}. {role_sentence(t['_role'])}"
        )
    rule_order = sorted(all_rules, key=lambda r: int("".join(ch for ch in r if ch.isdigit()) or 0))
    lines.append(f"Applicable COLREG rules given the current situation: {', '.join(rule_order)}.")
    lines.append("Conditions: visibility is clear, (range 10000 m).")
    lines.append("Allowed actions this cycle: maintain_course, alter_course, set_speed, stop, resume_cruising_speed.")
    lines.append("Candidate action parameters: maintain_course; alter_course (degrees between -30 and 30, "
                 f"lookahead distance 200 m); set_speed (up to {cruise:.1f}); stop; resume_cruising_speed.")
    return "\n".join(lines)


# ── LLM rendering (Anthropic -- gold_answer prose ONLY; action+params are fixed) ──
RENDER_SYSTEM_PROMPT = """\
You write the "gold_answer" for an autonomous ship's Officer-of-the-Watch (Navigation) \
agent's training/eval data. Each input record gives you an ALREADY-DECIDED action and \
COLREG facts -- you NEVER change or second-guess the decision, you only phrase it as \
one fluent paragraph a real officer would say, never a telegraphic label:value dump.

Each input record has:
  - action: the chosen action name (maintain_course | alter_course | set_speed | stop | resume_cruising_speed)
  - action_params_text: the exact parameters already decided (state these verbatim,
    e.g. "alter_course (+30 degrees to starboard, lookahead distance 200 m)")
  - role: the encounter role(s) (mutual | give_way | stand_on | overtaking_give_way |
    overtaking_stand_on | cleared, or a '+'-joined combination for multi-contact encounters)
  - rules: the COLREG rule numbers engaged (state them by name, e.g. "Rule 15")

Produce ONE fluent paragraph per record: state the action being taken (in the agent's
own voice, e.g. "I will alter course..." / "I will maintain course and speed..."),
using the EXACT given parameters, then justify it with the given role and rule(s) in
natural prose. Do not invent a different action, additional manoeuvres, or a different
rule than given.

OUTPUT FORMAT: for EVERY record in the batch, return exactly one line of JSON \
(JSON Lines), nothing else -- no markdown fences, no commentary:
{"id": "<copied verbatim from input>", "gold_answer": "..."}
"""


def parse_jsonl_response(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip().strip("`").rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"    [warn] unparseable line skipped: {line[:100]}...", file=sys.stderr)
            continue
        rid = obj.get("id")
        if rid is not None:
            out[str(rid)] = obj
    return out


def render_batch(client, model: str, batch: list[dict], max_tokens: int) -> dict[str, dict]:
    payload = [{"id": r["_id"], "action": r["action"], "action_params_text": r["action_params_text"],
                "role": r["role"], "rules": r["rules"]} for r in batch]
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=RENDER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}],
    )
    return parse_jsonl_response(resp.content[0].text)


def render_all(client, model: str, recs: list[dict], batch_size: int, max_tokens: int,
               label: str) -> None:
    """Renders gold_answer IN PLACE onto each rec (situation_report/question are
    already deterministic, set before this is called)."""
    todo = recs
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        t0 = time.time()
        got = render_batch(client, model, batch, max_tokens)
        n_ok = 0
        for r in batch:
            rendered = got.get(r["_id"])
            if rendered:
                r["gold_answer"] = rendered.get("gold_answer", "")
                n_ok += 1
        print(f"  [{label}] batch {start // batch_size + 1}: {n_ok}/{len(batch)} rendered "
              f"({time.time() - t0:.1f}s, total {start + len(batch)}/{len(todo)})", flush=True)


# ══════════════════════════════════════════════════════════════════════════════════
# Fase B3 (RAG-rebuild-v2 plan, 2026-09-22, 25-first review gate): the reasoning text
# must DERIVE the applicable rule(s) from the raw situation facts, never restate a rule
# it was handed -- unlike RENDER_SYSTEM_PROMPT above (which is v1/legacy-only: it hands
# Claude the already-computed role/rules and asks it to phrase around them, and must
# stay byte-for-byte as-is for v1 reproducibility). This is the unified-format
# reasoning generator: situation text only, NO role/rule/degrees-justification labels in
# the input -- EXCEPT `decisive_contact` (name + CPA), which is answer-side context for
# the TEACHER only (same status as the already-given action/degrees), never part of
# Qwen's own user-facing prompt (render_scenario_situation()/build_user_message() never
# take or embed it -- see test_oow_scenarios_task_format.py's
# test_qwen_user_content_never_contains_the_decisive_contact_hint). Output is
# schema-validated AND gated (pipeline.track2.b3_reasoning_gates) before acceptance.
# ══════════════════════════════════════════════════════════════════════════════════
REASONING_SYSTEM_PROMPT = """\
You are an expert deck officer analysing a COLREG close-quarters situation. You are \
given a FUSED SITUATION REPORT (own-ship state, contact bearing/range/CPA/TCPA/speed/ \
heading -- no rule numbers or role labels), the ACTION already decided (never change, \
second-guess, or invent a different action/degrees than given), and -- only when a real \
collision risk exists -- which contact (`decisive_contact`, by name) actually drove that \
decision. Never mention any OTHER contact as if it were the reason for the decision.

Write ONE fluent reasoning paragraph, in the officer's own voice, that:
  1. Describes what the geometry of the decisive contact shows (closing or opening, \
which side it is on, how much risk it poses) in your own words -- never a template or \
label dump. If other contacts are present, you may mention them, but the decisive \
contact must be the one your reasoning is actually built on.
  2. From THAT geometry alone, determines which vessel is give-way/stand-on (if \
either) and names BOTH: the encounter rule (Rule 13 overtaking / Rule 14 head-on / \
Rule 15 crossing / 'none' if no real risk), and the conduct rule that governs the \
SPECIFIC action being taken (Rule 16 give-way turn/speed change, Rule 14 head-on turn, \
Rule 17 stand-on, Rule 8 a give-way vessel's own emergency stop -- Rule 17(b) is the \
STAND-ON vessel's provision only, never cite it for a give-way vessel's stop -- Rule 19 \
restricted visibility, or 'none' if no real risk). Whole rule numbers only, no
sub-paragraphs.
  3. States the given action (and degrees, if any) and justifies it under the conduct
rule.

Each input record has:
  - situation: the full fused situation report text
  - action: the action name already decided
  - degrees: turn amount in degrees (only for turn_left/turn_right, else null)
  - decisive_contact: {"name", "cpa_m"} of the contact that drove the decision (omitted
    when there is no real risk)

OUTPUT FORMAT: return ONLY one JSON object, no markdown fences, no commentary:
{"id": "<copied verbatim from input>", "action": "<copied verbatim from input>",
 "degrees": <copied verbatim from input, null if not given>,
 "encounter_rule": "Rule N or 'none'", "conduct_rule": "Rule N or 'none'", "reasoning": "..."}
"""


def contact_name_for_index(i: int) -> str:
    return CONTACT_NAME_POOL[i] if i < len(CONTACT_NAME_POOL) else f"RANDOM_TS{i + 1}"


def build_teacher_payload(rec: dict, rec_id: str) -> tuple[dict, dict]:
    """The TEACHER-only payload (includes decisive_contact -- answer-side info, never
    sent to Qwen) plus the ground-truth dict used both for row assembly and gating."""
    truth = to_unified_action(rec)
    situation = render_scenario_situation(rec)
    idx = truth["decisive_contact_index"]
    cpa_m = rec["targets"][idx]["_cpa_m"] if idx is not None else None
    payload = {"id": rec_id, "situation": situation, "action": truth["action"], "degrees": truth["degrees"]}
    decisive_name = None
    if idx is not None:
        decisive_name = contact_name_for_index(idx)
        payload["decisive_contact"] = {"name": decisive_name, "cpa_m": round(cpa_m, 0) if cpa_m is not None else None}
    return payload, {
        "situation": situation, "expected_action": truth["action"], "expected_degrees": truth["degrees"],
        "expected_encounter_rule": truth["encounter_rule"], "expected_conduct_rule": truth["conduct_rule"],
        "real_risk": truth["encounter_rule"] != "none", "cpa_m": cpa_m, "safe_distance_m": SAFE_CPA_M,
        "decisive_contact_name": decisive_name,
    }


def render_reasoning_review_sample(client, model: str, recs: list[dict], n: int, seed: int,
                                   max_attempts: int, max_tokens: int) -> dict:
    """B3's 25-first review gate: stratified sample across categories (never the full
    population), REAL Anthropic calls, EVERY row passes through the gated
    generate/retry/drop loop (pipeline.track2.b3_reasoning_gates) before acceptance.
    Returns {"accepted": [...], "rejected": [...], "gate_rejection_counts": {...}} --
    writes NOTHING to any production file."""
    from pipeline.track2.b3_reasoning_gates import generate_gated_row, GATE_NAMES

    by_cat: dict[str, list[dict]] = {}
    for r in recs:
        by_cat.setdefault(r["category"], []).append(r)
    rnd = random.Random(seed)
    for v in by_cat.values():
        rnd.shuffle(v)
    cat_names = list(by_cat)
    sample: list[dict] = []
    i = 0
    while len(sample) < n and any(by_cat.values()):
        cat = cat_names[i % len(cat_names)]
        if by_cat[cat]:
            sample.append(by_cat[cat].pop())
        i += 1

    accepted, rejected = [], []
    gate_rejection_counts = {name: 0 for name in GATE_NAMES}
    for r in sample:
        payload, expected = build_teacher_payload(r, r["_id"])
        obj, attempt_log = generate_gated_row(
            client, model, REASONING_SYSTEM_PROMPT, payload, max_attempts=max_attempts,
            max_tokens=max_tokens, **expected,
        )
        for attempt in attempt_log:
            for gate_name in attempt["failures"]:
                gate_rejection_counts[gate_name] += 1
        row = {
            "id": r["_id"], "category": r["category"], "situation": expected["situation"],
            "ground_truth": {"action": expected["expected_action"], "degrees": expected["expected_degrees"],
                            "encounter_rule": expected["expected_encounter_rule"],
                            "conduct_rule": expected["expected_conduct_rule"]},
            "n_attempts": len(attempt_log), "attempt_log": attempt_log,
        }
        if obj is not None:
            row["model_response"] = obj
            accepted.append(row)
        else:
            rejected.append(row)
    return {"accepted": accepted, "rejected": rejected, "gate_rejection_counts": gate_rejection_counts}


B3_CHECKPOINT_FILE = CACHE / "oow_scenario_b3_checkpoint.jsonl"


def load_b3_checkpoint() -> dict[str, dict]:
    """_id -> {"accepted": bool, "reasoning": str|None}, from every row a prior
    --b3-full-population run already completed -- resumable across crashes/interrupts,
    same pattern as build_oow_scenarios_leo.py's load_checkpoint()."""
    if not B3_CHECKPOINT_FILE.exists():
        return {}
    out: dict[str, dict] = {}
    for line in B3_CHECKPOINT_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def append_b3_checkpoint(rows: list[dict]) -> None:
    B3_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with B3_CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


def render_reasoning_full_population(client, model: str, recs: list[dict], max_attempts: int,
                                     max_tokens: int) -> dict:
    """Full-population version of render_reasoning_review_sample(): EVERY record in
    `recs` (not a sample) passes through the SAME gated generate/retry/drop loop.
    Checkpointed (B3_CHECKPOINT_FILE) so a crash/interrupt only loses the single
    in-flight row, not the whole run -- re-running skips every _id already checkpointed.
    Sets r["reasoning"] IN PLACE on accepted records (never on dropped ones, which stay
    None and are naturally excluded by write_scenario_*_files()'s own
    `if not r.get("reasoning")` guard). Returns {"n_accepted", "n_rejected",
    "gate_rejection_counts"} -- the counts only cover THIS run's fresh generations, not
    rows resumed from a prior checkpoint."""
    from pipeline.track2.b3_reasoning_gates import generate_gated_row, GATE_NAMES

    done = load_b3_checkpoint()
    if done:
        print(f"  [B3] resume: {len(done)}/{len(recs)} row(s) already checkpointed")
    gate_rejection_counts = {name: 0 for name in GATE_NAMES}
    n_accepted = n_rejected = 0
    for i, r in enumerate(recs):
        cp = done.get(r["_id"])
        if cp is not None:
            if cp["accepted"]:
                r["reasoning"] = cp["reasoning"]
                n_accepted += 1
            else:
                n_rejected += 1
            continue
        payload, expected = build_teacher_payload(r, r["_id"])
        obj, attempt_log = generate_gated_row(
            client, model, REASONING_SYSTEM_PROMPT, payload, max_attempts=max_attempts,
            max_tokens=max_tokens, **expected,
        )
        for attempt in attempt_log:
            for gate_name in attempt["failures"]:
                gate_rejection_counts[gate_name] += 1
        if obj is not None:
            r["reasoning"] = obj["reasoning"]
            n_accepted += 1
            append_b3_checkpoint([{"id": r["_id"], "accepted": True, "reasoning": obj["reasoning"]}])
        else:
            n_rejected += 1
            append_b3_checkpoint([{"id": r["_id"], "accepted": False, "reasoning": None}])
        done_now = i + 1
        if done_now % 20 == 0 or done_now == len(recs):
            print(f"  [B3] {done_now}/{len(recs)} processed ({n_accepted} accepted, "
                 f"{n_rejected} dropped)", flush=True)
    return {"n_accepted": n_accepted, "n_rejected": n_rejected, "gate_rejection_counts": gate_rejection_counts}


def wrong_action_variant(decision: dict) -> dict:
    """A plausible but COLREG-INCORRECT alternative decision, for DPO 'rejected' answers
    -- deterministic field swap on the already-computed unified action (same principle,
    same code shape, as build_oow_scenarios_leo.py's own wrong_action_variant()). Replaces
    the old second-Claude-call render_wrong_all() prose-perturbation approach: there is no
    continuous prose to reformat here, the DPO 'rejected' side is a structured JSON action,
    so a deterministic field swap is the correct (not a shortcut) way to build it."""
    action, degrees = decision["action"], decision["degrees"]
    if action in ("turn_left", "turn_right"):
        wrong_action = "turn_left" if action == "turn_right" else "turn_right"
        return {"action": wrong_action, "degrees": degrees,
               "encounter_rule": decision["encounter_rule"], "conduct_rule": decision["conduct_rule"]}
    if action == "hold_course":
        # Wrong: manoeuvring when no real risk exists.
        return {"action": "turn_right", "degrees": MIN_TURN_DEG,
               "encounter_rule": "none", "conduct_rule": "none"}
    # Wrong response to a stop/speed-up-worthy situation: holding course instead.
    return {"action": "hold_course", "degrees": None, "encounter_rule": "none", "conduct_rule": "none"}


def _assistant_fields(unified: dict) -> dict:
    """Strips decisive_contact_index (teacher-only metadata) before a unified action
    dict is embedded as an assistant/chosen/rejected JSON response."""
    return {k: v for k, v in unified.items() if k != "decisive_contact_index"}


def write_scenario_sft_files(recs: list[dict], cache_dir: Path, mode: str = "w") -> None:
    """Writes oow_scenario_sft_direct.jsonl / _cot.jsonl in the UNIFIED task format (Fase
    B2, RAG-rebuild-v2 plan): system=SYSTEM_OOW_AGENT, user=user_message_for(r) (Fase B4's
    optional previous-decisions preamble + render_scenario_situation(r) -- the latter is
    the SAME renderer oow_colreg_scenarios_v2.json's eval records use, see
    tests/test_oow_scenarios_v2_shared_renderer_identity.py), assistant=the schema-validated
    JSON object {action, degrees, encounter_rule, conduct_rule, reasoning}. `reasoning`
    comes from Fase B3's gated Claude-authored text, set in place on each rec by
    render_reasoning_full_population() before this is called. direct/cot
    share the same content, matching the old writer's own rationale (one fixed question,
    one fused decision+reasoning paragraph, no separate terse/step-by-step version to
    write). `mode="a"` appends new-category rows onto already-committed files."""
    direct_path = cache_dir / "oow_scenario_sft_direct.jsonl"
    cot_path = cache_dir / "oow_scenario_sft_cot.jsonl"
    with direct_path.open(mode, encoding="utf-8") as fd, cot_path.open(mode, encoding="utf-8") as fc:
        for r in recs:
            if not r.get("reasoning"):
                continue
            unified = to_unified_action(r)
            assistant = {**_assistant_fields(unified), "reasoning": r["reasoning"]}
            row = {
                "category": r["category"], "action": r["action"],
                "messages": [
                    {"role": "system", "content": SYSTEM_OOW_AGENT},
                    {"role": "user", "content": user_message_for(r)},
                    {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
                ],
            }
            line = json.dumps(row, ensure_ascii=False) + "\n"
            fd.write(line)
            fc.write(line)
    print(f"Wrote {direct_path.name} and {cot_path.name}")


def user_message_for(r: dict) -> str:
    """The full user-turn text for record `r`: render_scenario_situation()'s deterministic
    rendering plus FIXED_QUESTION_UNIFIED, with Fase B4's real-history preamble prepended
    when r["prev_decisions"] is set (never touches render_scenario_situation()'s own
    output, so v2 eval-file identity is unaffected -- see
    test_v2_eval_record_situation_is_byte_identical_to_what_a_training_row_would_embed)."""
    history_prefix = render_previous_decisions(r.get("prev_decisions"))
    return f"Situation:\n{history_prefix}{render_scenario_situation(r)}\n\n{FIXED_QUESTION_UNIFIED}"


def write_scenario_dpo_file(recs: list[dict], cache_dir: Path, mode: str = "w") -> None:
    out_path = cache_dir / "oow_scenario_dpo_pairs.jsonl"
    n = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for r in recs:
            if not r.get("reasoning"):
                continue
            unified = _assistant_fields(to_unified_action(r))
            chosen = {**unified, "reasoning": r["reasoning"]}
            rejected_action = wrong_action_variant(unified)
            rejected = {**rejected_action, "reasoning": r["reasoning"]}
            user_msg = user_message_for(r)
            row = {
                "category": r["category"], "action": r["action"],
                "prompt": [{"role": "system", "content": SYSTEM_OOW_AGENT}, {"role": "user", "content": user_msg}],
                "chosen": [{"role": "assistant", "content": json.dumps(chosen, ensure_ascii=False)}],
                "rejected": [{"role": "assistant", "content": json.dumps(rejected, ensure_ascii=False)}],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {out_path.name} ({n} pairs)")


def write_scenario_reflection_file(recs: list[dict], cache_dir: Path, mode: str = "w") -> None:
    """Draft/Critique/Refined triples, same convention as build_reflection.py's output
    (see oow_reflection.jsonl): draft = the bare action name with no parameters or rule
    citation (deliberately vague, not wrong), critique = fixed text pointing out exactly
    that gap, refined = the unified JSON object with the full B3 reasoning text."""
    out_path = cache_dir / "oow_scenario_reflection.jsonl"
    n = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for r in recs:
            if not r.get("reasoning"):
                continue
            unified = _assistant_fields(to_unified_action(r))
            refined = json.dumps({**unified, "reasoning": r["reasoning"]}, ensure_ascii=False)
            draft = f"I will {r['action'].replace('_', ' ')}."
            critique = ("This response is too vague -- it must state the exact action parameters "
                       "and cite the specific COLREG rule(s) that justify the decision.")
            row = {
                "category": r["category"],
                "messages": [
                    {"role": "system", "content": SYSTEM_OOW_AGENT},
                    {"role": "user", "content": user_message_for(r)},
                    {"role": "assistant", "content": f"Draft: {draft}\n\nCritique: {critique}\n\n"
                                                      f"Refined: {refined}"},
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {out_path.name} ({n} rows)")


# ── Output assembly ────────────────────────────────────────────────────────────
def to_eval_record(rec: dict, idx: int) -> dict:
    contacts = [{"bearing_from_os_deg": t["bearing_from_os_deg"], "speed": t["speed"],
                 "heading_deg": t["heading_deg"], "cpa_m": round(t["_cpa_m"], 1),
                 "tcpa_min": t["_tcpa_min"], "role": t["_role"], "rules": t["_rules"]}
                for t in rec["targets"]]
    return {
        "id": f"oowcol_{idx:05d}", "category": rec["category"],
        "own_speed": rec["own_speed"], "contacts": contacts,
        "give_way_role": rec["role"], "colreg_rules": rec["rules"],
        "correct_action": rec["action"], "correct_action_params": rec["action_params"],
        "situation_report": rec["situation_report"],
        "question": FIXED_QUESTION, "expected_points": rec["pass_criteria"],
        "gold_answer": rec.get("gold_answer", ""),
    }


def to_trace_record(rec: dict, idx: int) -> dict:
    """Shared reasoning-trace schema (see § 5 extract_reasoning.py mapping) so
    build_sft.py / build_rlhf.py / build_reflection.py / build_multihop.py consume
    this with zero code changes -- same convention as the real-incident traces."""
    doc_id = f"oow_scenario_{rec['category']}_{idx:05d}"
    steps = [
        {"step": 1, "action": "Assess the fused contact picture (bearing/range/CPA/TCPA) "
                              "against own-ship's course and speed.",
         "why": "Rule 7 requires determining if risk of collision exists using all available means."},
        {"step": 2, "action": f"Determine own-ship's role: {rec['role'].replace('_', ' ')}.",
         "why": f"{', '.join(rec['rules'])} governs this encounter."},
        {"step": 3, "action": f"Execute the action: {rec['action_params_text']}.",
         "why": "The chosen action must comply with the applicable COLREG rule(s) above."},
    ]
    return {
        "document_id": doc_id, "chunk_id": doc_id,
        "source_file": f"oow_scenario_generator::{rec['category']}",
        "chapter_title": f"OOW Track 2 scenario: {rec['category']}",
        "chunk_concepts": [rec["category"], rec["role"]] + rec["rules"],
        "trace": {
            "situation": rec["situation_report"],
            "trigger": None,
            "procedures": steps,
            "constraints": rec["pass_criteria"],
            "prowords_used": [rec["role"]],
            "channels": rec["rules"],
            "regulations": ["COLREG 1972"],
            "warnings": [c for c in rec["pass_criteria"] if "not" in c.lower() or "avoid" in c.lower()],
            "outcomes": [f"Action taken: {rec['action_params_text']}."],
            "key_facts": [f"Own-ship speed {rec['own_speed']}.",
                         f"Chosen action: {rec['action']}."],
            "question_seeds": [{"angle": "what", "text": FIXED_QUESTION}],
        },
    }


# ── Contamination filter ───────────────────────────────────────────────────────
def filter_contamination(train_recs: list[dict], gold_questions: list[str]) -> list[dict]:
    """Filters training gold_answer text against Track 1's held-out exam questions ONLY
    (not against this script's own new Track 2 eval file). The deterministic narrative/
    action text here is intentionally formulaic across categories -- a self-comparison
    measured 0.89-0.95 MiniLM cosine similarity between DIFFERENT categories purely from
    shared sentence structure (false-positive-drops almost everything), whereas Track 1's
    naturally differently-phrased exam Q&A scores 0.48-0.57, a genuinely meaningful signal."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER_MODEL)
    gold_embs = model.encode(gold_questions, normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False)
    texts = [r.get("gold_answer", "") for r in train_recs]
    q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64,
                          show_progress_bar=False)
    kept, dropped = [], 0
    for rec, emb in zip(train_recs, q_embs):
        c_sim = float(np.max(gold_embs @ emb)) if gold_questions else 0.0
        if c_sim >= CONTAM_THRESH:
            dropped += 1
            continue
        kept.append(rec)
    print(f"contamination filter: kept={len(kept)}  dropped={dropped} (thresh={CONTAM_THRESH})")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-eval-per-category", type=int, default=N_EVAL_PER_CATEGORY_DEFAULT)
    ap.add_argument("--n-train-per-category", type=int, default=N_TRAIN_PER_CATEGORY_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default=MODEL_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    ap.add_argument("--gold-file", type=str,
                    default=str(paths.eval_dir / "colreg_qa_500_normalised.json"),
                    help="Track 1 held-out file to also filter training gold_answer text against")
    ap.add_argument("--smoke", action="store_true", help="tiny population, still real LLM calls")
    ap.add_argument("--skip-llm", action="store_true",
                    help="geometry+narrative only -- no API calls, no gold_answer, for free dry-run testing")
    ap.add_argument("--only-category", type=str, default=None,
                    help="restrict generation to a single named category (e.g. risk_cleared_resume) "
                         "instead of the full population -- combine with --append to add that one "
                         "category's data on top of already-committed files.")
    ap.add_argument("--append", action="store_true",
                    help="append to existing output files (train jsonl + reasoning traces in 'a' "
                         "mode; eval JSON is loaded, extended with continuing ids, and rewritten) "
                         "instead of overwriting them -- for adding one new category's data without "
                         "touching/regenerating already-reviewed categories.")
    ap.add_argument("--build-v2", action="store_true",
                    help="RAG-rebuild-v2 plan point 2: build oow_colreg_scenarios_v2.json (the "
                         "SAME 325 v1 geometries, re-rendered in the unified task format) and exit "
                         "-- an explicit, standalone step that NEVER runs alongside training-data "
                         "regeneration, so the eval set can never silently drift when training is "
                         "rebuilt. Writes via safe_write_jsonl-style no-overwrite protection "
                         "(pass --overwrite to allow replacing an existing v2 file).")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow --build-v2 to overwrite an existing oow_colreg_scenarios_v2.json")
    ap.add_argument("--b3-review-sample", type=int, default=None,
                    help="Fase B3 25-first review gate: generate this many REAL Anthropic "
                         "reasoning-derivation calls (REASONING_SYSTEM_PROMPT, stratified across "
                         "categories from the training pool) and write them to _review/ for human "
                         "review, then exit -- never runs the full population, never touches any "
                         "production file.")
    ap.add_argument("--b3-full-population", action="store_true",
                    help="Fase B3 full run (post 25-first-gate approval): gated reasoning "
                         "generation for EVERY training record (not a sample), checkpointed "
                         "(B3_CHECKPOINT_FILE) so it can resume after a crash/interrupt. Writes "
                         "oow_scenario_sft_direct/_cot/_dpo_pairs/_reflection.jsonl + the training "
                         "traces file -- never touches EVAL_OUT (v1, frozen) or gold_answer.")
    args = ap.parse_args()

    if args.build_v2:
        v2_records = build_v2_eval_records()
        out_path = EVAL_OUT_V2 if args.overwrite else review_path(EVAL_OUT_V2)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} already exists -- pass --overwrite for production")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(v2_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_path} ({len(v2_records)} v2 scenarios)"
             + ("" if args.overwrite else "  [dry-run/review path -- pass --overwrite for production]"))
        from collections import Counter
        print("category distribution:", Counter(r["category"] for r in v2_records))
        return

    if args.b3_review_sample is not None:
        load_env(paths.env_file)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set in .env")
        import anthropic
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        client = anthropic.Anthropic(api_key=key, default_headers=headers)
        pop = generate_population(N_EVAL_PER_CATEGORY_DEFAULT + N_TRAIN_PER_CATEGORY_DEFAULT, seed=args.seed)
        _, train_recs = split_eval_train(pop, N_EVAL_PER_CATEGORY_DEFAULT)
        for i, r in enumerate(train_recs):
            r["_id"] = f"t{i:05d}"
        result = render_reasoning_review_sample(
            client, args.model, train_recs, args.b3_review_sample, args.seed,
            max_attempts=3, max_tokens=args.max_tokens)
        out_path = review_path(CACHE / "oow_scenario_b3_review_sample.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in result["accepted"] + result["rejected"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_acc, n_rej = len(result["accepted"]), len(result["rejected"])
        print(f"Wrote {out_path} ({n_acc + n_rej} review rows: {n_acc} accepted, {n_rej} dropped)")
        print(f"Per-gate rejection counts (across all attempts): {result['gate_rejection_counts']}")
        if result["rejected"]:
            print(f"  DROPPED ids (exhausted retries): {[r['id'] for r in result['rejected']]}")
        print("This is a REVIEW-ONLY sample -- no production file was written. "
             "STOP here pending human review before running the full population.")
        return

    if args.b3_full_population:
        load_env(paths.env_file)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set in .env")
        import anthropic
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        client = anthropic.Anthropic(api_key=key, default_headers=headers)
        pop, rnd = generate_population(N_EVAL_PER_CATEGORY_DEFAULT + N_TRAIN_PER_CATEGORY_DEFAULT,
                                       seed=args.seed, return_rng=True)
        _, train_recs = split_eval_train(pop, N_EVAL_PER_CATEGORY_DEFAULT)
        held_out = load_held_out_signatures()
        if held_out:
            train_recs = resample_train_away_from_held_out(train_recs, rnd, held_out)
        for i, r in enumerate(train_recs):
            r["_id"] = f"t{i:05d}"
        # to_trace_record() needs situation_report -- 100% deterministic, no API call,
        # same line the OLD render_all()-based flow uses right after split_eval_train().
        for r in train_recs:
            r["situation_report"] = render_situation_narrative(r)
        print(f"Fase B3 full population: {len(train_recs)} training records...")
        result = render_reasoning_full_population(client, args.model, train_recs,
                                                   max_attempts=3, max_tokens=args.max_tokens)
        print(f"Done: {result['n_accepted']} accepted, {result['n_rejected']} dropped "
             f"(this run's fresh generations only)")
        print(f"Per-gate rejection counts (this run's fresh generations): "
             f"{result['gate_rejection_counts']}")
        # Fase B4: this synthetic generator has no real multi-step trajectories, so
        # "previous decisions" are reinforced self-consistency (the SAME action that is
        # already correct for this snapshot, shown as if already established) rather than
        # Leo's genuinely-real predecessor states -- applied to 1-in-5 real-risk rows only
        # (a no-risk hold_course history would be a trivial, uninformative signal).
        n_with_history = 0
        for i, r in enumerate(train_recs):
            unified = to_unified_action(r)
            if i % 5 == 0 and unified["encounter_rule"] != "none":
                r["prev_decisions"] = [{"action": unified["action"], "degrees": unified["degrees"]}] * 2
                n_with_history += 1
            else:
                r["prev_decisions"] = None
        print(f"Fase B4: {n_with_history}/{len(train_recs)} rows given a previous-decisions "
             "history preamble")
        trace_out = [to_trace_record(r, i + 1) for i, r in enumerate(train_recs)]
        TRACES_OUT.parent.mkdir(parents=True, exist_ok=True)
        with TRACES_OUT.open("w", encoding="utf-8") as f:
            for t in trace_out:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"Wrote {TRACES_OUT} ({len(trace_out)} training traces)")
        write_scenario_sft_files(train_recs, CACHE, mode="w")
        write_scenario_dpo_file(train_recs, CACHE, mode="w")
        write_scenario_reflection_file(train_recs, CACHE, mode="w")
        return

    n_eval = 2 if args.smoke else args.n_eval_per_category
    n_train = 2 if args.smoke else args.n_train_per_category

    categories_n = 1 if args.only_category else len(CATEGORIES) + len(MULTI_TARGET_TEMPLATES)
    print(f"Generating population: {n_eval + n_train} per category ({categories_n} categories)...")
    pop, rnd = generate_population(n_eval + n_train, seed=args.seed, only_category=args.only_category,
                                   return_rng=True)
    eval_recs, train_recs = split_eval_train(pop, n_eval)

    held_out = load_held_out_signatures()
    if held_out:
        n_before = len(train_recs)
        train_recs = resample_train_away_from_held_out(train_recs, rnd, held_out)
        print(f"Held-out contamination avoidance: checked {n_before} training records "
             f"against {len(held_out)} held-out (v1/v2) contact geometries.")
    print(f"eval pool: {len(eval_recs)}  train pool: {len(train_recs)}  "
          f"(geometrically disjoint by construction)")

    # situation_report is 100% deterministic -- computed for every record regardless
    # of --skip-llm, since it needs no API call at all.
    for r in eval_recs + train_recs:
        r["situation_report"] = render_situation_narrative(r)

    for i, r in enumerate(eval_recs):
        r["_id"] = f"e{i:05d}"
    for i, r in enumerate(train_recs):
        r["_id"] = f"t{i:05d}"

    if args.skip_llm:
        print("--skip-llm: situation_report/question/action are set; gold_answer left empty.")
    else:
        load_env(paths.env_file)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set in .env")
        import anthropic
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        client = anthropic.Anthropic(api_key=key, default_headers=headers)
        render_all(client, args.model, eval_recs, args.batch_size, args.max_tokens, "eval")
        render_all(client, args.model, train_recs, args.batch_size, args.max_tokens, "train")

    existing_eval: list[dict] = []
    if args.append and EVAL_OUT.exists():
        existing_eval = json.loads(EVAL_OUT.read_text(encoding="utf-8"))
    id_offset = len(existing_eval)
    eval_out = existing_eval + [to_eval_record(r, id_offset + i + 1) for i, r in enumerate(eval_recs)]
    # v1 is FROZEN (RAG-rebuild-v2 plan point 1) -- refuse to silently rewrite it. --append
    # (adding one new category on top) is still allowed since that's an intentional,
    # explicit extension, not an accidental full regeneration.
    if EVAL_OUT.exists() and not args.append:
        raise FileExistsError(
            f"{EVAL_OUT} is FROZEN (v1) and must never be silently regenerated -- pass "
            f"--append to add new categories on top, or delete it yourself if a full "
            f"regeneration is really intended (this will fail the v1 sha256 pin test)."
        )
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.write_text(json.dumps(eval_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {EVAL_OUT} ({len(eval_out)} held-out scenarios, {len(eval_recs)} new)")

    if not args.skip_llm:
        # Track 1 only -- see filter_contamination()'s docstring for why comparing against
        # this script's own eval file would false-positive almost everything.
        gold_questions = [g["question"] for g in json.loads(Path(args.gold_file).read_text(encoding="utf-8"))]
        train_recs = filter_contamination(train_recs, gold_questions)

    trace_mode = "a" if args.append else "w"
    trace_out = [to_trace_record(r, i + 1) for i, r in enumerate(train_recs)]
    TRACES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TRACES_OUT.open(trace_mode, encoding="utf-8") as f:
        for t in trace_out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Wrote {TRACES_OUT} ({len(trace_out)} {'new' if args.append else ''} training traces)")

    # NOTE (Fase B2 unification): the old second Claude pass (render_wrong_all(), building
    # a "wrong_answer" prose text for DPO's rejected side) is gone -- write_scenario_dpo_file
    # now builds the rejected side deterministically via wrong_action_variant() on the
    # unified JSON action, same principle as build_oow_scenarios_leo.py. One fewer LLM call.

    # NOTE (Fase B3, RAG-rebuild-v2 plan, 2026-09-22): the training-row writers below now
    # read r["reasoning"] (B3's gated, geometry-derived reasoning text), NOT the OLD
    # render_all()-produced r["gold_answer"] (still populated above -- v1's own eval file
    # needs it, unchanged). The B3 FULL-population reasoning generator is NOT implemented
    # yet -- only --b3-review-sample's review-only gate has landed, pending a second
    # 25-review per the plan's own gate. Until that lands, r["reasoning"] is never set
    # here, so these writers emit zero rows (their own `if not r.get("reasoning")` guard) --
    # an empty-but-honest output, not a silent fabrication.
    write_scenario_sft_files(train_recs, CACHE, mode=trace_mode)
    write_scenario_dpo_file(train_recs, CACHE, mode=trace_mode)
    write_scenario_reflection_file(train_recs, CACHE, mode=trace_mode)


if __name__ == "__main__":
    main()
