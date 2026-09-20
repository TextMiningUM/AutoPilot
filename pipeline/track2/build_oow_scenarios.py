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
- Data/OOW/OOW_Eval/oow_colreg_scenarios.json
      Held-out eval set (NEVER used for training).
- Data/OOW/OOW_Agents_Training/oow_scenario_sft_direct.jsonl / _cot.jsonl
      Training SFT rows, written DIRECTLY from the rendered situation_report/
      gold_answer (NOT via build_sft.py -- see write_scenario_sft_files()'s
      docstring for why that generic builder doesn't fit this fixed-question
      data shape).
- Data/OOW/OOW_Agents_Training/oow_scenario_dpo_pairs.jsonl
      DPO pairs; the rejected side is a second Claude pass over a
      deliberately-wrong action (see wrong_action_variant()).
- Data/OOW/OOW_Agents_Training/oow_scenario_reflection.jsonl
      Draft/Critique/Refined triples (draft = vague action, no parameters/rule).
- Data/OOW/OOW_Agents_Training/oow_scenario_reasoning_traces.jsonl
      Reasoning-trace-schema copy of the training scenarios, kept for
      build_pg.py (Procedural Graph) and build_multihop.py (cross-track
      pairing) ONLY -- not used for SFT/DPO/reflection (see above).

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
import sys
import time
from pathlib import Path

import numpy as np

from core import AgentPaths, load_env, EMBEDDER_MODEL, CONTAM_THRESH

paths = AgentPaths.oow()
W = paths.workspace
CACHE = paths.cache_dir
EVAL_OUT = paths.eval_dir / "oow_colreg_scenarios.json"
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


def relative_bearing(own_heading: float, true_bearing: float) -> float:
    """Bearing of a contact relative to own-ship's bow, -180..+180, positive = starboard."""
    return (true_bearing - own_heading + 540) % 360 - 180


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
    heuristic (see module docstring) -- NOT a full COLREG-compliant manoeuvre planner."""
    give_way_present = any(r in ("mutual", "give_way", "overtaking_give_way") for r in contact_roles)
    if not give_way_present:
        return "maintain_course", {}, "maintain_course"
    if worst_tcpa_min < 1.5:
        return "stop", {}, "stop"
    degrees, lookahead_m = 30, 200
    return ("alter_course", {"degrees": degrees, "lookahead_distance_m": lookahead_m},
            f"alter_course (+{degrees} degrees to starboard, lookahead distance {lookahead_m} m)")


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


def generate_population(n_per_category: int, seed: int = 0, only_category: str | None = None) -> list[dict]:
    """Deterministic (seeded) population of scenario-fact dicts, no prose yet.
    Same seed -> same population every time (reproducibility for train/eval split).
    `only_category` restricts generation to a single named category -- used to add ONE
    new category's data on top of already-committed files without touching/regenerating
    the other (already reviewed) categories."""
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


def wrong_action_variant(action: str, params: dict) -> tuple[str, dict, str]:
    """A plausible but COLREG-INCORRECT alternative action+params, for DPO 'rejected'
    answers -- e.g. turning the wrong way, or not acting as give-way vessel."""
    if action == "alter_course":
        degrees = params.get("degrees", 30)
        lookahead = params.get("lookahead_distance_m", 200)
        return ("alter_course", {"degrees": -degrees, "lookahead_distance_m": lookahead},
                f"alter_course ({degrees} degrees to port, lookahead distance {lookahead} m)")
    if action == "maintain_course":
        return ("alter_course", {"degrees": 30, "lookahead_distance_m": 200},
                "alter_course (+30 degrees to starboard, lookahead distance 200 m)")
    return "maintain_course", {}, "maintain_course"  # wrong response to a stop-worthy emergency


def render_wrong_all(client, model: str, recs: list[dict], batch_size: int, max_tokens: int) -> None:
    """Second pass: renders a 'wrong_answer' onto each rec by asking Claude to phrase the
    SAME facts but with `wrong_action_variant()`'s incorrect action -- Claude doesn't know
    it's wrong, it just phrases whatever action/params it's given, exactly like the real
    system prompt. This is the DPO 'rejected' side, built the same way as the 'chosen' side
    (never a string-edit of already-rendered prose)."""
    todo = recs
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        payload = []
        for r in batch:
            _, _, wrong_text = wrong_action_variant(r["action"], r["action_params"])
            payload.append({"id": r["_id"], "action": wrong_action_variant(r["action"], r["action_params"])[0],
                           "action_params_text": wrong_text, "role": r["role"], "rules": r["rules"]})
        t0 = time.time()
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=RENDER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}],
        )
        got = parse_jsonl_response(resp.content[0].text)
        n_ok = 0
        for r in batch:
            rendered = got.get(r["_id"])
            if rendered:
                r["wrong_answer"] = rendered.get("gold_answer", "")
                n_ok += 1
        print(f"  [dpo-rejected] batch {start // batch_size + 1}: {n_ok}/{len(batch)} rendered "
              f"({time.time() - t0:.1f}s, total {start + len(batch)}/{len(todo)})", flush=True)


def write_scenario_sft_files(recs: list[dict], cache_dir: Path, mode: str = "w") -> None:
    """Writes oow_scenario_sft_direct.jsonl / _cot.jsonl directly from the already-rendered
    situation_report/gold_answer -- NOT via build_sft.py, which (a) assumes per-record
    question DIVERSITY for its dedup filter (ours is one fixed question, so it would drop
    ~all rows as 'duplicates'), and (b) reconstructs its own generic answer text from the
    trace's structured fields instead of using the Claude-authored gold_answer. direct/cot
    share the same content: the gold_answer already fuses the decision with its reasoning
    in one paragraph, so there's no separate 'terse' vs 'step-by-step' version to write.
    No _rag.jsonl: there's no Track 2 retrieval corpus for this data to ground against.
    `mode="a"` appends new-category rows onto already-committed files instead of
    overwriting the other, already-reviewed categories."""
    direct_path = cache_dir / "oow_scenario_sft_direct.jsonl"
    cot_path = cache_dir / "oow_scenario_sft_cot.jsonl"
    with direct_path.open(mode, encoding="utf-8") as fd, cot_path.open(mode, encoding="utf-8") as fc:
        for r in recs:
            if not r.get("gold_answer"):
                continue
            row = {
                "category": r["category"], "action": r["action"],
                "messages": [
                    {"role": "system", "content": SYSTEM_OOW},
                    {"role": "user", "content": f"{r['situation_report']}\n\n{FIXED_QUESTION}"},
                    {"role": "assistant", "content": r["gold_answer"]},
                ],
            }
            line = json.dumps(row, ensure_ascii=False) + "\n"
            fd.write(line)
            fc.write(line)
    print(f"Wrote {direct_path.name} and {cot_path.name}")


def write_scenario_dpo_file(recs: list[dict], cache_dir: Path, mode: str = "w") -> None:
    out_path = cache_dir / "oow_scenario_dpo_pairs.jsonl"
    n = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for r in recs:
            if not r.get("gold_answer") or not r.get("wrong_answer"):
                continue
            user_msg = f"{r['situation_report']}\n\n{FIXED_QUESTION}"
            row = {
                "category": r["category"], "action": r["action"],
                "prompt": [{"role": "system", "content": SYSTEM_OOW}, {"role": "user", "content": user_msg}],
                "chosen": [{"role": "assistant", "content": r["gold_answer"]}],
                "rejected": [{"role": "assistant", "content": r["wrong_answer"]}],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {out_path.name} ({n} pairs)")


def write_scenario_reflection_file(recs: list[dict], cache_dir: Path, mode: str = "w") -> None:
    """Draft/Critique/Refined triples, same convention as build_reflection.py's output
    (see oow_reflection.jsonl): draft = the bare action name with no parameters or rule
    citation (deliberately vague, not wrong), critique = fixed text pointing out exactly
    that gap, refined = the gold_answer verbatim."""
    out_path = cache_dir / "oow_scenario_reflection.jsonl"
    n = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for r in recs:
            if not r.get("gold_answer"):
                continue
            draft = f"I will {r['action'].replace('_', ' ')}."
            critique = ("This response is too vague -- it must state the exact action parameters "
                       "and cite the specific COLREG rule(s) that justify the decision.")
            row = {
                "category": r["category"],
                "messages": [
                    {"role": "system", "content": SYSTEM_OOW},
                    {"role": "user", "content": f"{r['situation_report']}\n\n{FIXED_QUESTION}"},
                    {"role": "assistant", "content": f"Draft: {draft}\n\nCritique: {critique}\n\n"
                                                      f"Refined: {r['gold_answer']}"},
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
    args = ap.parse_args()

    n_eval = 2 if args.smoke else args.n_eval_per_category
    n_train = 2 if args.smoke else args.n_train_per_category

    categories_n = 1 if args.only_category else len(CATEGORIES) + len(MULTI_TARGET_TEMPLATES)
    print(f"Generating population: {n_eval + n_train} per category ({categories_n} categories)...")
    pop = generate_population(n_eval + n_train, seed=args.seed, only_category=args.only_category)
    eval_recs, train_recs = split_eval_train(pop, n_eval)
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

    if not args.skip_llm:
        # Second Claude pass: a plausible-but-wrong action, phrased the same way, for
        # DPO's rejected side (see render_wrong_all()'s docstring).
        render_wrong_all(client, args.model, train_recs, args.batch_size, args.max_tokens)

    write_scenario_sft_files(train_recs, CACHE, mode=trace_mode)
    write_scenario_dpo_file(train_recs, CACHE, mode=trace_mode)
    write_scenario_reflection_file(train_recs, CACHE, mode=trace_mode)


if __name__ == "__main__":
    main()
