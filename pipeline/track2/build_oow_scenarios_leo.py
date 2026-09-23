"""
================================================================================
build_oow_scenarios_leo.py — Track 2 augmentation from the Leo MOOS trajectories
================================================================================

Data/OOW/OOW_Scenarios_Leo/moos_temporal_narratives_final.jsonl holds 7928 REAL
MOOS-simulated bridge states across 402 multi-step trajectories (up to 50 cycles
each) -- richer than build_oow_scenarios.py's synthetic single-contact geometry:
0-6 simultaneous contacts, sensor-realism fields (track_quality/age_s/
track_reacquired), own-ship movement_status (underway/paused/stalled), and
encounter/risk vocabularies (parallel, diverging, stationary_contact, risk
uncertain/critical) build_oow_scenarios.py doesn't generate.

It has NO action/gold_answer labels -- it is pure situation state, exactly like
build_oow_scenarios.py's geometry stage before its BUILD pass. This script is
that BUILD pass for the Leo data: own-ship/contact FACTS are re-rendered in our
OWN house prose style (never the source file's ALL-CAPS "CURRENT BRIDGE STATE /
AUTOMATED TRAFFIC ASSESSMENT" label-dump narrative field, which is the exact
telegraphic-dump anti-pattern already identified as harmful for this project --
see copilot-instructions.md) via a deterministic action choice + a Claude-authored
justification, same BUILD-vs-VALIDATE split as build_oow_scenarios.py.

Kept in SEPARATE oow_scenario_Leo_*.jsonl files (never merged into the plain
oow_scenario_*.jsonl files) so this source's contribution stays traceable and
reviewable independently, per explicit user direction.

Action choice reuses the Leo record's OWN already-computed own_role/risk labels
(more informed than re-deriving from raw CPA/TCPA, since Leo's role/risk
computation already accounts for track quality etc. we don't model) rather than
build_oow_scenarios.py's classify_encounter()/choose_action() geometry, which
assumes a single synthetic contact this file's real, variable-count contacts
don't fit -- but the OUTPUT task format (system prompt, action vocabulary, JSON
response schema, GOAL COURSE CHECK line) is shared byte-for-byte with Basic
Simulator/app/agents.py and build_oow_scenarios.py via pipeline/oow_agent_spec.py
(Fase B2, RAG-rebuild-v2 plan, 2026-09-22) -- training on one task format and
evaluating on another is exactly what this project's second core principle forbids.

This is a SAMPLE/REVIEW pass, not the full integration: no eval-file merge, no
PG regeneration yet -- do that only after the sample's reviewed. Every dry-run
(--skip-llm) and review-gate pass writes to a `_review/` subfolder (core.review_path),
NEVER to the tracked production .jsonl files, and a real run refuses to overwrite an
existing production file without --overwrite (core.safe_write_jsonl) -- see this
repo's pipeline-notes.md for the 2026-09-22 incident that made this mandatory.

USAGE
-----
    python -m pipeline.track2.build_oow_scenarios_leo --n 25 --skip-llm   # free dry-run, writes to _review/
    python -m pipeline.track2.build_oow_scenarios_leo --n 25              # real LLM calls, writes to _review/
    python -m pipeline.track2.build_oow_scenarios_leo --n 7928 --overwrite  # full production run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from core import AgentPaths, load_env, review_path, safe_write_jsonl, CONTAM_THRESH, EMBEDDER_MODEL
from pipeline.oow_agent_spec import (
    SYSTEM_OOW_AGENT, ACTIONS, validate_action_json, goal_course_check_line, goal_course_action,
    classify_rules, render_previous_decisions, real_risk, STAND_ON_TCPA_S, classify_encounter,
    sample_row_limits, constraint_line, bearing_and_range, relative_bearing,
)

paths = AgentPaths.oow()
CACHE = paths.cache_dir
LEO_FILE = paths.workspace / "Data" / "OOW" / "OOW_Scenarios_Leo" / "moos_temporal_narratives_final.jsonl"
# One JSON line per fully-rendered record (gold_answer + wrong_answer already set),
# appended after EACH batch completes -- a crash/interrupt partway through a long run
# (e.g. --n 7928 is ~530 sequential batches, hours unattended) only loses the single
# in-flight batch, not everything already done. Re-running the same --n/--seed skips
# every leo_id already in here instead of re-spending API calls on it.
# NOTE: NOT oow_scenario_Leo_checkpoint.jsonl -- that file is stale data from a
# pre-Fase-B3-gates approach (committed 2026-09-21, predates the gated
# encounter_rule/conduct_rule schema entirely) and would silently make every row look
# already-processed-and-dropped if reused here.
CHECKPOINT_FILE = CACHE / "oow_scenario_Leo_b3_checkpoint.jsonl"

PASS_CRITERIA = {
    "stop": ["Own-ship stops given a genuinely close-range, still-closing, critical-risk "
             "contact that manoeuvre alone can no longer clear (Rule 17(b)).",
             "Own-ship does not simply hold course when a give-way contact is at critical risk."],
    "alter_course": ["Own-ship takes early, substantial action to keep clear as the give-way vessel, "
                     "scaled to how far CPA falls under the safe passing distance.",
                     "Own-ship does not simply hold course while she holds a give-way obligation.",
                     "Own-ship does not manoeuvre against a contact that poses no real CPA-based risk."],
    "stand_on": ["Own-ship holds course and speed while she is the stand-on vessel.",
                "Own-ship does not needlessly alter away from a stand-on obligation."],
    "stand_on_17b": ["Own-ship (stand-on) takes her own action once it is apparent the give-way "
                     "vessel isn't -- CPA under the safe distance AND the encounter is imminent."],
    "resume": ["Own-ship resumes cruise speed now that no contact requires give-way action.",
              "Own-ship does not needlessly remain at reduced speed with no active encounter."],
    "clear": ["Own-ship maintains course and speed; no contact currently requires action.",
             "Own-ship does not manoeuvre against a contact whose CPA is not below the safe distance."],
    "stationary": ["Own-ship takes early, substantial action to keep clear of a real-risk stationary/"
                  "non-vessel object, scaled to how far CPA falls under the safe passing distance.",
                  "Own-ship does not simply hold course while a stationary object poses a real risk.",
                  "Own-ship does not cite a COLREG encounter rule for a non-vessel object (Rule 8 only)."],
}

SAFE_CPA_M = 500.0        # matches Basic Simulator VesselConstraints' default min_cpa_m
CRITICAL_RANGE_M = 200.0  # "so close that collision cannot be avoided by the give-way
                          # vessel's action alone" (Rule 17(b)) -- a genuine GEOMETRY/range
                          # criterion for stop, never a TCPA cutoff.
# STAND_ON_TCPA_S imported from pipeline.oow_agent_spec (quality-review STAP 1) -- it is
# DERIVED from the same RISK_HORIZON_S real_risk() itself uses, never a second
# independent number: Rule 17(a)(ii)/(b)'s stand-on-may/must-act trigger needs BOTH a
# real CPA shortfall AND the encounter being imminent (a SHORTER TCPA than the general
# real_risk() horizon -- a contact can be a real risk while still too early to judge the
# give-way vessel as failing to act).
MIN_TURN_DEG = 15.0       # Rule 16's "early and substantial" rules out a token gesture.
MAX_TURN_DEG = 30.0       # matches Basic Simulator VesselConstraints' max_rudder_angle_deg
                          # (a single turn command beyond this is silently capped there).
RESUME_SPEED_MARGIN = 0.5  # own speed must be at least this far under target before
                           # "speed_up" is worth issuing (avoids churn on noise-level gaps)
MOVING_SPEED_THRESHOLD = 0.5  # quality-review STAP 2 (2026-09-23): a contact below this
                              # speed is treated as effectively stationary/noise, never
                              # fed into the geometric not_applicable-role fallback below.

# Fase B4 (2026-09-23): only this FRACTION of rows that actually HAVE 2-decision history
# available show it in their prompt TEXT (fixed seed per row id) -- the model must keep
# working on rows with no history too (a mission's real first/second decision never has
# any), so history-in-the-prompt itself must not become a learned shortcut. The label
# (including any 'past and clear' override) is ALWAYS computed with the full internal
# history regardless of whether the text shows it -- this fraction only controls prompt
# visibility, documented as metadata ("history_shown") on every row.
HISTORY_TEXT_FRACTION = 0.6


def _row_gets_history_text(leo_id: str, fraction: float = HISTORY_TEXT_FRACTION) -> bool:
    seed = int(hashlib.sha256(f"history::{leo_id}".encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed).random() < fraction

# Quality-review STAP 2 (2026-09-23): fallback limits for any caller that doesn't pass an
# explicit per-row `limits` dict (existing tests, ad-hoc scripts) -- the historical fixed
# 500/30/300 values, so nothing that doesn't opt in to sampling changes behaviour.
_DEFAULT_LIMITS = {"safe_distance_m": SAFE_CPA_M, "max_turn_deg": MAX_TURN_DEG,
                   "risk_horizon_s": STAND_ON_TCPA_S / 0.6, "stand_on_tcpa_s": STAND_ON_TCPA_S}


def limits_for_leo_record(r: dict) -> dict:
    """Deterministic per-row STAP-2 sampled limits for Leo record `r` -- seeded off the
    record's OWN stable `id` (never a running index), so the same source row always
    samples the same safe_distance_m/max_turn_deg/risk_horizon_s across separate runs."""
    return sample_row_limits(r["id"], r["state"]["own_ship"]["speed"])


def render_leo_narrative(state: dict, limits: dict | None = None) -> str:
    """Deterministic house-style narrative from a Leo `state` dict -- no LLM, and
    deliberately NOT the source file's own ALL-CAPS-headers 'narrative' field. Matches
    Basic Simulator's narrate.py convention byte-for-byte where the two overlap (own-ship/
    contact phrasing, the shared goal_course_check_line()) -- see Fase B2 (RAG-rebuild-v2
    plan). Deliberately leaves out own_role/encounter_type/active_encounter_rules per
    contact, exactly like narrate.py leaves out encounter classification and applicable
    rules: those are precisely what the model is being asked to derive from the numbers
    (bearing/range/CPA/TCPA) itself, per this project's core "rule knowledge comes from
    data, never handed to the model in the prompt" principle -- handing over "We are
    meeting head-on" or "Rules engaged: Rule 15, Rule 16" would leak the answer instead of
    testing whether the model can derive it."""
    limits = limits or _DEFAULT_LIMITS
    own, mission, contacts, cond = state["own_ship"], state["mission"], state["contacts"], state["conditions"]
    ox, oy = own["x"], own["y"]
    mx, my = mission["x"], mission["y"]
    dist = math.hypot(mx - ox, my - oy)
    bearing = math.degrees(math.atan2(mx - ox, my - oy)) % 360.0
    n = len(contacts)
    lines = [
        # own["name"] is always "LLM_SHIP" in the raw data, but the rendered text always
        # says the generic "Own-ship" (matching narrate.py) rather than that raw identifier.
        f"Own-ship is {own['movement_status']} at ({ox:.1f}, {oy:.1f}), heading {own['heading']:.1f} "
        f"degrees, speed {own['speed']:.2f}. Target cruise speed is {own['target_speed']:.1f}.",
        f"Own-ship vessel type is {own['colregs_vessel_type'].replace('_', ' ')}.",
        f"Mission waypoint is at ({mx:.1f}, {my:.1f}), {dist:.0f} m away, bearing {bearing:.1f} deg.",
        goal_course_check_line(ox, oy, own["heading"], mx, my),
        constraint_line(limits["safe_distance_m"], limits["max_turn_deg"], limits["risk_horizon_s"]),
        f"{n} other ship{'s' if n != 1 else ''}:" if n else "No other ships tracked.",
    ]
    for c in contacts:
        if c.get("cpa_distance_m") is not None and c.get("tcpa_s") is not None:
            cpa_txt = f", CPA {c['cpa_distance_m']:.0f} m, TCPA {c['tcpa_s']:.0f}s"
            if (c.get("closing_speed") or 0) <= 0:
                cpa_txt += " (already past closest point, ranges now increasing)"
        else:
            cpa_txt = " (ranges opening -- no future CPA/TCPA projected)"
        # Sensor-realism fact (data quality), NOT a rule/risk hint -- kept, unlike the
        # removed "Risk assessment: X"/own_role/encounter_type/rule-number fields above.
        track_txt = ""
        if c["risk"] == "uncertain":
            age_txt = f" (age {c['age_s']:.0f}s)" if c.get("age_s") is not None else ""
            track_txt = (f" Track on this contact is {c.get('track_quality', 'unreliable')}{age_txt} "
                        "-- treat its numbers with caution.")
        lines.append(
            f'  - Ship named "{c["name"]}" ({c["colregs_vessel_type"].replace("_", " ")}): range '
            f"{c['range_m']:.0f} m, rel.bearing {c['relative_bearing_deg']:.1f} deg, heading "
            f"{c['heading']:.1f}, speed {c['speed']:.2f}, closing speed {c['closing_speed']:.2f}"
            f"{cpa_txt}.{track_txt}"
        )
    vis = cond.get("visibility_condition", "clear")
    vis_range = cond.get("visibility_range_m")
    cond_line = f"Conditions: visibility is {vis}"
    if vis_range is not None:
        cond_line += f", assessed visibility range is {vis_range:.0f} m"
    cond_line += (", narrow-channel context is indicated" if cond.get("narrow_channel")
                 else ", narrow-channel context is not indicated")
    cond_line += (", traffic-separation context is indicated." if cond.get("traffic_separation_scheme")
                 else ", traffic-separation context is not indicated.")
    lines.append(cond_line)
    return "\n".join(lines)

# Primary rule cited per real-risk encounter type -- the ONE rule that actually drives
# the required action, not every standing/background rule (those stay in "rules" for
# metadata/PG tagging).
def _leo_role_for_classify(own_role: str, encounter_type: str) -> str:
    """Maps Leo's own (own_role, encounter_type) labels onto the shared classify_rules()
    role vocabulary (mutual/give_way/stand_on/overtaking_give_way/overtaking_stand_on) --
    the ONE place this translation happens, so leo_choose_action() and the B3 cross-check
    can never silently disagree about which rule pair a Leo record implies."""
    if own_role in ("give_way", "both_give_way"):
        if encounter_type == "head_on":
            return "mutual"
        if encounter_type == "we_are_overtaking_contact":
            return "overtaking_give_way"
        return "give_way"
    if own_role == "stand_on":
        if encounter_type == "contact_overtaking_us":
            return "overtaking_stand_on"
        return "stand_on"
    return "none"


# Quality-review STAP 2 (2026-09-23), decision "nu fixen, geometrisch, gescoped": Leo's
# own own_role=="not_applicable" on a MOVING contact (encounter_type != stationary) is a
# gap in the SOURCE data's own role classifier, not a "no risk" case -- 220/7928 frames
# had a genuine CPA/TCPA real_risk contact silently fall through to hold_course/speed_up
# because this generator only ever trusted Leo's own_role. Maps the shared, two-
# perspective classify_encounter() (pipeline.oow_agent_spec) onto Leo's own_role/
# encounter_type vocabulary -- the SAME translation _leo_role_for_classify() then applies
# identically to a "real" Leo-labelled contact, so this fallback can never diverge from
# how a normal frame is handled.
_GEOMETRIC_ENCOUNTER_MAP = {
    "head_on": ("both_give_way", "head_on"),
    "we_are_overtaking_target": ("give_way", "we_are_overtaking_contact"),
    "target_is_overtaking_us": ("stand_on", "contact_overtaking_us"),
    "crossing_target_on_starboard": ("give_way", "crossing"),
    "crossing_target_on_port": ("stand_on", "crossing"),
}


def _geometric_role_and_type(own: dict, c: dict) -> tuple[str, str] | None:
    """(own_role, encounter_type) in Leo's own vocabulary, geometrically derived via the
    shared classify_encounter() -- or None if the contact isn't actually CONVERGING
    (closing_speed<=0) despite passing the CPA/TCPA real_risk() gate, in which case the
    record should be EXCLUDED from training (see leo_choose_action()) rather than forced
    into a give-way/stand-on role that wouldn't make physical/COLREG sense."""
    if (c.get("closing_speed") or 0) <= 0:
        return None
    tx, ty = _contact_absolute_xy(own, c)
    enc, _rules, _rel = classify_encounter(own["x"], own["y"], own["heading"], tx, ty, c["heading"])
    return _GEOMETRIC_ENCOUNTER_MAP[enc]


def _real_risk(c: dict, limits: dict | None = None) -> bool:
    """Rule 7 gate: real collision risk, via the ONE shared real_risk() (pipeline.
    oow_agent_spec) -- CPA below the row's safe_distance_m AND TCPA within its
    risk_horizon_s (quality-review STAP 2, 2026-09-23: both now PER-ROW sampled values,
    `limits`, rather than the fixed SAFE_CPA_M/RISK_HORIZON_S), NEVER gated on Leo's own
    "risk" label (medium/high/critical): that label is a TCPA-urgency judgement from the
    source data, not a CPA-based risk-of-collision judgement -- see quality-review STAP 1
    (2026-09-23) for the concrete real-data example this fixes. Leo's risk label stays
    available as METADATA (c["risk"] itself, still surfaced in the rendered narrative for
    uncertain-track contacts) but is never a gate here again."""
    limits = limits or _DEFAULT_LIMITS
    return real_risk(c.get("cpa_distance_m"), c.get("tcpa_s"), limits["safe_distance_m"], limits["risk_horizon_s"])


def _turn_degrees(cpa_m: float, limits: dict | None = None) -> float:
    """Substantial, geometry-scaled turn -- bigger the further CPA falls below the row's
    safe_distance_m, always at least MIN_TURN_DEG, capped at the row's own max_turn_deg
    (quality-review STAP 2: both now PER-ROW sampled via `limits`, not the fixed
    SAFE_CPA_M/MAX_TURN_DEG). A fixed +30 regardless of shortfall was the original bug's
    fixed-degree half; this scales with how much clearance is actually missing."""
    limits = limits or _DEFAULT_LIMITS
    safe_distance_m, max_turn_deg = limits["safe_distance_m"], limits["max_turn_deg"]
    shortfall = min(1.0, max(0.0, (safe_distance_m - cpa_m) / safe_distance_m))
    return round(MIN_TURN_DEG + shortfall * (max_turn_deg - MIN_TURN_DEG), 1)


def _diverging_turn(c: dict, own: dict | None = None, other_real_risk_contacts: list[dict] | None = None,
                    mission: dict | None = None, degrees: float | None = None) -> str:
    """Turn direction when EITHER side is COLREG-permitted (Rule 13 overtaking, or a
    stationary/17(b) avoidance where either side clears): turn AWAY from whichever side
    the contact currently sits on, so the manoeuvre increases separation instead of
    cutting across the contact's bow.

    Quality-review STAP 3b (2026-09-23), MAJOR BUG FIX: Leo's own relative_bearing_deg is
    UNSIGNED [0, 360) (verified: min 0.0, max 359.9, zero negative values across all
    14225 contacts in the dataset), but the previous check used it directly as if it were
    SIGNED (-180, 180] -- an unsigned value is never negative, so the old code ALWAYS
    returned turn_left regardless of which side the contact was actually on (confirmed:
    100% turn_left across all 3 call sites -- stationary 684/684, stand_on_17b 440/440,
    overtaking 833/833 non-hardcoded cases). Converted to SIGNED here before the side
    check (positive = starboard, matching classify_encounter()'s own convention: `if
    rel_from_own > 0: crossing_target_on_starboard`).

    Within 5 degrees of dead ahead/astern the side is a genuine geometric tie (NOT a
    float-noise artefact -- a histogram of the affected bearings showed mass spread
    broadly across 0-90 and 270-360 degrees, not clustered near 0), resolved via the
    review's option (ii): the side that leaves the GREATEST minimum CPA to every OTHER
    real-risk contact after the turn, falling back to whichever side is closer to the
    mission's goal bearing when there are no other contacts to compare against (or
    `own`/`mission` aren't supplied, e.g. a unit test calling this in isolation)."""
    signed_bearing = ((c["relative_bearing_deg"] + 180) % 360) - 180
    if abs(signed_bearing) >= 5.0 or own is None:
        return "turn_left" if signed_bearing >= 0 else "turn_right"
    turn_deg = degrees if degrees is not None else MIN_TURN_DEG
    if other_real_risk_contacts:
        left_heading = (own["heading"] - turn_deg) % 360.0
        right_heading = (own["heading"] + turn_deg) % 360.0
        left_min = min(_cpa_with_own_heading(own, oc, left_heading) for oc in other_real_risk_contacts)
        right_min = min(_cpa_with_own_heading(own, oc, right_heading) for oc in other_real_risk_contacts)
        if left_min != right_min:
            return "turn_left" if left_min > right_min else "turn_right"
    if mission is not None:
        goal_brg, _ = bearing_and_range(own["x"], own["y"], mission["x"], mission["y"])
        off_course = relative_bearing(own["heading"], goal_brg)
        if off_course != 0:
            return "turn_right" if off_course > 0 else "turn_left"
    return "turn_left" if signed_bearing >= 0 else "turn_right"


def _should_stop(c: dict) -> bool:
    """Rule 17(b)'s own criterion for when manoeuvre alone no longer suffices: genuinely
    close range AND still closing AND risk already critical -- a geometry/range trigger,
    never a TCPA cutoff. This is deliberately rare (the original bug defaulted to "stop"
    on 875/7928 frames from a TCPA<1.5min OR risk=="critical" check alone)."""
    return (c.get("risk") == "critical" and c.get("range_m") is not None
            and c["range_m"] < CRITICAL_RANGE_M and (c.get("closing_speed") or 0) > 0)


def _contact_absolute_xy(own: dict, c: dict) -> tuple[float, float]:
    """Contact `c`'s absolute (x, y), derived from own-ship's CURRENT heading + `c`'s
    relative_bearing_deg/range_m -- Leo contacts don't carry absolute x/y themselves.
    Shared by _cpa_with_own_heading() and _geometric_role_and_type()."""
    brg_true = math.radians((own["heading"] + c["relative_bearing_deg"]) % 360.0)
    return own["x"] + c["range_m"] * math.sin(brg_true), own["y"] + c["range_m"] * math.cos(brg_true)


def _cpa_with_own_heading(own: dict, c: dict, own_heading_deg: float) -> float:
    """Recomputes contact `c`'s CPA (metres) assuming own-ship turns to
    `own_heading_deg` and both vessels then hold course/speed. Same dot-product CPA
    formula as narrate.py's cpa_tcpa()/build_oow_scenarios.py's cpa_tcpa_m() --
    reimplemented here rather than imported (Basic Simulator's path contains a space the
    pipeline package can't import across; see oow_agent_spec.py's own docstring) --
    deliberate small duplication, same pattern as REASONING_SYSTEM_PROMPT below."""
    tx, ty = _contact_absolute_xy(own, c)
    dx, dy = tx - own["x"], ty - own["y"]
    oh, th = math.radians(own_heading_deg), math.radians(c["heading"])
    vox, voy = own["speed"] * math.sin(oh), own["speed"] * math.cos(oh)
    vtx, vty = c["speed"] * math.sin(th), c["speed"] * math.cos(th)
    dvx, dvy = vtx - vox, vty - voy
    rel_sq = dvx ** 2 + dvy ** 2
    if rel_sq < 1e-6:
        return math.hypot(dx, dy)
    t = max(0.0, -(dx * dvx + dy * dvy) / rel_sq)
    return math.hypot(dx + dvx * t, dy + dvy * t)


def _turn_shrinks_other_cpa(own: dict, other: dict, action: str, degrees: float | None) -> bool:
    """True if turning `degrees` in `action`'s direction would reduce `other`'s CPA below
    its CURRENT recorded cpa_distance_m (a 1m tolerance avoids float-noise false
    positives on an unchanged/near-parallel course). Never fires for a non-turn action
    (stop/slow_down have no heading component to check)."""
    if action not in ("turn_left", "turn_right") or degrees is None:
        return False
    delta = -degrees if action == "turn_left" else degrees
    new_heading = (own["heading"] + delta) % 360.0
    new_cpa = _cpa_with_own_heading(own, other, new_heading)
    return new_cpa < other["cpa_distance_m"] - 1.0


def _any_shrinks(own: dict, others: list[dict], action: str, degrees: float | None) -> bool:
    """Rule 8(c) multi-contact check (quality-review STAP 3d, 2026-09-23, generalizing the
    STAP 1 stationary-vs-give-way scoped version): True if `action`/`degrees` would shrink
    the CPA of ANY of `others` (every OTHER real-risk contact in the current frame, not
    just one specific paired contact)."""
    return any(_turn_shrinks_other_cpa(own, oc, action, degrees) for oc in others)


def _rule17c_guard(c: dict, action: str) -> str:
    """Rule 17(c) (quality-review STAP 3c, 2026-09-23): a stand-on vessel taking her own
    17(a)(ii)/(b) action must never alter to PORT for a contact that is on her OWN port
    side (signed bearing < 0) -- previously true only 'by accident' (365x left/0x right)
    because the pre-STAP-3b bug always returned turn_left regardless of side; now that
    _diverging_turn() is fixed AND has a genuine near-zero free-space tie-break, this
    guard makes the constraint explicit and unconditional (the tie-break's free-space
    reasoning must never be allowed to override it)."""
    signed_bearing = ((c["relative_bearing_deg"] + 180) % 360) - 180
    if signed_bearing < 0 and action == "turn_left":
        return "turn_right"
    return action


def _rule_list(c: dict) -> list[str]:
    nums = sorted(set(c.get("active_encounter_rules") or []) | set(c.get("standing_rules") or []))
    return [f"Rule {n}" for n in nums]


def leo_choose_action(state: dict, limits: dict | None = None) -> dict:
    """Deterministic action from the record's OWN own_role/encounter/CPA/TCPA/risk labels
    (trusted, since Leo's own encounter/risk computation already accounts for track
    quality etc. this script does not re-derive). Rule 7 gates everything: a contact only
    drives the action if it poses REAL risk (_real_risk -- CPA vs the safe distance, never
    TCPA alone). No stop-default, no fixed +30-degree turn, no manoeuvre label for a
    stopped/paused own-ship -- see this repo's RAG-rebuild-v2 plan Fase B1 for the full
    diagnosis of what the previous role-only version got wrong (857 manoeuvres on
    low-risk-only contacts, 526 on opening-range contacts, 875 stop-labels, 883 paused
    frames mislabeled as manoeuvres).

    `limits` (quality-review STAP 2, 2026-09-23): the row's {safe_distance_m, max_turn_deg,
    risk_horizon_s, stand_on_tcpa_s} -- see pipeline.oow_agent_spec.sample_row_limits().
    Defaults to the historical fixed 500m/30deg/300s (_DEFAULT_LIMITS) when omitted, so
    every existing caller/test keeps working unchanged; the real production loop always
    passes an explicit per-row sampled dict (limits_for_leo_record()). Echoed back in the
    returned dict as "training_limits" -- METADATA carried alongside the row, never fed
    into `messages` beyond its already-rendered constraint_line() text.

    STATIONARY CONTACTS (quality-review STAP 1 extension, 2026-09-23): a real-risk contact
    with encounter_type=="stationary_contact" (own_role "not_applicable" -- an anchored
    vessel/fixed object) is NOT a COLREG give-way/stand-on encounter, so it gets its own
    branch (checked ALONGSIDE give_way, before stand_on) rather than falling through to
    "no risk"/speed_up as it silently did before (found via the STAP 1 full-dataset test:
    99/7928 frames). Priority when BOTH a real-risk stationary contact and a real-risk
    give-way contact are present: whichever has the SMALLER cpa_distance_m drives the
    decision; the LOSING contact's CPA is then protected by a scoped Rule 8(c) check
    (_turn_shrinks_other_cpa) -- if the winning turn would shrink it, the action falls
    back to slow_down (never stop, never a direction flip) rather than being taken anyway.

    NOT_APPLICABLE MOVING CONTACTS (quality-review STAP 2 (classify_encounter), 2026-09-23):
    own_role=="not_applicable" on a MOVING, real-risk contact (not stationary) is a gap in
    Leo's own role classifier, not a "no risk" case -- its role/encounter_type are
    geometrically re-derived via the shared classify_encounter() and it is then treated
    exactly like a normal give_way/stand_on contact. A contact whose geometry shows it
    isn't actually converging is too ambiguous to label at all -- the WHOLE frame is
    excluded (action=None, role="excluded"), never silently defaulted to hold_course.

    Returns simulator-format fields (action/degrees/encounter_rule/conduct_rule) directly,
    matching Basic Simulator/app/agents.py's SYSTEM_OOW_AGENT JSON contract -- see Fase B2.
    encounter_rule/conduct_rule come from pipeline.oow_agent_spec.classify_rules() (Fase
    B3) via _leo_role_for_classify()'s role translation -- the ONE shared mapping table,
    so this generator, build_oow_scenarios.py, and the B3 reasoning cross-check can never
    silently disagree. `decisive_contact_name` (Leo contacts already carry a "name" field)
    is the contact that actually drove the decision, or None when there is no real risk --
    fed to the B3 teacher prompt as answer-side context, never leaked to
    render_leo_narrative()'s user-facing text."""
    limits = limits or _DEFAULT_LIMITS
    own = state["own_ship"]
    contacts = state["contacts"]

    # Never a manoeuvre label for a ship that isn't moving -- there is no manoeuvre to take.
    if own.get("paused") or own.get("stopped"):
        return {"action": None, "degrees": None, "encounter_rule": None, "conduct_rule": None,
                "role": "paused", "rules": [], "category": "leo_paused", "bucket": "paused",
                "decisive_contact_name": None, "training_limits": limits}

    # Quality-review STAP 2 (2026-09-23): own_role=="not_applicable" on a MOVING,
    # real-risk contact is a gap in the SOURCE data's own role classifier (220/7928
    # frames), not a "no risk" case -- geometrically derive its role via the shared
    # classify_encounter() and treat it exactly like a normal Leo-labelled contact from
    # here on. If the geometry says it isn't actually converging (closing_speed<=0)
    # despite passing real_risk(), the record is too ambiguous to trust -- EXCLUDE the
    # whole frame from training rather than force a role/action that wouldn't make
    # physical/COLREG sense (never silently label it hold_course).
    contacts = [dict(c) for c in contacts]  # never mutate the caller's state
    for c in contacts:
        if (c["own_role"] == "not_applicable" and c.get("encounter_type") != "stationary_contact"
                and (c.get("speed") or 0) > MOVING_SPEED_THRESHOLD and _real_risk(c, limits)):
            geo = _geometric_role_and_type(own, c)
            if geo is None:
                return {"action": None, "degrees": None, "encounter_rule": None, "conduct_rule": None,
                        "role": "excluded", "rules": [], "category": "leo_excluded_ambiguous_geometry",
                        "bucket": "excluded", "decisive_contact_name": c["name"], "training_limits": limits,
                        "exclude_reason": "not_applicable moving contact not actually converging "
                                          "(closing_speed<=0) despite CPA/TCPA real_risk"}
            c["own_role"], c["encounter_type"] = geo

    stationary = [c for c in contacts if c.get("encounter_type") == "stationary_contact" and _real_risk(c, limits)]
    give_way = [c for c in contacts if c["own_role"] in ("give_way", "both_give_way") and _real_risk(c, limits)]
    stand_on = [c for c in contacts if c["own_role"] == "stand_on" and _real_risk(c, limits)]

    if stationary or give_way:
        worst_stationary = min(stationary, key=lambda c: c["cpa_distance_m"]) if stationary else None
        worst_give_way = min(give_way, key=lambda c: c["cpa_distance_m"]) if give_way else None
        give_way_wins = worst_give_way is not None and (
            worst_stationary is None or worst_give_way["cpa_distance_m"] < worst_stationary["cpa_distance_m"])

        if give_way_wins:
            worst = worst_give_way
            classify_role = _leo_role_for_classify(worst["own_role"], worst["encounter_type"])
            if _should_stop(worst):
                encounter_rule, conduct_rule = classify_rules(classify_role, "stop")
                return {"action": "stop", "degrees": None, "encounter_rule": encounter_rule,
                        "conduct_rule": conduct_rule, "role": worst["own_role"], "rules": _rule_list(worst),
                        "category": f"leo_{worst['encounter_type']}", "bucket": "stop",
                        "decisive_contact_name": worst["name"], "training_limits": limits}
            degrees = _turn_degrees(worst["cpa_distance_m"], limits)
            other = [x for x in (stationary + give_way + stand_on) if x is not worst]
            if worst["encounter_type"] in ("head_on", "crossing"):
                action = "turn_right"  # Rule 14/15+16 mandates starboard -- no alternate side
                if _any_shrinks(own, other, action, degrees):
                    action, degrees = "slow_down", None
            else:
                action = _diverging_turn(worst, own, other, state["mission"], degrees)  # Rule 13: either side permitted
                if _any_shrinks(own, other, action, degrees):
                    alt = "turn_right" if action == "turn_left" else "turn_left"
                    if not _any_shrinks(own, other, alt, degrees):
                        action = alt
                    else:
                        action, degrees = "slow_down", None
            encounter_rule, conduct_rule = classify_rules(classify_role, action)
            return {"action": action, "degrees": degrees, "encounter_rule": encounter_rule,
                    "conduct_rule": conduct_rule, "role": worst["own_role"], "rules": _rule_list(worst),
                    "category": f"leo_{worst['encounter_type']}", "bucket": "alter_course",
                    "decisive_contact_name": worst["name"], "training_limits": limits}

        # Stationary avoidance wins (no give-way contact, or its CPA is not smaller). Never
        # a stop-default, never a give-way/stand-on role -- Rule 8 covers the action alone.
        worst = worst_stationary
        degrees = _turn_degrees(worst["cpa_distance_m"], limits)
        other = [x for x in (stationary + give_way + stand_on) if x is not worst]
        action = _diverging_turn(worst, own, other, state["mission"], degrees)
        if _any_shrinks(own, other, action, degrees):
            alt = "turn_right" if action == "turn_left" else "turn_left"
            if not _any_shrinks(own, other, alt, degrees):
                action = alt
            else:
                action, degrees = "slow_down", None
        encounter_rule, conduct_rule = classify_rules("stationary", action)
        return {"action": action, "degrees": degrees, "encounter_rule": encounter_rule,
                "conduct_rule": conduct_rule, "role": "stationary",
                "rules": ["Rule 2", "Rule 5", "Rule 6", "Rule 7", "Rule 8"],
                "category": "leo_stationary_avoid", "bucket": "stationary",
                "decisive_contact_name": worst["name"], "training_limits": limits}

    if stand_on:
        # Rule 17(a)(ii)/(b): own-ship (stand-on) may/must act once it's apparent the
        # give-way vessel isn't -- requires BOTH a real CPA shortfall AND an imminent
        # encounter (short TCPA, scaled to THIS row's risk_horizon_s), never TCPA alone.
        triggered = [c for c in stand_on if (c.get("tcpa_s") or 1e9) < limits["stand_on_tcpa_s"]]
        if triggered:
            worst = min(triggered, key=lambda c: c["cpa_distance_m"])
            degrees = _turn_degrees(worst["cpa_distance_m"], limits)
            other = [x for x in stand_on if x is not worst]
            action = _rule17c_guard(worst, _diverging_turn(worst, own, other, state["mission"], degrees))
            if _any_shrinks(own, other, action, degrees):
                alt = _rule17c_guard(worst, "turn_right" if action == "turn_left" else "turn_left")
                if alt != action and not _any_shrinks(own, other, alt, degrees):
                    action = alt
                else:
                    action, degrees = "slow_down", None
            classify_role = _leo_role_for_classify("stand_on", worst["encounter_type"])
            encounter_rule, conduct_rule = classify_rules(classify_role, action)
            return {"action": action, "degrees": degrees, "encounter_rule": encounter_rule,
                    "conduct_rule": conduct_rule, "role": "stand_on", "rules": _rule_list(worst),
                    "category": f"leo_stand_on_{worst['encounter_type']}_17b", "bucket": "stand_on_17b",
                    "decisive_contact_name": worst["name"], "training_limits": limits}
        ref = stand_on[0]
        classify_role = _leo_role_for_classify("stand_on", ref["encounter_type"])
        encounter_rule, conduct_rule = classify_rules(classify_role, "hold_course")
        return {"action": "hold_course", "degrees": None, "encounter_rule": encounter_rule,
                "conduct_rule": conduct_rule, "role": "stand_on", "rules": _rule_list(ref),
                "category": f"leo_stand_on_{ref['encounter_type']}", "bucket": "stand_on",
                "decisive_contact_name": ref["name"], "training_limits": limits}

    # No contact poses real risk -- follow GOAL COURSE CHECK exactly (SYSTEM_OOW_AGENT's
    # decision procedure step 3/5), never a fixed hold_course default: only standing rules
    # (2/5/6/7/11) apply, never a specific action-driving rule.
    rule_nums = sorted({n for c in contacts for n in (c.get("standing_rules") or [])}) or [2, 5, 6, 7]
    rules = [f"Rule {n}" for n in rule_nums]
    mission = state["mission"]

    # Fase B4 "past and clear" (Rule 8(d)/13(d), 2026-09-23, ADOPTED per user decision):
    # a contact whose CPA has merely edged back above the safe distance is NOT yet
    # "finally past and clear" while it is still closing (closing_speed > 0) and its TCPA
    # is still >= 0 -- resuming the goal course/speed here is exactly the zigzag bug found
    # live (v0_base's R-L-L-R on Imazu01; the 401m Rule-8(c) shortfall in s01 from turning
    # back too early). Only the MOST RECENT previous decision's real-risk contacts are
    # checked (not both of the last two) -- matches the decision procedure text.
    prev_decisions = state.get("previous_decisions")
    if prev_decisions:
        last = prev_decisions[-1]
        prev_names = set(last.get("real_risk_contact_names") or [])
        if prev_names:
            cur_by_name = {c["name"]: c for c in contacts}
            pending_names = [
                name for name in prev_names
                if (cur_by_name.get(name) is not None
                    and (cur_by_name[name].get("closing_speed") or 0) > 0
                    and (cur_by_name[name].get("tcpa_s") if cur_by_name[name].get("tcpa_s") is not None else -1) >= 0)
            ]
            if pending_names:
                # STAP 5 gate a needs a NAME to check the teacher's reasoning against --
                # "decisive_contact_name" is repurposed here for that (never a give-way/
                # stand-on decisive contact in this branch, since encounter_rule is "none").
                return {"action": "hold_course", "degrees": None, "encounter_rule": "none",
                        "conduct_rule": "none", "role": "cleared", "rules": rules,
                        "category": "leo_clear_not_yet_past_and_clear", "bucket": "clear",
                        "decisive_contact_name": pending_names[0], "training_limits": limits,
                        "reason": "not yet past and clear"}

    goal_action, goal_degrees = goal_course_action(own["x"], own["y"], own["heading"], mission["x"], mission["y"])
    if goal_action != "hold_course":
        return {"action": goal_action, "degrees": goal_degrees, "encounter_rule": "none",
                "conduct_rule": "none", "role": "cleared", "rules": rules,
                "category": "leo_clear_goal_turn", "bucket": "clear", "decisive_contact_name": None,
                "training_limits": limits}
    if own["speed"] < own["target_speed"] - RESUME_SPEED_MARGIN:
        return {"action": "speed_up", "degrees": None, "encounter_rule": "none", "conduct_rule": "none",
                "role": "cleared", "rules": rules, "category": "leo_clear_resume", "bucket": "resume",
                "decisive_contact_name": None, "training_limits": limits}
    return {"action": "hold_course", "degrees": None, "encounter_rule": "none", "conduct_rule": "none",
            "role": "cleared", "rules": rules, "category": "leo_clear_maintain", "bucket": "clear",
            "decisive_contact_name": None, "training_limits": limits}



FIXED_QUESTION = "Recommend exactly ONE manoeuvre as the specified JSON object."


def build_user_message(situation_report: str) -> str:
    """The USER turn -- situation text only, matching Basic Simulator/app/agents.py's
    build_oow_prompt() convention exactly (Situation: ... + the same fixed closing
    instruction) so train and eval share one task format."""
    return f"Situation:\n{situation_report}\n\n{FIXED_QUESTION}"


def build_assistant_json(decision: dict, reasoning: str) -> dict:
    """The ASSISTANT turn -- the schema-validated JSON object itself (never free prose),
    reasoning supplied separately (Fase B3's gated [LLM] step derives it from bearing/
    CPA/rule, never hands the rule to the model to just restate)."""
    return {"action": decision["action"], "degrees": decision["degrees"],
            "encounter_rule": decision["encounter_rule"], "conduct_rule": decision["conduct_rule"],
            "reasoning": reasoning}


def wrong_action_variant(decision: dict) -> dict:
    """A plausible but COLREG-INCORRECT alternative decision, for DPO 'rejected' answers
    -- deterministic, same principle as build_oow_scenarios.py's own wrong_action_variant()."""
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


# ══════════════════════════════════════════════════════════════════════════════════
# Fase B3 (RAG-rebuild-v2 plan, 2026-09-22, 25-first review gate): reasoning text must
# DERIVE the applicable rule(s) from the raw situation facts alone, never restate a rule
# it was handed -- EXCEPT `decisive_contact` (name + CPA), which is answer-side context
# for the TEACHER only (same status as the already-given action/degrees), never part of
# Qwen's own user-facing prompt (build_user_message() never takes or embeds it -- see
# tests/test_oow_scenarios_leo_task_format.py's
# test_qwen_user_content_never_contains_the_decisive_contact_hint). Same design as
# build_oow_scenarios.py's REASONING_SYSTEM_PROMPT (kept as a separate constant here, not
# a cross-import, matching this file's established pattern of small deliberate
# duplication between the two parallel generators). Output is schema-validated AND gated
# (pipeline.track2.b3_reasoning_gates) before acceptance.
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
restricted visibility, or 'none' if no real risk). Whole rule numbers only, no \
sub-paragraphs.
  3. States the given action (and degrees, if any) and justifies it under the conduct \
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


def build_teacher_payload(rec: dict) -> tuple[dict, dict]:
    """The TEACHER-only payload (includes decisive_contact -- answer-side info, never
    sent to Qwen) plus the ground-truth dict used both for row assembly and gating.
    `rec` is one row from `recs` as built in main()/the --b3-review-sample branch below
    (already has action/degrees/encounter_rule/conduct_rule/decisive_contact_name set by
    leo_choose_action()).

    Quality-review STAP 5 gate-b fix (2026-09-23): `safe_distance_m` now reads the ROW'S
    OWN sampled limit (rec["training_limits"]) -- was previously hardcoded to the module
    constant SAFE_CPA_M regardless of what that row actually sampled, which would have
    silently mis-gated gate_threshold_wording's CPA-vs-safe-distance check for any row
    that didn't happen to sample 500m. `risk_horizon_s` is passed through too, for the
    matching TCPA-vs-horizon wording check. History (Fase B4): when rec["history_shown"]
    is True, the SAME render_previous_decisions() preamble a training row would embed is
    prepended to the teacher's own situation text too -- the teacher must be able to see
    the history it is meant to reason about (e.g. why a 'not yet past and clear'
    hold_course is correct), never withheld from it."""
    limits = rec.get("training_limits") or _DEFAULT_LIMITS
    situation = rec["situation_report"]
    if rec.get("history_shown"):
        situation = render_previous_decisions(rec.get("prev_decisions")) + situation
    decisive_name = rec.get("decisive_contact_name")
    cpa_m = None
    tcpa_s = None
    if decisive_name is not None:
        for c in rec["state"]["contacts"]:
            if c["name"] == decisive_name:
                cpa_m = c.get("cpa_distance_m")
                tcpa_s = c.get("tcpa_s")
                break
    payload = {"id": rec["_id"], "situation": situation, "action": rec["action"], "degrees": rec["degrees"]}
    if decisive_name is not None:
        payload["decisive_contact"] = {"name": decisive_name, "cpa_m": round(cpa_m, 0) if cpa_m is not None else None}
    return payload, {
        "situation": situation, "expected_action": rec["action"], "expected_degrees": rec["degrees"],
        "expected_encounter_rule": rec["encounter_rule"], "expected_conduct_rule": rec["conduct_rule"],
        "real_risk": rec["encounter_rule"] != "none", "cpa_m": cpa_m, "safe_distance_m": limits["safe_distance_m"],
        "tcpa_s": tcpa_s, "risk_horizon_s": limits["risk_horizon_s"],
        "decisive_contact_name": decisive_name,
    }


def render_reasoning_review_sample(client, model: str, recs: list[dict], max_attempts: int,
                                   max_tokens: int) -> dict:
    """B3's 25-first review gate for the Leo dataset: `recs` is already the output of
    stratified_sample() (never the full 7928 -- caller decides `n`). REAL Anthropic
    calls, EVERY row passes through the gated generate/retry/drop loop
    (pipeline.track2.b3_reasoning_gates) before acceptance. Returns {"accepted": [...],
    "rejected": [...], "gate_rejection_counts": {...}} -- writes NOTHING to any
    production or checkpoint file."""
    from pipeline.track2.b3_reasoning_gates import generate_gated_row, GATE_NAMES

    accepted, rejected = [], []
    gate_rejection_counts = {name: 0 for name in GATE_NAMES}
    for r in recs:
        payload, expected = build_teacher_payload(r)
        obj, attempt_log = generate_gated_row(
            client, model, REASONING_SYSTEM_PROMPT, payload, max_attempts=max_attempts,
            max_tokens=max_tokens, **expected,
        )
        for attempt in attempt_log:
            for gate_name in attempt["failures"]:
                gate_rejection_counts[gate_name] += 1
        row = {
            "id": r["_id"], "leo_id": r["leo_id"], "category": r["category"],
            "situation": expected["situation"],
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


def write_outputs(outputs: dict[str, list[dict]], cache_dir: Path, overwrite: bool) -> None:
    """Writes every {filename: rows} pair in `outputs` under `cache_dir` -- WITHOUT
    --overwrite, every file goes to core.review_path(cache_dir/filename) instead of the
    real production path, and safe_write_jsonl() additionally refuses to clobber an
    existing file even at the production path. `cache_dir` is an explicit parameter (not
    the module-level CACHE constant) so this is unit-testable against a tmp_path without
    ever touching real Data/ files."""
    for name, rows in outputs.items():
        prod_path = cache_dir / name
        out_path = prod_path if overwrite else review_path(prod_path)
        safe_write_jsonl(rows, out_path, overwrite=overwrite)
        print(f"Wrote {out_path} ({len(rows)} rows)"
             + ("" if overwrite else "  [dry-run/review path -- pass --overwrite for production]"))


def stratified_sample_by_safe_distance(recs: list[dict], n: int, seed: int = 0,
                                       decisions_by_id: dict[str, dict] | None = None) -> list[dict]:
    """Quality-review STAP 5 (2026-09-23): the SECOND 25-review sample stratifies across
    safe_distance_m values (round-robin over the 5 sampled values: 300/400/500/750/926m)
    instead of action bucket -- proves gate b's per-row threshold-wording check against a
    genuine spread of sampled safe distances, not just whichever a pure action-stratified
    sample happened to include. "paused"/"excluded" frames excluded (no manoeuvre decision
    to review; an excluded frame is never a valid training instance at all -- see
    leo_choose_action()'s not_applicable-moving exclusion))."""
    rnd = random.Random(seed)
    buckets: dict[float, list[dict]] = {}
    for r in recs:
        limits = limits_for_leo_record(r)
        decision = (decisions_by_id[r["id"]] if decisions_by_id is not None
                   else leo_choose_action(r["state"], limits))
        if decision["bucket"] in ("paused", "excluded"):
            continue
        buckets.setdefault(limits["safe_distance_m"], []).append(r)
    for b in buckets.values():
        rnd.shuffle(b)
    out: list[dict] = []
    bucket_keys = list(buckets)
    i = 0
    while len(out) < n and any(buckets.values()):
        b = bucket_keys[i % len(bucket_keys)]
        if buckets[b]:
            out.append(buckets[b].pop())
        i += 1
    return out


def stratified_sample(recs: list[dict], n: int, seed: int = 0, decisions_by_id: dict[str, dict] | None = None) -> list[dict]:
    """Round-robins across the action-outcome buckets so even a small sample covers
    stop/alter_course/stand_on/stand_on_17b/resume/clear -- pure random risks missing rare
    buckets (e.g. 'stop' is a small minority of records) in a first-look sample. "paused"
    frames (own-ship not underway) are excluded entirely -- there is no manoeuvre decision
    to make for a stopped ship, so they never become a training instance.

    `decisions_by_id` (Fase B4, 2026-09-23): the precomputed, correctly-history-threaded
    decisions from compute_all_decisions() -- REQUIRED so bucketing reflects the ACTUAL
    final decision (including any 'past and clear' override), not a naive independent
    leo_choose_action() call without history. Falls back to a bare (no-history) call only
    when omitted, for backward compatibility with any caller/test not yet passing it."""
    rnd = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for r in recs:
        decision = (decisions_by_id[r["id"]] if decisions_by_id is not None
                   else leo_choose_action(r["state"], limits_for_leo_record(r)))
        if decision["bucket"] in ("paused", "excluded"):
            continue
        buckets.setdefault(decision["bucket"], []).append(r)
    for b in buckets.values():
        rnd.shuffle(b)
    out: list[dict] = []
    bucket_names = list(buckets)
    i = 0
    while len(out) < n and any(buckets.values()):
        b = bucket_names[i % len(bucket_names)]
        if buckets[b]:
            out.append(buckets[b].pop())
        i += 1
    return out


def build_trajectory_index(all_recs: list[dict]) -> dict[str, list[dict]]:
    """Group raw Leo states by source_file, preserving the file's original (already
    chronological, per cycle_id) order -- used by real_previous_decisions() (Fase B4) to
    look up a state's real predecessors within the SAME multi-step trajectory."""
    trajectories: dict[str, list[dict]] = {}
    for r in all_recs:
        trajectories.setdefault(r["source_file"], []).append(r)
    return trajectories


def compute_all_decisions(all_recs: list[dict]) -> dict[str, dict]:
    """Fase B4 (2026-09-23): ONE forward pass per trajectory (grouped by source_file,
    preserving each file's real chronological cycle_id order) computing EVERY Leo state's
    FINAL decision via leo_choose_action() -- correctly threading each frame's own rolling
    2-decision history (state['previous_decisions']) so the 'past and clear' check (see
    leo_choose_action()) resolves against the REAL, correctly-history-aware decision of
    its own predecessors, not a naive independent re-derivation (which would get a
    predecessor's OWN past-and-clear override wrong, since history is itself recursive).
    Replaces the old real_previous_decisions() (re-derived predecessors one-off, without
    threading THEIR OWN history, and without the past-and-clear/richer-fields support).
    A paused/excluded predecessor resets the rolling history to empty -- no manoeuvre
    decision to report for it, matching the historical real_previous_decisions() semantics.
    Returns {leo_id: decision} for every one of `all_recs` (`decision` also carries the
    'previous_decisions' list actually used, under decision['_previous_decisions'], for
    the caller to render/attach without re-deriving it)."""
    trajectories = build_trajectory_index(all_recs)
    decisions: dict[str, dict] = {}
    for traj in trajectories.values():
        history: list[dict] = []
        for r in traj:
            limits = limits_for_leo_record(r)
            state_with_history = dict(r["state"])
            state_with_history["previous_decisions"] = list(history) if history else None
            decision = dict(leo_choose_action(state_with_history, limits))
            decision["_previous_decisions"] = state_with_history["previous_decisions"]
            decisions[r["id"]] = decision
            if decision["action"] is None:
                history = []  # paused/excluded predecessor breaks the chain
                continue
            if decision["category"] == "leo_clear_not_yet_past_and_clear" and history:
                # Still not finally past and clear -- PROPAGATE the pending contact
                # name(s) forward (never re-derive from the strict real_risk() gate,
                # which is empty here by construction) so a MULTI-step not-yet-clear
                # stretch keeps checking the SAME contact until it genuinely clears,
                # rather than losing track after a single hold_course step.
                real_risk_names = history[-1].get("real_risk_contact_names") or []
            else:
                real_risk_names = [c["name"] for c in r["state"]["contacts"] if _real_risk(c, limits)]
            history.append({"action": decision["action"], "degrees": decision["degrees"],
                            "encounter_rule": decision["encounter_rule"],
                            "conduct_rule": decision["conduct_rule"],
                            "real_risk_contact_names": real_risk_names})
            history = history[-2:]
    return decisions


def load_checkpoint() -> dict[str, dict]:
    """leo_id -> fully-rendered record, from every batch a prior run already completed."""
    if not CHECKPOINT_FILE.exists():
        return {}
    out: dict[str, dict] = {}
    for line in CHECKPOINT_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["leo_id"]] = rec
    return out


def append_checkpoint(recs: list[dict]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


def filter_contamination_batch(batch: list[dict], gold_embs, model) -> list[dict]:
    """Per-batch version of build_oow_scenarios.filter_contamination() -- takes a
    PRE-LOADED embedder + pre-computed gold_embs instead of reloading the
    SentenceTransformer on every call, since this now runs once per ~15-record batch
    (~530 times for a full --n 7928 run) instead of once for the whole sample."""
    if not batch:
        return batch
    texts = [r.get("reasoning", "") for r in batch]
    q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    kept, dropped = [], 0
    for rec, emb in zip(batch, q_embs):
        c_sim = float(np.max(gold_embs @ emb)) if len(gold_embs) else 0.0
        if c_sim >= CONTAM_THRESH:
            dropped += 1
            continue
        kept.append(rec)
    if dropped:
        print(f"    contamination filter: kept={len(kept)}  dropped={dropped} (thresh={CONTAM_THRESH})")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n", type=int, default=25, help="sample size (stratified across action buckets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default="claude-sonnet-4-5")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--gold-file", type=str, default=str(paths.eval_dir / "colreg_qa_500_normalised.json"))
    ap.add_argument("--skip-llm", action="store_true",
                    help="geometry+narrative+action only -- no API calls, no reasoning text, for free dry-run testing")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting existing production oow_scenario_Leo_*.jsonl files -- "
                         "without this, a run whose output already exists on disk refuses to write "
                         "(see core.safe_write_jsonl); dry-run/review passes never need this, they "
                         "always write to _review/ instead")
    ap.add_argument("--b3-review-sample", type=int, default=None,
                    help="Fase B3 25-first review gate: generate this many REAL Anthropic "
                         "reasoning-derivation calls (REASONING_SYSTEM_PROMPT, stratified across "
                         "action buckets via stratified_sample) and write them to _review/ for "
                         "human review, then exit -- never runs the full population, never touches "
                         "any production or checkpoint file.")
    args = ap.parse_args()

    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(all_recs)} Leo states; sampling {args.n} (stratified by action bucket)...")
    decisions_by_id = compute_all_decisions(all_recs)
    sample = stratified_sample(all_recs, args.n, seed=args.seed, decisions_by_id=decisions_by_id)

    recs: list[dict] = []
    for i, r in enumerate(sample):
        limits = limits_for_leo_record(r)
        decision = decisions_by_id[r["id"]]
        prev_decisions = decision["_previous_decisions"]
        show_history = prev_decisions is not None and _row_gets_history_text(r["id"])
        recs.append({
            "_id": f"leo{i:05d}", "leo_id": r["id"], "leo_source_file": r["source_file"],
            "category": decision["category"], "action": decision["action"],
            "degrees": decision["degrees"], "encounter_rule": decision["encounter_rule"],
            "conduct_rule": decision["conduct_rule"], "decisive_contact_name": decision["decisive_contact_name"],
            "role": decision["role"], "rules": decision["rules"], "training_limits": limits,
            "pass_criteria": PASS_CRITERIA[decision["bucket"]],
            "situation_report": render_leo_narrative(r["state"], limits), "state": r["state"],
            # Fase B4: real, correctly-history-threaded previous decisions (see
            # compute_all_decisions()) -- None when this row has no 2-decision history
            # available (e.g. the first frames of a trajectory). Even when available, only
            # HISTORY_TEXT_FRACTION of ELIGIBLE rows actually show it in the prompt text
            # (fixed seed per row id) -- "history_shown" records which, as metadata.
            "prev_decisions": prev_decisions if show_history else None,
            "history_available": prev_decisions is not None, "history_shown": show_history,
            "reasoning": None, "wrong_decision": None, "wrong_reasoning": None,
        })

    if args.b3_review_sample is not None:
        load_env(paths.env_file)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set in .env")
        import anthropic
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        client = anthropic.Anthropic(api_key=key, default_headers=headers)
        review_sample = stratified_sample_by_safe_distance(all_recs, args.b3_review_sample, seed=args.seed, decisions_by_id=decisions_by_id)
        review_recs: list[dict] = []
        for i, r in enumerate(review_sample):
            limits = limits_for_leo_record(r)
            decision = decisions_by_id[r["id"]]
            prev_decisions = decision["_previous_decisions"]
            show_history = prev_decisions is not None and _row_gets_history_text(r["id"])
            review_recs.append({
                "_id": f"leo{i:05d}", "leo_id": r["id"], "category": decision["category"],
                "action": decision["action"], "degrees": decision["degrees"],
                "encounter_rule": decision["encounter_rule"], "conduct_rule": decision["conduct_rule"],
                "decisive_contact_name": decision["decisive_contact_name"], "training_limits": limits,
                "situation_report": render_leo_narrative(r["state"], limits), "state": r["state"],
                "prev_decisions": prev_decisions if show_history else None, "history_shown": show_history,
            })
        result = render_reasoning_review_sample(client, args.model, review_recs,
                                                max_attempts=3, max_tokens=args.max_tokens)
        out_path = review_path(CACHE / "oow_scenario_Leo_b3_review_sample.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in result["accepted"] + result["rejected"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_acc, n_rej = len(result["accepted"]), len(result["rejected"])
        print(f"Wrote {out_path} ({n_acc + n_rej} review rows: {n_acc} accepted, {n_rej} dropped)")
        print(f"Per-gate rejection counts (across all attempts): {result['gate_rejection_counts']}")
        if result["rejected"]:
            print(f"  DROPPED ids (exhausted retries): {[r['id'] for r in result['rejected']]}")
        print("This is a REVIEW-ONLY sample -- no production/checkpoint file was written. "
             "STOP here pending human review before running the full population.")
        return

    # Fase B1 (leo_choose_action rewrite) + Fase B2 (unified task format -- system prompt/
    # user framing/JSON response schema, all from pipeline/oow_agent_spec.py) have landed.
    # Fase B3 (the actual [LLM] reasoning-text generation, gated behind the 25-first review
    # per this repo's RAG-rebuild-v2 plan) is approved and implemented below -- every rec
    # passes through the SAME gated generate/retry/drop loop as --b3-review-sample,
    # checkpointed (CHECKPOINT_FILE) so a crash/interrupt only loses the in-flight row.
    if not args.skip_llm:
        load_env(paths.env_file)
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY not set in .env")
        import anthropic
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        client = anthropic.Anthropic(api_key=key, default_headers=headers)
        from pipeline.track2.b3_reasoning_gates import generate_gated_row, GATE_NAMES
        checkpoint = load_checkpoint()
        print(f"Fase B3 full population: {len(recs)} training records "
             f"({len(checkpoint)} already checkpointed)...")
        gate_rejection_counts = {name: 0 for name in GATE_NAMES}
        n_accepted = n_rejected = 0
        for i, r in enumerate(recs):
            cp = checkpoint.get(r["leo_id"])
            if cp is not None:
                if cp.get("reasoning"):
                    r["reasoning"] = cp["reasoning"]
                    n_accepted += 1
                else:
                    n_rejected += 1
                continue
            payload, expected = build_teacher_payload(r)
            obj, attempt_log = generate_gated_row(
                client, args.model, REASONING_SYSTEM_PROMPT, payload,
                max_attempts=3, max_tokens=args.max_tokens, **expected,
            )
            for attempt in attempt_log:
                for gate_name in attempt["failures"]:
                    gate_rejection_counts[gate_name] += 1
            if obj is not None:
                r["reasoning"] = obj["reasoning"]
                n_accepted += 1
            else:
                n_rejected += 1
            append_checkpoint([{"leo_id": r["leo_id"], "reasoning": r.get("reasoning")}])
            done_now = i + 1
            if done_now % 20 == 0 or done_now == len(recs):
                print(f"  [B3] {done_now}/{len(recs)} processed ({n_accepted} accepted, "
                     f"{n_rejected} dropped)", flush=True)
        print(f"Fase B3 done: {n_accepted} accepted, {n_rejected} dropped "
             "(this run's fresh generations + resumed checkpoint)")
        print(f"Per-gate rejection counts (this run's fresh generations only): {gate_rejection_counts}")
    else:
        print("--skip-llm: situation_report/action/degrees/encounter_rule/conduct_rule are set; "
             "reasoning left None pending Fase B3.")
    final_recs = recs

    sft_rows, dpo_rows, reflect_rows, trace_rows = [], [], [], []
    n_with_history = 0
    for r in final_recs:
        if not r.get("reasoning"):
            continue
        # Fase B4: prepend the real-history preamble for rows that have one.
        history_prefix = render_previous_decisions(r.get("prev_decisions"))
        if history_prefix:
            n_with_history += 1
        user_msg = build_user_message(history_prefix + r["situation_report"])
        decision = {"action": r["action"], "degrees": r["degrees"],
                   "encounter_rule": r["encounter_rule"], "conduct_rule": r["conduct_rule"]}
        assistant_json = build_assistant_json(decision, r["reasoning"])
        sft_rows.append({"category": r["category"], "action": r["action"], "leo_id": r["leo_id"], "messages": [
            {"role": "system", "content": SYSTEM_OOW_AGENT}, {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
        ]})
        wrong_decision = wrong_action_variant(decision)
        rejected_json = build_assistant_json(wrong_decision, r["reasoning"])
        dpo_rows.append({
            "category": r["category"], "action": r["action"], "leo_id": r["leo_id"],
            "prompt": [{"role": "system", "content": SYSTEM_OOW_AGENT}, {"role": "user", "content": user_msg}],
            "chosen": [{"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}],
            "rejected": [{"role": "assistant", "content": json.dumps(rejected_json, ensure_ascii=False)}],
        })
        draft = f"I will {r['action'].replace('_', ' ')}." if r["action"] else "I will hold course."
        critique = ("This response is too vague -- it must state the exact action parameters "
                   "and cite the specific COLREG rule(s) that justify the decision.")
        reflect_rows.append({
            "category": r["category"], "leo_id": r["leo_id"],
            "messages": [
                {"role": "system", "content": SYSTEM_OOW_AGENT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"Draft: {draft}\n\nCritique: {critique}\n\n"
                                                  f"Refined: {json.dumps(assistant_json, ensure_ascii=False)}"},
            ],
        })
        trace_rows.append({
            "document_id": f"oow_scenario_leo_{r['leo_id']}", "chunk_id": f"oow_scenario_leo_{r['leo_id']}",
            "source_file": f"oow_scenario_generator_leo::{r['leo_source_file']}",
            "chapter_title": f"OOW Track 2 (Leo) scenario: {r['category']}",
            "chunk_concepts": [r["category"], r["role"]] + r["rules"],
            "trace": {
                "situation": r["situation_report"], "trigger": None,
                "procedures": [
                    {"step": 1, "action": "Assess the fused contact picture (bearing/range/CPA/TCPA) against "
                                          "own-ship's course and speed.",
                     "why": "Rule 7 requires determining if risk of collision exists using all available means."},
                    {"step": 2, "action": f"Determine own-ship's role: {r['role'].replace('_', ' ')}.",
                     "why": f"{', '.join(r['rules']) or 'no encounter rule'} governs this encounter."},
                    {"step": 3, "action": "Execute the action: " + r["action"] +
                                          (f" ({r['degrees']:.0f} deg)" if r["degrees"] is not None else "") + ".",
                     "why": "The chosen action must comply with the applicable COLREG rule(s) above."},
                ],
                "constraints": r["pass_criteria"], "prowords_used": [r["role"]], "channels": r["rules"],
                "regulations": ["COLREG 1972"], "warnings": [],
                "outcomes": [f"Action taken: {r['action']}."],
                "key_facts": [f"Chosen action: {r['action']}."],
                "question_seeds": [{"angle": "what", "text": FIXED_QUESTION}],
            },
        })

    outputs = {
        "oow_scenario_Leo_sft_direct.jsonl": sft_rows,
        "oow_scenario_Leo_sft_cot.jsonl": sft_rows,
        "oow_scenario_Leo_dpo_pairs.jsonl": dpo_rows,
        "oow_scenario_Leo_reflection.jsonl": reflect_rows,
        "oow_scenario_Leo_reasoning_traces.jsonl": trace_rows,
    }
    write_outputs(outputs, CACHE, overwrite=args.overwrite)

    from collections import Counter
    print("category distribution:", Counter(r["category"] for r in final_recs))
    print(f"Fase B4: {n_with_history}/{len(sft_rows)} SFT rows include a real previous-decisions "
         "history preamble")


if __name__ == "__main__":
    main()
