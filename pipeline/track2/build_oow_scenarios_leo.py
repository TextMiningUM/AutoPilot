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


def render_leo_narrative(state: dict) -> str:
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
        f"This mission's safe passing distance is {SAFE_CPA_M:.0f}m: CPA below that is a real "
        "collision risk, CPA well above it is safe regardless of how small it looks.",
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


def _real_risk(c: dict) -> bool:
    """Rule 7 gate: real collision risk, via the ONE shared real_risk() (pipeline.
    oow_agent_spec) -- CPA below SAFE_CPA_M AND TCPA within RISK_HORIZON_S, NEVER gated
    on Leo's own "risk" label (medium/high/critical): that label is a TCPA-urgency
    judgement from the source data, not a CPA-based risk-of-collision judgement -- see
    quality-review STAP 1 (2026-09-23) for the concrete real-data example this fixes.
    Leo's risk label stays available as METADATA (c["risk"] itself, still surfaced in the
    rendered narrative for uncertain-track contacts) but is never a gate here again."""
    return real_risk(c.get("cpa_distance_m"), c.get("tcpa_s"), SAFE_CPA_M)


def _turn_degrees(cpa_m: float) -> float:
    """Substantial, geometry-scaled turn -- bigger the further CPA falls below the safe
    distance, always at least MIN_TURN_DEG, capped at the physical per-command limit
    (MAX_TURN_DEG). A fixed +30 regardless of shortfall was the original bug's fixed-
    degree half; this scales with how much clearance is actually missing."""
    shortfall = min(1.0, max(0.0, (SAFE_CPA_M - cpa_m) / SAFE_CPA_M))
    return round(MIN_TURN_DEG + shortfall * (MAX_TURN_DEG - MIN_TURN_DEG), 1)


def _diverging_turn(c: dict) -> str:
    """Turn direction when EITHER side is COLREG-permitted (Rule 13 overtaking): turn
    AWAY from whichever side the contact currently sits on, so the manoeuvre increases
    separation instead of cutting across the contact's bow."""
    return "turn_left" if c["relative_bearing_deg"] >= 0 else "turn_right"


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
    """Minimal, SCOPED Rule 8(c) multi-contact check (quality-review STAP 1 extension,
    2026-09-23) -- covers only the stationary-vs-give-way priority case; STAP 3d
    generalizes this to every multi-real-risk-contact combination. True if turning
    `degrees` in `action`'s direction would reduce `other`'s CPA below its CURRENT
    recorded cpa_distance_m (a 1m tolerance avoids float-noise false positives on an
    unchanged/near-parallel course). Never fires for a non-turn action (stop/slow_down
    have no heading component to check)."""
    if action not in ("turn_left", "turn_right") or degrees is None:
        return False
    delta = -degrees if action == "turn_left" else degrees
    new_heading = (own["heading"] + delta) % 360.0
    new_cpa = _cpa_with_own_heading(own, other, new_heading)
    return new_cpa < other["cpa_distance_m"] - 1.0


def _rule_list(c: dict) -> list[str]:
    nums = sorted(set(c.get("active_encounter_rules") or []) | set(c.get("standing_rules") or []))
    return [f"Rule {n}" for n in nums]


def leo_choose_action(state: dict) -> dict:
    """Deterministic action from the record's OWN own_role/encounter/CPA/TCPA/risk labels
    (trusted, since Leo's own encounter/risk computation already accounts for track
    quality etc. this script does not re-derive). Rule 7 gates everything: a contact only
    drives the action if it poses REAL risk (_real_risk -- CPA vs the safe distance, never
    TCPA alone). No stop-default, no fixed +30-degree turn, no manoeuvre label for a
    stopped/paused own-ship -- see this repo's RAG-rebuild-v2 plan Fase B1 for the full
    diagnosis of what the previous role-only version got wrong (857 manoeuvres on
    low-risk-only contacts, 526 on opening-range contacts, 875 stop-labels, 883 paused
    frames mislabeled as manoeuvres).

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

    NOT_APPLICABLE MOVING CONTACTS (quality-review STAP 2, 2026-09-23): own_role==
    "not_applicable" on a MOVING, real-risk contact (not stationary) is a gap in Leo's own
    role classifier, not a "no risk" case -- its role/encounter_type are geometrically
    re-derived via the shared classify_encounter() and it is then treated exactly like a
    normal give_way/stand_on contact. A contact whose geometry shows it isn't actually
    converging is too ambiguous to label at all -- the WHOLE frame is excluded (action=
    None, role="excluded"), never silently defaulted to hold_course.

    Returns simulator-format fields (action/degrees/encounter_rule/conduct_rule) directly,
    matching Basic Simulator/app/agents.py's SYSTEM_OOW_AGENT JSON contract -- see Fase B2.
    encounter_rule/conduct_rule come from pipeline.oow_agent_spec.classify_rules() (Fase
    B3) via _leo_role_for_classify()'s role translation -- the ONE shared mapping table,
    so this generator, build_oow_scenarios.py, and the B3 reasoning cross-check can never
    silently disagree. `decisive_contact_name` (Leo contacts already carry a "name" field)
    is the contact that actually drove the decision, or None when there is no real risk --
    fed to the B3 teacher prompt as answer-side context, never leaked to
    render_leo_narrative()'s user-facing text."""
    own = state["own_ship"]
    contacts = state["contacts"]

    # Never a manoeuvre label for a ship that isn't moving -- there is no manoeuvre to take.
    if own.get("paused") or own.get("stopped"):
        return {"action": None, "degrees": None, "encounter_rule": None, "conduct_rule": None,
                "role": "paused", "rules": [], "category": "leo_paused", "bucket": "paused",
                "decisive_contact_name": None}

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
                and (c.get("speed") or 0) > MOVING_SPEED_THRESHOLD and _real_risk(c)):
            geo = _geometric_role_and_type(own, c)
            if geo is None:
                return {"action": None, "degrees": None, "encounter_rule": None, "conduct_rule": None,
                        "role": "excluded", "rules": [], "category": "leo_excluded_ambiguous_geometry",
                        "bucket": "excluded", "decisive_contact_name": c["name"],
                        "exclude_reason": "not_applicable moving contact not actually converging "
                                          "(closing_speed<=0) despite CPA/TCPA real_risk"}
            c["own_role"], c["encounter_type"] = geo

    stationary = [c for c in contacts if c.get("encounter_type") == "stationary_contact" and _real_risk(c)]
    give_way = [c for c in contacts if c["own_role"] in ("give_way", "both_give_way") and _real_risk(c)]
    stand_on = [c for c in contacts if c["own_role"] == "stand_on" and _real_risk(c)]

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
                        "decisive_contact_name": worst["name"]}
            degrees = _turn_degrees(worst["cpa_distance_m"])
            if worst["encounter_type"] in ("head_on", "crossing"):
                action = "turn_right"  # Rule 14/15+16: give-way alters to starboard
            else:
                action = _diverging_turn(worst)  # Rule 13: either side permitted
            if worst_stationary is not None and _turn_shrinks_other_cpa(own, worst_stationary, action, degrees):
                action, degrees = "slow_down", None
            encounter_rule, conduct_rule = classify_rules(classify_role, action)
            return {"action": action, "degrees": degrees, "encounter_rule": encounter_rule,
                    "conduct_rule": conduct_rule, "role": worst["own_role"], "rules": _rule_list(worst),
                    "category": f"leo_{worst['encounter_type']}", "bucket": "alter_course",
                    "decisive_contact_name": worst["name"]}

        # Stationary avoidance wins (no give-way contact, or its CPA is not smaller). Never
        # a stop-default, never a give-way/stand-on role -- Rule 8 covers the action alone.
        worst = worst_stationary
        action = _diverging_turn(worst)
        degrees = _turn_degrees(worst["cpa_distance_m"])
        if worst_give_way is not None and _turn_shrinks_other_cpa(own, worst_give_way, action, degrees):
            action, degrees = "slow_down", None
        encounter_rule, conduct_rule = classify_rules("stationary", action)
        return {"action": action, "degrees": degrees, "encounter_rule": encounter_rule,
                "conduct_rule": conduct_rule, "role": "stationary",
                "rules": ["Rule 2", "Rule 5", "Rule 6", "Rule 7", "Rule 8"],
                "category": "leo_stationary_avoid", "bucket": "stationary",
                "decisive_contact_name": worst["name"]}

    if stand_on:
        # Rule 17(a)(ii)/(b): own-ship (stand-on) may/must act once it's apparent the
        # give-way vessel isn't -- requires BOTH a real CPA shortfall AND an imminent
        # encounter (short TCPA), never TCPA alone.
        triggered = [c for c in stand_on if (c.get("tcpa_s") or 1e9) < STAND_ON_TCPA_S]
        if triggered:
            worst = min(triggered, key=lambda c: c["cpa_distance_m"])
            degrees = _turn_degrees(worst["cpa_distance_m"])
            action = _diverging_turn(worst)
            classify_role = _leo_role_for_classify("stand_on", worst["encounter_type"])
            encounter_rule, conduct_rule = classify_rules(classify_role, action)
            return {"action": action, "degrees": degrees, "encounter_rule": encounter_rule,
                    "conduct_rule": conduct_rule, "role": "stand_on", "rules": _rule_list(worst),
                    "category": f"leo_stand_on_{worst['encounter_type']}_17b", "bucket": "stand_on_17b",
                    "decisive_contact_name": worst["name"]}
        ref = stand_on[0]
        classify_role = _leo_role_for_classify("stand_on", ref["encounter_type"])
        encounter_rule, conduct_rule = classify_rules(classify_role, "hold_course")
        return {"action": "hold_course", "degrees": None, "encounter_rule": encounter_rule,
                "conduct_rule": conduct_rule, "role": "stand_on", "rules": _rule_list(ref),
                "category": f"leo_stand_on_{ref['encounter_type']}", "bucket": "stand_on",
                "decisive_contact_name": ref["name"]}

    # No contact poses real risk -- follow GOAL COURSE CHECK exactly (SYSTEM_OOW_AGENT's
    # decision procedure step 3/5), never a fixed hold_course default: only standing rules
    # (2/5/6/7/11) apply, never a specific action-driving rule.
    rule_nums = sorted({n for c in contacts for n in (c.get("standing_rules") or [])}) or [2, 5, 6, 7]
    rules = [f"Rule {n}" for n in rule_nums]
    mission = state["mission"]
    goal_action, goal_degrees = goal_course_action(own["x"], own["y"], own["heading"], mission["x"], mission["y"])
    if goal_action != "hold_course":
        return {"action": goal_action, "degrees": goal_degrees, "encounter_rule": "none",
                "conduct_rule": "none", "role": "cleared", "rules": rules,
                "category": "leo_clear_goal_turn", "bucket": "clear", "decisive_contact_name": None}
    if own["speed"] < own["target_speed"] - RESUME_SPEED_MARGIN:
        return {"action": "speed_up", "degrees": None, "encounter_rule": "none", "conduct_rule": "none",
                "role": "cleared", "rules": rules, "category": "leo_clear_resume", "bucket": "resume",
                "decisive_contact_name": None}
    return {"action": "hold_course", "degrees": None, "encounter_rule": "none", "conduct_rule": "none",
            "role": "cleared", "rules": rules, "category": "leo_clear_maintain", "bucket": "clear",
            "decisive_contact_name": None}



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
    leo_choose_action())."""
    situation = rec["situation_report"]
    decisive_name = rec.get("decisive_contact_name")
    cpa_m = None
    if decisive_name is not None:
        for c in rec["state"]["contacts"]:
            if c["name"] == decisive_name:
                cpa_m = c.get("cpa_distance_m")
                break
    payload = {"id": rec["_id"], "situation": situation, "action": rec["action"], "degrees": rec["degrees"]}
    if decisive_name is not None:
        payload["decisive_contact"] = {"name": decisive_name, "cpa_m": round(cpa_m, 0) if cpa_m is not None else None}
    return payload, {
        "situation": situation, "expected_action": rec["action"], "expected_degrees": rec["degrees"],
        "expected_encounter_rule": rec["encounter_rule"], "expected_conduct_rule": rec["conduct_rule"],
        "real_risk": rec["encounter_rule"] != "none", "cpa_m": cpa_m, "safe_distance_m": SAFE_CPA_M,
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


def stratified_sample(recs: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Round-robins across the action-outcome buckets so even a small sample covers
    stop/alter_course/stand_on/stand_on_17b/resume/clear -- pure random risks missing rare
    buckets (e.g. 'stop' is a small minority of records) in a first-look sample. "paused"
    frames (own-ship not underway) are excluded entirely -- there is no manoeuvre decision
    to make for a stopped ship, so they never become a training instance."""
    rnd = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for r in recs:
        decision = leo_choose_action(r["state"])
        if decision["bucket"] == "paused":
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


def real_previous_decisions(trajectories: dict[str, list[dict]], source_file: str, leo_id: str,
                            n: int = 2) -> list[dict] | None:
    """The real previous `n` helm decisions for the state `leo_id`, derived from its own
    trajectory's REAL preceding states (never synthesized) via leo_choose_action() --
    Fase B4's "previous decisions" history variant. Returns None (no history to add) if
    the state isn't found, doesn't have `n` predecessors in its own trajectory, or any
    predecessor was a paused frame (no manoeuvre decision to report for it)."""
    traj = trajectories.get(source_file)
    if traj is None:
        return None
    idx = next((i for i, r in enumerate(traj) if r["id"] == leo_id), None)
    if idx is None or idx < n:
        return None
    decisions = []
    for r in traj[idx - n:idx]:
        d = leo_choose_action(r["state"])
        if d["action"] is None:
            return None  # a paused predecessor breaks the history chain
        decisions.append({"action": d["action"], "degrees": d["degrees"]})
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
    sample = stratified_sample(all_recs, args.n, seed=args.seed)
    trajectories = build_trajectory_index(all_recs)

    recs: list[dict] = []
    for i, r in enumerate(sample):
        decision = leo_choose_action(r["state"])
        recs.append({
            "_id": f"leo{i:05d}", "leo_id": r["id"], "leo_source_file": r["source_file"],
            "category": decision["category"], "action": decision["action"],
            "degrees": decision["degrees"], "encounter_rule": decision["encounter_rule"],
            "conduct_rule": decision["conduct_rule"], "decisive_contact_name": decision["decisive_contact_name"],
            "role": decision["role"], "rules": decision["rules"],
            "pass_criteria": PASS_CRITERIA[decision["bucket"]],
            "situation_report": render_leo_narrative(r["state"]), "state": r["state"],
            # Fase B4: real previous decisions from this state's own trajectory, when
            # it has 2 real (non-paused) predecessors -- None for the rest (no history
            # preamble added for those rows).
            "prev_decisions": real_previous_decisions(trajectories, r["source_file"], r["id"]),
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
        review_sample = stratified_sample(all_recs, args.b3_review_sample, seed=args.seed)
        review_recs: list[dict] = []
        for i, r in enumerate(review_sample):
            decision = leo_choose_action(r["state"])
            review_recs.append({
                "_id": f"leo{i:05d}", "leo_id": r["id"], "category": decision["category"],
                "action": decision["action"], "degrees": decision["degrees"],
                "encounter_rule": decision["encounter_rule"], "conduct_rule": decision["conduct_rule"],
                "decisive_contact_name": decision["decisive_contact_name"],
                "situation_report": render_leo_narrative(r["state"]), "state": r["state"],
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
