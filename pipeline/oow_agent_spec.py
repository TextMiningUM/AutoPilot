"""Single source of truth for the OOW Navigation Agent's task format -- the system
prompt, action vocabulary, JSON response schema, and the GOAL COURSE CHECK computation
that Basic Simulator/app/{agents,narrate}.py AND every Track-2 training-data generator
(pipeline/track2/build_oow_scenarios*.py) must all reproduce byte-identically.

WHY THIS MODULE EXISTS (RAG-rebuild-v2 plan, Fase B2, 2026-09-22)
------------------------------------------------------------------
Before this module, build_oow_scenarios.py/build_oow_scenarios_leo.py trained on an
ENTIRELY different task than the live simulator evaluates on: a different action
vocabulary (maintain_course/alter_course/set_speed/stop/resume_cruising_speed vs the
simulator's turn_left/turn_right/hold_course/speed_up/slow_down/stop), a different
response shape (free-form justification prose vs a fixed JSON object), and no shared
GOAL COURSE CHECK computation at all. Training on one task and evaluating on another is
exactly what this project's second core principle forbids.

DEPENDENCY-FREE BY DESIGN
--------------------------
stdlib only (just `math`) so this can be imported from BOTH the data pipeline
(pipeline/, plain CPU scripts) and the Streamlit app ("Basic Simulator/app/agents.py",
which pulls in torch/streamlit/transformers) without EITHER side inheriting the other's
dependency stack. Deliberately NOT in core/ (core/__init__.py's own docstring reserves
that package for domain-AGNOSTIC building blocks; system prompts are explicitly called
out there as a per-agent/domain concern that belongs OUTSIDE core), and deliberately NOT
inside Basic Simulator/app/agents.py itself (that path contains a space, which the
pipeline package cannot import across cleanly, and would force torch/streamlit onto
every pipeline data-builder that just wants the prompt/schema constants).
"""
from __future__ import annotations
import hashlib
import math
import random

ACTIONS = ("turn_left", "turn_right", "hold_course", "speed_up", "slow_down", "stop")
_DEGREES_ONLY_FOR = ("turn_left", "turn_right")
# Same 1852m/NM constant as Basic Simulator/app/units.py -- duplicated (not imported)
# since this module must stay dependency-free/path-independent (see module docstring).
_NM_TO_M = 1852.0

# Quality-review STAP 1 (blocking bug, 2026-09-23): the ONE definition of "real collision
# risk", shared by BOTH Track-2 generators' labelers (leo_choose_action/_real_risk,
# build_oow_scenarios.py's to_unified_action) and Basic Simulator/app/measurement.py's
# Check A -- previously build_oow_scenarios_leo.py's _real_risk() ALSO required Leo's own
# "risk" label (medium/high/critical), but that label is a TCPA-urgency judgement from the
# SOURCE data, not a CPA-based risk-of-collision judgement -- e.g. a real source record
# with give_way role, CPA 187m (<500m), TCPA 709s carried risk="low" purely because TCPA
# was distant, so gating on it too silently dropped ~840 genuinely-close-CPA frames to
# hold_course/speed_up with no rule cited. Leo's risk label is METADATA from here on,
# never a gate.
#
# RISK_HORIZON_S matches colreg_llm_bridge.py's own risk_horizon default (300s) -- the
# live MOOS bridge already uses this exact number to decide when a contact is worth
# actively deciding about at all, so training/eval reuses it rather than inventing a
# second number. STAND_ON_TCPA_S (Rule 17(a)(ii)/(b): a stand-on vessel may/must act once
# it becomes "apparent" the give-way vessel isn't) is DERIVED as a shorter sub-horizon of
# RISK_HORIZON_S, not an independent constant: a contact can be a REAL risk (within
# RISK_HORIZON_S) while it is still too early to judge the give-way vessel as failing to
# act -- that stronger judgement only applies once the encounter is more imminent.
RISK_HORIZON_S = 300.0
STAND_ON_TCPA_S = RISK_HORIZON_S * 0.6  # == 180.0, matches the prior hardcoded value


def real_risk(cpa_m: float | None, tcpa_s: float | None, safe_distance_m: float,
             risk_horizon_s: float = RISK_HORIZON_S) -> bool:
    """The ONE gate for "does this contact pose a real risk of collision right now" --
    CPA below `safe_distance_m` (a per-row/per-mission PARAMETER, never a hardcoded
    constant, so training data never learns a shortcut against one fixed number) AND TCPA
    within [0, risk_horizon_s). Both conditions are required:
      - CPA alone is not enough: a contact can have a tiny CPA that is still hours away
        (not yet actionable) or -- more commonly -- already resolved.
      - TCPA alone is not enough: TCPA==0 is genuinely ambiguous on its own (it occurs
        BOTH for "collision right now" and "closest point already passed, now diverging"
        -- never disambiguate using TCPA alone).
      - TCPA < 0 (already past the closest point) is explicitly EXCLUDED (`0 <= tcpa_s`),
        never treated as a risk regardless of how small CPA was.
      - TCPA >= risk_horizon_s is a contact to monitor, not yet one to act on.
    `risk_horizon_s` defaults to the module-level RISK_HORIZON_S (300s, the live MOOS
    bridge's own default) for backward compatibility -- quality-review STAP 2 (2026-09-
    23) makes it an explicit PER-ROW parameter too: a fixed 300s horizon silently treated
    some genuinely-slow-manoeuvring missions' real risks as "no risk" (Imazu01 compliance
    finding: TCPA 1800s, CPA 0 -- see derive_risk_horizon_s() below).
    """
    return (cpa_m is not None and cpa_m < safe_distance_m
            and tcpa_s is not None and 0 <= tcpa_s < risk_horizon_s)


def risk_band(cpa_m: float | None, tcpa_s: float | None, safe_distance_m: float,
             risk_horizon_s: float = RISK_HORIZON_S) -> str:
    """Quality-review STOP-1-blocking-bug fix (2026-09-23): the ONE encounter-band
    classifier, shared by BOTH Track-2 labelers (leo_choose_action()/to_unified_action())
    and Basic Simulator/app/evaluation.py's auditor ground truth (previously duplicated
    there as `_contact_ground_truth()`'s own inline if/elif chain -- extracted here so
    all three call sites can never silently disagree). Four bands, one governing
    principle: identifying an encounter is required as soon as CPA < safe_distance_m; the
    horizon only decides when ACTING becomes mandatory (or when resuming is allowed), it
    never decides whether a rule applies at all.

      "passed": tcpa_s < 0 -- the closest point of approach is already behind us (the
                contact is diverging). Checked BEFORE "safe" so a contact still technically
                inside the safe distance while diverging is "passed", not "acute". Reports
                no rule, by convention (matches Fase B4's "past and clear" handling, which
                is what actually governs whether a DIFFERENT, still-pending contact must
                still be held against).
      "safe":   cpa_m is None, or cpa_m >= safe_distance_m -- no encounter at all.
      "acute":  cpa_m < safe_distance_m AND 0 <= tcpa_s < risk_horizon_s -- real_risk()
                itself; action is MANDATORY.
      "early":  cpa_m < safe_distance_m AND (tcpa_s is None OR tcpa_s >= risk_horizon_s)
                -- a real encounter that does not yet mandate action: identify the
                encounter/rule now, act early if convenient (never with "stop", which
                stays exclusive to "acute"), but the D-check (acting is REQUIRED) only
                fires once the band becomes "acute". Was previously silently folded into
                real_risk()==False ("no risk") by every caller, training a model that a
                CPA-0, TCPA-beyond-horizon collision course has no encounter at all --
                the blocking bug this function exists to fix (measured: 229/276 (83%) of
                Leo's genuine early-band frames mislabeled encounter_rule='none', 40 of
                those mislabeled 'speed_up'). tcpa_s=None is treated as "early", never
                "safe" -- CPA alone already says an encounter exists; missing TCPA data
                must never hide it.
    """
    if tcpa_s is not None and tcpa_s < 0:
        return "passed"
    if cpa_m is None or cpa_m >= safe_distance_m:
        return "safe"
    if tcpa_s is not None and tcpa_s < risk_horizon_s:
        return "acute"
    return "early"


# Bands that constitute a real encounter needing to be identified (I1/I2 invariants,
# quality-review STOP-1-blocking-bug fix) -- "passed" and "safe" both report no rule.
ENCOUNTER_BANDS = ("acute", "early")


# Quality-review STAP 2 (2026-09-23): THREE previously-hardcoded training-data constants
# (safe_distance_m, max_turn_deg, and now also the risk horizon) become PER-ROW sampled
# variables instead -- a value that never varies in training is learned as a constant
# (shortcut learning), so a user who configures the live simulator away from the old
# hardcoded defaults (500m/30deg) would find the fine-tuned model still silently judging
# against 500/30. Sampling weights favour the historical defaults (40-50%) while still
# giving real coverage to the simulator's full configurable range.
SAFE_DISTANCE_WEIGHTS: dict[float, float] = {300.0: 0.15, 400.0: 0.15, 500.0: 0.40, 750.0: 0.15, 926.0: 0.15}
MAX_TURN_DEG_WEIGHTS: dict[float, float] = {20.0: 0.15, 25.0: 0.20, 30.0: 0.50, 35.0: 0.15}
# Applied as a MULTIPLIER of the per-row geometry-derived risk_horizon_s default (see
# derive_risk_horizon_s()), never sampled as an absolute second independent number.
HORIZON_MULTIPLIER_WEIGHTS: dict[float, float] = {0.6: 0.15, 0.8: 0.20, 1.0: 0.40, 1.3: 0.15, 1.6: 0.10}
# t_manoeuvre = safe_distance_m / (v_own * sin(max_turn_deg)) is roughly how long own-ship
# takes to physically open the safe distance by turning at its per-command max; K is a
# safety multiple of that so the horizon covers deciding, executing, AND confirming
# separation -- not just the bare manoeuvre time. Verified against the two reference
# speeds given at Imazu01 review: 12 kt/30deg/500m -> t_manoeuvre~162s -> horizon~567s
# (~570s); a 2.5 m/s USV/30deg/500m -> t_manoeuvre~400s -> horizon~1400s (a MUCH longer
# horizon than the bridge's old fixed 300s -- genuinely correct for how slowly a USV can
# manoeuvre relative to that safe distance, not a bug; see STOP-1-supplement report).
RISK_HORIZON_K = 3.5


def derive_risk_horizon_s(safe_distance_m: float, max_turn_deg: float, own_speed_mps: float | None) -> float:
    """Per-row DEFAULT risk horizon, derived from how long own-ship actually takes to open
    the safe distance by turning at its per-command max (see RISK_HORIZON_K above) --
    replaces a single fixed RISK_HORIZON_S=300 for every mission regardless of speed/turn
    limit/safe distance. Falls back to the historical RISK_HORIZON_S when own_speed_mps is
    missing/zero/negative (a stopped/unknown-speed own-ship has no manoeuvre time to derive
    a horizon from at all).

    Quality-review STOP-1-blocking-bug fix (2026-09-23): the horizon's role is DELIBERATELY
    limited to exactly three things -- (a) the D-check, i.e. whether acting is MANDATORY
    right now (real_risk()/risk_band()=="acute"); (b) STAND_ON_TCPA_S, when a stand-on
    vessel's own Rule 17(a)(ii)/(b) action may/must trigger; (c) the auditor's safe/early/
    acute/passed band. It never decides whether a rule/encounter applies at all -- a
    contact with CPA below safe_distance_m is a real encounter (risk_band()=="early" or
    "acute") regardless of how far beyond the horizon its TCPA sits; see risk_band()."""
    if not own_speed_mps or own_speed_mps <= 0:
        return RISK_HORIZON_S
    t_manoeuvre = safe_distance_m / (own_speed_mps * math.sin(math.radians(max_turn_deg)))
    return RISK_HORIZON_K * t_manoeuvre


def _weighted_choice(rnd: random.Random, weights: dict[float, float]) -> float:
    keys = list(weights.keys())
    return rnd.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def sample_row_limits(row_id: str, own_speed_mps: float | None) -> dict:
    """Deterministic per-row sample of the three STAP-2 training-variable limits -- a
    FIXED seed derived from `row_id` (never the module's global RNG) so the exact same
    row always samples the exact same limits across separate regeneration runs. Returns
    {safe_distance_m, max_turn_deg, risk_horizon_s, risk_horizon_default_s,
    risk_horizon_multiplier, stand_on_tcpa_s} -- the sampled values themselves are
    METADATA (carried alongside a training row, never inside its `messages`), only their
    rendered constraint_line() text and their effect on the derived label are ever shown
    to the model."""
    seed = int(hashlib.sha256(str(row_id).encode("utf-8")).hexdigest()[:16], 16)
    rnd = random.Random(seed)
    safe_distance_m = _weighted_choice(rnd, SAFE_DISTANCE_WEIGHTS)
    max_turn_deg = _weighted_choice(rnd, MAX_TURN_DEG_WEIGHTS)
    horizon_default = derive_risk_horizon_s(safe_distance_m, max_turn_deg, own_speed_mps)
    multiplier = _weighted_choice(rnd, HORIZON_MULTIPLIER_WEIGHTS)
    risk_horizon_s = horizon_default * multiplier
    return {
        "safe_distance_m": safe_distance_m, "max_turn_deg": max_turn_deg,
        "risk_horizon_s": risk_horizon_s, "risk_horizon_default_s": horizon_default,
        "risk_horizon_multiplier": multiplier, "stand_on_tcpa_s": risk_horizon_s * 0.6,
    }


def fixed_limits(safe_distance_m: float, max_turn_deg: float, own_speed_mps: float | None,
                horizon_multiplier: float = 1.0) -> dict:
    """Non-sampled limits at an EXPLICIT (safe_distance_m, max_turn_deg) pair, with the
    risk horizon at its geometry-derived default (or an explicit multiple of it) -- for
    oow_colreg_scenarios_v2.json (500/30/derived-default) and its probe_{300,926} files
    (ONLY safe_distance_m changed; max_turn_deg and the horizon stay at each scenario's
    OWN derived default), as opposed to sample_row_limits()'s full per-row weighted
    sampling used for training data."""
    horizon_default = derive_risk_horizon_s(safe_distance_m, max_turn_deg, own_speed_mps)
    risk_horizon_s = horizon_default * horizon_multiplier
    return {
        "safe_distance_m": safe_distance_m, "max_turn_deg": max_turn_deg,
        "risk_horizon_s": risk_horizon_s, "risk_horizon_default_s": horizon_default,
        "risk_horizon_multiplier": horizon_multiplier, "stand_on_tcpa_s": risk_horizon_s * 0.6,
    }


def constraint_line(safe_distance_m: float, max_turn_deg: float, risk_horizon_s: float) -> str:
    """Single-source rendering of the per-row safe-distance/turn-cap FACTS -- used by BOTH
    Track-2 generators' situation-text renderers AND Basic Simulator/app/agents.py's live
    prompt, so a training row and a live simulator step given the SAME settings render
    byte-identical constraint text (see the parity test). Rendered in NM (`_NM_TO_M`,
    matching every other number in the live situation report).

    2026-09-24 simplification: states facts only (safe distance, the CPA+TCPA definition
    of real risk, the turn cap) -- no COLREG rule names, no "you must"/"becomes required"
    instructions on what to do about it. An earlier version also spelled out a stand-on-
    vessel escalation deadline and several imperatives here; that judgement is exactly the
    kind of situational reasoning meant to come from RAG/PG-retrieved COLREG text and
    eventual SFT/DPO/Reflection fine-tuning, not a hand-written rule buried in a shared
    prose function -- see SYSTEM_OOW_AGENT's DECISION PROCEDURE for the (now much
    shorter) high-level procedure that replaces it."""
    safe_distance_nm = safe_distance_m / _NM_TO_M
    return (
        f"This mission's safe passing distance is {safe_distance_nm:.3f} NM. A contact is "
        f"a real collision risk only when its CPA is below that distance AND its TCPA is "
        f"within this mission's risk horizon of {risk_horizon_s:.0f}s; beyond that horizon "
        "it is one to monitor, not yet one to act on. A single turn_left/turn_right "
        f"command may request at most {max_turn_deg:.0f} degrees."
    )

# Fixed response-format contract, byte-identical to what Basic Simulator/app/agents.py's
# build_oow_prompt() has always sent for v0-v9 (only the USER turn varies across
# configs). 'bare_qwen' is the one deliberate exception (its own separate, minimal
# BARE_SYSTEM prompt) and is unaffected by this constant.
SYSTEM_OOW_AGENT = """You are the navigator on a large commercial vessel. Decide the next helm order.

MISSION REQUIREMENTS -- this mission is only completed successfully if ALL of these hold, not just
the first one you happen to satisfy:
1. Every contact's CPA stays at or above this mission's safe passing distance (given further below)
   at all times -- exactly as mandatory as actually reaching the goal. There is no automatic safety
   net correcting your choice if you get this wrong: your own action each step is what the ship
   actually does.
2. Every manoeuvre you take while a real collision risk exists complies with COLREG.
3. You reach the mission goal.

PRIORITY ORDER when these pull in different directions -- always in this order, never reversed:
1. Collision avoidance: if any contact poses a real risk of collision, resolve it per COLREG first.
2. Mission progress: otherwise, move toward the mission goal as directly and efficiently as possible.

FACTS GIVEN TO YOU -- treat all of these as already correct; never recompute, re-derive, or
second-guess them:
- Positions/ranges are in nautical miles, bearings/headings in degrees (heading 0=north, clockwise,
  compass convention), speeds in knots.
- rel.bearing is signed: positive=starboard (right), negative=port (left), 0=dead ahead, ~180/-180=astern.
- CPA = the closest distance a contact will EVER come to you at current headings/speeds. TCPA = seconds
  until that closest point. TCPA=0 does NOT always mean an imminent collision -- it also happens once
  the closest point has already passed (the situation report says so explicitly when that's the case,
  e.g. "already past closest point, ranges now increasing").
- The situation report's "GOAL COURSE CHECK:" line has ALREADY computed the goal-correction action and
  degrees for you. Never substitute a contact's rel.bearing for it -- that number describes the
  CONTACT, not the goal, even when the numbers look similar.

DECISION PROCEDURE -- follow in order:
1. Check every contact's CPA and TCPA against this mission's safe passing distance and risk horizon
   (given further below). If no contact poses a real risk right now, go to step 3.
2. If any contact poses a real risk, pick the ONE action that satisfies the applicable COLREG rule
   for that contact. This step overrides everything below it.
3. Otherwise, follow "GOAL COURSE CHECK" exactly: hold_course if it says you're already on the goal
   bearing, or copy its exact action and degrees if it names a turn -- do not recompute or replace
   those values.
4. Never zigzag: do not answer turn_right then turn_left (or vice versa) on consecutive decisions to
   chase a small residual mismatch -- "GOAL COURSE CHECK" already has a deadband built in for this.
5. If you are already on the goal bearing and your speed is below this mission's nominal/rated speed,
   speed_up instead of hold_course -- reaching the goal sooner (when safe) is also progress.

Ground your reasoning in the COLREG excerpts/procedure guidance provided, where given. Reply with
ONLY a JSON object, no other text:
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": <float, only for turn_left/turn_right>,
 "encounter_rule": "<Rule 13, Rule 14, Rule 15, or 'none' if no encounter poses real risk>",
 "conduct_rule": "<Rule 8, Rule 13, Rule 14, Rule 16, Rule 17, Rule 19, or 'none' -- the rule that
   governs YOUR specific action, 'none' if no real risk>",
 "reasoning": "<one or two sentences>"}"""


# Fase B3 (RAG-rebuild-v2 plan, 2026-09-22): the ONE ground-truth mapping from (encounter role,
# action) to (encounter_rule, conduct_rule), shared by BOTH Track-2 generators' deterministic
# labelers (to_unified_action() / leo_choose_action()) and the B3 reasoning cross-check -- so
# none of the three can silently disagree about what the "correct" rule pair is for a given
# role/action combination. Replaces the single rule_applied field, which conflated "why are
# these two vessels in a give-way/stand-on relationship" with "which rule governs THIS action"
# (e.g. every stop was labelled Rule 17 even for a give-way vessel, when 17(b) is exclusively
# the stand-on vessel's provision -- Rule 8 is the correct citation for a give-way vessel's own
# emergency stop).
_ENCOUNTER_RULE_BY_ROLE = {
    "mutual": "Rule 14", "give_way": "Rule 15", "stand_on": "Rule 15",
    "overtaking_give_way": "Rule 13", "overtaking_stand_on": "Rule 13",
}
_STAND_ON_ROLES = ("stand_on", "overtaking_stand_on")


def classify_rules(role: str, action: str, restricted_visibility: bool = False) -> tuple[str, str]:
    """(encounter_rule, conduct_rule) for a given encounter role and the action taken.
    `role` is one of "mutual" (head-on) | "give_way" | "stand_on" | "overtaking_give_way" |
    "overtaking_stand_on" | "stationary" (a real-risk stationary/non-vessel object -- quality-
    review STAP 1 extension, 2026-09-23: NOT a COLREG vessel encounter, so encounter_rule stays
    "none" even though a real risk exists and an action IS required) | "cleared"/"none"/None (no
    real risk at all). `action` is any value from ACTIONS. `restricted_visibility=True` forces
    conduct_rule="Rule 19" (not yet produced by either generator, included for forward-
    compatibility with Basic Simulator's live agent).
    """
    if role == "stationary":
        return "none", "Rule 8"
    if role in (None, "none", "cleared"):
        return "none", "none"
    encounter_rule = _ENCOUNTER_RULE_BY_ROLE.get(role, "none")
    if restricted_visibility:
        return encounter_rule, "Rule 19"
    if role in _STAND_ON_ROLES:
        conduct_rule = "Rule 17"
    elif action == "stop":
        conduct_rule = "Rule 8"  # give-way emergency stop -- 17(b) is stand-on only
    elif role == "mutual":
        conduct_rule = "Rule 14"
    elif role == "overtaking_give_way":
        conduct_rule = "Rule 13"
    else:  # give_way (crossing), turning or a speed change
        conduct_rule = "Rule 16"
    return encounter_rule, conduct_rule


def validate_action_json(obj: dict) -> list[str]:
    """Returns a list of validation-error strings (empty == valid) for a candidate
    assistant-response dict -- the ONE schema check both training-data generators (their
    written assistant answers must pass this) and Basic Simulator/app/agents.py's
    _parse_json_action() (its acceptance test) share, so neither can silently drift from
    the other."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return [f"expected a dict, got {type(obj).__name__}"]
    for key in ("action", "degrees", "encounter_rule", "conduct_rule", "reasoning"):
        if key not in obj:
            errors.append(f"missing required key {key!r}")
    action = obj.get("action")
    if action not in ACTIONS:
        errors.append(f"action {action!r} not in {ACTIONS}")
    degrees = obj.get("degrees")
    if action in _DEGREES_ONLY_FOR:
        if not isinstance(degrees, (int, float)) or isinstance(degrees, bool):
            errors.append(f"action {action!r} requires a numeric 'degrees', got {degrees!r}")
    # 2026-09-23: found live on the cloud pod's post-Blackwell-upgrade stack -- Qwen3-8B
    # routinely writes "degrees": 0.0 (never null) for non-turn actions despite the schema
    # asking for null (95% of that sweep's parse errors, 65% of ALL its decisions, were
    # discarded for exactly this reason and silently replaced with a hold_course fallback).
    # 0/0.0/-0.0 means the same thing as "no turn requested" here, so accept it instead of
    # only the literal `null` -- a genuinely wrong nonzero degrees value (e.g. 45) for a
    # non-turn action still correctly fails below.
    elif degrees is not None and degrees != 0:
        errors.append(f"action {action!r} must have degrees=None (or 0), got {degrees!r}")
    encounter_rule = obj.get("encounter_rule")
    conduct_rule = obj.get("conduct_rule")
    for field_name, value in (("encounter_rule", encounter_rule), ("conduct_rule", conduct_rule)):
        if value != "none" and not (isinstance(value, str) and value.startswith("Rule ")
                                    and value.split(" ", 1)[-1].isdigit()):
            errors.append(f"{field_name} {value!r} must be 'none' or a whole 'Rule N'")
    # The ONE exception to "both none together": a real-risk STATIONARY/non-vessel object
    # (quality-review STAP 1 extension, 2026-09-23) is not a COLREG vessel encounter
    # (encounter_rule stays "none") even though conduct_rule is "Rule 8" (an action IS required).
    if not (encounter_rule == "none" and conduct_rule == "Rule 8"):
        if (encounter_rule == "none") != (conduct_rule == "none"):
            errors.append(f"encounter_rule {encounter_rule!r} and conduct_rule {conduct_rule!r} "
                         "must be 'none' together, never only one of them")
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        errors.append("'reasoning' must be a non-empty string")
    return errors


def bearing_and_range(ox: float, oy: float, tx: float, ty: float) -> tuple[float, float]:
    dx, dy = tx - ox, ty - oy
    return math.degrees(math.atan2(dx, dy)) % 360.0, math.hypot(dx, dy)


def relative_bearing(own_heading: float, true_bearing: float) -> float:
    return (true_bearing - own_heading + 540) % 360 - 180


def classify_encounter(own_x: float, own_y: float, own_hdg: float,
                       tgt_x: float, tgt_y: float, tgt_hdg: float) -> tuple[str, list[str], float]:
    """Single source of truth for COLREG encounter classification (quality-review STAP 2,
    2026-09-23 -- moved here from Basic Simulator/app/narrate.py, which now imports this
    instead of keeping its own copy). Correct Rule 13 check: overtaking is defined by the
    bearing of OWN-SHIP as seen from the TARGET (>112.5 deg abaft the target's beam), not
    by the bearing of the target as seen from own-ship -- using only the latter
    misclassifies real overtaking cases as ordinary crossing whenever the closing angle is
    fine/moderate rather than near-dead-astern."""
    brg_own_to_tgt, _ = bearing_and_range(own_x, own_y, tgt_x, tgt_y)
    rel_from_own = relative_bearing(own_hdg, brg_own_to_tgt)

    brg_tgt_to_own, _ = bearing_and_range(tgt_x, tgt_y, own_x, own_y)
    rel_from_tgt = relative_bearing(tgt_hdg, brg_tgt_to_own)

    course_diff = (tgt_hdg - own_hdg + 540) % 360 - 180
    if abs(rel_from_own) <= 6 and abs(abs(course_diff) - 180) <= 20:
        return "head_on", ["Rule 14"], rel_from_own
    if abs(rel_from_tgt) > 112.5:
        return "we_are_overtaking_target", ["Rule 13"], rel_from_own
    if abs(rel_from_own) > 112.5:
        return "target_is_overtaking_us", ["Rule 13"], rel_from_own
    if rel_from_own > 0:
        return "crossing_target_on_starboard", ["Rule 15", "Rule 16"], rel_from_own
    return "crossing_target_on_port", ["Rule 15", "Rule 17"], rel_from_own


GOAL_DEADBAND_DEG = 5.0


def goal_course_action(own_x: float, own_y: float, own_heading: float,
                       goal_x: float, goal_y: float) -> tuple[str, float | None]:
    """The action/degrees GOAL COURSE CHECK recommends -- ("hold_course", None) if already
    on the goal bearing (within GOAL_DEADBAND_DEG), else ("turn_left"/"turn_right", degrees).
    Used BOTH to render the check's text line (below) and as the actual ground-truth
    action a training-data generator picks when no real collision risk exists (SYSTEM_OOW_
    AGENT's decision procedure step 3) -- so the rendered text and the label a model is
    trained on can never silently disagree."""
    goal_brg, _ = bearing_and_range(own_x, own_y, goal_x, goal_y)
    off_course = relative_bearing(own_heading, goal_brg)
    if abs(off_course) <= GOAL_DEADBAND_DEG:
        return "hold_course", None
    return ("turn_right" if off_course > 0 else "turn_left"), round(abs(off_course), 1)


def goal_course_check_line(own_x: float, own_y: float, own_heading: float,
                           goal_x: float, goal_y: float,
                           max_turn_deg: float | None = None) -> str:
    """The exact 'GOAL COURSE CHECK: ...' situation-report line -- computed identically
    regardless of caller (Basic Simulator's live narrate(), or a Track-2 training-data
    generator's deterministic narrative). See this module's docstring for why a SECOND,
    independently-drifting implementation of this exact calculation is the same bug class
    already found and fixed once for `mission.targets` vs the simulator's live contacts.

    `max_turn_deg` (optional -- None preserves old behaviour for any caller not yet
    updated): when given and the needed correction exceeds it, states that fact plainly
    (2026-09-24 simplification: previously also instructed the model to "issue it now
    regardless" and "expect to repeat" -- moved that decision back to the model, it can
    reason from the stated cap and its own kinematics facts same as any other action)."""
    action, degrees = goal_course_action(own_x, own_y, own_heading, goal_x, goal_y)
    if action == "hold_course":
        return (f"GOAL COURSE CHECK: heading is ALREADY on the goal bearing (within "
                f"{GOAL_DEADBAND_DEG:.0f} deg) -- no turn needed for the goal.")
    goal_brg, _ = bearing_and_range(own_x, own_y, goal_x, goal_y)
    off_course = relative_bearing(own_heading, goal_brg)
    side = "starboard" if off_course > 0 else "port"
    line = (f"GOAL COURSE CHECK: heading is {abs(off_course):.0f} deg off the goal "
           f"bearing, to {side} -- to correct, use action \"{action}\" with "
           f"degrees={degrees:.0f} (unless a target poses a real collision "
           f"risk, which takes precedence).")
    if max_turn_deg is not None and degrees > max_turn_deg:
        line += f" This exceeds your per-command max of {max_turn_deg:.0f} deg."
    return line


# Fase B4 (RAG-rebuild-v2 plan): a "previous decisions" history preamble, prepended to a
# SUBSET of training rows' user turns, teaching the model not to reverse/re-derive an
# already-established decision from scratch every single step (the measured zigzag
# problem). Both Track-2 generators use this exact renderer so the wording a model is
# trained on never silently drifts between them.
def render_previous_decisions(decisions: list[dict]) -> str:
    """`decisions` is an ordered list (oldest first) of dicts with at least {"action",
    "degrees"} -- the helm orders actually given on the immediately preceding step(s) of
    THIS mission. Returns "" for an empty/None list (no history preamble to add).

    Quality-review B4 (2026-09-23): each dict MAY also carry "conduct_rule" (mentioned
    inline when not None/"none") and "real_risk_contact_names" (the contacts that posed a
    real risk AT THAT DECISION -- named explicitly so a reader, human or model, can check
    "is THIS SAME contact finally past and clear yet" without re-deriving it). Both are
    OPTIONAL and backward compatible -- a plain {"action", "degrees"} dict (e.g. the
    synthetic generator's self-consistency history, which has no per-contact real-risk
    data) renders exactly as before."""
    if not decisions:
        return ""
    labels = []
    for d in decisions:
        action, degrees = d["action"], d.get("degrees")
        label = f"{action} ({degrees:.0f} deg)" if degrees is not None else action
        conduct_rule = d.get("conduct_rule")
        if conduct_rule and conduct_rule != "none":
            label += f" [{conduct_rule}]"
        real_risk_names = d.get("real_risk_contact_names")
        if real_risk_names:
            label += f" -- real-risk contact(s) then: {', '.join(real_risk_names)}"
        labels.append(label)
    return (f"Your last {len(decisions)} helm decision(s), oldest first: {'; '.join(labels)}. "
           "Stay consistent with this unless the CURRENT situation below has genuinely "
           "changed enough to justify a different action -- do not reverse or re-derive an "
           "already-established decision from scratch every step.\n\n")

