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
import math

ACTIONS = ("turn_left", "turn_right", "hold_course", "speed_up", "slow_down", "stop")
_DEGREES_ONLY_FOR = ("turn_left", "turn_right")

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


def real_risk(cpa_m: float | None, tcpa_s: float | None, safe_distance_m: float) -> bool:
    """The ONE gate for "does this contact pose a real risk of collision right now" --
    CPA below `safe_distance_m` (a per-row/per-mission PARAMETER, never a hardcoded
    constant, so training data never learns a shortcut against one fixed number) AND TCPA
    within [0, RISK_HORIZON_S). Both conditions are required:
      - CPA alone is not enough: a contact can have a tiny CPA that is still hours away
        (not yet actionable) or -- more commonly -- already resolved.
      - TCPA alone is not enough: TCPA==0 is genuinely ambiguous on its own (it occurs
        BOTH for "collision right now" and "closest point already passed, now diverging"
        -- never disambiguate using TCPA alone).
      - TCPA < 0 (already past the closest point) is explicitly EXCLUDED (`0 <= tcpa_s`),
        never treated as a risk regardless of how small CPA was.
      - TCPA >= RISK_HORIZON_S is a contact to monitor, not yet one to act on.
    """
    return (cpa_m is not None and cpa_m < safe_distance_m
            and tcpa_s is not None and 0 <= tcpa_s < RISK_HORIZON_S)

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
- Positions/bearings/headings are in metres/degrees, heading 0=north, clockwise (compass convention).
- rel.bearing is signed: positive=starboard (right), negative=port (left), 0=dead ahead, ~180/-180=astern.
- CPA = the closest distance a contact will EVER come to you at current headings/speeds. TCPA = seconds
  until that closest point.
- A contact poses REAL collision risk only when BOTH hold: its CPA is below this mission's safe
  passing distance, AND its TCPA is within the real-risk time horizon (both given further below) --
  neither alone is enough. TCPA=0 does NOT always mean an imminent collision -- it also happens once
  the closest point has already passed (the situation report says so explicitly when that's the
  case, e.g. "already past closest point, ranges now increasing"); such a contact poses no real risk
  regardless of how small its CPA was. A contact whose TCPA is beyond the horizon is one to monitor,
  not yet one to act on.
- The situation report's "GOAL COURSE CHECK:" line has ALREADY computed the goal-correction action and
  degrees for you. Never substitute a contact's rel.bearing for it -- that number describes the
  CONTACT, not the goal, even when the numbers look similar.

DECISION PROCEDURE -- follow in order:
1. Check every contact against this mission's safe passing distance AND real-risk time horizon
   (given further below): a real risk exists only when a contact's CPA is below the safe distance
   AND its TCPA is within the horizon. If no contact meets both, there is no real collision risk
   right now -- go to step 3.
2. If any contact meets both conditions, pick the ONE action that satisfies the applicable COLREG
   rule for that contact. This step overrides everything below it.
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
   governs YOUR specific action (see below), 'none' if no real risk>",
 "reasoning": "<one or two sentences>"}

encounter_rule names which COLREG encounter you are in, from the geometry alone (Rule 13
overtaking, Rule 14 head-on, Rule 15 crossing, or 'none' if no contact poses real risk).
conduct_rule names the rule that governs the SPECIFIC action you are taking: Rule 16 for a
give-way vessel's turn/speed change (except overtaking, which stays Rule 13), Rule 14 for a
head-on turn, Rule 17 for a stand-on vessel (holding course, or its own 17(b) action), Rule 8
for a give-way vessel's emergency stop (Rule 17(b) is for the STAND-ON vessel only, never a
give-way vessel's own stop), Rule 19 in restricted visibility, or 'none' if no real risk. Both
fields are whole rule numbers only (no sub-paragraphs like "17(b)" -- put that detail in
reasoning instead), and both are 'none' together whenever no contact poses real risk -- standing
rules (2/5/6/7/11) are never cited in either field, they always apply and are not what these
fields are for."""


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
    "overtaking_stand_on" | "cleared"/"none"/None (no real risk). `action` is any value from
    ACTIONS. `restricted_visibility=True` forces conduct_rule="Rule 19" (not yet produced by
    either generator, included for forward-compatibility with Basic Simulator's live agent).
    """
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
    elif degrees is not None:
        errors.append(f"action {action!r} must have degrees=None, got {degrees!r}")
    encounter_rule = obj.get("encounter_rule")
    conduct_rule = obj.get("conduct_rule")
    for field_name, value in (("encounter_rule", encounter_rule), ("conduct_rule", conduct_rule)):
        if value != "none" and not (isinstance(value, str) and value.startswith("Rule ")
                                    and value.split(" ", 1)[-1].isdigit()):
            errors.append(f"{field_name} {value!r} must be 'none' or a whole 'Rule N'")
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


def goal_course_action(own_x: float, own_y: float, own_heading: float,
                       goal_x: float, goal_y: float) -> tuple[str, float | None]:
    """The action/degrees GOAL COURSE CHECK recommends -- ("hold_course", None) if already
    on the goal bearing (within a 10 deg deadband), else ("turn_left"/"turn_right", degrees).
    Used BOTH to render the check's text line (below) and as the actual ground-truth
    action a training-data generator picks when no real collision risk exists (SYSTEM_OOW_
    AGENT's decision procedure step 3) -- so the rendered text and the label a model is
    trained on can never silently disagree."""
    goal_brg, _ = bearing_and_range(own_x, own_y, goal_x, goal_y)
    off_course = relative_bearing(own_heading, goal_brg)
    if abs(off_course) <= 10:
        return "hold_course", None
    return ("turn_right" if off_course > 0 else "turn_left"), round(abs(off_course), 1)


def goal_course_check_line(own_x: float, own_y: float, own_heading: float,
                           goal_x: float, goal_y: float) -> str:
    """The exact 'GOAL COURSE CHECK: ...' situation-report line -- computed identically
    regardless of caller (Basic Simulator's live narrate(), or a Track-2 training-data
    generator's deterministic narrative). See this module's docstring for why a SECOND,
    independently-drifting implementation of this exact calculation is the same bug class
    already found and fixed once for `mission.targets` vs the simulator's live contacts."""
    action, degrees = goal_course_action(own_x, own_y, own_heading, goal_x, goal_y)
    if action == "hold_course":
        return ("GOAL COURSE CHECK: heading is ALREADY on the goal bearing (within "
                "10 deg) -- no turn needed for the goal.")
    goal_brg, _ = bearing_and_range(own_x, own_y, goal_x, goal_y)
    off_course = relative_bearing(own_heading, goal_brg)
    side = "starboard" if off_course > 0 else "port"
    return (f"GOAL COURSE CHECK: heading is {abs(off_course):.0f} deg off the goal "
           f"bearing, to {side} -- to correct, use action \"{action}\" with "
           f"degrees={degrees:.0f} (unless a target poses a real collision "
           f"risk, which takes precedence).")


# Fase B4 (RAG-rebuild-v2 plan): a "previous decisions" history preamble, prepended to a
# SUBSET of training rows' user turns, teaching the model not to reverse/re-derive an
# already-established decision from scratch every single step (the measured zigzag
# problem). Both Track-2 generators use this exact renderer so the wording a model is
# trained on never silently drifts between them.
def render_previous_decisions(decisions: list[dict]) -> str:
    """`decisions` is an ordered list (oldest first) of dicts with at least {"action",
    "degrees"} -- the helm orders actually given on the immediately preceding step(s) of
    THIS mission. Returns "" for an empty/None list (no history preamble to add)."""
    if not decisions:
        return ""
    labels = []
    for d in decisions:
        action, degrees = d["action"], d.get("degrees")
        labels.append(f"{action} ({degrees:.0f} deg)" if degrees is not None else action)
    return (f"Your last {len(decisions)} helm decision(s), oldest first: {'; '.join(labels)}. "
           "Stay consistent with this unless the CURRENT situation below has genuinely "
           "changed enough to justify a different action -- do not reverse or re-derive an "
           "already-established decision from scratch every step.\n\n")

