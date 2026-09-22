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
- TCPA=0 does NOT always mean an imminent collision -- it also happens once the closest point has
  already passed (the situation report says so explicitly when that's the case). Judge real risk from
  CPA alone, never from TCPA alone.
- The situation report's "GOAL COURSE CHECK:" line has ALREADY computed the goal-correction action and
  degrees for you. Never substitute a contact's rel.bearing for it -- that number describes the
  CONTACT, not the goal, even when the numbers look similar.

DECISION PROCEDURE -- follow in order:
1. Check every contact's CPA against this mission's safe passing distance (given further below). If
   none are below it, there is no real collision risk right now -- go to step 3.
2. If any contact's CPA is below the safe passing distance, pick the ONE action that satisfies the
   applicable COLREG rule for that contact. This step overrides everything below it.
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
 "rule_applied": "<e.g. Rule 15, or 'none' if no rule applies>",
 "reasoning": "<one or two sentences>"}"""


def validate_action_json(obj: dict) -> list[str]:
    """Returns a list of validation-error strings (empty == valid) for a candidate
    assistant-response dict -- the ONE schema check both training-data generators (their
    written assistant answers must pass this) and Basic Simulator/app/agents.py's
    _parse_json_action() (its acceptance test) share, so neither can silently drift from
    the other."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return [f"expected a dict, got {type(obj).__name__}"]
    for key in ("action", "degrees", "rule_applied", "reasoning"):
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
    rule_applied = obj.get("rule_applied")
    if rule_applied != "none" and not (isinstance(rule_applied, str) and rule_applied.startswith("Rule ")):
        errors.append(f"rule_applied {rule_applied!r} must be 'none' or 'Rule N'")
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

