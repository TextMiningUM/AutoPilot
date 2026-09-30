"""pipeline/action_tokens.py -- Stap 2 Step 10 (VLA-style action tokens) design/foundation.

Replaces the free-text-mediated {"action": "turn_right", "degrees": 15.3} pair with a
SINGLE closed-vocabulary special token (e.g. "<hdg_p15>") as the JSON "action" value --
the model commits to a discrete action TOKEN, not a linguistic byproduct of generating
the word "turn_right" followed by a separately-sampled float. This is the minimal-diff
version of "action tokens" (see repo memory / chat discussion, 2026-09-30): encounter_
rule/conduct_rule/reasoning stay exactly as free text (no change to rule-citation or
explanation scoring), only the action+degrees pair becomes one token.

WHY 1-DEGREE GRANULARITY, NOT THE ORACLE'S OWN 15-DEGREE SEARCH GRID
---------------------------------------------------------------------
app/oracle_planner.py's `_GRID_OFFSETS_DEG` (-90..90 step 15) is a COMPUTATIONAL
SHORTCUT for its own candidate search, not a claim that 15-degree precision is enough
for the action space the model should be able to EXPRESS -- oracle-labeled degrees
routinely come from the 6 baselines' own continuous first-step choices (e.g. Sawada's
CRI-proportional turn, DWA/MPC's own arbitrary picks) and live GOAL COURSE CHECK
corrections have been observed well past 90 degrees (e.g. 112 deg, see repo memory).
Collapsing to a 15-degree bucket here would be a real, avoidable precision loss.
1-degree steps over -180..180 (361 tokens) preserves effectively all the precision
existing training data already has (values are generated via `round(x, 1)` -- only the
first decimal is ever lost, immaterial for actual steering) while still giving DAgger/
GRPO a small, closed, directly-diffable/scoreable action space -- comparable in scale
to typical VLA action-discretization bin counts (e.g. OpenVLA's 256 bins per DoF).

WHAT THIS MODULE DOES NOT DO (YET)
-----------------------------------
This is the shared encode/decode contract ONLY -- it does not yet touch the tokenizer,
training data, SYSTEM_OOW_AGENT's schema description, agents.py's _parse_json_action(),
or reward_grpo.py/build_dagger_dpo.py's action extraction. Wiring those in is a real,
multi-file follow-up (regenerate SFT/DPO/reflection data in the new schema, resize the
model's embeddings, retrain) -- deliberately NOT done in this pass, see the chat
discussion this module was built from for the staged rollout plan.
"""
from __future__ import annotations

from pipeline.oow_agent_spec import ACTIONS

DEGREE_MIN = -180
DEGREE_MAX = 180

NON_TURN_TOKENS = {
    "hold_course": "<hold_course>",
    "speed_up": "<speed_up>",
    "slow_down": "<slow_down>",
    "stop": "<stop>",
}
_NON_TURN_TOKEN_TO_ACTION = {v: k for k, v in NON_TURN_TOKENS.items()}


def _heading_token(signed_degrees: int) -> str:
    """signed_degrees > 0 == starboard (turn_right), < 0 == port (turn_left) -- same sign
    convention as app/oracle_planner.py's own offset_deg/required_direction()."""
    sign = "p" if signed_degrees >= 0 else "m"
    return f"<hdg_{sign}{abs(signed_degrees)}>"


# The full closed vocabulary, in a stable/deterministic order (tokenizer.add_special_
# tokens() is order-sensitive for reproducible token IDs across separate calls/machines).
ALL_TOKENS: list[str] = (
    [_heading_token(d) for d in range(DEGREE_MIN, DEGREE_MAX + 1) if d != 0]
    + list(NON_TURN_TOKENS.values())
)


def encode_action(action: str, degrees: float | None) -> str:
    """(action, degrees) -- the existing ACTIONS-vocabulary pair -- -> one closed-
    vocabulary token string. Raises ValueError on an action outside ACTIONS or a degrees
    value outside [DEGREE_MIN, DEGREE_MAX] (callers should treat that as a data-quality
    issue to fix upstream, never silently clamp)."""
    if action not in ACTIONS:
        raise ValueError(f"action {action!r} not in {ACTIONS}")
    if action in NON_TURN_TOKENS:
        return NON_TURN_TOKENS[action]
    if degrees is None:
        raise ValueError(f"action {action!r} requires a numeric degrees value")
    signed = round(degrees) if action == "turn_right" else -round(degrees)
    if not (DEGREE_MIN <= signed <= DEGREE_MAX) or signed == 0:
        raise ValueError(f"degrees {degrees!r} (signed {signed}) out of range "
                         f"[{DEGREE_MIN}, {DEGREE_MAX}] or rounds to 0 (use hold_course instead)")
    return _heading_token(signed)


def decode_action(token: str) -> tuple[str, float | None]:
    """The reverse of encode_action() -- one closed-vocabulary token -> (action, degrees).
    Raises ValueError on a token outside ALL_TOKENS (a model that emits something else
    produced an invalid/unparseable action -- same "never silently coerce" convention as
    validate_action_json())."""
    if token in _NON_TURN_TOKEN_TO_ACTION:
        return _NON_TURN_TOKEN_TO_ACTION[token], None
    if token.startswith("<hdg_p") and token.endswith(">"):
        return "turn_right", float(token[len("<hdg_p"):-1])
    if token.startswith("<hdg_m") and token.endswith(">"):
        return "turn_left", float(token[len("<hdg_m"):-1])
    raise ValueError(f"{token!r} is not a recognised action token")
