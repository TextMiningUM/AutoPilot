"""Captain reasoning/planning skeleton (design_captain_missions.md Sec 16.2/16.4) --
renders a real `CaptainResponse` (Sec 13.B.6's own output schema: action/plan/reasoning)
from the already-computed deterministic facts (`RolloutResult` cost breakdown, `regret()`
against the oracle-best candidate, and the Mission Order's own deadline slack) -- never an
LLM, exactly like the rest of this walking skeleton's deterministic baseline Captain (Sec
14). This is the "scratchpad" Sec 8.4 describes ("the active plan ... individually
relaxed/abandoned as the mission proceeds") made concrete: `render_captain_response()`
re-derives goal achievability from the REAL remaining-deadline-slack number every time it's
called, and returns an updated goals list reflecting that -- not a static copy.

Pure pipeline/ module, no app/ dependency, no GPU/API key needed, safe to run locally.
"""
from __future__ import annotations
from typing import Any

from pipeline.captain_agent_spec import CaptainAction, CaptainResponse, RolloutResult

_DEADLINE_GOAL_MARKER = "deadline"  # matches "arrive within the stated ETA deadline" (Sec 14's scenarios)


def assess_goal_achievability(deadline_slack_h: float | None) -> tuple[str, str]:
    """Sec 9's own re-planning question ("is the original mission goal still achievable")
    made concrete from one real number -- returns (goal_status, note). `deadline_slack_h`
    is hours of margin ASSUMING the mission continues at its current speed from here
    (negative = already projected to miss the deadline); `None` means the Mission Order
    states no deadline at all."""
    if deadline_slack_h is None:
        return "open", "No stated ETA deadline to assess against."
    if deadline_slack_h >= 0:
        return "open", f"Goal still achievable -- projected to arrive with {deadline_slack_h:.1f}h to spare."
    return ("at_risk", f"Goal at risk -- projected to MISS the deadline by {abs(deadline_slack_h):.1f}h "
                      "unless the plan is revised (reroute/replenish) or the goal is formally abandoned.")


def _update_deadline_goal(goals: list[dict[str, Any]], new_status: str) -> list[dict[str, Any]]:
    """Returns a NEW goals list with whichever goal mentions the ETA deadline re-stamped
    with `new_status` -- never mutates the input list in place, so the caller decides
    whether/when to actually commit this back onto the live Mission State."""
    return [{**g, "status": new_status} if _DEADLINE_GOAL_MARKER in g["goal"].lower() else dict(g)
            for g in goals]


def _format_cost_breakdown(rollout: RolloutResult) -> str:
    dims = rollout.dims
    parts = [f"life_risk={dims.life_risk:.1f}", f"ship_env_risk={dims.ship_env_risk:.1f}",
             f"duty_breaches={dims.duty_breaches:.1f}", f"goal_shortfall={dims.goal_shortfall:.2f}",
             f"commercial={dims.commercial:.1f}"]
    if rollout.notes:
        parts.append("notes: " + ", ".join(f"{k}={v:.1f}" for k, v in rollout.notes.items()))
    return "; ".join(parts)


def render_captain_response(
    event_type: str, applied: CaptainAction, chosen_rollout: RolloutResult, regret_value: float,
    shield_substituted: bool, goals: list[dict[str, Any]], deadline_slack_h: float | None,
) -> CaptainResponse:
    """Builds the Sec 13.B.6 output schema's `plan`/`reasoning` fields from real, already-
    computed facts -- the deterministic-baseline-Captain counterpart of what a prompt-driven
    Captain would eventually produce as free text (Sec 9.2/13.C.10), grounded the same way
    (cost dimensions + regret + goal-achievability), never hand-written decision prose."""
    goal_status, achievability_note = assess_goal_achievability(deadline_slack_h)
    updated_goals = _update_deadline_goal(goals, goal_status)

    regret_note = (" (this IS the oracle-best choice, zero regret)" if regret_value == 0.0
                   else f" (regret={regret_value:.1f} vs the oracle-best candidate)")
    shield_note = (" The Captain's own proposal breached a safety margin and was substituted "
                   "by the shield." if shield_substituted else "")
    reasoning = (f"Selected '{applied.tool}' for the {event_type} event{regret_note}. "
                f"Cost breakdown -- {_format_cost_breakdown(chosen_rollout)}.{shield_note}")

    return CaptainResponse(
        action=applied,
        plan={"updated_goals": updated_goals, "resource_note": achievability_note},
        reasoning=reasoning,
    )
