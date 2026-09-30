"""pipeline/train/reward_grpo.py -- Stap 2 Step 9 prep: TRL GRPOTrainer reward functions
for the OOW agent.

Three composable reward functions (combined via GRPOConfig's reward_weights), each
`Callable[..., list[float]]` matching TRL's documented contract: called once per
training step with `prompts`/`completions` (length = batch_size * num_generations) plus
every OTHER train_dataset column passed through by name as a kwarg (here: mission/own/
targets/constraints, one dict per row -- see build_grpo_dataset.py).

  1. reward_schema      -- can the completion even be parsed as the required JSON action
                            schema? (reuses pipeline.track2.b3_reasoning_gates' own
                            validate_action_json(), the SAME schema check every other
                            OOW training-data gate already enforces)
  2. reward_legality    -- does the chosen action's direction violate
                            app.oracle_planner.required_direction()'s HARD COLREG gate
                            (the same gate build_oow_scenarios_rnd.py's oracle-labeled
                            data is filtered by -- here used as a reward penalty instead
                            of an exclusion filter, since GRPO needs a continuous signal
                            for every sampled completion, not a binary keep/drop)?
  3. reward_quality     -- app.oracle_planner._rollout_cost()'s own 60s Nomoto-physics
                            rollout cost (goal-deviation + clearance-penalty + turn-
                            effort), evaluated on the COMPLETION's own offset instead of
                            a fixed grid candidate -- pure Python trig, no GPU, cheap
                            enough to run k times per training step.

SCOPE CAVEAT (same as oracle_planner.py itself): turn-quality only. A speed_up/
slow_down/stop completion is scored as offset=0.0 (same as hold_course) -- this reward
set does not yet distinguish "correctly held course" from "should have used an emergency
stop/speed change instead" beyond what reward_legality's direction check already catches.
Flagged as a known first-version limitation, not silently ignored.
"""
from __future__ import annotations

import dataclasses
import re

from core import AgentPaths

paths = AgentPaths.oow()
APP_ROOT = paths.ensure_basic_simulator_importable()

from app import oracle_planner  # noqa: E402
from app.missions import Mission, Vessel  # noqa: E402
from app.simulation import VesselConstraints  # noqa: E402
from pipeline.oow_agent_spec import validate_action_json  # noqa: E402
from pipeline.track2.b3_reasoning_gates import extract_first_json_object  # noqa: E402

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Penalties are fixed, large-magnitude negatives so they dominate reward_quality's much
# smaller-magnitude continuous cost (oracle_planner's own W_GOAL/W_CLEARANCE/W_EFFORT
# weights keep _rollout_cost() typically in the 0-5 range) -- a schema/legality failure
# must always outrank any quality difference between two otherwise-valid completions.
SCHEMA_FAIL_REWARD = -3.0
LEGALITY_FAIL_REWARD = -2.0


def _completion_text(completion) -> str:
    """A GRPO completion is either a plain string (text prompts) or a list of chat
    messages (conversational prompts, what this dataset uses) -- return the assistant
    text either way."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


def _parse_action(completion) -> dict | None:
    text = _completion_text(completion)
    m = _THINK_RE.search(text)
    tail = text[m.end():] if m else text
    return extract_first_json_object(tail)


def _offset_from_action(obj: dict) -> float:
    """Signed heading offset matching oracle_planner._offset_direction()'s own
    convention -- positive=starboard, negative=port, 0.0 for any non-turn action
    (hold_course/speed_up/slow_down/stop alike, see module docstring's scope caveat)."""
    action, degrees = obj.get("action"), obj.get("degrees")
    if action == "turn_right":
        return float(degrees or 0.0)
    if action == "turn_left":
        return -float(degrees or 0.0)
    return 0.0


def _reconstruct_state(mission_d: dict, own_d: dict, targets_d: list[dict],
                       constraints_d: dict) -> tuple[Mission, Vessel, list[Vessel], VesselConstraints]:
    mission = Mission(**{**mission_d, "own_ship": Vessel(**mission_d["own_ship"]),
                        "targets": [Vessel(**t) for t in mission_d["targets"]]})
    own = Vessel(**own_d)
    targets = [Vessel(**t) for t in targets_d]
    constraints = VesselConstraints(**constraints_d)
    return mission, own, targets, constraints


def reward_schema(prompts, completions, **kwargs) -> list[float]:
    out = []
    for c in completions:
        obj = _parse_action(c)
        errors = validate_action_json(obj) if obj is not None else ["no parseable JSON action object"]
        out.append(1.0 if not errors else SCHEMA_FAIL_REWARD)
    return out


def reward_legality(prompts, completions, mission, own, targets, constraints, **kwargs) -> list[float]:
    out = []
    for i, c in enumerate(completions):
        obj = _parse_action(c)
        if obj is None:
            out.append(LEGALITY_FAIL_REWARD)
            continue
        _m, o, t, cons = _reconstruct_state(mission[i], own[i], targets[i], constraints[i])
        required = oracle_planner.required_direction(o, t, cons)
        direction = oracle_planner._offset_direction(_offset_from_action(obj))
        illegal = required is not None and direction != required
        out.append(LEGALITY_FAIL_REWARD if illegal else 1.0)
    return out


def reward_quality(prompts, completions, mission, own, targets, constraints, **kwargs) -> list[float]:
    out = []
    for i, c in enumerate(completions):
        obj = _parse_action(c)
        if obj is None:
            out.append(SCHEMA_FAIL_REWARD)
            continue
        m, o, t, cons = _reconstruct_state(mission[i], own[i], targets[i], constraints[i])
        cost, _clearance = oracle_planner._rollout_cost(m, o, _offset_from_action(obj), t, cons)
        out.append(-cost)
    return out


# Suggested starting weights for GRPOConfig(reward_weights=...), same order as
# REWARD_FUNCS below -- legality dominates (the hard COLREG gate), schema is a close
# second (an unparseable completion can't be scored by anything else), quality is the
# fine-grained shaping term. Tune empirically once a pilot run's reward distribution is
# actually observed -- these are principled starting points, not measured-optimal.
REWARD_FUNCS = [reward_schema, reward_legality, reward_quality]
REWARD_WEIGHTS = [0.3, 1.0, 0.3]
