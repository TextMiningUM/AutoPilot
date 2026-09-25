"""Deterministic baseline collision-avoidance decision functions -- the non-LLM
counterparts to app.agents.ask_oow(), sharing the exact same Simulation.apply_action()
interface so run_baseline_scenario.py can reuse run_llm_scenario.py's orchestration
pattern almost unchanged.

Each decide() function has the signature:
    decide(mission, own, targets, constraints) -> (decision, debug)
where `decision` is the SAME {"action","degrees","encounter_rule","conduct_rule",
"reasoning"} shape ask_oow() returns (validated by the same
pipeline.oow_agent_spec.validate_action_json()), and `debug` carries at least
{"situation": <narrate() text>} for parity with the LLM run logs' checkpoint schema.

BASELINE_CONFIGS: dict[str, str] -- registry of {config_name: human label}, mirrors
app.agents.MODEL_CONFIGS. DECISION_FUNCS: dict[str, Callable] -- the actual functions.
Add a new baseline by dropping in a module with its own decide() and registering both
dicts below -- nothing else (run_baseline_scenario.py, sweep tooling) needs to change.
"""
from __future__ import annotations
from typing import Callable

from app.missions import Mission, Vessel
from app.simulation import VesselConstraints

from .ruletree import decide as decide_ruletree
from .velocity_obstacle import decide as decide_vo
from .potential_field import decide as decide_apf
from .dynamic_window import decide as decide_dwa
from .mpc import decide as decide_mpc
from .sawada import decide as decide_sawada

BASELINE_CONFIGS: dict[str, str] = {
    "baseline_ruletree": "Rule-based COLREG decision tree (deterministic) -- IMO COLREGS (1972)",
    "baseline_vo": "Velocity Obstacle (VO) / Collision Cone (deterministic) -- "
                  "Fiorini & Shiller (1998), Int. J. Robotics Research 17(7)",
    "baseline_apf": "Artificial Potential Field (APF) (deterministic) -- "
                    "Khatib (1986), Int. J. Robotics Research 5(1)",
    "baseline_dwa": "Dynamic Window Approach (DWA) (deterministic) -- "
                    "Fox, Burgard & Thrun (1997), IEEE Robotics & Automation Magazine 4(1)",
    "baseline_mpc": "Model Predictive Control (MPC), rollout-based (deterministic) -- "
                    "Garcia, Prett & Morari (1989), Automatica 25(3)",
    "baseline_sawada": "Sawada et al. (2021) conventional method -- BEST-EFFORT "
                       "reconstruction (CRI-based), J. Mar. Sci. Technol. 26(2), 509-524 "
                       "-- see app/baselines/sawada.py docstring",
}

DECISION_FUNCS: dict[str, Callable[[Mission, Vessel, list[Vessel], VesselConstraints], tuple[dict, dict]]] = {
    "baseline_ruletree": decide_ruletree,
    "baseline_vo": decide_vo,
    "baseline_apf": decide_apf,
    "baseline_dwa": decide_dwa,
    "baseline_mpc": decide_mpc,
    "baseline_sawada": decide_sawada,
}
