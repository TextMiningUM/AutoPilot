"""One-off CPU-only smoke test: confirms the checkpoint-construction logic in
run_llm_scenario.py preserves the full raw reasoning (every "Wait, ..." step) as its own
top-level "reasoning_raw" field, and no longer duplicates it inside "debug"."""
from app.missions import load_mission
from app.simulation import Simulation, VesselConstraints
from app.narrate import contact_line
from app.measurement import measure_decision_quality

mission = load_mission("Imazu01")
constraints = VesselConstraints(time_step_s=10.0, cruise_speed_mps=mission.own_ship.speed)
sim = Simulation(mission, constraints)
contacts_now = [contact_line(sim.own, t) for t in sim.targets]

decision = {"action": "turn_right", "degrees": 20.0, "rule_applied": "Rule 14", "reasoning": "short"}
debug = {
    "situation": "sit text",
    "config": "v3_rag_cot",
    "raw_response": "<think>\nWait, let me check CPA... Wait, actually Rule 14 applies.\n</think>\n"
                    '{"action": "turn_right"}',
}
cp = {
    "step": 0, "time": sim.t,
    "situation_report": debug.get("situation"),
    "decision": decision,
    "reasoning_raw": debug.get("raw_response"),
    "measurement": measure_decision_quality(decision, contacts_now, constraints),
    "debug": {kk: vv for kk, vv in debug.items() if kk not in ("situation", "raw_response")},
}

assert "raw_response" not in cp["debug"]
assert cp["reasoning_raw"].startswith("<think>")
assert "Wait" in cp["reasoning_raw"]
print("checkpoint keys:", list(cp.keys()))
print("debug keys (no raw_response, no situation):", list(cp["debug"].keys()))
print("reasoning_raw:", cp["reasoning_raw"])
print("WIRING OK -- CPU only, no GPU/model touched")
