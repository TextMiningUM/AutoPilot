"""Captain Track 2 held-out eval -- captain_mission_scenarios.json (design_captain_missions.md
Sec 13.C.9/13.C.13, mirrors vhf_colreg_scenarios.json/oow_colreg_scenarios.json's role).

Basic Simulator/generate_captain_missions.py already implements Sec 13.C.9's strongest
held-out guarantee: a WHOLE route template ("long") is permanently reserved and never
drawn for a training mission, regardless of which events/severities attach to it -- every
generated mission already carries its own `held_out: bool` flag. This script performs NO
generation of its own; it only selects the already-`held_out: true` missions out of
Data/Captain/Scenarios/generated/*.json and compiles them into the proper held-out
location (Captain_Eval/), exactly as design_captain_missions.md Sec 13.C.13 specifies: the
scenario file stores only what actually happened (events/severity/context/trigger/
world-responder) -- the "correct answer" for each brown envelope is derived AT EVAL TIME
from the procedure library (pipeline/captain_agent_spec.py), never stored redundantly
here, so this compiler needs no ground-truth-labelling logic of its own.

If too few held-out missions exist yet, first generate more (additive, deterministic from
(seed, index) -- reruns at the same seed never change already-written files):
    python "Basic Simulator/generate_captain_missions.py" --n 200 --seed 100

Safe to run locally (pure JSON selection, no GPU/API key needed).

Run with: python -m pipeline.eval.build_captain_mission_scenarios
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from core import AgentPaths

paths = AgentPaths.captain()
GENERATED_DIR = paths.data_root / "Scenarios" / "generated"
OUT_FILE = paths.eval_file("captain_mission_scenarios.json")


def load_held_out_missions() -> list[dict]:
    """Every already-generated mission with held_out == True, sorted by mission_id for a
    reproducible file (generation order on disk is not guaranteed stable across platforms)."""
    missions = []
    for path in sorted(GENERATED_DIR.glob("*.json")):
        mission = json.loads(path.read_text(encoding="utf-8"))
        if mission.get("held_out"):
            missions.append(mission)
    missions.sort(key=lambda m: m["mission_id"])
    return missions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = ap.parse_args()

    if not GENERATED_DIR.exists():
        raise SystemExit(f"{GENERATED_DIR} does not exist -- run generate_captain_missions.py first")

    missions = load_held_out_missions()
    print(f"Held-out missions found: {len(missions)}")
    if not missions:
        raise SystemExit("0 held-out missions -- generate more with a larger --n first")

    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(missions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUT_FILE}")

    route_counts = Counter(m["route_template"] for m in missions)
    print(f"Route templates: {dict(route_counts)}")
    event_counts = Counter(ev["type"] for m in missions for ev in m["brown_envelopes"])
    print(f"Brown-envelope event types: {dict(event_counts)}")


if __name__ == "__main__":
    main()
