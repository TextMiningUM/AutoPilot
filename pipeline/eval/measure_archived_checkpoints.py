"""
================================================================================
measure_archived_checkpoints.py -- Fase C0 (RAG-rebuild-v2 plan)
================================================================================

Retroactively runs Basic Simulator/app/measurement.py's deterministic, read-only
Check A (fabricated risk) / Check B (wrong turn direction) / Check C (physically
impossible turn) over the 147 archived units_v1 checkpoints under
"Basic Simulator/Data/missions/Missions data v2 20260922/" -- per explicit user
direction, the ONLY mission-log directory used for Fase C (older archives predate
fixes already made this session and are useless for DPO mining).

WHY THIS IS A RETROFIT, NOT JUST A RE-RUN
------------------------------------------
These 147 files predate measurement.py itself: no "measurement" field was ever
attached (run_llm_scenario.py only started doing that later), and no structured
per-checkpoint contact list was ever persisted (only the free-text
"situation_report" the model actually saw). So, per checkpoint:
  - per-contact CPA is regex-parsed straight out of situation_report's own
    "CPA X.XXX NM" text (the exact value app/narrate.py's contact_line() computed
    live, just formatted for the model -- see app/units.py's m_to_nm/nm_to_m) --
    NOT re-simulated, which would risk silently drifting from what the model was
    actually shown.
  - VesselConstraints is reconstructed from each file's own params.dt, mirroring
    exactly how run_llm_scenario.py built it at generation time (only
    time_step_s/cruise_speed_mps were ever overridden there; min_cpa_m=500.0 and
    turn_rate_deg_s=3.0 always stayed at VesselConstraints' own defaults).
  - decisions use the OLD pre-B3 single "rule_applied" field --
    measure_decision_quality()'s Fase-C0 backward-compat fallback reads it as
    both encounter_rule/conduct_rule (documented there, not duplicated here).

Writes a "measurement" field into each checkpoint IN PLACE (these files are
git-tracked, so this is reversible) -- the same field name/shape
run_llm_scenario.py already attaches live, so downstream tooling never needs to
special-case archived vs. fresh runs. Prints Check A/B/C fire-rates per config.

USAGE
-----
    python -m pipeline.eval.measure_archived_checkpoints            # writes results
    python -m pipeline.eval.measure_archived_checkpoints --dry-run  # report only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from core import AgentPaths

# app/measurement.py + app/simulation.py are stdlib-only (no torch/streamlit/
# transformers) -- safe to import into this CPU-light pipeline script. The space in
# "Basic Simulator" prevents a normal dotted import, so this mirrors the exact
# sys.path pattern Basic Simulator/tests/test_measurement.py already uses.
APP_ROOT = AgentPaths.oow().workspace / "Basic Simulator"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.measurement import measure_decision_quality  # noqa: E402
from app.simulation import VesselConstraints  # noqa: E402
from app.units import nm_to_m  # noqa: E402

MISSIONS_DIR = APP_ROOT / "Data" / "missions" / "Missions data v2 20260922"
CPA_RE = re.compile(r"CPA (-?[\d.]+) NM")
CHECK_NAMES = ("A_fabricated_risk", "B_wrong_direction", "C_degrees_over_limit")


def parse_situation_cpas_m(situation_report: str) -> list[dict]:
    """The per-contact situation list measure_decision_quality() expects, reconstructed
    from situation_report's free text -- no structured per-checkpoint contact data was
    ever persisted in these pre-measurement.py archives."""
    return [{"cpa_m": nm_to_m(float(m))} for m in CPA_RE.findall(situation_report)]


def measure_file(data: dict) -> dict:
    """Attaches a "measurement" field to every checkpoint in `data` IN PLACE, returns
    this file's own Check A/B/C/n counts."""
    dt = data.get("params", {}).get("dt", 10.0)
    constraints = VesselConstraints(time_step_s=dt)
    counts = {name: 0 for name in CHECK_NAMES}
    counts["n"] = 0
    for cp in data.get("checkpoints", []):
        situation = parse_situation_cpas_m(cp.get("situation_report") or "")
        measurement = measure_decision_quality(cp["decision"], situation, constraints)
        cp["measurement"] = measurement
        counts["n"] += 1
        for check in measurement["checks_fired"]:
            counts[check] = counts.get(check, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true", help="report only, never write files")
    args = ap.parse_args()

    files = sorted(MISSIONS_DIR.glob("*__*__units_v1.json"))
    if not files:
        sys.exit(f"No checkpoint files found under {MISSIONS_DIR}")
    print(f"Found {len(files)} archived units_v1 mission files under {MISSIONS_DIR.name}")

    per_config: dict[str, dict] = defaultdict(lambda: {name: 0 for name in CHECK_NAMES} | {"n": 0})
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = measure_file(data)
        config = data.get("config", "?")
        for k, v in counts.items():
            per_config[config][k] += v
        if not args.dry_run:
            path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'config':<20}{'n':>6}{'A%':>8}{'B%':>8}{'C%':>8}")
    for config in sorted(per_config):
        c = per_config[config]
        n = c["n"] or 1
        print(f"{config:<20}{c['n']:>6}{100 * c['A_fabricated_risk'] / n:>7.1f}%"
             f"{100 * c['B_wrong_direction'] / n:>7.1f}%{100 * c['C_degrees_over_limit'] / n:>7.1f}%")
    if args.dry_run:
        print("\n--dry-run: no files written")


if __name__ == "__main__":
    main()
