"""Conversion pass (2026-09-24): rewrites evaluation.compliance/.explanation_compliance in
every existing Basic Simulator/Data/missions/_llm_runs/*.json run log to the NEW manoeuvre/
explanation split (see Evaluation Functions/evaluate_run.py's COMPLIANCE_CATEGORY split +
app/evaluation.py's A_fabricated_risk/E_role_fabrication dedup). Unlike
_tmp_compare_compliance_split.py (read-only, derived the split from the OLD stored
breakdown), this ACTUALLY RE-RUNS score_trajectory() against each file's own stored
trajectory/checkpoints -- the exact same deterministic recompute the current code would
produce, no re-simulation/no model call needed since trajectory+checkpoints are already on
disk. Only the "evaluation" key is replaced in place; mission/trajectory/checkpoints/
colreg_llm_check/params/outcome are left untouched.

Run from the "Basic Simulator" directory: python ../_tmp_convert_llm_runs_evaluation.py
"""
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent / "Basic Simulator"
sys.path.insert(0, str(APP_ROOT))

from app.evaluation import score_trajectory  # noqa: E402
from app.missions import mission_from_dict, load_mission  # noqa: E402
from app.simulation import VesselConstraints  # noqa: E402
from app.sweep_llm_params import score_one, _save_row  # noqa: E402

RUNS_DIR = APP_ROOT / "Data" / "missions" / "_llm_runs"
_SKIP = {"_sweep_status.json", "_sweep_summary.json"}
_DEFAULTS = VesselConstraints()  # no run in this codebase has ever overridden these


def main() -> None:
    files = sorted(p for p in RUNS_DIR.glob("*.json") if p.name not in _SKIP)
    print(f"Converting {len(files)} run file(s) in {RUNS_DIR}\n")
    print(f"{'file':60s} {'old':>6s} {'new_M':>6s} {'new_E':>6s} {'oldC':>6s} {'newC':>6s}")
    n_ok, n_skip = 0, 0
    for p in files:
        log = json.loads(p.read_text(encoding="utf-8"))
        if not all(k in log for k in ("mission", "trajectory", "checkpoints")):
            print(f"  [skip] {p.name}: missing mission/trajectory/checkpoints")
            n_skip += 1
            continue
        old_ev = log.get("evaluation") or {}
        old_score = old_ev.get("compliance", {}).get("score")
        old_composite = old_ev.get("composite_score")

        mission = mission_from_dict(log["mission"])
        try:
            new_ev = score_trajectory(
                log["trajectory"], start_xy=(mission.own_ship.x, mission.own_ship.y),
                goal_xy=mission.goal, nominal_speed=mission.own_ship.speed,
                safe_distance_m=_DEFAULTS.min_cpa_m, max_turn_deg=_DEFAULTS.max_rudder_angle_deg,
                checkpoints=log["checkpoints"],
            )
        except Exception as exc:
            print(f"  [FAIL] {p.name}: {exc}")
            n_skip += 1
            continue
        log["evaluation"] = new_ev
        p.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

        print(f"{p.name[:60]:60s} "
              f"{(old_score if old_score is not None else float('nan')):6.3f} "
              f"{new_ev['compliance']['score']:6.3f} "
              f"{new_ev['explanation_compliance']['score']:6.3f} "
              f"{(old_composite if old_composite is not None else float('nan')):6.3f} "
              f"{new_ev['composite_score']:6.3f}")
        n_ok += 1

    print(f"\nConverted {n_ok} file(s), skipped {n_skip}.")

    # Refresh _sweep_summary.json's per-(mission,config,tag) rows from the just-converted
    # evaluations too -- the dashboard reads this file directly, not the individual logs.
    n_refreshed = 0
    for p in files:
        try:
            log = json.loads(p.read_text(encoding="utf-8"))
            mission_id = log.get("mission_id")
            if not mission_id:
                continue
            mission = load_mission(mission_id)
            row = score_one(mission, log)
            _save_row(mission_id, row)
            n_refreshed += 1
        except Exception as exc:
            print(f"  [summary refresh FAIL] {p.name}: {exc}")
    print(f"Refreshed {n_refreshed} row(s) in _sweep_summary.json.")


if __name__ == "__main__":
    main()
