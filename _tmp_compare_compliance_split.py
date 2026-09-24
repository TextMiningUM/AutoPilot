"""Read-only comparison: recompute the NEW manoeuvre/explanation compliance split from
data ALREADY stored in each Basic Simulator/Data/missions/_llm_runs/*.json run log
(old evaluation.compliance.breakdown -- one flat, blended list of every deducted code),
without touching the files themselves and without re-running any model/simulation.

For each run:
  - old_score      = evaluation.compliance.score (as originally computed, blended)
  - new_manoeuvre  = 1.0 - sum(weight for manoeuvre-category codes, after A/E dedup), clipped
  - new_explanation= 1.0 - sum(weight for explanation-category codes, after A/E dedup), clipped
  - old_composite / new_composite recomputed with DEFAULT_WEIGHTS, swapping in new_manoeuvre
    for the "compliance" term (explanation never feeds composite).

Does NOT write anything back to the run files -- purely a diff report.
"""
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "Basic Simulator" / "Data" / "missions" / "_llm_runs"

_spec = importlib.util.spec_from_file_location(
    "evaluate_run", ROOT / "Basic Simulator" / "Evaluation Functions" / "evaluate_run.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
COMPLIANCE_CATEGORY = _mod.COMPLIANCE_CATEGORY
DEFAULT_WEIGHTS = _mod.DEFAULT_WEIGHTS


def resplit(old_breakdown: list[dict]) -> tuple[float, float, list[dict], list[dict]]:
    """old_breakdown: evaluation.compliance.breakdown as originally stored (blended, may
    include the single {"code": "collision", ...} hard-gate entry). Returns
    (new_manoeuvre_score, new_explanation_score, manoeuvre_breakdown, explanation_breakdown)."""
    if len(old_breakdown) == 1 and old_breakdown[0]["code"] == "collision":
        return 0.0, 0.0, old_breakdown, old_breakdown

    # A/E dedup: group by "at" (step or contact), drop A_fabricated_risk when
    # E_role_fabrication also fired at the same "at".
    by_at: dict = {}
    for f in old_breakdown:
        by_at.setdefault(f["at"], []).append(f)
    deduped = []
    for at, entries in by_at.items():
        codes_here = {e["code"] for e in entries}
        if "E_role_fabrication" in codes_here and "A_fabricated_risk" in codes_here:
            entries = [e for e in entries if e["code"] != "A_fabricated_risk"]
        deduped.extend(entries)

    manoeuvre = [f for f in deduped if COMPLIANCE_CATEGORY.get(f["code"]) == "manoeuvre"]
    explanation = [f for f in deduped if COMPLIANCE_CATEGORY.get(f["code"]) == "explanation"]
    m_score = max(0.0, min(1.0, 1.0 + sum(f["deduction"] for f in manoeuvre)))
    e_score = max(0.0, min(1.0, 1.0 + sum(f["deduction"] for f in explanation)))
    return m_score, e_score, manoeuvre, explanation


def recompute_composite(ev: dict, new_manoeuvre_score: float) -> float:
    w = DEFAULT_WEIGHTS
    man = ev["manoeuvre"]
    if not ev["safety"]["passed"]:
        return 0.0
    if not ev["temporal"]["arrived"]:
        return min(0.2, w["compliance"] * new_manoeuvre_score +
                        w["manoeuvre"] * man["manoeuvre_score"] +
                        w["smoothness"] * man["smoothness_score"])
    return (w["compliance"] * new_manoeuvre_score +
            w["temporal"] * ev["temporal"]["temporal_score"] +
            w["spatial"] * ev["spatial"]["spatial_score"] +
            w["manoeuvre"] * man["manoeuvre_score"] +
            w["smoothness"] * man["smoothness_score"])


def main() -> None:
    files = sorted(RUNS_DIR.glob("*.json"))
    print(f"Found {len(files)} run file(s) in {RUNS_DIR}\n")
    rows = []
    code_counter = Counter()
    for p in files:
        try:
            log = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [skip] {p.name}: {exc}")
            continue
        ev = log.get("evaluation")
        if not ev or "compliance" not in ev:
            continue
        old_breakdown = ev["compliance"]["breakdown"]
        old_score = ev["compliance"]["score"]
        old_composite = ev["composite_score"]
        for f in old_breakdown:
            code_counter[f["code"]] += 1
        m_score, e_score, m_bd, e_bd = resplit(old_breakdown)
        new_composite = recompute_composite(ev, m_score)
        rows.append({
            "file": p.name, "mission_id": log.get("mission_id"),
            "config": log.get("config"), "verdict": ev.get("verdict"),
            "completed": str(ev.get("verdict", "")).startswith("PASS"),
            "old_score": old_score, "new_manoeuvre": round(m_score, 3),
            "new_explanation": round(e_score, 3), "old_composite": round(old_composite, 3),
            "new_composite": round(new_composite, 3), "n_old": len(old_breakdown),
            "n_manoeuvre": len(m_bd), "n_explanation": len(e_bd),
        })

    if not rows:
        print("No runs with compliance data found.")
        return

    print(f"{'file':60s} {'old':>6s} {'new_M':>6s} {'new_E':>6s} {'oldC':>6s} {'newC':>6s}")
    for r in rows:
        print(f"{r['file'][:60]:60s} {r['old_score']:6.3f} {r['new_manoeuvre']:6.3f} "
              f"{r['new_explanation']:6.3f} {r['old_composite']:6.3f} {r['new_composite']:6.3f}")

    n = len(rows)
    avg_old = sum(r["old_score"] for r in rows) / n
    avg_new_m = sum(r["new_manoeuvre"] for r in rows) / n
    avg_new_e = sum(r["new_explanation"] for r in rows) / n
    avg_old_c = sum(r["old_composite"] for r in rows) / n
    avg_new_c = sum(r["new_composite"] for r in rows) / n
    n_zero_old = sum(1 for r in rows if r["old_score"] == 0.0)
    n_zero_new_m = sum(1 for r in rows if r["new_manoeuvre"] == 0.0)

    print(f"\n--- Summary over {n} run(s) ---")
    print(f"avg old compliance (blended):       {avg_old:.3f}")
    print(f"avg new manoeuvre compliance:        {avg_new_m:.3f}")
    print(f"avg new explanation compliance:      {avg_new_e:.3f}")
    print(f"avg old composite:                   {avg_old_c:.3f}")
    print(f"avg new composite:                   {avg_new_c:.3f}")
    print(f"runs with old compliance == 0.0:      {n_zero_old} / {n}")
    print(f"runs with new manoeuvre == 0.0:       {n_zero_new_m} / {n}")
    print(f"\nCode frequency across all old breakdowns:")
    for code, cnt in code_counter.most_common():
        cat = COMPLIANCE_CATEGORY.get(code, "n/a (hard-gate)")
        print(f"  {code:30s} {cnt:5d}  [{cat}]")

    # --- Per (mission, config) -- only COMPLETED runs (verdict PASS / PASS_WITH_CPA_VIOLATION) ---
    completed = [r for r in rows if r["completed"]]
    by_mission: dict[str, list[dict]] = {}
    by_mission_config: dict[tuple[str, str], list[dict]] = {}
    for r in completed:
        by_mission.setdefault(r["mission_id"], []).append(r)
        by_mission_config.setdefault((r["mission_id"], r["config"]), []).append(r)

    print(f"\n--- Per completed mission ({len(completed)}/{n} runs completed, "
          f"{len(by_mission)} distinct mission(s)) ---")
    print(f"{'mission':16s} {'n':>3s} {'old':>6s} {'new_M':>6s} {'new_E':>6s} "
          f"{'oldC':>6s} {'newC':>6s}")
    for mission_id in sorted(by_mission):
        rs = by_mission[mission_id]
        k = len(rs)
        print(f"{mission_id:16s} {k:3d} "
              f"{sum(r['old_score'] for r in rs) / k:6.3f} "
              f"{sum(r['new_manoeuvre'] for r in rs) / k:6.3f} "
              f"{sum(r['new_explanation'] for r in rs) / k:6.3f} "
              f"{sum(r['old_composite'] for r in rs) / k:6.3f} "
              f"{sum(r['new_composite'] for r in rs) / k:6.3f}")

    print(f"\n--- Per completed mission x model variation (config) ---")
    print(f"{'mission':16s} {'config':26s} {'n':>3s} {'old':>6s} {'new_M':>6s} "
          f"{'new_E':>6s} {'oldC':>6s} {'newC':>6s}")
    for mission_id, config in sorted(by_mission_config):
        rs = by_mission_config[(mission_id, config)]
        k = len(rs)
        print(f"{mission_id:16s} {config:26s} {k:3d} "
              f"{sum(r['old_score'] for r in rs) / k:6.3f} "
              f"{sum(r['new_manoeuvre'] for r in rs) / k:6.3f} "
              f"{sum(r['new_explanation'] for r in rs) / k:6.3f} "
              f"{sum(r['old_composite'] for r in rs) / k:6.3f} "
              f"{sum(r['new_composite'] for r in rs) / k:6.3f}")


if __name__ == "__main__":
    main()
