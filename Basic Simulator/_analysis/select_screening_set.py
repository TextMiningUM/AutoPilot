"""Pick a small "screening set" of missions from the EXISTING _llm_runs sweep data that
best discriminates between OOW-agent configs (bare_qwen..v6_pg_scenario) -- pure
read-only analysis, zero GPU, zero new runs. See OOW_Mission_Sim_Analysis_V1.md Sec.2 for
the failure patterns this is meant to screen for cheaply before promoting a winning
candidate (fine-tuned model / new RAG corpus / reranker) to the full mission set.

WHY VARIANCE IS THE CRITERION
A mission where every config passes (or every config fails) never discriminates between
techniques -- it still costs compute on every future sweep. A mission where configs
strongly DISAGREE in outcome is informative. So: rank missions by how much their configs'
results spread, not by how well/badly any single config did.

METHOD (per mission, over whichever configs are actually present):
  a. stdev of composite_score across configs
  b. stdev of compliance.score across configs
  c. verdict disagreement: number of DISTINCT evaluation.verdict strings seen across
     configs (a mission with mixed verdicts is a strong discriminator by definition)
Each signal is min-max normalized to [0,1] across all missions, then averaged into one
combined_score -- reported alongside the three raw signals, never hidden behind it.

TWO MANDATORY CORRECTIONS ON PURE VARIANCE-SELECTION
  a. Encounter-type coverage: the final 5 must include at least one mission of every
     encounter type present in the sweep (head_on / crossing_give_way /
     crossing_stand_on / overtaking / multi_target), derived from each mission
     DEFINITION's rule_refs/own_ship_role/targets -- never from the mission id string.
  b. One quiet "canary": UM01/UM02 (the only genuine no-risk missions) score near-zero
     variance by construction and would never survive ranking, but they're the only test
     for degeneracy/overreacting (the DTU-paper risk) -- always add the one with the
     lowest observed min_cpa_m (the "hardest" quiet mission), on top of, not instead of,
     the 5 variance-ranked picks.

INCOMPLETE DATA: the sweep may still be partial (fewer than 8 configs for a given
mission). Every mission's variance is still computed and shown in the full ranking table
(never silently dropped), but a mission with fewer than all 8 configs present is EXCLUDED
from automatic selection eligibility (both the variance top-N and the type-coverage
backfill) -- its signal is real but less trustworthy with fewer samples, so it is flagged,
not silently trusted at full weight either. The full table always shows exactly how many
configs each mission had.

Run: .venv\\Scripts\\python.exe "Basic Simulator/_analysis/select_screening_set.py"
Outputs (both written next to this script):
  - screening_set_ranking.md -- the full ranking table (also printed to console)
  - screening_set.json -- the final selected mission-id list + reasons + scores,
    directly consumable as a --missions list by run_llm_scenario.py/sweep_llm_params.py
"""
from __future__ import annotations
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent.parent  # Basic Simulator/
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
from app.llm_runs import parse_run_filename  # noqa: E402 -- needs sys.path set first

MISSIONS_DIR = APP_DIR / "Data" / "missions"
TAG = "units_v1"
CONFIGS_ALL = ["bare_qwen", "v0_base", "v1_rag", "v2_cot", "v3_rag_cot",
              "v4_pg", "v5_pg_incident", "v6_pg_scenario"]
N_CONFIGS_EXPECTED = len(CONFIGS_ALL)
QUIET_MISSIONS = {"UM01", "UM02"}  # the only genuine no-real-risk missions in this sweep
REQUIRED_TYPES = ["head_on", "crossing_give_way", "crossing_stand_on", "overtaking", "multi_target"]
MAX_SCREEN = 5  # + 1 canary = 6 maximum, per explicit instruction (not a second full sweep)

OUT_DIR = Path(__file__).resolve().parent
TABLE_MD = OUT_DIR / "screening_set_ranking.md"
SET_JSON = OUT_DIR / "screening_set.json"


def find_run_files() -> list[Path]:
    # Recursive: run logs live under whichever archive subfolder they were last moved
    # into (folder names/locations have changed more than once this project) -- rglob
    # + the tag filter finds them regardless of exactly where they currently sit.
    return sorted(MISSIONS_DIR.rglob(f"*__*__{TAG}.json"))


def load_mission_def(mission_id: str) -> dict:
    p = MISSIONS_DIR / f"{mission_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def classify_encounter_type(mission_def: dict) -> str:
    """Derived ONLY from targets/own_ship_role/rule_refs -- never from the mission id
    string. Multi-target missions are their own bucket regardless of the individual
    rules involved (matches the user's own "UM10 / de Imazu-clusters" framing) since a
    multi-contact decision is qualitatively a different, harder test than any single
    single-target encounter type."""
    targets = mission_def.get("targets", [])
    role = str(mission_def.get("own_ship_role", "")).lower()
    rules = mission_def.get("rule_refs", [])
    if not targets or role.startswith("none"):
        return "quiet"
    if len(targets) > 1:
        return "multi_target"
    if "Rule 13" in rules or "overtak" in role:
        return "overtaking"
    if "Rule 14" in rules or "mutual" in role:
        return "head_on"
    if "give" in role:
        return "crossing_give_way"
    if "stand" in role:
        return "crossing_stand_on"
    return "other"


def collect_mission_stats() -> dict[str, dict]:
    """{mission_id: {config: {composite, compliance, verdict, min_cpa_m, weights}}}. Config/
    weights are read from the filename via parse_run_filename() (understands both the current
    4-segment {mission}__{config}__{weights}__{tag}.json form and the older 2/3-segment forms
    the units_v1 archive used, which have no weights segment -- all implicitly "W0_base", the
    only checkpoint any run has ever used so far) rather than the JSON body, since older logs
    never had a "weights" field at all."""
    by_mission: dict[str, dict] = defaultdict(dict)
    for f in find_run_files():
        doc = json.loads(f.read_text(encoding="utf-8"))
        parsed = parse_run_filename(f)
        mission_id, config = parsed["mission_id"], parsed["config"]
        if not mission_id or not config:
            continue
        ev = doc.get("evaluation", {})
        by_mission[mission_id][config] = {
            "composite": ev.get("composite_score"),
            "compliance": ev.get("compliance", {}).get("score"),
            "verdict": ev.get("verdict"),
            "min_cpa_m": ev.get("safety", {}).get("min_cpa_m"),
            "weights": doc.get("weights") or parsed["weights"],
        }
    return by_mission


def normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def build_rows(by_mission: dict[str, dict]) -> list[dict]:
    rows = []
    for mission_id, cfgs in by_mission.items():
        composites = [v["composite"] for v in cfgs.values() if v["composite"] is not None]
        compliances = [v["compliance"] for v in cfgs.values() if v["compliance"] is not None]
        verdicts = [v["verdict"] for v in cfgs.values() if v["verdict"] is not None]
        min_cpas = [v["min_cpa_m"] for v in cfgs.values() if v["min_cpa_m"] is not None]
        n_present = len(cfgs)
        composite_std = statistics.pstdev(composites) if len(composites) > 1 else 0.0
        compliance_std = statistics.pstdev(compliances) if len(compliances) > 1 else 0.0
        verdict_set = sorted(set(verdicts))
        mission_def = load_mission_def(mission_id)
        rows.append({
            "mission_id": mission_id,
            "n_configs_present": n_present,
            "full_coverage": n_present >= N_CONFIGS_EXPECTED,
            "composite_std": round(composite_std, 4),
            "compliance_std": round(compliance_std, 4),
            "verdict_disagreement": len(verdict_set),
            "verdicts_seen": verdict_set,
            "mean_composite": round(statistics.mean(composites), 3) if composites else None,
            "min_cpa_m": min(min_cpas) if min_cpas else None,
            "encounter_type": classify_encounter_type(mission_def),
            "all_pass": len(verdict_set) == 1 and verdict_set and verdict_set[0] == "PASS",
            "all_fail": len(verdict_set) == 1 and verdict_set and verdict_set[0] != "PASS",
        })
    return rows


def score_rows(rows: list[dict]) -> None:
    """Adds combined_score IN PLACE (min-max normalize each of the 3 raw signals across
    ALL missions, then average)."""
    n_composite = normalize([r["composite_std"] for r in rows])
    n_compliance = normalize([r["compliance_std"] for r in rows])
    n_verdict = normalize([r["verdict_disagreement"] for r in rows])
    for r, a, b, c in zip(rows, n_composite, n_compliance, n_verdict):
        r["norm_composite_std"] = round(a, 3)
        r["norm_compliance_std"] = round(b, 3)
        r["norm_verdict_disagreement"] = round(c, 3)
        r["combined_score"] = round((a + b + c) / 3, 4)


def select_screening_set(rows: list[dict]) -> tuple[list[dict], dict]:
    """Returns (selected_rows, reasons) -- reasons keyed by mission_id, explaining why
    each one was picked (variance-top-N / type-coverage / canary)."""
    reasons: dict[str, str] = {}

    # Eligible for AUTOMATIC selection = full 8/8 config coverage only (low-coverage
    # missions still get scored/shown above, never silently trusted at full weight for
    # picking the set itself).
    eligible = [r for r in rows if r["full_coverage"] and r["encounter_type"] != "quiet"]
    eligible_sorted = sorted(eligible, key=lambda r: -r["combined_score"])

    selected = eligible_sorted[:MAX_SCREEN]
    for r in selected:
        reasons[r["mission_id"]] = "variance-top-N"

    # Type-coverage backfill.
    for req_type in REQUIRED_TYPES:
        if any(r["encounter_type"] == req_type for r in selected):
            continue
        candidates = [r for r in eligible
                     if r["encounter_type"] == req_type
                     and r["mission_id"] not in {s["mission_id"] for s in selected}]
        if not candidates:
            continue  # no mission of this type exists in the sweep at all -- can't backfill
        best = max(candidates, key=lambda r: r["combined_score"])
        type_counts = Counter(s["encounter_type"] for s in selected)
        removable = sorted(selected, key=lambda r: r["combined_score"])
        victim = next((r for r in removable if type_counts[r["encounter_type"]] > 1), removable[0])
        selected.remove(victim)
        reasons.pop(victim["mission_id"], None)
        selected.append(best)
        reasons[best["mission_id"]] = f"type-coverage ({req_type})"

    # Canary: the quiet mission (UM01/UM02) with the lowest observed min_cpa_m -- the
    # "hardest" quiet mission -- ALWAYS included, on top of the 5 above.
    quiet_rows = [r for r in rows if r["mission_id"] in QUIET_MISSIONS and r["min_cpa_m"] is not None]
    if quiet_rows:
        canary = min(quiet_rows, key=lambda r: r["min_cpa_m"])
        if canary["mission_id"] not in reasons:
            selected.append(canary)
            reasons[canary["mission_id"]] = "canary (quiet mission, lowest min_cpa_m)"

    selected_sorted = sorted(selected, key=lambda r: -r["combined_score"])
    return selected_sorted, reasons


def render_table(rows: list[dict]) -> str:
    rows_sorted = sorted(rows, key=lambda r: -r["combined_score"])
    header = ("| Rank | Mission | Type | Configs | Composite std | Compliance std | "
             "Verdict disagreement | Combined score | Mean composite | Verdicts seen | Note |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for i, r in enumerate(rows_sorted, 1):
        note_bits = []
        if not r["full_coverage"]:
            note_bits.append(f"LOW COVERAGE ({r['n_configs_present']}/{N_CONFIGS_EXPECTED} configs)")
        if r["all_pass"]:
            note_bits.append("ALL CONFIGS PASS")
        if r["all_fail"]:
            note_bits.append("ALL CONFIGS FAIL")
        note = "; ".join(note_bits)
        lines.append(
            f"| {i} | {r['mission_id']} | {r['encounter_type']} | {r['n_configs_present']}/{N_CONFIGS_EXPECTED} | "
            f"{r['composite_std']} | {r['compliance_std']} | {r['verdict_disagreement']} | "
            f"**{r['combined_score']}** | {r['mean_composite']} | {', '.join(r['verdicts_seen'])} | {note} |"
        )
    return "\n".join(lines)


def main() -> None:
    by_mission = collect_mission_stats()
    print(f"Found {len(by_mission)} missions with run data (tag={TAG})")
    rows = build_rows(by_mission)
    score_rows(rows)
    table_md = render_table(rows)

    print()
    print(table_md.replace("**", ""))  # plain console rendering, no markdown bold markers
    print()

    all_fail_rows = [r for r in rows if r["all_fail"]]
    all_pass_rows = [r for r in rows if r["all_pass"]]
    print("--- missions where EVERY config failed (no prompt variant fixes this -- training-data candidate) ---")
    for r in all_fail_rows:
        print(f"  {r['mission_id']:10s} type={r['encounter_type']:18s} verdict={r['verdicts_seen']}")
    print("--- missions where EVERY config passed (never discriminates -- safe to drop from future sweeps) ---")
    for r in all_pass_rows:
        print(f"  {r['mission_id']:10s} type={r['encounter_type']:18s} verdict={r['verdicts_seen']}")

    selected, reasons = select_screening_set(rows)
    print()
    print(f"--- proposed screening set ({len(selected)} missions) ---")
    for r in selected:
        print(f"  {r['mission_id']:10s} type={r['encounter_type']:18s} combined_score={r['combined_score']} "
              f"reason={reasons[r['mission_id']]}")

    TABLE_MD.write_text(
        "# Screening-set mission ranking (all missions in the current sweep)\n\n"
        f"Tag: `{TAG}`. {len(by_mission)} missions with run data at generation time.\n\n"
        + table_md + "\n\n"
        "## Missions where every available config failed\n\n"
        + ("\n".join(f"- `{r['mission_id']}` ({r['encounter_type']}): {r['verdicts_seen']}"
                    for r in all_fail_rows) or "(none)") + "\n\n"
        "## Missions where every available config passed\n\n"
        + ("\n".join(f"- `{r['mission_id']}` ({r['encounter_type']}): {r['verdicts_seen']}"
                    for r in all_pass_rows) or "(none)") + "\n",
        encoding="utf-8",
    )

    screening_payload = {
        "tag": TAG,
        "n_missions_in_sweep": len(by_mission),
        "n_configs_expected": N_CONFIGS_EXPECTED,
        "mission_ids": [r["mission_id"] for r in selected],
        "missions": [
            {
                "mission_id": r["mission_id"],
                "reason": reasons[r["mission_id"]],
                "encounter_type": r["encounter_type"],
                "n_configs_present": r["n_configs_present"],
                "combined_score": r["combined_score"],
                "composite_std": r["composite_std"],
                "compliance_std": r["compliance_std"],
                "verdict_disagreement": r["verdict_disagreement"],
                "min_cpa_m": r["min_cpa_m"],
            }
            for r in selected
        ],
    }
    SET_JSON.write_text(json.dumps(screening_payload, indent=2), encoding="utf-8")
    print(f"\nWrote {TABLE_MD.relative_to(APP_DIR)} and {SET_JSON.relative_to(APP_DIR)}")


if __name__ == "__main__":
    main()
