"""Quality-review STOP-1-blocking-bug fix (2026-09-23): permanent, full-dataset
acceptance tests for the five hard invariants the early-band labeler fix must satisfy,
across BOTH Track-2 sources (Leo's 7928 real MOOS-trajectory frames and the synthetic
generator's full population):

  I1  CPA < safe_distance for some contact (band early/acute)  => encounter_rule != "none"
  I2  CPA < safe_distance for some contact (band early/acute)  => action != "speed_up"
  I3  band acute, own give-way  => action in {turn_left, turn_right, slow_down, stop}
  I4  band early, own give-way  => action in {turn_left, turn_right, slow_down} (never "stop")
  I5  encounter_rule == "none"  <=>  conduct_rule == "none" (the ONE documented exception:
      a real-risk stationary/non-vessel object, encounter_rule="none" + conduct_rule="Rule 8")

Before this fix, measured on Leo's data: 229/276 (83%) of genuine early-band frames
violated I1 (mislabeled encounter_rule='none'), 40 of those also violated I2
(mislabeled 'speed_up'). See risk_band()'s own docstring in pipeline/oow_agent_spec.py.
"""
import json

from pipeline.oow_agent_spec import risk_band
from pipeline.track2.build_oow_scenarios import (
    N_EVAL_PER_CATEGORY_DEFAULT, N_TRAIN_PER_CATEGORY_DEFAULT, generate_population,
    limits_for_scenario_record, to_unified_action,
)
from pipeline.track2.build_oow_scenarios_leo import LEO_FILE, leo_choose_action, limits_for_leo_record


def _check_i1_i2_i5(encounter_rule: str, conduct_rule: str, action: str, has_encounter: bool,
                    violations: dict, row_id: str) -> None:
    both_none = encounter_rule == "none" and conduct_rule == "none"
    stationary_exception = encounter_rule == "none" and conduct_rule == "Rule 8"
    if has_encounter and both_none:
        violations["I1"].append(row_id)
    if has_encounter and action == "speed_up":
        violations["I2"].append(row_id)
    if (encounter_rule == "none") != (conduct_rule == "none") and not stationary_exception:
        violations["I5"].append(row_id)


def test_full_leo_dataset_satisfies_i1_i2_i3_i4_i5() -> None:
    all_recs = [json.loads(l) for l in LEO_FILE.read_text(encoding="utf-8").splitlines()]
    violations: dict[str, list] = {"I1": [], "I2": [], "I3": [], "I4": [], "I5": []}
    bucket_counts: dict[str, int] = {}
    for r in all_recs:
        limits = limits_for_leo_record(r)
        d = leo_choose_action(r["state"], limits)
        bucket_counts[d["bucket"]] = bucket_counts.get(d["bucket"], 0) + 1
        if d["bucket"] in ("paused", "excluded"):
            continue  # no action was taken at all -- nothing to check
        contacts = r["state"]["contacts"]
        bands = [risk_band(c.get("cpa_distance_m"), c.get("tcpa_s"),
                           limits["safe_distance_m"], limits["risk_horizon_s"]) for c in contacts]
        has_encounter = any(b in ("acute", "early") for b in bands)
        _check_i1_i2_i5(d["encounter_rule"], d["conduct_rule"], d["action"], has_encounter,
                        violations, r["id"])
        if d["bucket"] in ("alter_course", "stop"):
            if d["action"] not in ("turn_left", "turn_right", "slow_down", "stop"):
                violations["I3"].append((r["id"], d["action"]))
        if d["bucket"] == "early_give_way":
            if d["action"] not in ("turn_left", "turn_right", "slow_down"):
                violations["I4"].append((r["id"], d["action"]))
    for code, bad in violations.items():
        assert bad == [], f"{code} violated in {len(bad)} Leo frames: {bad[:5]}"
    # Sanity: the early-band buckets exist and carry real volume (never zero -- a zero
    # count here would mean the widened band filter silently isn't firing at all).
    assert bucket_counts.get("early_give_way", 0) > 0
    assert bucket_counts.get("early_stand_on", 0) > 0


def test_full_synthetic_dataset_satisfies_i1_i2_i5() -> None:
    """Synthetic generator: I3/I4 bucket names aren't surfaced by to_unified_action()
    (no bucket field), so only the role-agnostic I1/I2/I5 invariants are checked here --
    matches classify_rules()'s own encounter_rule/conduct_rule contract, which is the
    same shared table Leo's labeler uses."""
    pop = generate_population(N_EVAL_PER_CATEGORY_DEFAULT + N_TRAIN_PER_CATEGORY_DEFAULT, seed=0)
    for i, rec in enumerate(pop):
        rec["_id"] = f"t{i:05d}"  # limits_for_scenario_record() needs a stable per-row id
    violations: dict[str, list] = {"I1": [], "I2": [], "I5": []}
    for rec in pop:
        limits = limits_for_scenario_record(rec)
        unified = to_unified_action(rec, limits)
        bands = [risk_band(t["_cpa_m"], t["_tcpa_min"] * 60.0,
                           limits["safe_distance_m"], limits["risk_horizon_s"])
                for t in rec["targets"]]
        has_encounter = any(b in ("acute", "early") for b in bands)
        _check_i1_i2_i5(unified["encounter_rule"], unified["conduct_rule"], unified["action"],
                        has_encounter, violations, rec.get("category", "?"))
    for code, bad in violations.items():
        assert bad == [], f"{code} violated in {len(bad)} synthetic rows: {bad[:5]}"
