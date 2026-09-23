"""
================================================================================
build_measurement_reflection.py -- Fase C3 (RAG-rebuild-v2 plan)
================================================================================

Redesigned reflection data: the critique asks the SAME 3 diagnostic questions every
time (does a real collision risk exist? does the manoeuvre direction match the cited
rule? is the requested manoeuvre physically achievable?) and names EXACTLY which one(s)
the draft fails -- never the old "too vague, add parameters" critique.

draft    = the model's own REAL wrong decision from a measured archived units_v1
           checkpoint (Fase C0's Check A/B/C hits -- see app/measurement.py and
           pipeline/eval/measure_archived_checkpoints.py), same source as Fase C2(ii)'s
           anti-fabricated-risk DPO pairs (build_measurement_dpo.py) but reused here for
           reflection instead of preference pairs.
critique = walks through the 3 check questions, naming the SPECIFIC number(s)
           (CPA/safe-distance, cited rule, requested/limit degrees) that fail, using
           the checkpoint's own real measurement details -- never invented.
refined  = the deterministic correct answer with every fired issue corrected at once.

Only rows where >=1 of Check A/B/C actually fired are used -- a clean decision has
nothing to reflect on.

USAGE
-----
    python -m pipeline.track2.build_measurement_reflection
"""
from __future__ import annotations

import json

from core import AgentPaths
from pipeline.eval.measure_archived_checkpoints import MISSIONS_DIR
from pipeline.track2.build_measurement_dpo import build_chosen, build_rejected, FIXED_QUESTION

paths = AgentPaths.oow()
CACHE = paths.cache_dir
OUT_FILE = CACHE / "oow_measurement_reflection.jsonl"

Q1 = "does a real collision risk exist?"
Q2 = "does the manoeuvre direction match the cited rule?"
Q3 = "is the requested manoeuvre physically achievable?"


def build_critique(measurement: dict) -> str:
    checks = set(measurement["checks_fired"])
    details = measurement["details"]
    lines = []
    if "A_fabricated_risk" in checks:
        a = details["A"]
        lines.append(
            f"Check 1 ({Q1}): NO -- the closest contact's CPA is {a['min_cpa_m']:.0f} m, at or "
            f"above the {a['safe_distance_m']:.0f} m safe passing distance, yet the draft still "
            f"cites {a['cited_encounter_rule']}/{a['cited_conduct_rule']} -- a rule cannot be "
            "cited when no real risk of collision exists."
        )
    else:
        lines.append(f"Check 1 ({Q1}): consistent with the draft's own risk assessment.")
    if "B_wrong_direction" in checks:
        b = details["B"]
        lines.append(
            f"Check 2 ({Q2}): NO -- {b['cited_conduct_rule']} requires this give-way vessel to "
            f"turn to starboard, but the draft turned to port."
        )
    else:
        lines.append(f"Check 2 ({Q2}): consistent.")
    if "C_degrees_over_limit" in checks:
        c = details["C"]
        lines.append(
            f"Check 3 ({Q3}): NO -- {c['requested_degrees']:.0f} degrees exceeds the "
            f"{c['limit_degrees']:.0f} degree physical turn limit available for one decision step."
        )
    else:
        lines.append(f"Check 3 ({Q3}): consistent.")
    return " ".join(lines)


def build_refined(decision: dict, measurement: dict) -> dict:
    """Fixes every check that fired, all at once."""
    checks = set(measurement["checks_fired"])
    details = measurement["details"]
    if "A_fabricated_risk" in checks:
        a = details["A"]
        return build_chosen(a["min_cpa_m"], a["safe_distance_m"])
    rejected = build_rejected(decision)
    if "B_wrong_direction" in checks:
        rejected["action"] = "turn_right"
    if "C_degrees_over_limit" in checks and rejected.get("degrees") is not None:
        rejected["degrees"] = details["C"]["limit_degrees"]
    rejected["reasoning"] = (
        f"Correcting the issue(s) identified above: {rejected['action']}"
        + (f" ({rejected['degrees']:.0f} deg)" if rejected.get("degrees") is not None else "")
        + f", citing {rejected['encounter_rule']}/{rejected['conduct_rule']}."
    )
    return rejected


def mine_rows() -> list[dict]:
    files = sorted(MISSIONS_DIR.glob("*__*__units_v1.json"))
    rows = []
    seen_per_file: dict[str, set[str]] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        system_prompt = data.get("params", {}).get("system_prompt", "")
        seen = seen_per_file.setdefault(path.name, set())
        for cp in data.get("checkpoints", []):
            measurement = cp.get("measurement")
            if not measurement or not measurement["checks_fired"]:
                continue
            if ("A_fabricated_risk" in measurement["checks_fired"]
                and measurement["details"]["A"]["min_cpa_m"] is None):
                continue  # no contacts at all -- can't build a faithful refined answer
            decision = cp["decision"]
            draft = build_rejected(decision)
            dedup_key = (tuple(sorted(measurement["checks_fired"])), draft["reasoning"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            refined = build_refined(decision, measurement)
            critique = build_critique(measurement)
            user_msg = (cp.get("debug", {}).get("user_msg")
                       or f"Situation:\n{cp.get('situation_report', '')}\n\n{FIXED_QUESTION}")
            draft_text = f"I will {draft['action']}." if draft.get("action") else "I will hold course."
            refined_json = json.dumps(refined, ensure_ascii=False)
            rows.append({
                "source_file": path.name, "step": cp.get("step"),
                "checks_fired": measurement["checks_fired"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"Draft: {draft_text}\n\nCritique: {critique}\n\n"
                                                      f"Refined: {refined_json}"},
                ],
            })
    return rows


def main() -> None:
    rows = mine_rows()
    print(f"Mined {len(rows)} measurement-based reflection rows (deduped) from Check A/B/C hits")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_FILE} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
