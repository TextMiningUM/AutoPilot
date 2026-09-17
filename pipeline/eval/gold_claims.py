"""Schema + structural validator for ``gold_claims`` (RAGAS migration, Phase 1).

A ``gold_claims`` entry decomposes one gold record's answer into atomic,
independently verifiable claims so that claim-level metrics (AnswerCorrectness
F1, ContextRecall, role-aware NumericF1, LitHit) can be computed against it.

Claim object schema::

    {"claim": "<one atomic statement>",
     "type":  "fact" | "procedure" | "number" | "literal" | "direction",
     "value": "<required for number/literal/direction>",
     "role":  "<required for number/direction, from the vocab below>"}

This module is the single source of truth for the claim vocabulary. It is
imported by enrich_gold_claims.py (to validate model output before accepting
it) and later by the ragas_metrics module (to consume the claims).

CLI check of an enriched gold file::

    python -m pipeline.eval.gold_claims --check Data/VHF/VHF_Eval/vhf_gold_answers_claims.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CLAIM_TYPES = {"fact", "procedure", "number", "literal", "direction"}

NUMBER_ROLES = {
    "vhf_channel", "colreg_rule", "distance_nm", "speed_kn", "bearing_deg",
    "course_deg", "power_watt", "repeat_count", "time_interval",
    "frequency_mhz", "gross_tonnage", "sea_area", "mmsi", "other_number",
}

DIRECTION_ROLES = {"turn_direction", "pass_side", "light_colour"}

QA_FLAG_PREFIX = "QA_FLAG:"

# The enrichment prompt asks for 3-10 claims, but genuinely rich gold records
# (channel tables, multi-step procedures) legitimately decompose into 15-22
# atoms — capping those would force lossy merging, so the ceiling is generous.
MIN_CLAIMS, MAX_CLAIMS = 1, 24


def validate_claims(claims: object) -> list[str]:
    """Return a list of structural problems ([] = valid) for one gold_claims array."""
    errors: list[str] = []
    if not isinstance(claims, list):
        return [f"gold_claims is {type(claims).__name__}, expected list"]
    if not (MIN_CLAIMS <= len(claims) <= MAX_CLAIMS):
        errors.append(f"{len(claims)} claims, expected {MIN_CLAIMS}-{MAX_CLAIMS}")
    for i, c in enumerate(claims):
        where = f"claim[{i}]"
        if not isinstance(c, dict):
            errors.append(f"{where}: not an object")
            continue
        text = c.get("claim")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{where}: missing/empty 'claim' text")
            continue
        if text.strip().startswith(QA_FLAG_PREFIX):
            continue  # QA flags are free-form review notes, only need text
        ctype = c.get("type")
        if ctype not in CLAIM_TYPES:
            errors.append(f"{where}: type {ctype!r} not in {sorted(CLAIM_TYPES)}")
            continue
        if ctype == "number":
            if not str(c.get("value", "")).strip():
                errors.append(f"{where}: number claim without 'value'")
            if c.get("role") not in NUMBER_ROLES:
                errors.append(f"{where}: number role {c.get('role')!r} invalid")
        elif ctype == "direction":
            if not str(c.get("value", "")).strip():
                errors.append(f"{where}: direction claim without 'value'")
            if c.get("role") not in DIRECTION_ROLES:
                errors.append(f"{where}: direction role {c.get('role')!r} invalid")
        elif ctype == "literal":
            if not str(c.get("value", "")).strip():
                errors.append(f"{where}: literal claim without 'value'")
    return errors


def check_file(path: Path) -> int:
    """Validate every record of an enriched gold file; print a report, return #bad records."""
    records = json.loads(path.read_text(encoding="utf-8"))
    n_bad = 0
    type_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    qa_flags: list[tuple[str, str]] = []
    claim_lens: list[int] = []

    for rec in records:
        rid = rec.get("id", "<no id>")
        claims = rec.get("gold_claims")
        if claims is None:
            print(f"  [MISSING] {rid}: no gold_claims field")
            n_bad += 1
            continue
        errs = validate_claims(claims)
        if errs:
            n_bad += 1
            for e in errs:
                print(f"  [INVALID] {rid}: {e}")
        claim_lens.append(len(claims))
        for c in claims:
            if not isinstance(c, dict):
                continue
            text = str(c.get("claim", ""))
            if text.strip().startswith(QA_FLAG_PREFIX):
                qa_flags.append((str(rid), text.strip()))
            else:
                type_counts[str(c.get("type"))] += 1
                if c.get("role"):
                    role_counts[str(c.get("role"))] += 1

    print(f"\n{path.name}: {len(records)} records, {n_bad} with problems")
    if claim_lens:
        print(f"  claims/record: min={min(claim_lens)} "
              f"mean={sum(claim_lens)/len(claim_lens):.1f} max={max(claim_lens)}")
    print(f"  claim types: {dict(type_counts.most_common())}")
    print(f"  roles:       {dict(role_counts.most_common())}")
    if qa_flags:
        print(f"  QA flags ({len(qa_flags)}) — gold-file inconsistencies to review:")
        for rid, flag in qa_flags:
            print(f"    {rid}: {flag}")
    return n_bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=Path, required=True, nargs="+",
                    help="Enriched gold file(s) to validate")
    args = ap.parse_args()
    total_bad = sum(check_file(p) for p in args.check)
    raise SystemExit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
