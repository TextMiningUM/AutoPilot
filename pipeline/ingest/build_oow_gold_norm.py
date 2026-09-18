"""colreg_qa_500.json -> colreg_qa_500_normalised.json (OOW notebook § 1, script form).

Standalone version of the notebook § 1 cell, so cloud/run_all_oow.sh (and any
other headless caller) can produce the normalised gold file every downstream
script expects ({question, gold_answer, expected_points}) without needing the
notebook. Idempotent -- safe to rerun, always rewrites the output in full.

Run with: python -m pipeline.ingest.build_oow_gold_norm
"""
from __future__ import annotations
import json

from core import AgentPaths

paths = AgentPaths.oow()
GOLD_FILE = paths.eval_file("colreg_qa_500.json")
NORM_FILE = paths.eval_dir / "colreg_qa_500_normalised.json"


def main() -> None:
    raw_gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(raw_gold)} COLREG Q&A records from {GOLD_FILE.name}")

    normalised = [
        {
            "id":              r["id"],
            "section_id":      r.get("rule_ref", r.get("category", "")),
            "section_title":   r.get("category", ""),
            "type":            r.get("question_type", ""),
            "question":        r["question"],
            "gold_answer":     r.get("answer", ""),
            "expected_points": [r["explanation"]] if r.get("explanation") else [],
        }
        for r in raw_gold
    ]
    NORM_FILE.write_text(json.dumps(normalised, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote normalised copy: {NORM_FILE}  ({len(normalised)} rows)")


if __name__ == "__main__":
    main()
