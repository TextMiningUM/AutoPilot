"""
================================================================================
analyze_gaps.py — data-coverage gap analysis for any eval-metrics JSONL file
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Runs *after* an eval script (eval_finetuned.py, eval_colreg_scenarios.py, or
any future eval script with the same output shape) and turns the per-question
metrics into an actionable "where should I add more training data?" report.

It groups scored rows by a caller-supplied field (e.g. section_id, category),
ranks the groups by mean Composite score, and for the worst groups prints:
  - which single metric is most often the lowest one (the dominant failure
    mode for that group — e.g. "Cover" low means the model states plausible
    but non-specific things; "ChannelProc" low means it skips the channel
    discipline, etc.)
  - the worst-scoring individual rows (question / gold / model answer /
    per-metric scores) so a human or an LLM can read the actual failure and
    decide whether it is a missing-content gap (no source document teaches
    this at all), an under-represented style gap (the content exists but
    training examples mostly explain it rather than demonstrate it), or a
    training/behavioural issue unrelated to data coverage (e.g. wrong
    escalation level, format violations) that more data will not fix.

WHY THIS IS A SEPARATE, REPEATABLE PIPELINE STAGE
--------------------------------------------------
Once a base model's raw capability is decent, the ceiling on how much a
fine-tune (SFT/DPO/reflection) can still improve a given topic is set by
whether the underlying knowledge or demonstration exists in the training
data at all — no amount of extra training steps, learning-rate tuning, or
epochs can teach the model something it was never shown. In other words,
DATA COVERAGE eventually becomes the bottleneck on model quality, not the
training procedure itself. Running this analysis on the *base* model's eval
results, before any fine-tuning starts, is the cheapest possible point in
the whole pipeline to catch this: it costs one eval pass (already done for
the baseline) and zero additional GPU-hours, versus discovering the same
gap after a multi-hour training run only shows disappointing scores on
exactly these topics.

This script is intentionally generic (no VHF/COLREG-specific field names
hardcoded) so it can be reused on any eval output that has the shape:
    {"<group_field>": ..., "question": ..., "gold_answer": ...,
     "answer": ..., "metrics": {"Composite": <float>, <other metrics>: ...}}
Point it at a Track 1 file (grouped by section_id), a Track 2 file (grouped
by category), or any other domain's eval file with the same shape.

USAGE
-----
    python -m pipeline.eval.analyze_gaps --eval-file Data/VHF/VHF_Agents_Training/eval_qwen_base.jsonl --group-by section_id
    python -m pipeline.eval.analyze_gaps --eval-file Data/VHF/VHF_Agents_Training/eval_qwen_base_colreg.jsonl --group-by category
    python -m pipeline.eval.analyze_gaps --eval-file ... --top-n-groups 5 --samples-per-group 2

This is meant to be run interactively, as many times as useful (once before
any fine-tuning, then again after adding more source data, then again after
the next eval) — there is no fixed number of iterations; the user decides
when the returns from adding more data have diminished enough to move on to
training.

OUTPUT
------
Prints a ranked table + per-group failure-mode breakdown + worst-row samples
to stdout, and writes the same structured data to:
    <eval_file_dir>/gap_analysis_<eval_file_stem>.json
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _is_missing(v) -> bool:
    """True for None or NaN — used to skip metrics that weren't computed for a row."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def load_eval_rows(path: Path) -> list[dict]:
    """Load one eval JSONL file (list of {question, gold_answer, answer, metrics, ...})."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_rows(rows: list[dict], group_by: str) -> dict[str, list[dict]]:
    """Group eval rows by the given field, tolerating a missing field (bucketed as '?')."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(group_by, "?"))].append(r)
    return groups


def rank_groups(groups: dict[str, list[dict]]) -> list[tuple[str, float, int]]:
    """Return (group_key, mean_composite, n) sorted worst-first."""
    stats = []
    for key, rs in groups.items():
        comp = [r["metrics"]["Composite"] for r in rs
                if not _is_missing(r.get("metrics", {}).get("Composite"))]
        if not comp:
            continue
        stats.append((key, sum(comp) / len(comp), len(comp)))
    stats.sort(key=lambda x: x[1])
    return stats


def dominant_failure_metrics(rows: list[dict], top_k: int = 20) -> dict[str, int]:
    """Across the top_k worst rows in this group, count which single metric
    (excluding Composite itself) was the lowest for each row. The metric that
    shows up most often is the group's dominant failure mode."""
    sorted_rows = sorted(rows, key=lambda r: r.get("metrics", {}).get("Composite", 1.0))
    counts: dict[str, int] = defaultdict(int)
    for r in sorted_rows[:top_k]:
        m = r.get("metrics", {})
        candidates = {k: v for k, v in m.items() if k != "Composite" and not _is_missing(v)}
        if not candidates:
            continue
        worst_metric = min(candidates, key=candidates.get)
        counts[worst_metric] += 1
    return dict(counts)


def worst_samples(rows: list[dict], n: int) -> list[dict]:
    """The n lowest-Composite rows in this group, trimmed to the fields useful for diagnosis."""
    sorted_rows = sorted(rows, key=lambda r: r.get("metrics", {}).get("Composite", 1.0))
    out = []
    for r in sorted_rows[:n]:
        out.append({
            "question":    r.get("question", "")[:300],
            "gold_answer": r.get("gold_answer", "")[:300],
            "answer":      r.get("answer", "")[:300],
            "metrics":     r.get("metrics", {}),
        })
    return out


def build_report(rows: list[dict], group_by: str, top_n_groups: int, samples_per_group: int) -> dict:
    """Build the full gap-analysis report as a plain dict (JSON-serialisable)."""
    groups = group_rows(rows, group_by)
    ranked = rank_groups(groups)

    report_groups = []
    for key, mean_comp, n in ranked[:top_n_groups]:
        rs = groups[key]
        report_groups.append({
            "group":                 key,
            "mean_composite":        round(mean_comp, 3),
            "n":                     n,
            "dominant_failure_metrics": dominant_failure_metrics(rs),
            "worst_samples":         worst_samples(rs, samples_per_group),
        })

    return {
        "group_by":        group_by,
        "n_groups_total":   len(ranked),
        "n_rows_total":     len(rows),
        "worst_groups":     report_groups,
        "all_group_scores": [{"group": k, "mean_composite": round(v, 3), "n": n} for k, v, n in ranked],
    }


def print_report(report: dict) -> None:
    """Human-readable rendering of build_report()'s output."""
    print("=" * 100)
    print(f"DATA-COVERAGE GAP ANALYSIS  —  grouped by {report['group_by']!r}  "
          f"({report['n_rows_total']} rows across {report['n_groups_total']} groups)")
    print("=" * 100)

    print(f"\n{'Group':<45} {'n':>4} {'MeanComposite':>14}")
    for g in report["all_group_scores"]:
        print(f"{g['group'][:44]:<45} {g['n']:>4} {g['mean_composite']:>14.3f}")

    print("\n" + "=" * 100)
    print(f"WORST {len(report['worst_groups'])} GROUPS — likely data-coverage gaps, worst first")
    print("=" * 100)
    for g in report["worst_groups"]:
        print(f"\n--- {g['group']}  (mean Composite={g['mean_composite']}, n={g['n']}) ---")
        print(f"  Dominant failure metric(s): {g['dominant_failure_metrics']}")
        for s in g["worst_samples"]:
            print(f"\n  Q: {s['question']}")
            print(f"  Gold: {s['gold_answer']}")
            print(f"  A: {s['answer']}")
            print(f"  metrics: {s['metrics']}")

    print("\n" + "=" * 100)
    print("NEXT STEP: for each worst group above, decide whether the failure is —")
    print("  (a) a missing-content gap  (no source document teaches this at all)")
    print("  (b) an under-represented demonstration-style gap  (content is explained")
    print("      but training examples rarely show the exact expected wording/format)")
    print("  (c) a training/behavioural issue unrelated to data coverage  (e.g. wrong")
    print("      escalation level, format violations) that more source data will not fix")
    print("Re-run this script after adding source data + rebuilding the training files")
    print("as many times as useful — there is no fixed number of iterations.")
    print("=" * 100)


def main() -> None:
    """CLI entry point: load an eval JSONL file, rank groups by mean Composite, and report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", type=str, required=True,
                    help="path to an eval_*.jsonl file (eval_finetuned.py or eval_colreg_scenarios.py output)")
    ap.add_argument("--group-by", type=str, default="section_id",
                    help="field name to group rows by (e.g. section_id for Track 1, category for Track 2)")
    ap.add_argument("--top-n-groups", type=int, default=8,
                    help="how many of the worst-scoring groups to show in detail")
    ap.add_argument("--samples-per-group", type=int, default=3,
                    help="how many worst individual rows to show per detailed group")
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON path (default: <eval_file_dir>/gap_analysis_<eval_file_stem>.json)")
    args = ap.parse_args()

    eval_path = Path(args.eval_file)
    rows = load_eval_rows(eval_path)
    print(f"Loaded {len(rows)} rows from {eval_path}")

    report = build_report(rows, args.group_by, args.top_n_groups, args.samples_per_group)
    print_report(report)

    out_path = Path(args.out) if args.out else eval_path.parent / f"gap_analysis_{eval_path.stem}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
