"""
================================================================================
build_outcome_reflection.py -- Track 2 outcome-tied reflection triples
================================================================================

Same outcome-driven mining as build_outcome_dpo.py (collision / CPA-violation / goal-
not-reached run verdicts, PLUS the verdict-independent "rule13_mislabel" citation-bias
category -- see that module's docstring for the full rationale and the deterministic
ground-truth machinery both scripts share), reused here for reflection instead of
preference pairs:

  draft    = the model's own REAL wrong decision from a mined checkpoint.
  critique = states the same two check questions every time (does a real collision risk
             exist? does the action address it?) using the checkpoint's own real
             CPA/TCPA/goal-bearing numbers -- never invented.
  refined  = build_outcome_dpo.py's deterministic "chosen" answer for the same
             checkpoint (imported and reused unchanged, so the two scripts can never
             silently disagree about what the correct answer was).

USAGE
-----
    python -m pipeline.track2.build_outcome_reflection
"""
from __future__ import annotations

import json

from core import AgentPaths, CONTAM_THRESH, EMBEDDER_MODEL
from pipeline.track2.build_measurement_dpo import FIXED_QUESTION, build_rejected, load_gold_texts
from pipeline.track2.build_outcome_dpo import iter_run_paths, mined_events, CURRENT_PROMPT_VERSION

paths = AgentPaths.oow()
CACHE = paths.cache_dir
OUT_FILE = CACHE / "oow_outcome_reflection.jsonl"


def build_critique(event: dict) -> str:
    outcome, decisive, chosen = event["outcome"], event["decisive"], event["chosen"]
    if outcome == "goal_not_reached":
        return (
            "Check 1 (does a real collision risk exist?): NO -- no contact is below the safe "
            "passing distance. Check 2 (does the action address the situation?): NO -- the draft "
            f"does not correct heading toward the goal the way GOAL COURSE CHECK requires."
        )
    if outcome == "rule13_mislabel":
        true_kind = "head-on" if decisive["expected_encounter_rule"] == "Rule 14" else "crossing"
        return (
            f"Check 1 (does a real collision risk exist?): YES -- the closest contact's CPA is "
            f"{decisive['cpa_m']:.0f} m with TCPA {decisive['tcpa_s']:.0f} s. Check 2 (is the cited "
            f"rule correct?): NO -- the draft cites Rule 13 (overtaking), but this is actually a "
            f"{true_kind} encounter -- {decisive['expected_encounter_rule']}/"
            f"{decisive['expected_conduct_rule']} applies instead."
        )
    return (
        f"Check 1 (does a real collision risk exist?): YES -- the closest contact's CPA is "
        f"{decisive['cpa_m']:.0f} m with TCPA {decisive['tcpa_s']:.0f} s, below the safe distance "
        f"and within the acting horizon, citing {decisive['expected_encounter_rule']}/"
        f"{decisive['expected_conduct_rule']}. Check 2 (does the action address it?): NO -- the "
        f"draft does not take the required {chosen['action']} manoeuvre."
    )


def mine_rows() -> list[dict]:
    rows = []
    n_scanned = n_stale = 0
    for p in iter_run_paths():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "checkpoints" not in data or "evaluation" not in data:
            continue
        n_scanned += 1
        if data.get("params", {}).get("prompt_version") != CURRENT_PROMPT_VERSION:
            n_stale += 1
            continue
        system_prompt = data.get("params", {}).get("system_prompt", "")
        seen: set[tuple[str, str]] = set()
        for event in mined_events(data):
            cp = event["cp"]
            decision = cp.get("decision") or {}
            draft = build_rejected(decision)
            dedup_key = (event["outcome"], draft["reasoning"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            user_msg = (cp.get("debug", {}).get("user_msg")
                       or f"Situation:\n{cp.get('situation_report', '')}\n\n{FIXED_QUESTION}")
            draft_text = f"I will {draft['action']}." if draft.get("action") else "I will hold course."
            critique = build_critique(event)
            refined_json = json.dumps(event["chosen"], ensure_ascii=False)
            rows.append({
                "source_file": p.name, "step": cp.get("step"), "outcome": event["outcome"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"Draft: {draft_text}\n\nCritique: {critique}\n\n"
                                                      f"Refined: {refined_json}"},
                ],
            })
    print(f"Scanned {n_scanned} run logs ({n_stale} predate the current prompt, skipped)")
    return rows


def filter_contamination(rows: list[dict], gold_texts: list[str]) -> list[dict]:
    """Same mandatory contamination rule as build_measurement_dpo.py's own filter
    (copilot-instructions.md), adapted for this file's {"messages": [...]} shape --
    the user turn is always messages[1]."""
    if not rows or not gold_texts:
        return rows
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER_MODEL)
    gold_embs = model.encode(gold_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    texts = [r["messages"][1]["content"] for r in rows]
    q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    kept, dropped = [], 0
    for r, emb in zip(rows, q_embs):
        if float(np.max(gold_embs @ emb)) >= CONTAM_THRESH:
            dropped += 1
            continue
        kept.append(r)
    print(f"contamination filter: kept={len(kept)}  dropped={dropped} (thresh={CONTAM_THRESH})")
    return kept


def main() -> None:
    rows = mine_rows()
    by_outcome: dict[str, int] = {}
    for r in rows:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
    print(f"Mined {len(rows)} outcome-tied reflection rows (deduped): {by_outcome}")
    rows = filter_contamination(rows, load_gold_texts())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_FILE} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
