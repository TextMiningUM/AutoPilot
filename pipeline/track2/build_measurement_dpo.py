"""
================================================================================
build_measurement_dpo.py -- Fase C2(ii) "anti-fabricated-risk" DPO pairs
================================================================================

Mines DPO pairs DETERMINISTICALLY (no LLM) from Fase C0's measurement results on the
152 archived units_v1 checkpoints under
"Basic Simulator/Data/missions/Missions data v2 20260922/" -- per explicit user
direction, the ONLY mission-log directory used for Fase C (older archives predate
fixes already made this session and are useless for DPO mining).

Every checkpoint where Check A ("A_fabricated_risk") fired is a REAL model mistake: it
cited a COLREG rule even though no contact's CPA was below this mission's safe passing
distance.
  rejected = the model's OWN actual (wrong) decision, taken verbatim from the checkpoint.
  chosen   = the deterministic correct answer (hold_course, encounter_rule/
             conduct_rule="none", a template reasoning citing the SAME real CPA/safe-
             distance numbers the checkpoint's own situation carried -- never invented).

Mandatory contamination filter (see copilot-instructions.md): embeds each pair's user
prompt and drops any row scoring >=CONTAM_THRESH cosine similarity against EITHER
held-out file (colreg_qa_500_normalised.json's questions AND oow_colreg_scenarios_v1.
json's situation_report text) -- never just one.

USAGE
-----
    python -m pipeline.track2.build_measurement_dpo
"""
from __future__ import annotations

import json

import numpy as np

from core import AgentPaths, CONTAM_THRESH, EMBEDDER_MODEL
from pipeline.eval.measure_archived_checkpoints import MISSIONS_DIR

paths = AgentPaths.oow()
CACHE = paths.cache_dir
OUT_PATH = CACHE / "oow_measurement_dpo_pairs.jsonl"

FIXED_QUESTION = "Recommend exactly ONE manoeuvre as the specified JSON object."


def build_chosen(min_cpa_m: float, safe_distance_m: float) -> dict:
    """The deterministic correct answer for a fabricated-risk mistake: no contact is a
    real risk, so hold course and cite no rule -- reasoning cites the SAME real numbers
    the checkpoint's own situation carried, never invented ones."""
    return {
        "action": "hold_course", "degrees": None,
        "encounter_rule": "none", "conduct_rule": "none",
        "reasoning": (f"The closest contact's CPA is {min_cpa_m:.0f} m, above this "
                     f"mission's {safe_distance_m:.0f} m safe passing distance, so no "
                     "contact poses a real risk of collision -- no COLREG rule applies; "
                     "hold course toward the mission goal."),
    }


def build_rejected(decision: dict) -> dict:
    """The model's own actual mistake, re-expressed in the current encounter_rule/
    conduct_rule schema -- old archived decisions only ever had "rule_applied" (read as
    both fields, same backward-compat interpretation as measurement.py's own fallback)."""
    encounter_rule = decision.get("encounter_rule") or decision.get("rule_applied") or "none"
    conduct_rule = decision.get("conduct_rule") or decision.get("rule_applied") or "none"
    return {
        "action": decision.get("action"), "degrees": decision.get("degrees"),
        "encounter_rule": encounter_rule, "conduct_rule": conduct_rule,
        "reasoning": decision.get("reasoning", ""),
    }


def mine_pairs() -> list[dict]:
    files = sorted(MISSIONS_DIR.glob("*__*__units_v1.json"))
    pairs = []
    seen_per_file: dict[str, set[str]] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        system_prompt = data.get("params", {}).get("system_prompt", "")
        seen = seen_per_file.setdefault(path.name, set())
        for cp in data.get("checkpoints", []):
            measurement = cp.get("measurement")
            if not measurement or "A_fabricated_risk" not in measurement["checks_fired"]:
                continue
            details = measurement["details"]["A"]
            if details["min_cpa_m"] is None:
                continue  # no contacts at all -- can't build a faithful chosen answer
            rejected = build_rejected(cp["decision"])
            # De-dup near-identical repeats within the SAME slow-moving mission (a static
            # geometry can repeat the exact same mistake for many consecutive checkpoints)
            # -- keep diversity ACROSS missions/configs, drop redundant reinforcement.
            dedup_key = rejected["reasoning"]
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            user_msg = (cp.get("debug", {}).get("user_msg")
                       or f"Situation:\n{cp.get('situation_report', '')}\n\n{FIXED_QUESTION}")
            chosen = build_chosen(details["min_cpa_m"], details["safe_distance_m"])
            pairs.append({
                "source_file": path.name, "step": cp.get("step"),
                "prompt": [{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_msg}],
                "chosen": [{"role": "assistant", "content": json.dumps(chosen, ensure_ascii=False)}],
                "rejected": [{"role": "assistant", "content": json.dumps(rejected, ensure_ascii=False)}],
            })
    return pairs


def load_gold_texts() -> list[str]:
    """BOTH held-out files, per copilot-instructions.md's mandatory contamination rule --
    never just Track 1's exam questions."""
    texts: list[str] = []
    qa_path = paths.eval_dir / "colreg_qa_500_normalised.json"
    if qa_path.exists():
        texts += [g["question"] for g in json.loads(qa_path.read_text(encoding="utf-8"))]
    scenarios_v1_path = paths.eval_dir / "oow_colreg_scenarios_v1.json"
    if scenarios_v1_path.exists():
        texts += [g["situation_report"] for g in json.loads(scenarios_v1_path.read_text(encoding="utf-8"))]
    return texts


def filter_contamination(pairs: list[dict], gold_texts: list[str]) -> list[dict]:
    if not pairs:
        return pairs
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER_MODEL)
    gold_embs = model.encode(gold_texts, normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False)
    texts = [p["prompt"][-1]["content"] for p in pairs]
    q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    kept, dropped = [], 0
    for p, emb in zip(pairs, q_embs):
        c_sim = float(np.max(gold_embs @ emb)) if gold_texts else 0.0
        if c_sim >= CONTAM_THRESH:
            dropped += 1
            continue
        kept.append(p)
    print(f"contamination filter: kept={len(kept)}  dropped={dropped} (thresh={CONTAM_THRESH})")
    return kept


def main() -> None:
    pairs = mine_pairs()
    print(f"Mined {len(pairs)} anti-fabricated-risk DPO pairs from Check A hits (deduped)")
    gold_texts = load_gold_texts()
    pairs = filter_contamination(pairs, gold_texts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_PATH} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
