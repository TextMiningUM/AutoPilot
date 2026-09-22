"""Recalibrate core/embedding.py's cosine-similarity thresholds for a NEW embedder
against the OLD one -- absolute cosine similarity is NOT portable between embedding
models (each has a different random-pair "noise floor"), so every threshold is
re-derived by matching its percentile-rank within the OLD model's own random-pair
similarity distribution, then reading off the NEW model's value at that SAME
percentile (same relative strictness, new absolute scale). Same methodology used for
the 2026-09-18 MiniLM->bge-base migration (see core/embedding.py's own history note).

Two distributions are calibrated separately (question-type vs procedure-label-type
thresholds are NOT interchangeable -- confirmed necessary during the bge-base
migration):
  - question<->question (VHF gold + OOW colreg_qa_500 questions): CONTAM_THRESH, DEDUP_THRESH
  - PG-node-label<->label (all 4 non-backup OOW PG files): CANON_THRESH, MATCH_THRESH, ANCHOR_THRESH

Run with: python -m pipeline.ingest.calibrate_embedder_thresholds
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core import AgentPaths

paths_vhf = AgentPaths.vhf()
paths_oow = AgentPaths.oow()

OLD_MODEL = "BAAI/bge-base-en-v1.5"
NEW_MODEL = "BAAI/bge-large-en-v1.5"

OLD_THRESHOLDS = {
    "CONTAM_THRESH": 0.90, "DEDUP_THRESH": 0.93,
    "CANON_THRESH": 0.89, "MATCH_THRESH": 0.84, "ANCHOR_THRESH": 0.65,
}

PG_FILES = ["oow_pg.json", "oow_pg_rule.json", "oow_pg_incident.json", "oow_pg_scenario.json"]

random.seed(42)


def load_questions(n: int = 400) -> list[str]:
    vhf = json.loads((paths_vhf.eval_dir / "vhf_gold_answers.json").read_text(encoding="utf-8"))
    oow = json.loads((paths_oow.eval_dir / "colreg_qa_500.json").read_text(encoding="utf-8"))
    qs = [r["question"] for r in vhf if r.get("question")] + [r["question"] for r in oow if r.get("question")]
    random.shuffle(qs)
    return qs[:n]


def load_pg_labels() -> list[str]:
    labels: set[str] = set()
    for name in PG_FILES:
        p = paths_oow.cache_dir / name
        if not p.exists():
            continue
        pg = json.loads(p.read_text(encoding="utf-8"))
        labels.update(n["label"] for n in pg["nodes"].values())
    return sorted(labels)


def random_pair_similarities(embs: np.ndarray, n_pairs: int = 20000) -> np.ndarray:
    n = embs.shape[0]
    rng = np.random.default_rng(42)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    return np.einsum("ij,ij->i", embs[i], embs[j])  # embeddings already L2-normalized


def percentile_of(value: float, distribution: np.ndarray) -> float:
    return float((distribution < value).mean() * 100)


def value_at_percentile(pct: float, distribution: np.ndarray) -> float:
    return float(np.percentile(distribution, pct))


def main() -> None:
    questions = load_questions()
    pg_labels = load_pg_labels()
    print(f"Calibration corpora: {len(questions)} questions, {len(pg_labels)} PG node labels")

    results = {}
    for model_name, tag in ((OLD_MODEL, "old"), (NEW_MODEL, "new")):
        print(f"\nLoading {model_name} ...")
        model = SentenceTransformer(model_name)
        q_embs = model.encode(questions, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        pg_embs = model.encode(pg_labels, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        results[tag] = {
            "question_dist": random_pair_similarities(q_embs),
            "pg_label_dist": random_pair_similarities(pg_embs),
            "dim": model.get_sentence_embedding_dimension(),
        }
        del model

    print(f"\n{OLD_MODEL} dim={results['old']['dim']}  {NEW_MODEL} dim={results['new']['dim']}")

    new_thresholds = {}
    for name, dist_key in (
        ("CONTAM_THRESH", "question_dist"), ("DEDUP_THRESH", "question_dist"),
        ("CANON_THRESH", "pg_label_dist"), ("MATCH_THRESH", "pg_label_dist"),
        ("ANCHOR_THRESH", "pg_label_dist"),
    ):
        old_val = OLD_THRESHOLDS[name]
        pct = percentile_of(old_val, results["old"][dist_key])
        new_val = value_at_percentile(pct, results["new"][dist_key])
        new_thresholds[name] = round(new_val, 3)
        print(f"{name:16s} old={old_val:.3f} (p{pct:5.1f} of {OLD_MODEL.split('/')[-1]}) "
              f"-> new={new_val:.3f} (same percentile of {NEW_MODEL.split('/')[-1]})")

    out = paths_oow.cache_dir / "embedder_recalibration_bge_large.json"
    out.write_text(json.dumps({
        "old_model": OLD_MODEL, "new_model": NEW_MODEL,
        "old_dim": results["old"]["dim"], "new_dim": results["new"]["dim"],
        "old_thresholds": OLD_THRESHOLDS, "new_thresholds": new_thresholds,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
