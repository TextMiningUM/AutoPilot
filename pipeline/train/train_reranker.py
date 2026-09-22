"""OOW reranker prototype, step 2 — fine-tune a small cross-encoder on the pairs
mined by build_reranker_pairs.py.

Deliberately NOT modeled on train_sft.py/train_dpo.py's QLoRA machinery: this is a
~22M-parameter classification head over a small BERT-style encoder, not an 8B causal
LM, so (like build_rag.py's SentenceTransformer embedding step) it is safe and fast
to run LOCALLY on CPU -- no GPU/cloud required for this stage.

Reports a zero-Qwen "dev accuracy@1" proxy (does the top-scored candidate among each
dev query's candidates carry label=1?) before AND after fine-tuning, so the prototype's
effect is visible immediately. This is NOT a substitute for the real ablation (wiring
the reranker into kg_retrieve's hit list and measuring the composite RAGAS score with
Qwen3-8B) -- it's a cheap first signal for whether fine-tuning is worth pursuing that
far, per the "prototype first" direction.

USAGE
-----
    .venv\\Scripts\\python.exe -m pipeline.train.train_reranker
    .venv\\Scripts\\python.exe -m pipeline.train.train_reranker --epochs 5

OUTPUT
------
_models/OOW/oow_reranker/ -- a CrossEncoder save_pretrained() directory, plus
TRAIN_INFO.json recording the before/after dev accuracy@1.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict

from datasets import Dataset

from core import AgentPaths

paths = AgentPaths.from_env()
PAIRS_FILE = paths.cache_dir / "oow_reranker_pairs.jsonl"
OUT_DIR    = paths.domain_models_dir / "oow_reranker"

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_rows() -> list[dict]:
    return [json.loads(l) for l in PAIRS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def to_dataset(rows: list[dict], split: str) -> Dataset:
    sub = [r for r in rows if r["split"] == split]
    return Dataset.from_dict({
        "query": [r["query"] for r in sub],
        "text":  [r["text"] for r in sub],
        "label": [float(r["label"]) for r in sub],
    })


def accuracy_at_1(model, dev_rows: list[dict]) -> tuple[float, int]:
    """Group dev rows by query; a query counts as "correct" if the single
    highest-scoring candidate among its own (pos + neg) rows has label=1."""
    by_query: dict[str, list[dict]] = defaultdict(list)
    for r in dev_rows:
        by_query[r["query"]].append(r)
    correct, n = 0, 0
    for q, cands in by_query.items():
        if not any(c["label"] == 1 for c in cands) or not any(c["label"] == 0 for c in cands):
            continue  # need both a positive and a negative to be a meaningful ranking test
        scores = model.predict([(q, c["text"]) for c in cands])
        best = max(range(len(cands)), key=lambda i: scores[i])
        correct += int(cands[best]["label"] == 1)
        n += 1
    return (correct / n if n else 0.0), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args()

    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows   = [r for r in rows if r["split"] == "dev"]
    print(f"train={len(train_rows)} dev={len(dev_rows)}", flush=True)

    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    print(f"Loading base model {args.base_model} ...", flush=True)
    model = CrossEncoder(args.base_model, num_labels=1, model_kwargs={"torch_dtype": "float32"})

    base_acc, n_eval = accuracy_at_1(model, dev_rows)
    print(f"BEFORE fine-tuning: dev accuracy@1 = {base_acc:.3f}  (n={n_eval} queries)", flush=True)

    train_ds = to_dataset(rows, "train")
    dev_ds   = to_dataset(rows, "dev")
    loss = BinaryCrossEntropyLoss(model)

    train_args = CrossEncoderTrainingArguments(
        output_dir=str(OUT_DIR / "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
    )
    trainer = CrossEncoderTrainer(
        model=model, args=train_args,
        train_dataset=train_ds, eval_dataset=dev_ds, loss=loss,
    )
    trainer.train()

    tuned_acc, _ = accuracy_at_1(model, dev_rows)
    print(f"AFTER fine-tuning:  dev accuracy@1 = {tuned_acc:.3f}  (n={n_eval} queries)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    (OUT_DIR / "TRAIN_INFO.json").write_text(json.dumps({
        "base_model": args.base_model, "epochs": args.epochs,
        "n_train": len(train_rows), "n_dev": len(dev_rows), "n_eval_queries": n_eval,
        "dev_accuracy_at_1_before": base_acc, "dev_accuracy_at_1_after": tuned_acc,
    }, indent=2), encoding="utf-8")
    print(f"Saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
