"""Shared --dry-run helpers for train_*.py / merge_adapter.py / compress_*.py.

Unlike a smoke test (small data, few steps, but STILL loads the real multi-GB
model onto the GPU), --dry-run skips model loading and training entirely and
writes a stub output that looks real enough for the next stage's existence
checks to pass -- so the whole SFT -> DPO -> Reflection -> merge -> compress
chain can be wired-checked locally in seconds, with zero GPU/model memory.

It does NOT replace the smoke test: dry-run only catches argument/data/path
bugs, never GPU/CUDA/numerical issues -- those still need a real (smoke or
full) run on the cloud GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_dry_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate data/args/paths and write a stub output WITHOUT loading or "
             "training/quantizing/pruning the real model. Zero GPU memory used.",
    )


def write_stub_output(out_dir: Path, tokenizer_source: str | Path | None = None) -> None:
    """Write a minimal directory that looks like a real adapter/model to every
    downstream `.exists()` check in this pipeline, without ever touching a GPU.

    Copies the real tokenizer files from `tokenizer_source` when given (cheap,
    useful for later real tokenization) and always writes an empty
    `adapter_model.safetensors` marker + a `DRY_RUN` manifest so nobody mistakes
    this for a real checkpoint.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_model.safetensors").write_bytes(b"")
    (out_dir / "adapter_config.json").write_text(
        json.dumps({"dry_run": True, "peft_type": "LORA"}), encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps({"dry_run": True}), encoding="utf-8",
    )
    (out_dir / "DRY_RUN.txt").write_text(
        "This directory was produced by --dry-run: no real model was loaded or "
        "trained. It exists only so downstream stages' file-existence checks "
        "pass during a local pipeline sanity check. Never evaluate or deploy it.\n",
        encoding="utf-8",
    )
    if tokenizer_source is not None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(tokenizer_source))
            tok.save_pretrained(str(out_dir))
        except Exception as e:
            print(f"[dry-run] Could not copy real tokenizer files ({e}); stub dir still valid for existence checks.")
    print(f"[dry-run] Wrote stub output to {out_dir} -- no model was loaded or trained.")
