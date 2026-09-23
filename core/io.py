"""Tiny I/O helpers shared by every build/train/eval script.

Previously copy-pasted verbatim into ~9 different scripts (build_sft.py,
build_rlhf.py, build_reflection.py, build_multihop.py, compress_distill.py,
extract_reasoning.py, extract_conversation_reasoning.py, eval_finetuned.py,
build_vhf_colreg_scenarios.py). One copy here now.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Iterator


def load_jsonl(path: Path) -> Iterator[dict]:
    """Yield one dict per non-empty line of a JSONL file. Skips malformed lines."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [load_jsonl] skipping malformed line in {path.name}: {e}", file=sys.stderr)
                continue


def load_jsonl_keyed(path: Path, key: str) -> dict[str, dict]:
    """Load an existing JSONL file into ``{str(row[key]): row}`` (``{}`` if the file
    doesn't exist), skipping rows missing `key`. Used by resume-safe generation/scoring
    loops: check ``str(item_id) in done`` to skip work already completed before a crash
    or a deliberate early interruption, instead of starting over from scratch."""
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    for row in load_jsonl(path):
        k = row.get(key)
        if k is not None:
            done[str(k)] = row
    return done


def load_jsonl_rows_capped(path: Path, max_rows: int | None, seed: int = 0) -> list[dict]:
    """Load a JSONL file into a list of dicts, deterministically subsampled down to at
    most `max_rows` (random.Random(seed).sample, so repeated loads of an unchanged file
    are stable) when the file has more rows than that. `max_rows=None` loads everything.
    Used to cap one data source's share of a final train-mix (e.g. the Leo MOOS-
    trajectory files, which have far more raw material than the other Track 2 sources)
    without touching how many rows the underlying generator actually wrote to disk.
    Returns [] if `path` doesn't exist."""
    if not path.exists():
        return []
    rows = list(load_jsonl(path))
    if max_rows is not None and len(rows) > max_rows:
        rows = random.Random(seed).sample(rows, max_rows)
    return rows


def load_env(path: Path) -> None:
    """Minimal .env loader: sets os.environ from KEY=VALUE lines, never overwriting existing vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_messages_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file of {"messages": [...]} rows into a plain list, ready
    for ``datasets.Dataset.from_list()``. Used by every training/distillation
    script that consumes chat-formatted SFT/reflection JSONL."""
    return [{"messages": r["messages"]} for r in load_jsonl(path)]


def review_path(path: Path) -> Path:
    """Redirects a production output path into a sibling `_review/` folder -- every
    dry-run and [LLM] "25-first" review-gate pass MUST write here, never to the real
    production path, so a review sample can never be mistaken for (or accidentally
    clobber) the real committed dataset."""
    return path.parent / "_review" / path.name


def safe_write_jsonl(rows: list[dict], path: Path, *, overwrite: bool = False) -> None:
    """Writes `rows` as JSONL to `path`, refusing to silently clobber an existing file
    unless `overwrite=True`. Raises FileExistsError instead of writing otherwise.

    Added after a real incident (2026-09-22): running build_oow_scenarios_leo.py
    --skip-llm as a quick smoke test overwrote 5 TRACKED production .jsonl files with
    near-empty content (0 rows, since --skip-llm never sets gold_answer) -- caught via
    `git status` and restored via `git checkout --` only because the previous version
    happened to be committed. This guard makes that class of mistake impossible instead
    of relying on noticing it afterward."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists -- refusing to overwrite without overwrite=True "
            f"(pass --overwrite on the CLI if this is really intended, or write to "
            f"review_path({path.name}) instead for a dry-run/review pass)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


