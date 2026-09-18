"""Tiny I/O helpers shared by every build/train/eval script.

Previously copy-pasted verbatim into ~9 different scripts (build_sft.py,
build_rlhf.py, build_reflection.py, build_multihop.py, compress_distill.py,
extract_reasoning.py, extract_conversation_reasoning.py, eval_finetuned.py,
build_vhf_colreg_scenarios.py). One copy here now.
"""
from __future__ import annotations

import json
import os
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

