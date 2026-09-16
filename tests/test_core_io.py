"""Unit tests for core/io.py — the shared JSONL/.env loading helpers.

Run with: python -m tests.test_core_io
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from core.io import load_env, load_jsonl, load_messages_jsonl


def test_load_jsonl_skips_blank_and_malformed_lines() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rows.jsonl"
        p.write_text(
            '{"a": 1}\n'
            "\n"
            "not json at all\n"
            '{"a": 2}\n',
            encoding="utf-8",
        )
        rows = list(load_jsonl(p))
        assert rows == [{"a": 1}, {"a": 2}]


def test_load_messages_jsonl_extracts_messages_key() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sft.jsonl"
        p.write_text(
            '{"messages": [{"role": "user", "content": "hi"}], "extra": "ignored"}\n',
            encoding="utf-8",
        )
        rows = load_messages_jsonl(p)
        assert rows == [{"messages": [{"role": "user", "content": "hi"}]}]


def test_load_env_sets_vars_without_overwriting_existing() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / ".env"
        p.write_text(
            "# a comment\n"
            "\n"
            'NEW_TEST_VAR="hello"\n'
            "ALREADY_SET_VAR=from_file\n",
            encoding="utf-8",
        )
        os.environ.pop("NEW_TEST_VAR", None)
        os.environ["ALREADY_SET_VAR"] = "from_process"
        try:
            load_env(p)
            assert os.environ["NEW_TEST_VAR"] == "hello"
            # setdefault semantics: existing process env wins over the .env file.
            assert os.environ["ALREADY_SET_VAR"] == "from_process"
        finally:
            os.environ.pop("NEW_TEST_VAR", None)
            os.environ.pop("ALREADY_SET_VAR", None)


def test_load_env_missing_file_is_a_noop() -> None:
    load_env(Path("this/file/does/not/exist.env"))  # must not raise


if __name__ == "__main__":
    test_load_jsonl_skips_blank_and_malformed_lines()
    test_load_messages_jsonl_extracts_messages_key()
    test_load_env_sets_vars_without_overwriting_existing()
    test_load_env_missing_file_is_a_noop()
    print("OK — core/io.py helpers behave as expected.")
