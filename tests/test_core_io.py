"""Unit tests for core/io.py — the shared JSONL/.env loading helpers.

Run with: python -m tests.test_core_io
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from core.io import load_env, load_jsonl, load_messages_jsonl, review_path, safe_write_jsonl


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


def test_review_path_redirects_into_a_review_subfolder() -> None:
    p = Path("/some/dir/oow_scenario_sft_direct.jsonl")
    assert review_path(p) == Path("/some/dir/_review/oow_scenario_sft_direct.jsonl")


def test_safe_write_jsonl_refuses_to_overwrite_an_existing_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prod.jsonl"
        p.write_text('{"a": 1}\n', encoding="utf-8")
        try:
            safe_write_jsonl([{"a": 2}], p)
            assert False, "expected FileExistsError"
        except FileExistsError:
            pass
        # Untouched -- the guard must fail BEFORE writing anything.
        assert p.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_safe_write_jsonl_overwrites_when_explicitly_told_to() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prod.jsonl"
        p.write_text('{"a": 1}\n', encoding="utf-8")
        safe_write_jsonl([{"a": 2}], p, overwrite=True)
        assert list(load_jsonl(p)) == [{"a": 2}]


def test_safe_write_jsonl_writes_freely_to_a_new_path() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "_review" / "new.jsonl"
        safe_write_jsonl([{"a": 1}], p)
        assert list(load_jsonl(p)) == [{"a": 1}]


if __name__ == "__main__":
    test_load_jsonl_skips_blank_and_malformed_lines()
    test_load_messages_jsonl_extracts_messages_key()
    test_load_env_sets_vars_without_overwriting_existing()
    test_load_env_missing_file_is_a_noop()
    print("OK — core/io.py helpers behave as expected.")
