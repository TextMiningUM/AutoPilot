"""Thin HTTP client for the Chief Engineer inference server (cloud/chief_engineer_server.py)
running on the cloud GPU pod, reached via an SSH tunnel -- lets the LOCAL Streamlit
dashboard get real generated answers from the fine-tuned 8B model without loading/running
it on the laptop's 8GB GPU (confirmed too slow there). Safe to import/run locally: pure
stdlib HTTP client, no torch/model loading in this module.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8800"


def ask_chief_engineer_remote(
    question: str, max_new_tokens: int = 512, base_url: str = DEFAULT_URL, timeout_s: float = 90.0,
) -> str:
    """POST a question to the cloud inference server and return its generated answer.

    Raises ConnectionError (with an actionable hint) if the SSH tunnel/server isn't
    reachable, rather than hanging or returning a confusing low-level socket error.
    """
    payload = json.dumps({"question": question, "max_new_tokens": max_new_tokens}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/ask", data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read())
            return body["answer"]
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read()).get("error", str(e))
        raise RuntimeError(f"Chief Engineer inference server returned an error: {detail}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not reach the Chief Engineer inference server at {base_url} -- is the SSH "
            "tunnel open? (ssh -N -L 8800:127.0.0.1:8800 -i <key> ubuntu@<pod-ip>)"
        ) from e


def is_remote_server_up(base_url: str = DEFAULT_URL, timeout_s: float = 3.0) -> bool:
    """Quick /health check -- used by the UI to decide whether to show the remote-model
    option at all, instead of letting a slow question silently time out."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False
