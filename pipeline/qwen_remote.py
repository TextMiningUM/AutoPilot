"""Thin HTTP client for cloud/qwen_inference_server.py -- a GENERIC (domain-agnostic)
cloud inference server for base/LoRA/merged Qwen3-8B checkpoints. Lets any domain's
Streamlit app (VHF, OOW, Captain, Chief Engineer, ...) get real generated text from the
cloud GPU instead of loading/running Qwen3-8B locally (confirmed too slow for interactive
use on an 8 GB laptop GPU). Reached via an SSH tunnel, never called directly over the
public internet:
    ssh -N -L 8801:127.0.0.1:8801 -i <key> ubuntu@<pod-ip>

Pure stdlib HTTP client -- no torch/model loading happens in this module, safe to import
anywhere locally.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8801"


def generate_remote(
    domain: str, messages: list[dict], weights: str = "W0_base", max_new_tokens: int = 400,
    enable_thinking: bool = False, base_url: str = DEFAULT_URL, timeout_s: float = 90.0,
) -> str:
    """POST a chat-style `messages` list to the cloud server and return the generated text.

    `domain` selects which AgentPaths layout to resolve `weights` against (e.g. "VHF",
    "OOW", "Captain", "ChiefEngineer"); `weights` is the same string
    core.qwen_loader.load_qwen() already accepts ("W0_base", an adapter name, or
    "MERGED:<dir>[+<adapter>...]"). Raises ConnectionError (with an actionable hint) if
    the SSH tunnel/server isn't reachable, rather than hanging or returning a confusing
    low-level socket error.
    """
    payload = json.dumps({
        "domain": domain, "weights": weights, "messages": messages,
        "max_new_tokens": max_new_tokens, "enable_thinking": enable_thinking,
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/generate", data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())["text"]
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read()).get("error", str(e))
        raise RuntimeError(f"Qwen inference server returned an error: {detail}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not reach the Qwen inference server at {base_url} -- is the SSH tunnel "
            "open? (ssh -N -L 8801:127.0.0.1:8801 -i <key> ubuntu@<pod-ip>)"
        ) from e


def is_remote_server_up(base_url: str = DEFAULT_URL, timeout_s: float = 3.0) -> bool:
    """Quick /health check -- used by a UI toggle to decide whether the remote option is
    actually usable right now, instead of letting a slow request silently time out."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False
