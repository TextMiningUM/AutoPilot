"""Generic (domain-agnostic) cloud inference server for base/LoRA/merged Qwen3-8B
checkpoints -- reuses core.qwen_loader.load_qwen() UNCHANGED, so ANY domain (VHF, OOW,
Captain, ChiefEngineer, or any future one) is served by this ONE process without touching
this file again, per the project's "reuse before rebuild"/"never fork a shared script"
convention. Models are loaded lazily on first request per (domain, weights) key and kept
cached in memory for subsequent requests.

Deliberately stdlib-only (no FastAPI/uvicorn) to avoid adding any new pip dependency to
the shared cloud venv's carefully-pinned torch/transformers/bitsandbytes stack (see
/memories/repo/cloud_sync.md's gptqmodel incident).

CLOUD-ONLY. Binds to 127.0.0.1 ONLY (never 0.0.0.0) -- this process must never be directly
reachable from the public internet. Access it from a local machine via an SSH tunnel
(`ssh -N -L 8801:127.0.0.1:8801 -i <key> ubuntu@<pod-ip>`), never by opening a firewall
port on the pod. No authentication is implemented because the SSH tunnel itself (which
already requires the private key) is the access control -- do not change this to bind
0.0.0.0 without adding real authentication first.

Separate port (8801) from cloud/chief_engineer_server.py (8800, a single-domain server
with a fixed system prompt) so both can run side by side without conflict; that one is
left as-is rather than migrated here, to avoid any risk to its already-working setup.

Run (inside tmux, so it survives SSH disconnects):
    ./.venv/bin/python -u cloud/qwen_inference_server.py
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # Auto Pilot/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.paths import AgentPaths
from core.qwen_loader import load_qwen

HOST, PORT = "127.0.0.1", 8801

_DOMAIN_FACTORY = {
    "VHF": AgentPaths.vhf,
    "OOW": AgentPaths.oow,
    "Captain": AgentPaths.captain,
    "ChiefEngineer": AgentPaths.chief_engineer,
}

_model_cache: dict[tuple[str, str], tuple] = {}
_cache_lock = threading.Lock()
_gen_lock = threading.Lock()  # one generation at a time -- a single GPU can't usefully parallelise anyway


def _get_model(domain: str, weights: str):
    key = (domain, weights)
    with _cache_lock:
        if key not in _model_cache:
            factory = _DOMAIN_FACTORY.get(domain)
            if factory is None:
                raise ValueError(f"Unknown domain {domain!r} -- known: {sorted(_DOMAIN_FACTORY)}")
            print(f"Loading {domain}/{weights} ...", flush=True)
            _model_cache[key] = load_qwen(weights, factory())
            print(f"Loaded {domain}/{weights}.", flush=True)
        return _model_cache[key]


@torch.inference_mode()
def _generate(tok, mdl, messages: list[dict], max_new_tokens: int, enable_thinking: bool) -> str:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
    with _gen_lock:
        out = mdl.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id, repetition_penalty=1.15)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/generate":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            domain = body["domain"]
            weights = body.get("weights", "W0_base")
            messages = body["messages"]
            max_new_tokens = int(body.get("max_new_tokens", 400))
            enable_thinking = bool(body.get("enable_thinking", False))
            tok, mdl = _get_model(domain, weights)
            text = _generate(tok, mdl, messages, max_new_tokens, enable_thinking)
            payload = json.dumps({"text": text}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:  # noqa: BLE001 -- always report the real error back to the client
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args) -> None:  # quieter default stderr access log
        print("[server]", fmt % args, flush=True)


if __name__ == "__main__":
    print(f"Serving on {HOST}:{PORT} (models load lazily per domain/weights on first request)", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
