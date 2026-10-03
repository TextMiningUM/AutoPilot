"""Minimal stdlib-only HTTP inference server for the fine-tuned Chief Engineer model
(_models/ChiefEngineer/ChiefEngineer-QWEN), so a question typed into the LOCAL Streamlit
dashboard can get a real generated answer from the 8B model running on the cloud's RTX
6000 instead of loading/running it on the laptop's 8GB GPU (too slow there). Deliberately
stdlib-only (no FastAPI/uvicorn) to avoid adding any new pip dependency to the shared
cloud venv's carefully-pinned torch/transformers/bitsandbytes stack (see
/memories/repo/cloud_sync.md's gptqmodel incident).

CLOUD-ONLY. Binds to 127.0.0.1 ONLY (never 0.0.0.0) -- this process must never be directly
reachable from the public internet. Access it from a local machine via an SSH tunnel
(`ssh -N -L 8800:127.0.0.1:8800 -i <key> ubuntu@<pod-ip>`), never by opening a firewall
port on the pod. No authentication is implemented because the SSH tunnel itself (which
already requires the private key) is the access control -- do not change this to bind
0.0.0.0 without adding real authentication first.

Run (inside tmux, so it survives SSH disconnects):
    AUTOPILOT_DOMAIN=ChiefEngineer ./.venv/bin/python -u cloud/chief_engineer_server.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import AgentPaths

paths = AgentPaths.chief_engineer()
MODEL_DIR = paths.domain_models_dir / "ChiefEngineer-QWEN"
HOST, PORT = "127.0.0.1", 8800

SYSTEM_PROMPT = (
    "You are the Chief Engineer, an AI engine-room agent responsible for condition "
    "monitoring and malfunction diagnosis aboard a MAN B&W-class main propulsion engine. "
    "Answer accurately, cite the affected system/component, and follow manufacturer "
    "limits and safety procedures. Be concise."
)

print(f"Loading {MODEL_DIR} ...", flush=True)
_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
_model.eval()
_lock = threading.Lock()  # one generation at a time -- a single GPU can't usefully parallelise anyway
print(f"Model loaded. Serving on {HOST}:{PORT}", flush=True)


@torch.inference_mode()
def _generate(question: str, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    text = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = _tok(text, return_tensors="pt", truncation=True, max_length=4096).to(_model.device)
    with _lock:
        out = _model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=_tok.eos_token_id, repetition_penalty=1.15)
    return _tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/ask":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            question = (body.get("question") or "").strip()
            max_new_tokens = int(body.get("max_new_tokens", 512))
            if not question:
                raise ValueError("empty question")
            answer = _generate(question, max_new_tokens)
            payload = json.dumps({"answer": answer}).encode("utf-8")
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
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
