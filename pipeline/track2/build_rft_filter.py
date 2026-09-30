"""
================================================================================
build_rft_filter.py -- Stap 2 Step 5: RFT self-consistency filter for <think> traces
================================================================================

Samples k candidate <think>...</think>+JSON completions per prompt from BASE Qwen3-8B
(facts-only prompt, no RAG/CoT scaffolding, no teacher-forced answer -- unlike Fase B3's
gated Claude reasoning generation, the model must freely DECIDE the action itself, not
just restate an already-given one), keeps only completions where
pipeline.track2.b3_reasoning_gates.run_gates() passes with ZERO failures AND the
<think> block is under MAX_THINK_TOKENS, dedupes per prompt, formats as SFT rows.

Prompt source: the SAME oracle-labeled rollout as build_oow_scenarios_rnd.py
(rollout_mission(), reused unchanged), restricted to the "acute_action" bucket -- RFT's
value is catching cases the base model SOMETIMES gets right, which is the rare/hard
acute-decision rows, not the trivial always-right quiet-cruise rows (same reasoning as
that module's own stratification cap).

KNOWN RISK (flagged BEFORE running the full population, not discovered after): this
project's own `full report evaluations V1.md` measured base Qwen3-8B (no fine-tuning, no
RAG/CoT scaffolding) at DirectionCorrect=0.0 on Track 2 -- it never independently picked
the correct turn direction unaided. Since this script samples base Qwen with a comparably
bare (facts-only) prompt, a LOW or even near-zero gate-pass rate is a real possibility --
ALWAYS run --pilot first and inspect the measured pass rate before committing to the full
population. Needs a GPU (cloud pod) -- see copilot-instructions.md's local/cloud split.

USAGE
-----
    python -m pipeline.track2.build_rft_filter --pilot 15 --k 5      # pilot, no writes
    python -m pipeline.track2.build_rft_filter --n 200 --k 5 --overwrite
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core import AgentPaths, CONTAM_THRESH, EMBEDDER_MODEL
from pipeline.oow_agent_spec import SYSTEM_OOW_AGENT, render_previous_decisions
from pipeline.track2.b3_reasoning_gates import GATE_NAMES, extract_first_json_object, run_gates
from pipeline.track2.build_oow_scenarios_rnd import FIXED_QUESTION, _bucket, build_user_message, rollout_mission

paths = AgentPaths.oow()
CACHE = paths.cache_dir
MODEL_ID = "Qwen/Qwen3-8B"
# Empirically measured via a diagnostic sample (see repo memory): Qwen3-8B's OWN unaided
# <think> traces on this facts-only prompt commonly run 1500-2800 tokens -- MAX_NEW_TOKENS
# used to be 1024, which truncated almost every completion before it ever closed </think>
# or emitted the JSON answer, producing a false ~0% pilot pass rate that looked like (but
# was NOT) a genuine model-incapability finding. MAX_THINK_TOKENS is now a generous safety
# ceiling against runaway/degenerate repetition, not a tight quality filter.
MAX_THINK_TOKENS = 3000
MAX_NEW_TOKENS = 3584
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

APP_ROOT = paths.ensure_basic_simulator_importable()

from app.missions import list_mission_ids, load_mission  # noqa: E402


def even_sample(rows: list[dict], n: int) -> list[dict]:
    """Evenly-spaced index subsample (not just the first N) so a capped run still spans
    the full mission/category range -- same principle as
    build_oow_scenarios_rnd.py's stratify_mission_rows()."""
    if len(rows) <= n or n <= 0:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def load_acute_prompts(n: int | None) -> list[dict]:
    all_ids = [m for m in list_mission_ids() if m.startswith("RND")]
    missions = [load_mission(m) for m in all_ids]
    trainable = [m for m in missions if not json.loads(
        (APP_ROOT / "Data" / "missions" / f"{m.id}.json").read_text(encoding="utf-8")).get("held_out")]
    recs: list[dict] = []
    for m in trainable:
        recs.extend(r for r in rollout_mission(m) if _bucket(r) == "acute_action")
    print(f"{len(recs)} acute_action prompts available across {len(trainable)} trainable missions")
    return even_sample(recs, n) if n is not None else recs


def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    print(f"Loading {MODEL_ID} in 4-bit NF4...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto",
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    mdl.eval()
    return tok, mdl


def sample_k(tok, mdl, messages: list[dict], k: int) -> list[str]:
    """k DIVERSE completions in one batched generate() call -- do_sample=True (unlike
    Basic Simulator/app/agents.py's own _generate(), which is greedy/do_sample=False by
    design for the live single-shot agent; RFT specifically needs sampling diversity
    across the k candidates, not k identical greedy repeats)."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.8, top_p=0.9,
                           num_return_sequences=k, pad_token_id=tok.eos_token_id, repetition_penalty=1.15)
    prompt_len = inp["input_ids"].shape[1]
    return [tok.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]


def gate_one_completion(tok, completion: str, expected: dict) -> tuple[bool, str | None, int]:
    """(passed, think_text_or_None, think_token_count) for one raw completion -- checks
    the token-length gate itself (MAX_THINK_TOKENS), then delegates schema/decision/rule/
    contact/number/threshold/risk checks to the SHARED b3_reasoning_gates.run_gates()."""
    m = _THINK_RE.search(completion)
    if not m:
        return False, None, 0
    think_text = m.group(1).strip()
    think_tokens = len(tok(think_text, add_special_tokens=False)["input_ids"])
    if think_tokens >= MAX_THINK_TOKENS:
        return False, think_text, think_tokens
    obj = extract_first_json_object(completion[m.end():])
    reasoning = obj.get("reasoning") if obj else None
    # The model's OWN <think> trace is what RFT is meant to keep -- if it didn't also
    # restate a "reasoning" field in the final JSON, fall back to the think text itself
    # for the gates that need reasoning PROSE (contact/number/threshold/risk consistency).
    reasoning_for_gates = reasoning or think_text
    failures = run_gates(obj, reasoning_for_gates, **expected)
    return not failures, think_text, think_tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pilot", type=int, default=None,
                    help="measure the gate-pass rate on this many prompts, print stats, exit -- no writes")
    ap.add_argument("--n", type=int, default=None, help="full-run prompt count (default: all acute_action rows)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gold-file", type=str, default=str(paths.eval_dir / "colreg_qa_500_normalised.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    n = args.pilot if args.pilot is not None else args.n
    rows = load_acute_prompts(n)
    print(f"Sampling k={args.k} completions each for {len(rows)} prompts...")
    tok, mdl = load_model()

    out_rows, n_total, n_passed = [], 0, 0
    for i, r in enumerate(rows):
        history_prefix = render_previous_decisions(r.get("prev_decisions") or [])
        user_msg = build_user_message(history_prefix + r["situation_report"])
        messages = [{"role": "system", "content": SYSTEM_OOW_AGENT}, {"role": "user", "content": user_msg}]
        expected = {
            "situation": history_prefix + r["situation_report"], "expected_action": r["action"],
            "expected_degrees": r["degrees"], "expected_encounter_rule": r["encounter_rule"],
            "expected_conduct_rule": r["conduct_rule"], "real_risk": r["encounter_rule"] != "none",
            "cpa_m": r["cpa_m"], "safe_distance_m": 500.0, "tcpa_s": r["tcpa_s"], "risk_horizon_s": 400.0,
            "decisive_contact_name": r["decisive_contact_name"],
        }
        completions = sample_k(tok, mdl, messages, args.k)
        seen: set[str] = set()
        for completion in completions:
            n_total += 1
            passed, think_text, think_tokens = gate_one_completion(tok, completion, expected)
            if not passed or think_text in seen:
                continue
            seen.add(think_text)
            n_passed += 1
            out_rows.append({"rnd_id": r["rnd_id"], "action": r["action"], "messages": messages + [
                {"role": "assistant", "content": completion.strip()}]})
        if (i + 1) % 5 == 0 or i + 1 == len(rows):
            print(f"  [RFT] {i + 1}/{len(rows)} prompts -- running pass rate: "
                 f"{n_passed}/{n_total} ({n_passed / max(1, n_total):.1%})", flush=True)

    print(f"\nFinal: {n_passed}/{n_total} completions passed ({n_passed / max(1, n_total):.1%}), "
         f"{len(out_rows)} deduped rows from {len(rows)} prompts")
    if args.pilot is not None:
        print("This was a PILOT run -- no file written. Inspect the pass rate above before "
             "deciding the full run's --n.")
        return

    if out_rows:
        gold_questions = [g["question"] for g in json.loads(Path(args.gold_file).read_text(encoding="utf-8"))]
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer(EMBEDDER_MODEL)
        gold_embs = model.encode(gold_questions, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        texts = [r["messages"][-1]["content"] for r in out_rows]
        q_embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        kept = []
        for r, emb in zip(out_rows, q_embs):
            if float(np.max(gold_embs @ emb)) < CONTAM_THRESH:
                kept.append(r)
        print(f"contamination filter: kept={len(kept)}  dropped={len(out_rows) - len(kept)} (thresh={CONTAM_THRESH})")
        out_rows = kept

    from core import safe_write_jsonl, review_path
    out_path = CACHE / "oow_rft_sft.jsonl"
    out_path = out_path if args.overwrite else review_path(out_path)
    safe_write_jsonl(out_rows, out_path, overwrite=args.overwrite)
    print(f"Wrote {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
