"""
================================================================================
probe_dpo.py — stage-attribution probes: DPO axis win-rates + reflection Δ
================================================================================

Aggregate eval deltas can't say WHICH training stage helped. These two probes
measure each stage's specific job directly (RAGAS migration plan, step 6):

DPO PROBE ("did preference training do its job?")
  The DPO training pairs perturb gold answers along five axes. This probe
  builds a HELD-OUT probe set with the same five axes from the gold eval file
  (never used in training) and measures, per axis, how often the model assigns
  a higher mean per-token log-likelihood to the good answer than to the
  perturbed one. DPO should raise these win-rates over the SFT-only
  checkpoint; axes: wrong_channel, wrong_proword, missing_step,
  dropped_regulation, dropped_warning.

REFLECTION PROBE ("can the model critique and repair a draft?")
  Feeds the model a flawed draft (the perturbed answer) and asks for critique
  + revision. Metric = Δ AnswerCorrectness (revision − draft) via the RAGAS
  suite. The reflection checkpoint should show a larger positive Δ than
  base/SFT — if not, the stage is dead weight.

The probe set is deterministic (seeded) and cached in
<cache_dir>/probe_set.json, so every checkpoint is scored on identical items.

USAGE (run per checkpoint, cloud or --force-4bit locally)
    python -X utf8 -m pipeline.eval.probe_dpo --model Qwen/Qwen3-8B --tag qwen_base
    python -X utf8 -m pipeline.eval.probe_dpo --model _models/VHF/VHF-QWEN-SFT --tag sft_only --probe dpo

OUTPUT
    <cache_dir>/probe_{tag}.json  per-axis win-rates + reflection deltas
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from core import AgentPaths, load_env, EMBEDDER_MODEL

paths = AgentPaths.from_env()
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))

PROBE_FILE = paths.cache_dir / "probe_set.json"
SEED = 20260917

AXES = ["wrong_channel", "wrong_proword", "missing_step",
        "dropped_regulation", "dropped_warning", "swapped_step_order"]

_CHANNEL_RE = re.compile(r"\b([Cc]hannel\s+)(\d{1,2})\b")
# COLREG "Rule N" plus the VHF-domain citation forms (ITU/SOLAS/GMDSS)
_RULE_RE = re.compile(
    r"\b(?:Rule\s+\d{1,2}|ITU(?:\s+Radio)?\s+Regulations?|SOLAS(?:\s+Chapter\s+[IVX\d]+)?"
    r"|GMDSS\s+(?:regulations?|requirements?)|Radio\s+Regulations?)\b")
_WARN_RE = re.compile(r"\b(never|must not|do not|don't|warning|caution|danger|avoid)\b",
                      re.IGNORECASE)
_VALID_CHANNELS = ["06", "08", "10", "12", "13", "16", "67", "68", "69", "72", "73", "77"]
_PROWORD_SWAPS = [("MAYDAY", "PAN PAN"), ("PAN PAN", "SECURITE"), ("PAN-PAN", "SECURITE"),
                  ("OVER", "OUT"), ("SAY AGAIN", "REPEAT"), ("WILCO", "ROGER")]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]


# ── perturbations: return the flawed answer, or None if axis not applicable ──
def perturb_wrong_channel(gold: str, rng: random.Random) -> str | None:
    m = _CHANNEL_RE.search(gold)
    if not m:
        return None
    wrong = rng.choice([c for c in _VALID_CHANNELS if int(c) != int(m.group(2))])
    return gold[:m.start()] + m.group(1) + str(int(wrong)) + gold[m.end():]


def perturb_wrong_proword(gold: str, rng: random.Random) -> str | None:
    up = gold.upper()
    options = [(a, b) for a, b in _PROWORD_SWAPS if a in up]
    if not options:
        return None
    a, b = rng.choice(options)
    idx = up.index(a)
    return gold[:idx] + b + gold[idx + len(a):]


def perturb_missing_step(gold: str, rng: random.Random) -> str | None:
    sents = _sentences(gold)
    if len(sents) < 3:
        return None
    drop = rng.randrange(1, len(sents) - 1)  # keep first and last
    return " ".join(s for i, s in enumerate(sents) if i != drop)


def perturb_dropped_regulation(gold: str, rng: random.Random) -> str | None:
    sents = _sentences(gold)
    rule_idx = [i for i, s in enumerate(sents) if _RULE_RE.search(s)]
    if not rule_idx or len(sents) < 2:
        return None
    # strip the rule citation from one sentence (keep the sentence's action text)
    i = rng.choice(rule_idx)
    stripped = _RULE_RE.sub("the applicable regulations", sents[i])
    stripped = re.sub(r"\b([Tt]he)\s+the\b", r"\1", stripped)  # "The the applicable..." → "The applicable..."
    return " ".join(stripped if j == i else s for j, s in enumerate(sents))


def perturb_dropped_warning(gold: str, rng: random.Random) -> str | None:
    sents = _sentences(gold)
    warn_idx = [i for i, s in enumerate(sents) if _WARN_RE.search(s)]
    if not warn_idx or len(sents) < 2:
        return None
    drop = rng.choice(warn_idx)
    return " ".join(s for i, s in enumerate(sents) if i != drop)


def perturb_swapped_step_order(gold: str, rng: random.Random) -> str | None:
    """Swap two adjacent sentences -- procedure order is safety-critical."""
    sents = _sentences(gold)
    if len(sents) < 3:
        return None
    i = rng.randrange(0, len(sents) - 1)
    sents[i], sents[i + 1] = sents[i + 1], sents[i]
    return " ".join(sents)


PERTURBERS = {
    "wrong_channel": perturb_wrong_channel,
    "wrong_proword": perturb_wrong_proword,
    "missing_step": perturb_missing_step,
    "dropped_regulation": perturb_dropped_regulation,
    "dropped_warning": perturb_dropped_warning,
    "swapped_step_order": perturb_swapped_step_order,
}


def build_probe_set(per_axis: int, gold_file: Path | None = None) -> list[dict]:
    """Deterministic held-out probe set from the gold eval file. Cached."""
    if PROBE_FILE.exists():
        cached = json.loads(PROBE_FILE.read_text(encoding="utf-8"))
        if cached.get("per_axis") == per_axis and cached.get("seed") == SEED:
            return cached["items"]
    gold = json.loads((gold_file or paths.gold_file).read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    order = list(range(len(gold)))
    rng.shuffle(order)
    items: list[dict] = []
    for axis in AXES:
        arng = random.Random(f"{SEED}-{axis}")
        n = 0
        for idx in order:
            if n >= per_axis:
                break
            g = gold[idx]
            bad = PERTURBERS[axis](g["gold_answer"], arng)
            if bad is None or bad == g["gold_answer"]:
                continue
            items.append({"id": g["id"], "axis": axis, "question": g["question"],
                          "good": g["gold_answer"], "bad": bad})
            n += 1
    PROBE_FILE.write_text(json.dumps({"seed": SEED, "per_axis": per_axis, "items": items},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Probe set: {len(items)} items "
          f"({ {a: sum(1 for i in items if i['axis'] == a) for a in AXES} })")
    return items


# ── DPO probe: paired log-likelihood ─────────────────────────────────────
@torch.inference_mode()
def mean_logprob(tok, model, system: str, question: str, answer: str) -> float:
    """Mean per-token log-likelihood of `answer` given the chat prompt."""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full_ids = tok(prompt + answer, return_tensors="pt").input_ids.to(model.device)
    n_prompt = p_ids.shape[1]
    logits = model(full_ids).logits[0, :-1]  # predicts token t+1
    targets = full_ids[0, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp[torch.arange(len(targets)), targets][n_prompt - 1:]
    return float(tok_lp.mean())


def run_dpo_probe(tok, model, system: str, items: list[dict]) -> dict:
    per_axis: dict[str, list[int]] = {a: [] for a in AXES}
    margins: dict[str, list[float]] = {a: [] for a in AXES}
    log_every = 1 if len(items) <= 20 else 20
    for i, it in enumerate(items, 1):
        lp_good = mean_logprob(tok, model, system, it["question"], it["good"])
        lp_bad = mean_logprob(tok, model, system, it["question"], it["bad"])
        per_axis[it["axis"]].append(1 if lp_good > lp_bad else 0)
        margins[it["axis"]].append(lp_good - lp_bad)
        if i % log_every == 0 or i == len(items):
            print(f"  dpo probe {i}/{len(items)}  axis={it['axis']}", flush=True)
    out = {}
    for a in AXES:
        wins = per_axis[a]
        out[a] = {"win_rate": round(float(np.mean(wins)), 3) if wins else None,
                  "mean_margin": round(float(np.mean(margins[a])), 4) if wins else None,
                  "n": len(wins)}
    all_wins = [w for a in AXES for w in per_axis[a]]
    out["overall"] = {"win_rate": round(float(np.mean(all_wins)), 3) if all_wins else None,
                      "n": len(all_wins)}
    return out


# ── Reflection probe: critique-and-revise Δ AnswerCorrectness ────────────
REVISE_PROMPT = (
    "Below is a draft answer to the question. Critique the draft for factual, "
    "procedural, channel/rule-number and proword errors, then produce a corrected, "
    "complete final answer.\n\nQuestion: {q}\n\nDraft answer: {draft}\n\n"
    "Respond in the form:\nCRITIQUE: <your critique>\nFINAL ANSWER: <the corrected answer>"
)


@torch.inference_mode()
def generate_revision(tok, model, system: str, question: str, draft: str) -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": REVISE_PROMPT.format(q=question, draft=draft)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(**inp, max_new_tokens=512, do_sample=False,
                         temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id)
    full = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # score only the revised answer, not the critique
    m = re.search(r"FINAL ANSWER:\s*(.+)", full, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else full


def run_reflection_probe(tok, model, system: str, items: list[dict], n: int) -> dict:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    from pipeline.eval.ragas_metrics import RagasScorer, load_gold_claims

    claims_by_id = load_gold_claims(paths.gold_file)
    usable = [it for it in items if str(it["id"]) in claims_by_id][:n]
    if not usable:
        print("  [reflection probe] no probe items with gold_claims yet -- skipped")
        return {"skipped": "no gold_claims"}
    print(f"  reflection probe on {len(usable)} items")

    revisions = []
    log_every = 1 if len(usable) <= 20 else 10
    for i, it in enumerate(usable, 1):
        revisions.append(generate_revision(tok, model, system, it["question"], it["bad"]))
        if i % log_every == 0 or i == len(usable):
            print(f"  reflect gen {i}/{len(usable)}", flush=True)

    # generation done -> free VRAM before judge phase
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    scorer = RagasScorer(OpenAI(), embedder, paths)
    deltas, draft_scores, rev_scores = [], [], []
    for it, rev in zip(usable, revisions):
        claims = claims_by_id[str(it["id"])]
        ac_draft = scorer.answer_correctness(it["question"], it["bad"], it["good"], claims)
        ac_rev = scorer.answer_correctness(it["question"], rev, it["good"], claims)
        if not (math.isnan(ac_draft) or math.isnan(ac_rev)):
            draft_scores.append(ac_draft)
            rev_scores.append(ac_rev)
            deltas.append(ac_rev - ac_draft)
    return {
        "n": len(deltas),
        "draft_answer_correctness": round(float(np.mean(draft_scores)), 3) if draft_scores else None,
        "revision_answer_correctness": round(float(np.mean(rev_scores)), 3) if rev_scores else None,
        "delta": round(float(np.mean(deltas)), 3) if deltas else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--probe", choices=["dpo", "reflection", "both"], default="both")
    ap.add_argument("--per-axis", type=int, default=30)
    ap.add_argument("--n-reflect", type=int, default=50)
    ap.add_argument("--gold-file", type=Path, default=None,
                    help="gold eval file for the probe set (default: paths.gold_file)")
    ap.add_argument("--force-4bit", action="store_true")
    args = ap.parse_args()

    load_env(paths.env_file)
    from pipeline.eval.eval_finetuned import load_lm, SYSTEM_PLAIN
    from pipeline.eval.ragas_metrics import summary_stamp

    items = build_probe_set(args.per_axis, args.gold_file)
    tok, model = load_lm(args.model, force_4bit=args.force_4bit)

    result = {"model": args.model, "tag": args.tag, **summary_stamp(),
              "probe_seed": SEED, "per_axis": args.per_axis}
    if args.probe in ("dpo", "both"):
        result["dpo_probe"] = run_dpo_probe(tok, model, SYSTEM_PLAIN, items)
    if args.probe in ("reflection", "both"):
        result["reflection_probe"] = run_reflection_probe(
            tok, model, SYSTEM_PLAIN, items, args.n_reflect)

    del model
    torch.cuda.empty_cache()
    out = paths.cache_dir / f"probe_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    if "dpo_probe" in result:
        print("DPO probe win-rates:")
        for a, v in result["dpo_probe"].items():
            print(f"  {a:<20} {v['win_rate']} (n={v['n']})")
    if "reflection_probe" in result:
        print(f"Reflection probe: {result['reflection_probe']}")


if __name__ == "__main__":
    main()
