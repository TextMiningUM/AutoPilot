"""
================================================================================
eval_colreg_scenarios.py — Track 2 evaluation: VHF conversational compliance
================================================================================

WHAT THIS SCRIPT DOES
---------------------
This is the SECOND, separate evaluation track for VHF-QWEN (see notebook § 13.5
for the full explanation of why rules-knowledge and conversational-compliance
are evaluated separately).

  Track 1 (eval_finetuned.py)   -> vhf_gold_answers.json (540 SRC exam Q's)
                                    "does the model KNOW the VHF/GMDSS rules?"
  Track 2 (THIS script)         -> vhf_colreg_scenarios.json (498 scenarios)
                                    "can the model actually CONDUCT a correct,
                                     compliant VHF exchange during a real
                                     collision-avoidance situation?"

For each scenario the model receives the situational briefing + the question
("As the Auto Pilot aboard <ship>, what do you transmit and what action do you
take?") -- NOT the gold vhf_channel/colreg_rules fields, those are answer-only.

METRICS COMPUTED (different from Track 1 -- these test procedure, not recall)
------------------------------------------------------------------------------
SemSim        : cosine(embed(gold_answer), embed(pred))         -- semantic similarity
AnsRel        : cosine(embed(question), embed(pred))            -- on-topic-ness
Cover         : fraction of expected_points covered (embedding threshold)
ChannelProc   : rule-based -- did the model hail on 16 AND name a distinct
                working channel to move to? (1.0 both / 0.5 only 16 / 0.0 neither)
CallFormatOK  : rule-based -- does the transmission call the OTHER vessel first
                ("<target>, <target>, THIS IS <own>, <own>, OVER"), i.e. NOT the
                self-hailing bug caught during data generation? (1.0 / 0.0)
ColregCorrect : gpt-4o-mini judge (0-1) -- does the stated action actually
                comply with the cited COLREG rule(s) and expected_points?
Composite     : weighted mean, ColregCorrect weighted heaviest (this track
                cares most about whether the ACTION is COLREG-compliant,
                not just fluent-sounding)

USAGE
-----
    python eval_colreg_scenarios.py --model _models/VHF/VHF-QWEN --tag vhf_qwen
    python eval_colreg_scenarios.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen_base
    python eval_colreg_scenarios.py --n 50   # smoke test

OUTPUT
------
Data/VHF/VHF_Agents_Training/eval_{tag}_colreg.jsonl
Data/VHF/VHF_Agents_Training/eval_{tag}_colreg_summary.json
"""
from __future__ import annotations
import os, json, argparse, math, re, time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from pipeline.eval.eval_finetuned import load_env, load_lm, semsim, cover, _latency_stats
from core import AgentPaths

paths = AgentPaths.vhf()
W = paths.workspace
CACHE = paths.cache_dir
MODELS = paths.models_root
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))

SCENARIOS_FILE = paths.eval_file("vhf_colreg_scenarios.json")

SYSTEM_COLREG = (
    "You are the Auto Pilot, an automated VHF radio watch-keeper aboard a vessel. "
    "Respond with the actual VHF transmission you would make, in quotation marks, "
    "using correct radio procedure (hail on the international calling/distress "
    "channel, then move to a working channel), followed by the collision-avoidance "
    "action you take and the COLREG rule that justifies it. Never call out your "
    "own vessel's name as the station being hailed -- always call the OTHER "
    "station first, then identify yourself."
)

JUDGE_MODEL = "gpt-4o-mini"

COLREG_JUDGE_PROMPT = """You are a COLREG examiner grading whether a candidate VHF response \
takes the CORRECT, COMPLIANT collision-avoidance action for the given situation. Return JSON: \
{{"correct": <float in [0,1]>, "reason": "..."}}. Consider: does the candidate apply the cited \
rule(s) correctly (right give-way/stand-on behaviour, right side to alter towards, correct \
priority), does it avoid treating a VHF arrangement as overriding the Rules, and does it cover \
the expected points? A wrong give-way/stand-on call, wrong turn direction, or an answer that \
relies on VHF agreement INSTEAD of the COLREG-required action should score low.

Scenario: {scenario}
Question: {question}
COLREG rule(s) engaged: {rules}
Expected points: {ep}
Candidate response: {p}"""

_JUDGE_ERR_LOGGED = 0


def judge_colreg_correct(client: OpenAI, scenario: str, question: str, rules: list[str],
                         ep: list[str], pred: str) -> float:
    """LLM-judge correctness score in [0,1] for a COLREG scenario answer (nan on judge failure)."""
    global _JUDGE_ERR_LOGGED
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": COLREG_JUDGE_PROMPT.format(
                scenario=scenario, question=question, rules=", ".join(rules), ep=ep, p=pred)}],
        )
        return float(json.loads(r.choices[0].message.content).get("correct", math.nan))
    except Exception as e:
        if _JUDGE_ERR_LOGGED < 3:
            print(f"  [judge error] {type(e).__name__}: {e}", flush=True)
            _JUDGE_ERR_LOGGED += 1
        return math.nan


# ── Rule-based procedure checks (no LLM needed) ──────────────────────────
_CHANNEL_NUM_RE = re.compile(r"\bchannel\s+(\d{1,2}[A]?)\b", re.IGNORECASE)
_QUOTE_RE = re.compile(r'["\']([^"\']{5,250})["\']')


def channel_procedure_score(pred: str) -> float:
    """1.0 if hailed on 16 AND named a distinct working channel; 0.5 if only 16; 0.0 if neither."""
    nums = {m.upper() for m in _CHANNEL_NUM_RE.findall(pred)}
    has_16 = "16" in nums or "one-six" in pred.lower() or "one six" in pred.lower()
    other = nums - {"16"}
    if has_16 and other:
        return 1.0
    if has_16 or other:
        return 0.5
    return 0.0


def call_format_score(own_vessel: str, pred: str) -> float | None:
    """1.0 if the quoted transmission calls the OTHER station first (not self-hailing).
    None if no quoted transmission is found (can't judge -- excluded from the mean)."""
    own = own_vessel.split(",")[0].replace("MV ", "").replace("M/V ", "").strip().upper()
    if not own:
        return None
    # Combine all quoted fragments -- models sometimes split one transmission across
    # multiple short quotes (e.g. 'x' 'y' instead of one "x ... y").
    quotes = _QUOTE_RE.findall(pred)
    combined = " ".join(quotes).upper()
    this_is = re.search(r"THIS IS ([A-Z][A-Z ]{2,40})", combined)
    if not this_is:
        return None
    # Bad: own vessel's name appears in the "THIS IS <caller>" slot AND also as the
    # first-named (called) station -- i.e. own vessel calling itself.
    called_first = combined.split("THIS IS")[0]
    self_hail = own in this_is.group(1) and own in called_first
    return 0.0 if self_hail else 1.0


def composite_colreg(sem: float, ans_rel: float, cov: float, chan: float,
                     callfmt: float | None, colreg: float) -> float:
    """(legacy v1) Weighted average of the per-metric COLREG scores, skipping any nan/missing metric."""
    weights = {"SemSim": 0.10, "AnsRel": 0.10, "Cover": 0.15,
               "ChannelProc": 0.15, "CallFormatOK": 0.15, "ColregCorrect": 0.35}
    vals = {"SemSim": sem, "AnsRel": ans_rel, "Cover": cov,
            "ChannelProc": chan, "CallFormatOK": callfmt, "ColregCorrect": colreg}
    num, den = 0.0, 0.0
    for k, w in weights.items():
        v = vals[k]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        num += w * v
        den += w
    return num / den if den > 0 else math.nan


# Track 2 suite-v2 composite: behavioral graders keep the majority of the
# weight (they are the point of this track); claim-level metrics replace the
# embedding proxies. Fixed weights -- a nan judge call makes the row's
# Composite nan (excluded + flagged) instead of silently re-weighting.
# CallFormatOK None (no quoted transmission found) counts as 0.0: the system
# prompt explicitly demands a quoted transmission.
COMPOSITE_V2_WEIGHTS = {"ColregCorrect": 0.30, "AnswerCorrectness": 0.25,
                        "ChannelProc": 0.15, "CallFormatOK": 0.10,
                        "NumericF1": 0.10, "CorpusGrounded": 0.10}


def composite_colreg_v2(m: dict) -> float:
    num = 0.0
    for k, w in COMPOSITE_V2_WEIGHTS.items():
        v = m.get(k)
        if k == "CallFormatOK" and v is None:
            v = 0.0
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return math.nan
        num += w * v
    return num / sum(COMPOSITE_V2_WEIGHTS.values())


@torch.inference_mode()
def generate_colreg(tok, model, scenario: str, question: str, max_new_tokens: int = 400) -> tuple[str, float]:
    """Greedy-decode one COLREG scenario answer; returns (answer_text, latency_seconds)."""
    messages = [
        {"role": "system", "content": SYSTEM_COLREG},
        {"role": "user",   "content": f"{scenario}\n\n{question}"},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    t0 = time.time()
    out = model.generate(
        **inp, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=1.0, top_p=1.0,
        pad_token_id=tok.eos_token_id,
    )
    dt = time.time() - t0
    ans = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ans, dt


def main() -> None:
    """CLI entry point: generate + score answers to the 498 held-out COLREG scenarios for one model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(MODELS / "VHF" / "VHF-QWEN"))
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--n", type=int, default=None, help="how many scenarios (default: all)")
    ap.add_argument("--force-4bit", action="store_true",
                    help="skip the bf16 attempt and load directly in 4-bit NF4")
    ap.add_argument("--legacy", action="store_true",
                    help="score with the old v1 metric suite instead of the RAGAS suite")
    args = ap.parse_args()

    tag = args.tag or Path(args.model).name.replace("/", "_")
    out_file    = CACHE / f"eval_{tag}_colreg.jsonl"
    summary_out = CACHE / f"eval_{tag}_colreg_summary.json"

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing (needed for ColregCorrect judge)")
    judge = OpenAI()

    scenarios = json.loads(SCENARIOS_FILE.read_text(encoding="utf-8"))
    if args.n:
        scenarios = scenarios[:args.n]
    print(f"Evaluating {len(scenarios)} COLREG scenarios (Track 2: conversational compliance)")

    # --- 1) Generation phase (only LM in VRAM) ---
    log_every = 1 if len(scenarios) <= 20 else 20
    tok, model = load_lm(args.model, force_4bit=args.force_4bit)
    gen_records = []
    for i, s in enumerate(scenarios, 1):
        print(f"  [gen {i}/{len(scenarios)}] {s['id']}: {s['question'][:80]!r}", flush=True)
        try:
            ans, dt = generate_colreg(tok, model, s["scenario"], s["question"])
        except Exception as e:
            ans, dt = f"[ERROR:{e}]", 0.0
        gen_records.append({**s, "answer": ans, "latency_s": round(dt, 2)})
        if i % log_every == 0 or i == len(scenarios):
            print(f"  gen {i}/{len(scenarios)} done  last {dt:.1f}s", flush=True)

    del model
    torch.cuda.empty_cache()

    # --- 2) Metric phase (embedder + judge) ---
    print("\nLoading embedder for metrics...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if args.legacy:
        suite_stamp = {"metric_suite": "legacy_v1", "judge_model": JUDGE_MODEL}
        keys = ["SemSim", "AnsRel", "Cover", "ChannelProc", "CallFormatOK", "ColregCorrect", "Composite"]
    else:
        from pipeline.eval.ragas_metrics import RagasScorer, load_gold_claims, summary_stamp
        claims_by_id = load_gold_claims(SCENARIOS_FILE)
        if not claims_by_id:
            raise SystemExit(f"No gold_claims next to {SCENARIOS_FILE} -- run "
                             "pipeline.eval.enrich_gold_claims first, or pass --legacy.")
        scorer = RagasScorer(judge, embedder, paths)
        suite_stamp = summary_stamp()
        keys = ["AnswerCorrectness", "ClaimPrec", "ClaimRec", "ClaimF1",
                "CorpusGrounded", "AnswerRelevancy", "NumericF1", "NumericPrec", "NumericRec",
                "LitHit", "Cover", "SemSim", "AnsRelCos", "ProcOrder",
                "ChannelProc", "CallFormatOK", "ColregCorrect", "Composite"]

    print("Scoring...")
    n_no_claims = 0
    with out_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(gen_records, 1):
            chan = channel_procedure_score(r["answer"])
            cfmt = call_format_score(r.get("own_vessel", ""), r["answer"])
            creg = judge_colreg_correct(judge, r["scenario"], r["question"],
                                        r.get("colreg_rules", []), r.get("expected_points", []),
                                        r["answer"])
            if args.legacy:
                sem  = round(semsim(embedder, r["gold_answer"], r["answer"]), 3)
                ans_rel = round(semsim(embedder, r["question"], r["answer"]), 3)
                cov  = round(cover(embedder, r["answer"], r.get("expected_points", [])), 3) \
                           if r.get("expected_points") else None
                m = {
                    "SemSim": sem, "AnsRel": ans_rel, "Cover": cov,
                    "ChannelProc": chan, "CallFormatOK": cfmt,
                    "ColregCorrect": None if math.isnan(creg) else round(creg, 3),
                }
                m["Composite"] = round(composite_colreg(sem, ans_rel, cov, chan, cfmt, creg), 3)
            else:
                claims = claims_by_id.get(str(r["id"]))
                if claims is None:
                    n_no_claims += 1
                    m = {"no_gold_claims": True}
                else:
                    m = scorer.score_row(r["question"], r["answer"], r["gold_answer"],
                                         r.get("expected_points"), claims, contexts=None)
                    m.pop("Composite", None)  # replaced by the Track 2 composite
                    m.update({"ChannelProc": chan, "CallFormatOK": cfmt,
                              "ColregCorrect": None if math.isnan(creg) else round(creg, 3)})
                    m["Composite"] = round(composite_colreg_v2({**m, "ColregCorrect": creg}), 3)
            row = {**r, "metrics": m}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(gen_records):
                print(f"  scored {i}/{len(gen_records)}  Composite={m.get('Composite')}", flush=True)
    if n_no_claims:
        print(f"  [warn] {n_no_claims} rows skipped: no gold_claims yet (enrichment incomplete)")

    # --- 3) Summarize ---
    sums, cnts = {k: 0.0 for k in keys}, {k: 0 for k in keys}
    with out_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)["metrics"]
            for k in keys:
                v = m.get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                sums[k] += v
                cnts[k] += 1
    means = {k: round(sums[k] / cnts[k], 3) if cnts[k] else None for k in keys}

    latencies = sorted(r["latency_s"] for r in gen_records if r.get("latency_s"))
    lat_stats = _latency_stats(latencies)

    summary = {"model": args.model, "tag": tag, "n": len(gen_records),
               "track": "conversational_compliance", **suite_stamp,
               "means": means, "counts": cnts, "latency": lat_stats}
    summary_out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"Track 2 (conversational compliance) summary — {tag} (n={len(gen_records)})")
    print("=" * 60)
    for k, v in means.items():
        print(f"  {k:<14} {v if v is not None else 'n/a':>6}    (n={cnts[k]})")
    print(f"\nDetails: {out_file}")
    print(f"Summary: {summary_out}")


if __name__ == "__main__":
    main()
