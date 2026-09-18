"""
================================================================================
eval_oow_scenarios.py — OOW Track 2 evaluation: applied helm/engine-order decisions
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Track 2 companion to eval_finetuned.py (Track 1: does the model KNOW the
rules?). This asks: given a fused MOOS-style situation report (own-ship
state + per-contact bearing/range/CPA/TCPA/risk, see
pipeline.track2.build_oow_scenarios), can the model DECIDE and STATE the
correct action?

Unlike VHF's Track 2 (eval_colreg_scenarios.py, which needs a GPT-4o-mini
judge for ColregCorrect because "is this VHF exchange procedurally correct"
has no closed form), OOW's ground truth here IS closed-form -- the
`correct_action`/`give_way_role`/`colreg_rules` fields were computed
deterministically by build_oow_scenarios.py from CPA/TCPA geometry. So the
three OOW-specific metrics below are RULE-BASED (regex/keyword parsing of
the model's own free-text decision), no judge call needed for them:

  ActionCorrect    : does the model's stated action (parsed from its answer)
                     match the ground-truth action name (maintain_course /
                     alter_course / stop / set_speed / resume)?
  DirectionCorrect : for alter_course actions only, does the model say
                     "starboard" (never "port") with a degree figure in a
                     plausible range? Vacuous 1.0 when the ground truth
                     action isn't alter_course.
  RuleCite         : fraction of the situational COLREG rule numbers (i.e.
                     colreg_rules minus the always-applicable BASE_RULES)
                     that the model's answer cites by number.

The claim-level RAGAS suite v2 (AnswerCorrectness/CorpusGrounded/
AnswerRelevancy/NumericF1/LitHit/Cover, closed-book kind since there's no
Track 2 RAG corpus yet) is computed exactly like every other eval script in
this repo, via OpenAI (gpt-4o-mini judge) -- same suite, same judge cache.

USAGE
-----
    python -m pipeline.eval.eval_oow_scenarios --model _models/OOW/OOW-QWEN --tag oow_qwen
    python -m pipeline.eval.eval_oow_scenarios --model Qwen/Qwen3-8B --tag oow_qwen_base
    python -m pipeline.eval.eval_oow_scenarios --n 20   # smoke test

OUTPUT
------
Data/OOW/OOW_Agents_Training/eval_{tag}_colreg.jsonl
Data/OOW/OOW_Agents_Training/eval_{tag}_colreg_summary.json
"""
from __future__ import annotations
import os, json, argparse, math, re, time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from pipeline.eval.eval_finetuned import load_env, load_lm, _latency_stats
from core import AgentPaths, EMBEDDER_MODEL

paths = AgentPaths.oow()
W = paths.workspace
CACHE = paths.cache_dir
MODELS = paths.models_root
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))

SCENARIOS_FILE = paths.eval_file("oow_colreg_scenarios.json")
BASE_RULES = {"Rule 2", "Rule 5", "Rule 6", "Rule 7", "Rule 8"}  # always-applicable, not scenario-specific

SYSTEM_OOW = (
    "You are the Navigation Agent aboard an autonomous surface vessel. You receive a fused "
    "situation report (own-ship state, tracked contacts with bearing/range/CPA/TCPA/risk, "
    "applicable COLREG rules, and the allowed actions this cycle with their parameters). "
    "Reply by stating the ONE action you take this cycle (maintain_course, alter_course with "
    "a degree figure and direction, set_speed, stop, or resume), then justify it citing the "
    "COLREG rule(s) that apply. Never invent an action outside the allowed list."
)

_ACTION_PATTERNS = {
    "maintain_course": re.compile(r"\bmaintain(ing)?\s+(course|speed)|\bhold(ing)?\s+course\b", re.I),
    "stop": re.compile(r"\bstop(ping)?\b(?!\s*course)", re.I),
    "resume": re.compile(r"\bresum(e|ing)\b", re.I),
    "set_speed": re.compile(r"\bset[\s_]?speed|\breduc(e|ing)\s+speed|\bslacken", re.I),
    "alter_course": re.compile(r"\balter(ing)?\s+course|\bturn(ing)?\s+(to\s+)?(starboard|port)", re.I),
}
_DEGREE_RE = re.compile(r"(\d{1,3})\s*deg", re.I)


@torch.inference_mode()
def generate_oow(tok, model, situation_report: str, question: str, max_new_tokens: int = 300) -> tuple[str, float]:
    """Greedy-decode one OOW Track 2 scenario answer; returns (answer_text, latency_seconds)."""
    messages = [
        {"role": "system", "content": SYSTEM_OOW},
        {"role": "user", "content": f"{situation_report}\n\n{question}"},
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


def parse_action(answer: str) -> str | None:
    """Best-effort extraction of the FIRST action the model claims to take. Checked in an
    order that resolves overlaps sensibly (e.g. "alter_course" wording beats a stray
    "speed" mention that would otherwise also match set_speed)."""
    for name in ("alter_course", "stop", "resume", "set_speed", "maintain_course"):
        if _ACTION_PATTERNS[name].search(answer):
            return name
    return None


def action_correct_score(answer: str, correct_action: str) -> float:
    return 1.0 if parse_action(answer) == correct_action else 0.0


def direction_correct_score(answer: str, correct_action: str, correct_params: dict) -> float | None:
    """Vacuous 1.0 (not applicable) unless the ground truth action is alter_course."""
    if correct_action != "alter_course":
        return None
    low = answer.lower()
    if "port" in low and "starboard" not in low:
        return 0.0
    if "starboard" not in low:
        return 0.0
    m = _DEGREE_RE.search(answer)
    if not m:
        return 0.0
    degrees = int(m.group(1))
    expected = correct_params.get("degrees", 30)
    return 1.0 if abs(degrees - expected) <= 10 else 0.0


def rule_cite_score(answer: str, colreg_rules: list[str]) -> float:
    """Fraction of the SITUATIONAL rules (colreg_rules minus the always-applicable BASE_RULES)
    the answer cites by number. Vacuous 1.0 if there are no situational rules to check
    (shouldn't happen in practice, but keeps the metric well-defined)."""
    situational = [r for r in colreg_rules if r not in BASE_RULES]
    if not situational:
        return 1.0
    cited = set(re.findall(r"Rule\s+(\d+)", answer, re.I))
    hits = sum(1 for r in situational if re.search(r"\d+", r) and re.search(r"\d+", r).group(0) in cited)
    return round(hits / len(situational), 3)


def main() -> None:
    """CLI entry point: generate + score answers to the held-out OOW Track 2 scenarios for one model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(MODELS / "OOW" / "OOW-QWEN"))
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--n", type=int, default=None, help="how many scenarios (default: all)")
    ap.add_argument("--force-4bit", action="store_true",
                    help="skip the bf16 attempt and load directly in 4-bit NF4")
    ap.add_argument("--legacy", action="store_true",
                    help="skip the RAGAS suite v2 -- rule-based metrics only (no judge/embedder needed)")
    args = ap.parse_args()

    tag = args.tag or Path(args.model).name.replace("/", "_")
    out_file    = CACHE / f"eval_{tag}_colreg.jsonl"
    summary_out = CACHE / f"eval_{tag}_colreg_summary.json"

    scenarios = json.loads(SCENARIOS_FILE.read_text(encoding="utf-8"))
    if args.n:
        scenarios = scenarios[:args.n]
    print(f"Evaluating {len(scenarios)} OOW scenarios (Track 2: applied helm/engine-order decisions)")

    # --- 1) Generation phase (only LM in VRAM) ---
    log_every = 1 if len(scenarios) <= 20 else 20
    tok, model = load_lm(args.model, force_4bit=args.force_4bit)
    gen_records = []
    for i, s in enumerate(scenarios, 1):
        print(f"  [gen {i}/{len(scenarios)}] {s['id']}: {s['category']}", flush=True)
        try:
            ans, dt = generate_oow(tok, model, s["situation_report"], s["question"])
        except Exception as e:
            ans, dt = f"[ERROR:{e}]", 0.0
        gen_records.append({**s, "answer": ans, "latency_s": round(dt, 2)})
        if i % log_every == 0 or i == len(scenarios):
            print(f"  gen {i}/{len(scenarios)} done  last {dt:.1f}s", flush=True)

    del model
    torch.cuda.empty_cache()

    # --- 2) Metric phase (rule-based always; RAGAS suite v2 unless --legacy) ---
    keys = ["ActionCorrect", "DirectionCorrect", "RuleCite"]
    scorer = None
    if not args.legacy:
        load_env(W / ".env")
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY missing (needed for the RAGAS suite v2 judge)")
        judge = OpenAI()
        print("\nLoading embedder for metrics...")
        embedder = SentenceTransformer(EMBEDDER_MODEL)
        from pipeline.eval.ragas_metrics import RagasScorer, load_gold_claims, summary_stamp
        claims_by_id = load_gold_claims(SCENARIOS_FILE)
        if not claims_by_id:
            raise SystemExit(f"No gold_claims next to {SCENARIOS_FILE} -- run "
                             "pipeline.eval.enrich_gold_claims first, or pass --legacy.")
        # Scenario-scoped PG (build_pg.py --traces-file oow_scenario_reasoning_traces.jsonl)
        # for ProcOrder -- falls back to the merged PG if it hasn't been built yet.
        scenario_pg = CACHE / "oow_pg_scenario.json"
        scorer = RagasScorer(judge, embedder, paths,
                             pg_file=scenario_pg if scenario_pg.exists() else None)
        suite_stamp = summary_stamp()
        keys += ["AnswerCorrectness", "ClaimPrec", "ClaimRec", "ClaimF1", "CorpusGrounded",
                "AnswerRelevancy", "NumericF1", "NumericPrec", "NumericRec", "LitHit", "Cover",
                "SemSim", "AnsRelCos", "Composite"]
    else:
        suite_stamp = {"metric_suite": "legacy_v1"}

    print("Scoring...")
    n_no_claims = 0
    with out_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(gen_records, 1):
            m: dict = {
                "ActionCorrect": action_correct_score(r["answer"], r["correct_action"]),
                "DirectionCorrect": direction_correct_score(r["answer"], r["correct_action"],
                                                            r["correct_action_params"]),
                "RuleCite": rule_cite_score(r["answer"], r["colreg_rules"]),
            }
            if scorer is not None:
                claims = claims_by_id.get(str(r["id"]))
                if claims is None:
                    n_no_claims += 1
                    m["no_gold_claims"] = True
                else:
                    ragas_m = scorer.score_row(r["question"], r["answer"], r["gold_answer"],
                                               r.get("expected_points"), claims, contexts=None)
                    m.update(ragas_m)
            row = {**r, "metrics": m}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(gen_records):
                print(f"  scored {i}/{len(gen_records)}  ActionCorrect={m['ActionCorrect']}", flush=True)
    if n_no_claims:
        print(f"  [warn] {n_no_claims} rows skipped RAGAS scoring: no gold_claims yet (enrichment incomplete)")

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
               "track": "applied_helm_engine_decisions", **suite_stamp,
               "means": means, "counts": cnts, "latency": lat_stats}
    summary_out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"Track 2 (applied helm/engine-order decisions) summary — {tag} (n={len(gen_records)})")
    print("=" * 60)
    for k, v in means.items():
        print(f"  {k:<18} {v if v is not None else 'n/a':>6}    (n={cnts[k]})")
    print(f"\nDetails: {out_file}")
    print(f"Summary: {summary_out}")


if __name__ == "__main__":
    main()
