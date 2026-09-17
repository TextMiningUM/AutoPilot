"""Enrich the held-out gold-standard files with atomic ``gold_claims`` via the
Anthropic API (RAGAS migration, Phase 1).

Claude — not GPT — authors the claim decomposition so the GPT-4o-mini judge
used at eval time never grades annotations produced by its own model family
(same reason the gold files themselves were authored with Claude).

Design:
- The model returns ONLY ``{"id": ..., "gold_claims": [...]}`` per record;
  the original record is merged locally, so every existing field stays
  byte-identical by construction.
- Resume-safe: accepted claims are appended to
  ``<stem>_claims_progress.jsonl`` next to the gold file; on restart, records
  whose id is already there are skipped. Records whose claims fail the
  structural validator (pipeline.eval.gold_claims) are NOT written and are
  retried on the next run.
- When every record is covered, the final enriched file
  ``<stem>_claims.json`` is written: the original array, original order,
  original fields, plus ``gold_claims`` appended to each record.

Run (needs ANTHROPIC_API_KEY in .env)::

    python -X utf8 -m pipeline.eval.enrich_gold_claims                 # both VHF gold files
    python -X utf8 -m pipeline.eval.enrich_gold_claims --limit 5       # smoke test
    python -X utf8 -m pipeline.eval.enrich_gold_claims --files Data/OOW/OOW_Eval/colreg_qa_500_normalised.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from core.io import load_env, load_jsonl
from core.paths import AgentPaths
from pipeline.eval.gold_claims import validate_claims

MODEL_DEFAULT = "claude-sonnet-4-5"
BATCH_SIZE_DEFAULT = 20
MAX_TOKENS_DEFAULT = 16000

SYSTEM_PROMPT = """\
You are enriching gold-standard evaluation records for a marine-VHF/COLREG AI
evaluation suite with atomic claims, so that RAGAS-style claim-level metrics
(answer correctness F1, context recall) and a deterministic role-aware numeric
check can be computed against them.

Each user message contains a JSON array of records. Record schemas you may see:

SCHEMA A (gold Q&A): fields id, section_id, section_title, type, question,
gold_answer, expected_points.
SCHEMA B (COLREG scenarios): fields category, colreg_rules, region, own_vessel,
target_vessel, scenario, vhf_channel {hailing, working, settings, note},
question, expected_points, gold_answer, id.

YOUR TASK, per record: derive its "gold_claims" — an array of 3-10 objects:
  {"claim": "<atomic statement>", "type": "<fact|procedure|number|literal|direction>",
   "value": "<only for number/literal/direction>", "role": "<only for number/direction>"}

RULES FOR CLAIMS
1. Atomic: exactly one verifiable statement per claim. Split compound sentences.
   Wrong: "Hail on channel 16 then switch to channel 72."
   Right: two claims, one per channel, each with its own role.
2. Self-contained: no pronouns or references to other claims ("it", "this
   channel"). A grader must be able to verify each claim in isolation.
3. Grounded ONLY in the record itself (gold_answer, expected_points, and for
   Schema B also vhf_channel/colreg_rules/scenario). NEVER add outside
   knowledge, even if true. If the gold_answer is silent on something, there is
   no claim for it.
4. Coverage: every expected_point must be represented by at least one claim.
   Every safety-critical number in the gold_answer (channel, rule number,
   distance, timing, power) must appear as a type:"number" claim.
5. De-duplicate: if gold_answer and expected_points state the same fact, emit
   one claim, not two.

TYPE DEFINITIONS
- "fact":      declarative knowledge ("The ITU Radio Regulations are a binding
               international treaty").
- "procedure": an action/step the operator or vessel must take ("The give-way
               vessel alters course to starboard").
- "number":    any claim whose correctness hinges on a numeric value. MUST
               include "value" (the number as a string, e.g. "16", "14",
               "20-30") and "role" from the ROLE LIST below.
- "literal":   an exact required token: proword or fixed phrase ("MAYDAY",
               "PAN PAN", "THIS IS", "OVER"). MUST include "value" with the
               exact token, uppercase.
- "direction": a side/turn/colour whose exactness is safety-critical. MUST
               include "value" ("starboard", "port", "red", "green") and
               "role" ("turn_direction", "pass_side", "light_colour").

ROLE LIST for type:"number" (use exactly these strings; pick the closest,
or "other_number" if none fits):
  vhf_channel | colreg_rule | distance_nm | speed_kn | bearing_deg |
  course_deg | power_watt | repeat_count | time_interval | frequency_mhz |
  gross_tonnage | sea_area | mmsi | other_number

EXAMPLE (Schema B record, abbreviated input):
  gold_answer: "... I transmit \\"GULF EXPLORER, GULF EXPLORER, THIS IS TEXAS
  SPIRIT, TEXAS SPIRIT, OVER.\\" After establishing contact I alter course to
  starboard to comply with Rule 14, ensuring both vessels pass port to port."
  vhf_channel: {"hailing": "16", "working": "71"}

  "gold_claims": [
    {"claim": "The initial hail is transmitted on VHF channel 16",
     "type": "number", "value": "16", "role": "vhf_channel"},
    {"claim": "After contact is established, communications move to working channel 71",
     "type": "number", "value": "71", "role": "vhf_channel"},
    {"claim": "The transmission hails the other vessel by name before identifying own vessel",
     "type": "procedure"},
    {"claim": "The transmission uses the proword THIS IS to introduce the calling vessel",
     "type": "literal", "value": "THIS IS"},
    {"claim": "The transmission ends with the proword OVER",
     "type": "literal", "value": "OVER"},
    {"claim": "Rule 14 governs this head-on encounter",
     "type": "number", "value": "14", "role": "colreg_rule"},
    {"claim": "Own vessel alters course to starboard",
     "type": "direction", "value": "starboard", "role": "turn_direction"},
    {"claim": "The vessels pass port to port",
     "type": "direction", "value": "port", "role": "pass_side"}
  ]

OUTPUT FORMAT
- For EVERY record in the batch return exactly one line of JSON (JSON Lines):
  {"id": "<the record's id, copied verbatim>", "gold_claims": [ ... ]}
- Do NOT echo any other record fields. No commentary, no markdown fences,
  no trailing commas.
- Preserve all Unicode exactly (e.g. "SÉCURITÉ", "°", "–").
- If a record's gold_answer contains an internal contradiction or a number
  that conflicts with its own vhf_channel/colreg_rules fields, still emit the
  claims from the gold_answer, and append a final object
  {"claim": "QA_FLAG: <one-line description of the inconsistency>", "type": "fact"}
  so it can be reviewed.
"""


def progress_path(gold_path: Path) -> Path:
    return gold_path.with_name(gold_path.stem + "_claims_progress.jsonl")


def output_path(gold_path: Path) -> Path:
    return gold_path.with_name(gold_path.stem + "_claims.json")


def parse_response(text: str) -> dict[str, list]:
    """Parse the model's JSONL reply into {id: gold_claims}. Tolerates fences."""
    out: dict[str, list] = {}
    for line in text.splitlines():
        line = line.strip().strip("`").rstrip(",")
        if not line or line in ("json", "[", "]") or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"    [warn] unparseable line skipped: {line[:100]}...", file=sys.stderr)
            continue
        rid, claims = obj.get("id"), obj.get("gold_claims")
        if rid is not None and claims is not None:
            out[str(rid)] = claims
    return out


def enrich_file(client, gold_path: Path, model: str, batch_size: int,
                max_tokens: int, limit: int | None) -> None:
    records = json.loads(gold_path.read_text(encoding="utf-8"))
    if limit:
        records = records[:limit]
    prog = progress_path(gold_path)
    done: dict[str, list] = {str(r["id"]): r["gold_claims"] for r in load_jsonl(prog)} if prog.exists() else {}
    todo = [r for r in records if str(r["id"]) not in done]
    print(f"\n=== {gold_path.name}: {len(records)} records, "
          f"{len(done)} already enriched, {len(todo)} to do ===")

    n_ok = n_bad = 0
    with prog.open("a", encoding="utf-8") as prog_f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            ids = [str(r["id"]) for r in batch]
            t0 = time.time()
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(batch, ensure_ascii=False, indent=1)}],
            )
            got = parse_response(resp.content[0].text)
            for rid in ids:
                claims = got.get(rid)
                errs = validate_claims(claims) if claims is not None else ["missing from response"]
                if errs:
                    n_bad += 1
                    print(f"    [reject] {rid}: {errs[0]} (will retry on next run)")
                    continue
                prog_f.write(json.dumps({"id": rid, "gold_claims": claims}, ensure_ascii=False) + "\n")
                done[rid] = claims
                n_ok += 1
            prog_f.flush()
            print(f"  batch {start // batch_size + 1}: +{len([i for i in ids if i in done])}/{len(ids)} "
                  f"ok ({time.time() - t0:.1f}s, total {len(done)}/{len(records)})")

    print(f"  run result: {n_ok} accepted, {n_bad} rejected")
    missing = [str(r["id"]) for r in records if str(r["id"]) not in done]
    if missing:
        print(f"  {len(missing)} records still missing (rerun to retry): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        return

    # All covered -> assemble final enriched file, original order + fields intact.
    enriched = []
    for r in records:
        r2 = dict(r)  # preserves insertion order; gold_claims appended last
        r2["gold_claims"] = done[str(r["id"])]
        enriched.append(r2)
    out = output_path(gold_path)
    out.write_text(json.dumps(enriched, ensure_ascii=False, indent=1), encoding="utf-8")
    n_flags = sum(1 for r in enriched for c in r["gold_claims"]
                  if str(c.get("claim", "")).startswith("QA_FLAG:"))
    print(f"  WROTE {out} ({len(enriched)} records, {n_flags} QA flags)")
    print(f"  validate with: python -m pipeline.eval.gold_claims --check \"{out}\"")


def main() -> None:
    paths = AgentPaths.vhf()
    load_env(paths.env_file)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        sys.exit("ANTHROPIC_API_KEY not set in .env")

    default_files = [paths.gold_file, paths.eval_file("vhf_colreg_scenarios.json")]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", type=Path, nargs="+", default=default_files,
                    help="Gold files to enrich (default: both VHF eval files)")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N records per file (smoke test)")
    args = ap.parse_args()

    import anthropic
    # Keys not scoped to a workspace must send the workspace ID with every request.
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    headers = {"anthropic-workspace-id": ws} if ws else None
    client = anthropic.Anthropic(api_key=key, default_headers=headers)
    for f in args.files:
        enrich_file(client, f, args.model, args.batch_size, args.max_tokens, args.limit)


if __name__ == "__main__":
    main()
