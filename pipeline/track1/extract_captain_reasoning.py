"""Captain domain, Track 1 (design_captain_missions.md Sec 16.6 step 3) — extract a
structured reasoning trace from every captain_rag_chunks.json chunk.

Mirrors extract_reasoning.py's per-chunk mechanism/resume-safety and
extract_incident_reasoning.py's field-naming conventions (prowords_used/channels reused,
not VHF/COLREG-literal), but with NO collision/COLREG-only pre-filter -- every Captain
chunk (ISM/SOLAS/BMP5/CHIRP/MARS/TSB/Navy-yearbook) is eligible, since the Captain domain
covers general command judgement, not just collision avoidance. Per Sec 16.1's own schema
table, the trace is extended with 3 fields the procedure library (pipeline/captain_agent_spec.py)
is keyed on: mapped_event_type, severity, context_flags (Sec 13.C.10) -- most chunks will
correctly map to "unmapped" (only 5 of Sec 4's ~70 event types are specified in v1); that is
expected, not a failure, and such rows still train the general reasoning/citation pool.

Uses Claude (claude-sonnet-5, Sec 16.4's "a stronger model than VHF/OOW's gpt-4o-mini" --
this is a genuine generative-judgement extraction, not pure extractive mining) via the
Anthropic API. Safe to run locally (API calls only, no GPU/model training).

Output: _cache/captain_reasoning_traces.jsonl
  one JSON line per chunk:
  {
    chunk_id, source_file, chapter_title, section_types, chunk_concepts, chunk_topics,
    trace: {
      situation, trigger, procedures: [{step, action, why}], constraints,
      prowords_used, channels, regulations, warnings, outcomes, key_facts,
      question_seeds: [{angle, text}],
      mapped_event_type, severity, context_flags
    }
  }

Resume-safe: skips chunk_ids already present in the output file.

25-first review gate before a full run (established convention, see this file's siblings):
    python -m pipeline.track1.extract_captain_reasoning --limit 25   # review sample
    python -m pipeline.track1.extract_captain_reasoning              # full run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core import AgentPaths, load_env
from pipeline.captain_agent_spec import PROCEDURE_LIBRARY

paths = AgentPaths.from_env()
W = paths.workspace
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

CHUNKS_FILE = CACHE / f"{_PFX}_rag_chunks.json"
OUT_FILE = CACHE / f"{_PFX}_reasoning_traces.jsonl"

MIN_TOKENS = 40
MAX_WORKERS = 4
MODEL = "claude-sonnet-5"  # Sec 16.4's stronger-than-gpt-4o-mini teacher model

EVENT_TYPES = tuple(PROCEDURE_LIBRARY.keys())  # engine_failure/fog/distress_call/whale_zone/commercial_instruction
SEVERITIES = ("minor", "moderate", "serious", "catastrophic")  # captain_types.py BrownEnvelopeEvent.severity

# Sec 13.C.10's "cheap keyword guard" against a hallucinated mapped_event_type -- each
# event_type requires at least one of its own keywords to appear in the source text,
# otherwise the mapping is downgraded to "unmapped" (never trusted on the LLM's say-so alone).
EVENT_KEYWORD_GUARDS: dict[str, tuple[str, ...]] = {
    "engine_failure": ("engine", "propulsion", "machinery", "breakdown", "power loss", "main engine"),
    "fog": ("fog", "visibility", "restricted visibility", "mist"),
    "distress_call": ("distress", "mayday", "assist", "rescue", "survivor", "sos"),
    "whale_zone": ("whale", "marine mammal", "cetacean", "protected", "speed restriction zone"),
    "commercial_instruction": ("charter", "schedule", "owner", "company instruction", "commercial pressure", "dpa"),
}

SYSTEM_PROMPT = f"""You are extracting the reasoning structure of a ship Captain's command-level
reference excerpt. This may be ISM/SOLAS/MARPOL/STCW statutory text, BMP5 anti-piracy guidance, a
real reported marine incident/near-miss (CHIRP/MARS/TSB), or a Royal Netherlands Navy annual report
narrative. A downstream AI Captain-agent will be trained on the extracted reasoning to make correct,
compliant, and well-justified command decisions during a mission. Extract everything from the
excerpt only -- no external knowledge, no invented details.

Return STRICT JSON with this exact schema:
{{
  "situation": "one sentence describing the scenario/context the excerpt covers",
  "trigger": "what event or condition makes this content apply (or null)",
  "procedures": [
    {{"step": 1, "action": "concise imperative", "why": "rationale grounded in excerpt"}}
  ],
  "constraints": ["precondition or rule that must hold", "..."],
  "prowords_used": ["master", "chief_engineer", "dpa", "..."]  (key roles/actors involved, reused field name -- NOT VHF prowords),
  "channels": ["ISM Code Art. 5", "SOLAS V/34", "..."]  (specific article/regulation numbers engaged, reused field name -- NOT VHF channels),
  "regulations": ["ISM Code", "SOLAS", "MARPOL", "BMP5", "STCW", "..."]  (broader instruments cited),
  "warnings": ["safety warning or prohibition", "..."],
  "outcomes": ["expected or actual result of following/not following the procedure", "..."],
  "key_facts": ["standalone factual statement 1", "standalone factual statement 2"],
  "question_seeds": [
    {{"angle": "what|when|how|why|which|who", "text": "realistic Captain question about THIS excerpt"}}
  ],
  "mapped_event_type": one of {list(EVENT_TYPES) + ["unmapped"]},
  "severity": one of {list(SEVERITIES)}, or null if mapped_event_type is "unmapped",
  "context_flags": ["short_snake_case_tag", "..."]  (0-3 free-form situational tags, e.g. "schedule_pressure", "restricted_visibility", "deadline_pressure"; [] if mapped_event_type is "unmapped")
}}

Rules:
- All fields are required. Use [] for lists that don't apply. Use null only for "trigger"/"severity".
- procedures: only if the excerpt describes actions to take; empty [] otherwise.
- key_facts: 2-5 discrete facts a Captain would need to know.
- question_seeds: 2-4 realistic questions a Captain might ask about THIS excerpt.
- mapped_event_type: only map to one of the 5 named event types if the excerpt is GENUINELY about
  that situation (e.g. a real engine breakdown, real fog/restricted-visibility regime, a real
  distress call, a charted whale/marine-mammal zone, or a commercial-vs-safety instruction
  conflict). Most excerpts (general statutory text, unrelated incident types, historical
  narrative) should correctly be "unmapped" -- do not force a mapping.
- Do not include markdown, code fences, or commentary outside the JSON.
- If the excerpt is boilerplate (page footer, ToC, copyright, url list) return {{"skip": true, "reason": "..."}}"""


def user_prompt(chunk: dict) -> str:
    """Build the user-turn message describing one RAG chunk for the extraction prompt."""
    return (
        f"Source: {chunk['source_file']}\n"
        f"Chapter: {chunk['chapter_title']}\n"
        f"Section types (from parser): {chunk.get('types', [])}\n"
        f"Topics (from parser): {chunk.get('topics', [])}\n"
        f"Concepts (from parser): {chunk.get('concepts', [])}\n\n"
        f"Excerpt:\n\"\"\"\n{chunk['text']}\n\"\"\""
    )


def parse_response(raw: str) -> dict | None:
    """Parse a JSON object from a model response, tolerating surrounding prose."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def cross_validate_event_type(trace: dict, text: str) -> None:
    """Sec 13.C.10's keyword-guard cross-check -- mutates trace in place, downgrading a
    hallucinated/out-of-vocabulary mapped_event_type to "unmapped" either when it isn't one
    of the 5 v1 event types at all, or when none of its guard keywords appear in the source
    text. Records the LLM's original guess for audit rather than silently discarding it."""
    event_type = trace.get("mapped_event_type")
    if event_type != "unmapped" and event_type not in EVENT_TYPES:
        trace["mapped_event_type_llm_guess"] = event_type
        trace["mapped_event_type"] = "unmapped"
    else:
        guard = EVENT_KEYWORD_GUARDS.get(event_type)
        if guard and not any(kw in text.lower() for kw in guard):
            trace["mapped_event_type_llm_guess"] = event_type
            trace["mapped_event_type"] = "unmapped"
    if trace.get("mapped_event_type") == "unmapped":
        trace["severity"] = None
        trace["context_flags"] = []


def _extract_text(resp) -> str:
    """claude-sonnet-5 returns extended-thinking blocks (ThinkingBlock, no `.text`) ahead of
    the actual TextBlock in resp.content -- find the first real text block rather than
    assuming position 0 (see extract_chief_engineer_known_issues_crossval.py's same fix)."""
    for block in resp.content or []:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def process_chunk(client, chunk: dict) -> tuple[str, dict | None, str | None]:
    """Call Claude to extract a reasoning trace from one chunk; returns (chunk_id, trace_or_None, error_or_None)."""
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt(chunk)}],
        ) as stream:
            resp = stream.get_final_message()
        raw = _extract_text(resp)
        obj = parse_response(raw)
        if obj is None:
            return chunk["chunk_id"], None, "parse-failure"
        return chunk["chunk_id"], obj, None
    except Exception as e:
        return chunk["chunk_id"], None, str(e)[:200]


def load_done_ids(path: Path) -> set[str]:
    """Read already-processed chunk_ids from an output JSONL file, for resume support."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    """CLI entry point: mine Captain reasoning traces from RAG chunks via the Anthropic API, with resume support."""
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N eligible chunks (review gate)")
    ap.add_argument("--model", default=MODEL, help=f"Anthropic model id (default: {MODEL})")
    args = ap.parse_args()

    load_env(W / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        sys.exit("ANTHROPIC_API_KEY not set in .env")
    import anthropic
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    headers = {"anthropic-workspace-id": ws} if ws else None
    client = anthropic.Anthropic(api_key=key, default_headers=headers)
    MODEL = args.model

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Total chunks: {len(chunks)}")
    eligible = [c for c in chunks if c["token_count"] >= MIN_TOKENS]
    print(f"Eligible (>={MIN_TOKENS} tok): {len(eligible)}")

    done = load_done_ids(OUT_FILE)
    todo = [c for c in eligible if c["chunk_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Already done: {len(done)}   Todo: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Extracting with {MODEL}, workers={MAX_WORKERS}...")
    t0 = time.time()
    errs = 0
    skipped = 0
    written = 0
    corrected = 0
    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_chunk, client, c): c for c in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                cid, obj, err = fut.result()
                chunk = next(c for c in todo if c["chunk_id"] == cid)
                if err or obj is None:
                    errs += 1
                    f_out.write(json.dumps({
                        "chunk_id": cid, "error": err or "empty", "trace": None,
                    }) + "\n")
                    continue
                if obj.get("skip"):
                    skipped += 1
                    f_out.write(json.dumps({
                        "chunk_id": cid,
                        "source_file": chunk["source_file"],
                        "chapter_title": chunk["chapter_title"],
                        "section_types": chunk["types"],
                        "chunk_concepts": chunk.get("concepts", []),
                        "skip": True,
                        "reason": obj.get("reason", ""),
                        "trace": None,
                    }) + "\n")
                    continue
                before = obj.get("mapped_event_type")
                cross_validate_event_type(obj, chunk["text"])
                if obj.get("mapped_event_type") != before:
                    corrected += 1
                row = {
                    "chunk_id": cid,
                    "source_file": chunk["source_file"],
                    "chapter_title": chunk["chapter_title"],
                    "section_types": chunk["types"],
                    "chunk_concepts": chunk.get("concepts", []),
                    "chunk_topics": chunk.get("topics", []),
                    "trace": obj,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if i % 25 == 0:
                    rate = i / (time.time() - t0)
                    eta = (len(todo) - i) / rate if rate else 0
                    print(f"  {i}/{len(todo)}  rate={rate:.2f}/s  "
                          f"ETA={eta:.0f}s  errs={errs}  skipped={skipped}  corrected={corrected}",
                          flush=True)

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s")
    print(f"  written   : {written}")
    print(f"  skipped   : {skipped}  (boilerplate)")
    print(f"  errors    : {errs}")
    print(f"  corrected : {corrected}  (mapped_event_type downgraded to unmapped by the keyword guard)")
    print(f"\nOutput: {OUT_FILE}  ({OUT_FILE.stat().st_size/1024:.1f} KB)")

    # Sample: first successful trace
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("trace") and not r.get("skip") and not r.get("error"):
                print("\n=== Sample reasoning trace ===")
                print(f"  chunk_id : {r['chunk_id']}")
                print(f"  source   : {r['source_file']}")
                print(f"  chapter  : {r['chapter_title']}")
                t = r["trace"]
                print(f"  situation: {t.get('situation','')[:120]}")
                print(f"  mapped_event_type: {t.get('mapped_event_type')}  severity: {t.get('severity')}  context_flags: {t.get('context_flags')}")
                print(f"  procedures ({len(t.get('procedures', []))} steps):")
                for st in t.get("procedures", [])[:3]:
                    print(f"    {st.get('step')}. {st.get('action','')[:80]} — {st.get('why','')[:80]}")
                print(f"  key_facts:")
                for kf in t.get("key_facts", [])[:3]:
                    print(f"    - {kf[:120]}")
                print(f"  question_seeds:")
                for qs in t.get("question_seeds", [])[:3]:
                    print(f"    [{qs.get('angle','?')}] {qs.get('text','')[:120]}")
                break


if __name__ == "__main__":
    main()
