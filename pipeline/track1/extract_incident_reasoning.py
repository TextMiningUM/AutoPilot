"""Track 1 (incidents) — extract structured COLREG-relevant reasoning from
each shortlisted incident-report excerpt (see build_incident_excerpts.py).

Unlike extract_reasoning.py (which mines one reasoning trace per ~400-token
RAG chunk of the COLREG rule text), each incident excerpt is already a single
bounded, coherent unit (report opening + its Analysis/Conclusions pages), so
here we extract ONE trace per incident document, not per chunk.

Output schema reuses the VHF/OOW reasoning-trace field names (situation,
trigger, procedures, constraints, prowords_used, channels, regulations,
warnings, outcomes, key_facts, question_seeds) so build_sft.py / build_multihop.py
/ build_rlhf.py / build_reflection.py need NO changes to consume these traces
alongside the rule-text ones. Additional incident-specific fields are nested
under "incident" and are additive (ignored by the existing builders, but kept
for a possible dedicated incident-DPO builder later: real actual_actions_taken
vs. correct_actions is a ready-made chosen/rejected pair).

Output: _cache/oow_incident_reasoning_traces.jsonl
  one JSON line per incident document:
  {
    document_id, source_file, screening_net_score,
    trace: {
      situation, trigger, procedures, constraints, prowords_used, channels,
      regulations, warnings, outcomes, key_facts, question_seeds,
      incident: {
        vessels, situation_type, colreg_rules_applicable,
        actual_actions_taken, fault_attribution, fault_type,
        avoidance_summary, exceptional_circumstances, confidence,
        outcome_severity
      }
    }
  }

Resume-safe: skips document_ids already present in the output file.

Run with: python -m pipeline.track1.extract_incident_reasoning [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from core import AgentPaths, load_env

paths = AgentPaths.from_env()
W = paths.workspace
JSON_DIR = paths.json_dir
CACHE = paths.cache_dir
_PFX = paths.domain.lower()

OUT_FILE = CACHE / f"{_PFX}_incident_reasoning_traces.jsonl"

MAX_WORKERS = 4
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are extracting the COLREG (collision-avoidance / Rule-of-the-Road) reasoning
structure from an excerpt of a real marine accident-investigation report (opening summary +
Analysis/Conclusions/Findings pages). A downstream Officer-of-the-Watch AI assistant will be
trained on this to recognise real collision-risk situations and apply COLREG correctly. Extract
everything from the excerpt only -- no external knowledge, no invented details.

Return STRICT JSON with this exact schema:
{
  "situation": "2-3 sentence neutral summary of the incident (vessels, context, what happened)",
  "trigger": "the situational trigger that made COLREG apply (e.g. 'crossing situation between two power-driven vessels in restricted visibility') or null",
  "procedures": [
    {"step": 1, "action": "the CORRECT action that should have been taken", "why": "COLREG rule/rationale grounding it"}
  ],
  "constraints": ["applicable COLREG rule stated as a requirement, e.g. 'Rule 15 obliges the vessel with the other on her own starboard side to keep out of the way'"],
  "prowords_used": ["give-way vessel", "stand-on vessel", "..."]  (vessel-role tags, reused field name -- NOT VHF prowords),
  "channels": ["Rule 5", "Rule 15", "..."]  (COLREG rule numbers engaged, reused field name -- NOT VHF channels),
  "regulations": ["COLREG 1972", "..."]  (instruments/annexes cited),
  "warnings": ["root cause / what went wrong, framed as a cautionary statement", "..."],
  "outcomes": ["actual outcome/severity", "official recommendation issued, if any", "..."],
  "key_facts": ["standalone factual statement (vessel types, visibility, distances, timings)", "..."],
  "question_seeds": [
    {"angle": "what|when|how|why|which|who", "text": "realistic OOW question about THIS incident"}
  ],
  "incident": {
    "vessels": [{"name": "...", "type": "e.g. bulk carrier / fishing vessel / yacht", "role": "give-way|stand-on|both|unclear|not-applicable"}],
    "situation_type": "crossing|head-on|overtaking|restricted-visibility|narrow-channel|traffic-separation-scheme|anchored|other",
    "colreg_rules_applicable": ["Rule 5", "Rule 15", "..."],
    "actual_actions_taken": [{"actor": "vessel name or role", "action": "what they ACTUALLY did (the failure)"}],
    "fault_attribution": "give-way vessel|stand-on vessel|both|third-party/other|undetermined",
    "fault_type": ["lookout_failure", "distraction", "equipment_failure", "procedural_failure", "communication_failure", "fatigue", "environmental", "other"],
    "avoidance_summary": "1-2 sentence counterfactual: what, done differently and when, would have avoided the incident",
    "exceptional_circumstances": ["unusual factor, e.g. defective AIS, distress situation, pilot error"]  (empty list if none -- do not omit),
    "confidence": "high|partial|undetermined"  (undetermined/partial if the source itself says one side's actions/reasoning could not be established),
    "outcome_severity": "near-miss|collision-no-damage|collision-damage|collision-fatality|pollution|other"
  }
}

Rules:
- All top-level fields are required. Use [] for lists that don't apply, null only for "trigger".
- "incident" is required and every one of its sub-fields is required (use "unclear"/"undetermined"/[] rather than omitting).
- procedures / actual_actions_taken: ground every action in the excerpt text, do not invent detail.
- key_facts: 2-5 discrete facts an Officer of the Watch would need to know.
- question_seeds: 2-4 realistic questions an OOW might ask, framed around THIS incident's specifics.
- Do not include markdown, code fences, or commentary outside the JSON.
- If the excerpt turns out to NOT be about a vessel-vs-vessel collision-avoidance situation (e.g. it's actually about grounding, fire, flooding, or machinery failure with no COLREG angle), return {"skip": true, "reason": "..."}"""


def user_prompt(doc: dict) -> str:
    text_parts = []
    for ch in doc.get("chapters", []):
        for sec in ch.get("sections", []):
            text_parts.append(f"[{sec['title']}]\n{sec['text']}")
    full_text = "\n\n".join(text_parts)
    return (
        f"Source report: {doc['source_file']}\n"
        f"Screening relevance score: {doc.get('screening_net_score')}\n\n"
        f"Excerpt:\n\"\"\"\n{full_text}\n\"\"\""
    )


def chapter_title_for(doc: dict) -> str:
    """Human-readable stand-in for 'chapter_title' (Track 1's rule-text field name,
    reused so build_sft.py/build_multihop.py/build_rlhf.py/build_reflection.py need no changes)."""
    stem = Path(doc["source_file"]).stem
    return f"Incident report: {stem}"


def chunk_concepts_for(doc: dict) -> list[str]:
    """Union of the tag_text() concept tags already computed per-section in
    build_incident_excerpts.py -- same CONCEPT_KEYWORDS vocabulary as the COLREG rule-text
    parser, so build_multihop.py's concept_signature() can pair an incident excerpt with a
    rule-text chunk (different source documents, shared concept)."""
    concepts: set[str] = set()
    for ch in doc.get("chapters", []):
        for sec in ch.get("sections", []):
            concepts.update(sec.get("concepts", []))
    return sorted(concepts)


def parse_response(raw: str) -> dict | None:
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


def process_doc(client: OpenAI, doc: dict) -> tuple[str, dict | None, str | None]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(doc)},
            ],
            max_tokens=1800,
        )
        raw = resp.choices[0].message.content or ""
        obj = parse_response(raw)
        if obj is None:
            return doc["document_id"], None, "parse-failure"
        return doc["document_id"], obj, None
    except Exception as e:
        return doc["document_id"], None, str(e)[:200]


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["document_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_incident_docs(limit: int | None) -> list[dict]:
    docs = []
    for path in sorted(JSON_DIR.glob("incident_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("source_type") == "incident_report":
            docs.append(doc)
    if limit:
        docs = docs[:limit]
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N incident docs (smoke-testing).")
    args = ap.parse_args()

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing from .env")
    client = OpenAI()

    docs = load_incident_docs(args.limit)
    print(f"Incident documents found: {len(docs)}")

    done = load_done_ids(OUT_FILE)
    todo = [d for d in docs if d["document_id"] not in done]
    print(f"Already done: {len(done)}   Todo: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Extracting with {MODEL}, workers={MAX_WORKERS}...")
    t0 = time.time()
    errs = skipped = written = 0
    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_doc, client, d): d for d in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                did, obj, err = fut.result()
                doc = next(d for d in todo if d["document_id"] == did)
                if err or obj is None:
                    errs += 1
                    f_out.write(json.dumps({
                        "document_id": did, "source_file": doc["source_file"],
                        "error": err or "empty", "trace": None,
                    }) + "\n")
                    continue
                if obj.get("skip"):
                    skipped += 1
                    f_out.write(json.dumps({
                        "document_id": did, "source_file": doc["source_file"],
                        "skip": True, "reason": obj.get("reason", ""), "trace": None,
                    }) + "\n")
                    continue
                row = {
                    "document_id": did,
                    "chunk_id": did,
                    "source_file": doc["source_file"],
                    "chapter_title": chapter_title_for(doc),
                    "chunk_concepts": chunk_concepts_for(doc),
                    "screening_net_score": doc.get("screening_net_score"),
                    "trace": obj,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if i % 10 == 0 or i == len(todo):
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)}  ({rate:.2f}/s)  written={written} skipped={skipped} errs={errs}")

    print(f"\nDone in {time.time()-t0:.0f}s. written={written} skipped={skipped} errs={errs}")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
