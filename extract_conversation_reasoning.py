"""extract_conversation_reasoning.py -- structured reasoning traces from Track 2
conversation training data (vhf_conversations.jsonl).

WHY THIS EXISTS
---------------
vhf_conversations.jsonl (360 multi-turn incoming/outgoing dialogues) currently
only trains the model on raw dialogue style. It never flows through the same
RAG / KG / SFT-variants / DPO / Reflection machinery that the VHF protocol
documents go through (see extract_reasoning.py + build_sft.py/build_rlhf.py/
build_reflection.py/build_multihop.py) -- so all the "agentic" derived training
signal (retrieval-augmented Q&A, preference pairs, self-critique triples) that
Track 1 gets, Track 2 never did.

This script closes that gap WITHOUT touching the held-out eval sets: it reads
the 360 TRAINING conversations only (never vhf_colreg_scenarios.json, which
stays 100% held out for Track 2 evaluation) and extracts ONE reasoning trace
per conversation, in the EXACT SAME schema extract_reasoning.py produces:

  {
    chunk_id, source_file, chapter_title, section_types, chunk_concepts,
    trace: {
      situation, trigger,
      procedures: [ {step, action, why} ],
      constraints, prowords_used, channels, regulations,
      warnings, outcomes, key_facts,
      question_seeds: [ {angle, text} ]
    }
  }

Because the schema is identical, build_sft.py / build_rlhf.py / build_reflection.py
/ build_multihop.py can be pointed at this file (via --traces-file) completely
unchanged, producing Track-2-labeled outputs (vhf_colreg_sft_*.jsonl,
vhf_colreg_dpo_pairs.jsonl, vhf_colreg_reflection.jsonl) that feed the SAME
fine-tune as Track 1 -- see notebook § 12.6.

Known fields (colreg_rules, vhf_channel, category) are passed to the LLM as
hints to ground extraction and reduce hallucination; everything else is
extracted from the actual dialogue text.

Output: Data/VHF/VHF_Agents_Training/vhf_conversation_traces.jsonl
Resume-safe: skips chunk_ids already present in the output file.
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"

CONVERSATIONS_FILE = CACHE / "vhf_conversations.jsonl"
OUT_FILE            = CACHE / "vhf_conversation_traces.jsonl"

MAX_WORKERS = 6
MODEL       = "gpt-4o-mini"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


SYSTEM_PROMPT = """You are extracting the reasoning structure of a VHF collision-avoidance \
radio conversation between an AI "Auto Pilot" and another vessel/VTS. A downstream AI will be \
trained on the extracted reasoning to answer mariner questions correctly. Extract everything \
from the conversation and the given hints only -- no external knowledge.

Return STRICT JSON with this exact schema:
{
  "situation": "one sentence describing the encounter (vessels, geometry, COLREG rule engaged)",
  "trigger": "what event or condition made the Auto Pilot act or respond (or null)",
  "procedures": [
    {"step": 1, "action": "concise imperative describing what the Auto Pilot actually did/said", "why": "COLREG-grounded rationale"}
  ],
  "constraints": ["precondition or rule that must hold", "..."],
  "prowords_used": ["OVER", "THIS IS", "..."],
  "channels": ["16", "66", "..."],
  "regulations": ["Rule 14", "Rule 34", "..."],
  "warnings": ["safety warning or prohibition implied by the exchange, if any", "..."],
  "outcomes": ["what the exchange achieved (e.g. both vessels agreed to alter course to starboard)", "..."],
  "key_facts": ["standalone factual statement 1", "standalone factual statement 2"],
  "question_seeds": [
    {"angle": "what|when|how|why|which|who", "text": "realistic mariner question this conversation could train an answer for"}
  ]
}

Rules:
- All fields are required. Use [] for lists that don't apply. Use null only for "trigger" if none.
- procedures: one entry per Auto Pilot (assistant) transmission/action in the conversation, in order.
- channels/regulations: ground these in the provided hints AND anything explicitly mentioned in the dialogue.
- key_facts: 2-5 discrete facts a mariner would want to know from this exchange.
- question_seeds: 2-4 realistic questions a mariner might ask that this conversation answers.
- Do not include markdown, code fences, or commentary outside the JSON."""


def user_prompt(rec: dict) -> str:
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in rec["messages"]
    )
    return (
        f"Category: {rec.get('category')}\n"
        f"Direction: {rec.get('direction')} (incoming = other station hailed the Auto Pilot first; "
        f"outgoing = the Auto Pilot initiated the call)\n"
        f"COLREG rules engaged (hint): {rec.get('colreg_rules')}\n"
        f"VHF channels (hint): {rec.get('vhf_channel')}\n"
        f"Own vessel: {rec.get('own_vessel')}\n"
        f"Target vessel: {rec.get('target_vessel')}\n\n"
        f"Conversation transcript:\n\"\"\"\n{transcript}\n\"\"\""
    )


def parse_response(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def process_conversation(client: OpenAI, rec: dict) -> tuple[str, dict | None, str | None]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt(rec)},
            ],
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content or ""
        obj = parse_response(raw)
        if obj is None:
            return rec["id"], None, "parse-failure"
        return rec["id"], obj, None
    except Exception as e:
        return rec["id"], None, str(e)[:200]


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["chunk_id"])
            except Exception:
                pass
    return done


def main():
    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing from .env")
    client = OpenAI()

    conversations = list(load_jsonl(CONVERSATIONS_FILE))
    print(f"Total conversations: {len(conversations)}")

    done = load_done_ids(OUT_FILE)
    todo = [r for r in conversations if r["id"] not in done]
    print(f"Already done: {len(done)}   Todo: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Extracting with {MODEL}, workers={MAX_WORKERS}...")
    t0 = time.time()
    errs = 0
    written = 0
    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_conversation, client, r): r for r in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                rid, obj, err = fut.result()
                if err or obj is None:
                    errs += 1
                    f_out.write(json.dumps({"chunk_id": rid, "error": err or "empty", "trace": None}) + "\n")
                    continue
                rec = next(r for r in todo if r["id"] == rid)
                row = {
                    "chunk_id":       rid,
                    "source_file":    "vhf_conversations.jsonl",
                    "chapter_title":  f"{rec.get('category')} ({rec.get('direction')})",
                    "section_types":  ["dialogue"],
                    "chunk_concepts": list(rec.get("colreg_rules") or []) + [rec.get("category", "")],
                    "chunk_topics":   [rec.get("category", ""), rec.get("direction", "")],
                    "trace":          obj,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if i % 25 == 0:
                    rate = i / (time.time() - t0)
                    eta = (len(todo) - i) / rate if rate else 0
                    print(f"  {i}/{len(todo)}  rate={rate:.1f}/s  ETA={eta:.0f}s  errs={errs}", flush=True)

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s")
    print(f"  written : {written}")
    print(f"  errors  : {errs}")
    print(f"\nOutput: {OUT_FILE}  ({OUT_FILE.stat().st_size/1024:.1f} KB)")

    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("trace") and not r.get("error"):
                print("\n=== Sample conversation trace ===")
                print(f"  chunk_id : {r['chunk_id']}")
                print(f"  chapter  : {r['chapter_title']}")
                t = r["trace"]
                print(f"  situation: {t.get('situation','')[:120]}")
                print(f"  channels : {t.get('channels', [])}")
                print(f"  regulations: {t.get('regulations', [])}")
                print(f"  procedures ({len(t.get('procedures', []))} steps):")
                for st in t.get("procedures", [])[:3]:
                    print(f"    {st.get('step')}. {st.get('action','')[:80]} -- {st.get('why','')[:80]}")
                print(f"  question_seeds:")
                for qs in t.get("question_seeds", [])[:3]:
                    print(f"    [{qs.get('angle','?')}] {qs.get('text','')[:120]}")
                break


if __name__ == "__main__":
    main()
