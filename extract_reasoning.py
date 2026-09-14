"""Tutorial 13 § 9 (step 1) — extract structured reasoning from every chunk.

Uses gpt-4o-mini to convert each chunk of the 30 VHF protocol / instruction
documents into a rich reasoning trace. This is the SINGLE LLM pass; all downstream
SFT / multi-hop / RLHF / reflection datasets are derived deterministically from
these traces (no more LLM calls after this).

Output: _cache/vhf_reasoning_traces.jsonl
  one JSON line per chunk:
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

Resume-safe: skips chunk_ids already present in the output file.

The 540 gold questions in vhf_gold_answers.json are NOT touched here.
They are held out for evaluation only.
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"

CHUNKS_FILE = CACHE / "vhf_rag_chunks.json"
OUT_FILE    = CACHE / "vhf_reasoning_traces.jsonl"

MIN_TOKENS  = 40
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


SYSTEM_PROMPT = """You are extracting the reasoning structure of a VHF marine-radio reference excerpt.
A downstream AI assistant will be trained on the extracted reasoning to answer mariner questions
correctly. Extract everything from the excerpt only — no external knowledge.

Return STRICT JSON with this exact schema:
{
  "situation": "one sentence describing the scenario/context the excerpt covers",
  "trigger": "what event or condition makes this content apply (or null)",
  "procedures": [
    {"step": 1, "action": "concise imperative", "why": "rationale grounded in excerpt"}
  ],
  "constraints": ["precondition or rule that must hold", "..."],
  "prowords_used": ["MAYDAY", "OVER", "..."],
  "channels": ["16", "70", "..."],
  "regulations": ["ITU RR", "GMDSS", "SOLAS", "Ofcom", "..."],
  "warnings": ["safety warning or prohibition", "..."],
  "outcomes": ["expected result of following the procedure", "..."],
  "key_facts": ["standalone factual statement 1", "standalone factual statement 2"],
  "question_seeds": [
    {"angle": "what|when|how|why|which|who", "text": "realistic mariner question"}
  ]
}

Rules:
- All fields are required. Use [] for lists that don't apply. Use null only for "trigger" if none.
- procedures: only if the excerpt describes actions to take; empty [] otherwise
- key_facts: 2-5 discrete facts a mariner would want to know
- question_seeds: 2-4 realistic questions a mariner might ask about THIS excerpt
- Do not include markdown, code fences, or commentary outside the JSON.
- If the excerpt is boilerplate (page footer, ToC, copyright, url list) return {"skip": true, "reason": "..."}"""


def user_prompt(chunk: dict) -> str:
    return (
        f"Source: {chunk['source_file']}\n"
        f"Chapter: {chunk['chapter_title']}\n"
        f"Section types (from parser): {chunk.get('types', [])}\n"
        f"Topics (from parser): {chunk.get('topics', [])}\n"
        f"Concepts (from parser): {chunk.get('concepts', [])}\n\n"
        f"Excerpt:\n\"\"\"\n{chunk['text']}\n\"\"\""
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


def process_chunk(client: OpenAI, chunk: dict) -> tuple[str, dict | None, str | None]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt(chunk)},
            ],
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content or ""
        obj = parse_response(raw)
        if obj is None:
            return chunk["chunk_id"], None, "parse-failure"
        return chunk["chunk_id"], obj, None
    except Exception as e:
        return chunk["chunk_id"], None, str(e)[:200]


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

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Total chunks: {len(chunks)}")
    eligible = [c for c in chunks if c["token_count"] >= MIN_TOKENS]
    print(f"Eligible (>={MIN_TOKENS} tok): {len(eligible)}")

    done = load_done_ids(OUT_FILE)
    todo = [c for c in eligible if c["chunk_id"] not in done]
    print(f"Already done: {len(done)}   Todo: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Extracting with {MODEL}, workers={MAX_WORKERS}...")
    t0 = time.time()
    errs = 0
    skipped = 0
    written = 0
    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_chunk, client, c): c for c in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                cid, obj, err = fut.result()
                if err or obj is None:
                    errs += 1
                    f_out.write(json.dumps({
                        "chunk_id": cid, "error": err or "empty", "trace": None,
                    }) + "\n")
                    continue
                chunk = next(c for c in todo if c["chunk_id"] == cid)
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
                row = {
                    "chunk_id":       cid,
                    "source_file":    chunk["source_file"],
                    "chapter_title": chunk["chapter_title"],
                    "section_types":  chunk["types"],
                    "chunk_concepts": chunk.get("concepts", []),
                    "chunk_topics":   chunk.get("topics", []),
                    "trace":          obj,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if i % 25 == 0:
                    rate = i / (time.time() - t0)
                    eta = (len(todo) - i) / rate if rate else 0
                    print(f"  {i}/{len(todo)}  rate={rate:.1f}/s  "
                          f"ETA={eta:.0f}s  errs={errs}  skipped={skipped}",
                          flush=True)

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s")
    print(f"  written : {written}")
    print(f"  skipped : {skipped}  (boilerplate)")
    print(f"  errors  : {errs}")
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
                print(f"  trigger  : {t.get('trigger','')}")
                print(f"  channels : {t.get('channels', [])}")
                print(f"  prowords : {t.get('prowords_used', [])}")
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
