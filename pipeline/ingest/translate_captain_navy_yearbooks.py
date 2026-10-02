"""Translates Data/Captain/Captain_JSON_NL/*.json (Royal Netherlands Navy yearbooks,
Dutch -- see build_captain_navy_yearbooks_json.py) section-by-section into English,
writing the result into Data/Captain/Captain_JSON/ -- the directory build_rag.py
actually globs. This is the ONLY step that turns this source's Dutch extraction into
something the RAG pipeline can use; RAG chunks must be in English, matching every
other Captain/VHF/OOW source.

Translation granularity: PER SECTION (one yearbook page each -- a few hundred words),
via OpenAI gpt-4o-mini (same model tier as every other extraction script in this
project), so each unit is cheap, independently resumable (checkpointed in a JSONL
cache keyed by document_id/section_id, same load_done_ids() convention as
extract_incident_reasoning.py), and trivially parallelizable. Estimated cost for all 7
yearbooks (~341K words, ~1,400 sections): well under $1 total at gpt-4o-mini rates.

A document is only written into Captain_JSON/ once EVERY one of its sections has a
cached translation -- never a partial document mixing Dutch and English text (which
would confuse an English-tuned embedder). Re-run without --limit to fill any gaps.

25-first review gate (same convention as every other [LLM] step in this project):
    python -m pipeline.ingest.translate_captain_navy_yearbooks --limit 25   # review sample
    python -m pipeline.ingest.translate_captain_navy_yearbooks              # full run
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from core import AgentPaths, load_env

paths = AgentPaths.captain()
W = paths.workspace
JSON_NL_DIR = paths.data_root / "Captain_JSON_NL"
JSON_EN_DIR = paths.json_dir  # Captain_JSON/ -- what build_rag.py actually globs
CACHE_FILE = paths.cache_dir / "captain_navy_yearbooks_translation_cache.jsonl"

MAX_WORKERS = 4
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are translating an excerpt from an official Royal Netherlands Navy
(Koninklijke Marine) annual report from Dutch into natural, fluent English. This is real
operational/administrative prose (ship deployments, exercises, maintenance, personnel,
Caribbean operations) -- translate it faithfully, preserving every concrete fact (names,
dates, ranks, ship names, exercise names, locations, numbers). Do not summarize, omit, or
add commentary -- translate the full text.

Keep ship names as-is (e.g. "Zr.Ms. Rotterdam" stays "Zr.Ms. Rotterdam"). For Dutch
military rank abbreviations, translate to the closest English-language equivalent rank
followed by the original Dutch abbreviation in parentheses on first use in a section, e.g.
"Commander (kltz)" / "Captain (ktz)" / "Lieutenant Colonel (ltkolmarns)". Common
abbreviations: kltz=kapitein-luitenant ter zee (Lieutenant Commander/Commander), ktz=
kapitein ter zee (Captain), ltkolmarns=luitenant-kolonel der mariniers (Lieutenant
Colonel, Marines), CZSK=Commando Zeestrijdkrachten (Royal Netherlands Navy Command). For
any other Dutch organizational abbreviation you are not confident about, keep it as-is
(do not guess at an expansion).

Respond with ONLY the translated English text -- no preamble, no notes, no markdown."""


def translate_section(
    client: OpenAI, document_id: str, section_id: str, text: str,
) -> tuple[str, str, str | None, str | None]:
    """Returns (document_id, section_id, translated_text_or_None, error_or_None)."""
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.2,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": text}],
            max_tokens=2000,
        )
        translated = (resp.choices[0].message.content or "").strip()
        if not translated:
            return document_id, section_id, None, "empty-response"
        return document_id, section_id, translated, None
    except Exception as e:
        return document_id, section_id, None, str(e)[:200]


def load_done_keys(path: Path) -> dict[str, str]:
    """{"document_id::section_id": translated_text} loaded from the checkpoint cache."""
    done: dict[str, str] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("translated_text"):
                    done[f"{row['document_id']}::{row['section_id']}"] = row["translated_text"]
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_nl_sections(limit: int | None) -> list[dict]:
    """Flat list of {document_id, section_id, text} across every parsed yearbook doc."""
    out = []
    for path in sorted(JSON_NL_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for chapter in doc["chapters"]:
            for section in chapter["sections"]:
                out.append({"document_id": doc["document_id"], "section_id": section["section_id"],
                           "text": section["text"]})
    if limit:
        out = out[:limit]
    return out


def write_assembled_english_docs(done: dict[str, str]) -> tuple[int, int]:
    """Writes Captain_JSON/<doc_id>.json for every yearbook doc whose sections are ALL
    present in `done` -- never a partial document mixing Dutch and English text. Returns
    (n_written, n_incomplete)."""
    JSON_EN_DIR.mkdir(parents=True, exist_ok=True)
    written = incomplete = 0
    for path in sorted(JSON_NL_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        missing = 0
        for chapter in doc["chapters"]:
            for section in chapter["sections"]:
                key = f"{doc['document_id']}::{section['section_id']}"
                if key in done:
                    section["text"] = done[key]
                else:
                    missing += 1
        if missing:
            incomplete += 1
            print(f"  [incomplete] {doc['document_id']}: {missing} section(s) still untranslated -- not written")
            continue
        doc["language"] = "en"
        doc["translated_from"] = "nl"
        out = JSON_EN_DIR / f"{doc['document_id']}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
        print(f"  {doc['source_file']:<35} -> {out.name}")
    return written, incomplete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="25-first review gate: only translate the first N sections (smoke-testing); "
                         "does NOT write the assembled Captain_JSON/ output.")
    args = ap.parse_args()

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing from .env")
    client = OpenAI()

    sections = load_nl_sections(args.limit)
    print(f"Sections to translate (requested scope): {len(sections)}")

    done = load_done_keys(CACHE_FILE)
    todo = [s for s in sections if f"{s['document_id']}::{s['section_id']}" not in done]
    print(f"Already translated (cached): {len(done)}   Todo: {len(todo)}")

    if todo:
        t0 = time.time()
        errs = 0
        with CACHE_FILE.open("a", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {
                    ex.submit(translate_section, client, s["document_id"], s["section_id"], s["text"]): s
                    for s in todo
                }
                for i, fut in enumerate(as_completed(futures), 1):
                    doc_id, sec_id, translated, err = fut.result()
                    if err or translated is None:
                        errs += 1
                        print(f"  [err] {doc_id}/{sec_id}: {err}")
                        continue
                    f_out.write(json.dumps(
                        {"document_id": doc_id, "section_id": sec_id, "translated_text": translated},
                        ensure_ascii=False) + "\n")
                    f_out.flush()
                    if i % 50 == 0 or i == len(todo):
                        rate = i / (time.time() - t0)
                        print(f"  {i}/{len(todo)}  ({rate:.1f}/s)  errs={errs}")
        print(f"Translation pass done in {time.time()-t0:.0f}s, errs={errs}")

    if args.limit:
        print(f"\n--limit {args.limit} set -- review the cache file before a full run: {CACHE_FILE}")
        return

    done = load_done_keys(CACHE_FILE)
    written, incomplete = write_assembled_english_docs(done)
    print(f"\n{written} English-language documents written -> {JSON_EN_DIR}")
    if incomplete:
        print(f"{incomplete} document(s) still incomplete -- re-run without --limit to fill gaps.")


if __name__ == "__main__":
    main()
