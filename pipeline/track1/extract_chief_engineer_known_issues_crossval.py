"""Chief Engineer known-issues CROSS-MODEL agreement filter (2026-10-03).

The 655 `llm_synthesized` rows in `chief_engineer_known_issues_traces.jsonl` were generated
by ONE model (gpt-4o-mini). This script answers the SAME synthesis prompts independently with
Claude Sonnet (the user confirmed a working Anthropic key this session, model id
"claude-sonnet-5" -- confirmed available via `client.models.list()`), then keeps ONLY the
GPT-synthesized items that have a semantically-matching Claude-synthesized counterpart for the
same engine system -- a cross-model agreement filter, same spirit as this project's existing
contamination/dedup embedding-similarity checks (core/embedding.py), applied here to filter
LLM-synthesized content down to what two independent models agree is a real, plausible known
problem for this engine family (not a quality guarantee, but a real second opinion).

Two phases, each independently re-runnable (resume-safe):

  1. generate  -- re-runs build_all_seeds()'s SAME synthesize-mode seeds (same grounding
                  context, same prompt) through Claude instead of OpenAI, writing
                  chief_engineer_known_issues_traces_claude.jsonl (own file, never touches
                  the original gpt-4o-mini file).
  2. crossval  -- embeds every GPT item and every Claude item (BAAI/bge-large-en-v1.5, same
                  embedder as the rest of this project) restricted to the SAME engine system,
                  computes each GPT item's max cosine similarity to same-system Claude items,
                  prints the similarity distribution (measure before picking a threshold --
                  same discipline as calibrate_embedder_thresholds.py), and writes
                  chief_engineer_known_issues_traces_agreed.jsonl containing only GPT items
                  above the threshold, each annotated with cross_validated_by/
                  agreement_similarity/matched_claude_chunk_id.

Reuses build_all_seeds()/SYNTHESIZE_SYSTEM_PROMPT/user_prompt_synthesize/parse_response from
extract_chief_engineer_known_issues.py UNCHANGED -- this script only swaps which API answers
the same prompts, never forks the seed-building or prompt logic.

Safe to run LOCALLY (API calls + CPU embedding only, no GPU/model training).

Run with: python -m pipeline.track1.extract_chief_engineer_known_issues_crossval generate
          python -m pipeline.track1.extract_chief_engineer_known_issues_crossval crossval
          python -m pipeline.track1.extract_chief_engineer_known_issues_crossval crossval --threshold 0.80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import numpy as np

from core import load_env
from core.embedding import EMBEDDER_MODEL
from pipeline.track1.extract_chief_engineer_known_issues import (
    CACHE, OUT_FILE, W,
    SYNTHESIZE_SYSTEM_PROMPT, user_prompt_synthesize, parse_response,
    build_all_seeds, load_done_ids,
)

MODEL = "claude-sonnet-5"  # newest Sonnet tier confirmed available via the user's key, 2026-10-03
MAX_WORKERS = 4
CLAUDE_OUT_FILE = CACHE / "chief_engineer_known_issues_traces_claude.jsonl"
AGREED_OUT_FILE = CACHE / "chief_engineer_known_issues_traces_agreed.jsonl"
# Starting point pending calibration (core/embedding.py's own documented convention for every
# new threshold) -- the actual measured similarity distribution is printed before filtering so
# this number can be sanity-checked against real data, not just asserted.
DEFAULT_THRESHOLD = 0.80


def _extract_text(resp) -> str:
    """claude-sonnet-5 returns extended-thinking blocks (ThinkingBlock, no `.text`) ahead of
    the actual TextBlock in resp.content -- resp.content[0] is NOT reliably the answer with
    this model (unlike every older Claude model already used elsewhere in this repo). Find
    the first real text block instead of assuming position 0."""
    for block in resp.content or []:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def process_seed_claude(client: anthropic.Anthropic, seed: dict) -> tuple[str, dict | list | None, str | None]:
    # This installed anthropic SDK (1.5.0)'s messages.create() has no temperature/top_p
    # param at all -- matches the rest of this repo's existing Claude call sites
    # (enrich_gold_claims.py/build_oow_scenarios.py), which also omit it.
    #
    # Empirically confirmed (2026-10-03): a plain (non-streaming) create() call with this
    # synthesis prompt hit the 16000-token output cap EVEN AT 15 items/call (truncated,
    # unparseable JSON) -- claude-sonnet-5's extended thinking + verbose per-item output
    # consumes far more of the budget than expected, and the API REJECTS max_tokens=32000
    # on a non-streaming call outright ("Streaming is required for operations that may take
    # longer than 10 minutes"). Fix: use messages.stream() (required above ~16000 tokens
    # anyway) with a much bigger ceiling so truncation stops being the limiting factor.
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=32000,
            system=SYNTHESIZE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt_synthesize(seed)}],
        ) as stream:
            resp = stream.get_final_message()
        raw = _extract_text(resp)
        obj = parse_response(raw)
        if obj is None:
            return seed["document_id"], None, "parse-failure"
        return seed["document_id"], obj, None
    except Exception as e:
        return seed["document_id"], None, str(e)[:200]


def cmd_generate(args: argparse.Namespace) -> None:
    import os
    load_env(W / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        sys.exit("ANTHROPIC_API_KEY not set in .env")
    # Keys not scoped to a workspace must send the workspace ID with every request
    # (same requirement already documented in pipeline/eval/enrich_gold_claims.py).
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    headers = {"anthropic-workspace-id": ws} if ws else None
    client = anthropic.Anthropic(api_key=key, default_headers=headers)

    seeds = [s for s in build_all_seeds(args.synth_per_system, args.synth_passes) if s["mode"] == "synthesize"]
    # Give Claude's seeds their own document_id namespace so they never collide with the
    # GPT run's "CE-SYN-..." ids in load_done_ids()'s resume check.
    for s in seeds:
        s["document_id"] = s["document_id"].replace("CE-SYN-", "CE-SYN-CLAUDE-")
    if args.limit:
        seeds = seeds[: args.limit]
    print(f"Claude synthesis seeds: {len(seeds)}")

    CACHE.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(CLAUDE_OUT_FILE)
    todo = [s for s in seeds if s["document_id"] not in done]
    print(f"Already done: {len(done)}   Todo: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    t0 = time.time()
    written = errs = 0
    with CLAUDE_OUT_FILE.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_seed_claude, client, s): s for s in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                did, obj, err = fut.result()
                seed = next(s for s in todo if s["document_id"] == did)
                if err or obj is None:
                    errs += 1
                    f_out.write(json.dumps({"chunk_id": did, "source_file": "llm_synthesized_claude",
                                             "error": err or "empty", "trace": None}) + "\n")
                    continue
                items = obj.get("items") if isinstance(obj, dict) else obj
                if not isinstance(items, list):
                    errs += 1
                    f_out.write(json.dumps({"chunk_id": did, "source_file": "llm_synthesized_claude",
                                             "error": "bad-items-shape", "trace": None}) + "\n")
                    continue
                for j, item in enumerate(items):
                    row = {
                        "chunk_id": f"{did}-{j:03d}",
                        "source_file": "llm_synthesized_claude",
                        "chapter_title": seed["chapter_title"],
                        "provenance": "llm_synthesized_claude",
                        "engine_model": seed["engine_model"],
                        "system": seed["system"],
                        "trace": item,
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                f_out.write(json.dumps({"chunk_id": did, "batch_complete": True, "n": len(items)}) + "\n")
                if i % 5 == 0 or i == len(todo):
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)}  ({rate:.2f}/s)  written={written} errs={errs}")

    print(f"\nDone in {time.time()-t0:.0f}s. written={written} errs={errs}")
    print(f"Saved: {CLAUDE_OUT_FILE}")


def _item_text(trace: dict) -> str:
    ki = trace.get("known_issue", {})
    actions = "; ".join(p.get("action", "") for p in (trace.get("procedures") or []))
    return f"{ki.get('failure_mode', '')}. {trace.get('situation', '')} Corrective actions: {actions}"


def _load_rows(path: Path, provenance_filter: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("trace") and r.get("provenance") == provenance_filter:
                rows.append(r)
    return rows


def cmd_crossval(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    if not CLAUDE_OUT_FILE.exists():
        sys.exit(f"{CLAUDE_OUT_FILE} does not exist -- run the 'generate' subcommand first.")

    gpt_rows = _load_rows(OUT_FILE, "llm_synthesized")
    claude_rows = _load_rows(CLAUDE_OUT_FILE, "llm_synthesized_claude")
    # GPT rows don't carry "system" at top level -- it lives under trace.known_issue.system.
    for r in gpt_rows:
        r["system"] = r["trace"].get("known_issue", {}).get("system", "")
    print(f"GPT items: {len(gpt_rows)}   Claude items: {len(claude_rows)}")

    print(f"Loading embedder ({EMBEDDER_MODEL})...")
    model = SentenceTransformer(EMBEDDER_MODEL)
    gpt_texts = [_item_text(r["trace"]) for r in gpt_rows]
    claude_texts = [_item_text(r["trace"]) for r in claude_rows]
    gpt_embs = model.encode(gpt_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    claude_embs = model.encode(claude_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    claude_by_system: dict[str, list[int]] = {}
    for idx, r in enumerate(claude_rows):
        claude_by_system.setdefault(r.get("system", ""), []).append(idx)

    best_sims = np.zeros(len(gpt_rows))
    best_idx = [-1] * len(gpt_rows)
    for i, r in enumerate(gpt_rows):
        cand_idx = claude_by_system.get(r["system"], [])
        if not cand_idx:
            continue
        cand_embs = claude_embs[cand_idx]
        sims = cand_embs @ gpt_embs[i]
        j = int(np.argmax(sims))
        best_sims[i] = float(sims[j])
        best_idx[i] = cand_idx[j]

    print("\nSimilarity distribution (GPT item -> best same-system Claude item):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p}: {np.percentile(best_sims, p):.3f}")
    print(f"  mean: {best_sims.mean():.3f}  min: {best_sims.min():.3f}  max: {best_sims.max():.3f}")

    threshold = args.threshold
    kept = [i for i in range(len(gpt_rows)) if best_sims[i] >= threshold]
    print(f"\nThreshold {threshold}: keeping {len(kept)}/{len(gpt_rows)} GPT items "
          f"({100*len(kept)/max(len(gpt_rows),1):.1f}%)")

    with AGREED_OUT_FILE.open("w", encoding="utf-8") as f_out:
        for i in kept:
            row = dict(gpt_rows[i])
            row["cross_validated_by"] = MODEL
            row["agreement_similarity"] = round(float(best_sims[i]), 4)
            row["matched_claude_chunk_id"] = claude_rows[best_idx[i]]["chunk_id"]
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved: {AGREED_OUT_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Answer the same synthesis prompts with Claude.")
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument("--synth-per-system", type=int, default=20,
                      help="Items requested per call. Streaming + max_tokens=32000 (see "
                           "process_seed_claude) gives enough headroom for this at 20 items/call.")
    gen.add_argument("--synth-passes", type=int, default=3,
                      help="Independent passes per engine system (matches the GPT run's 3).")
    gen.set_defaults(func=cmd_generate)

    cv = sub.add_parser("crossval", help="Embed + filter GPT items to those Claude agrees with.")
    cv.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    cv.set_defaults(func=cmd_crossval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
