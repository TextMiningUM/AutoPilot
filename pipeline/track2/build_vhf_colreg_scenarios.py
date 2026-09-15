"""
================================================================================
build_vhf_colreg_scenarios.py — realistic VHF + COLREG collision-avoidance eval set
================================================================================

WHY THIS EXISTS
---------------
vhf_gold_answers.json (Data/VHF/VHF_Eval/) is built from SRC-license EXAM questions —
it tests *knowledge about* VHF rules ("what is Channel 70 used for?"), not whether an
agent can actually *conduct a VHF radio exchange* in a live collision-avoidance situation
while applying COLREG correctly.

This script builds a SECOND, complementary eval set: realistic ship-encounter scenarios
(head-on, crossing, overtaking, narrow channel, restricted visibility, TSS, vessel not
under command, etc.) grounded in the actual COLREG rule text (Data/OfficeroftheWatch/
colregs_all.json), each requiring the agent to:
  1. pick the correct VHF channel(s) and procedure (hailing on 16, working channel,
     low power where conventional, etc.)
  2. produce an actual radio transmission (prowords, "THIS IS ... OVER", station ID)
  3. take the COLREG-correct collision-avoidance action (give-way/stand-on, alter
     course/speed, sound signal where relevant) and justify it by rule number.

Output: Data/VHF/VHF_Eval/vhf_colreg_scenarios.json
Schema per record: see `RECORD_SCHEMA_DOC` below.

USAGE
-----
    python build_vhf_colreg_scenarios.py --smoke        # ~10 records, quick sanity check
    python build_vhf_colreg_scenarios.py                # full 500, batched LLM calls
    python build_vhf_colreg_scenarios.py --n 500 --resume  # top up to 500, skip existing
"""
from __future__ import annotations
import os, json, argparse, random, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from core import AgentPaths, load_env

paths = AgentPaths.vhf()
W = paths.workspace
COLREG_FILE = W / "Data" / "OfficeroftheWatch" / "colregs_all.json"
OUT_DIR     = paths.eval_dir
OUT_FILE    = OUT_DIR / "vhf_colreg_scenarios.json"

MODEL       = "gpt-4o-mini"
BATCH_SIZE  = 5     # scenarios generated per LLM call
MAX_WORKERS = 6

RECORD_SCHEMA_DOC = """
{
  "id": "vhfcol_00001",
  "category": "head_on",                     # one of CATEGORIES below
  "colreg_rules": ["Rule 14"],                # rule numbers actually engaged
  "region": "English Channel / Dover Strait", # geographic flavour, drives channel choice
  "own_vessel": "container ship OCEAN VOYAGER, 300m, bound west",
  "target_vessel": "tanker NORDIC STAR, bearing 005, range 4 NM, closing",
  "scenario": "2-4 sentence situational briefing: positions, bearings, CPA/TCPA, geography",
  "vhf_channel": {
    "hailing": "16",
    "working": "13",
    "settings": "1 W low power once on working channel; continue Ch16 watch",
    "note": "regional convention -- verify local Channel Plan / VTS instructions"
  },
  "question": "As the Auto Pilot aboard <own_vessel>, what do you transmit over VHF and what collision-avoidance action do you take, and why?",
  "expected_points": ["...", "..."],           # 4-7 atomic facts a correct answer needs
  "gold_answer": "full natural-language answer: actual radio script (THIS IS / OVER)
                   + the COLREG-justified manoeuvre, in fluent prose (NOT label:value dumps)"
}
"""

# ── Encounter categories, weighted toward realistic Auto-Pilot use cases ───────
CATEGORIES: list[dict] = [
    {"name": "head_on",                    "rules": [14, 34],         "n": 32},
    {"name": "crossing_give_way",          "rules": [15, 16],         "n": 32},
    {"name": "crossing_stand_on",          "rules": [15, 17],         "n": 32},
    {"name": "overtaking_open_water",      "rules": [13, 16],         "n": 28},
    {"name": "overtaking_narrow_channel",  "rules": [9, 13, 34],      "n": 28},
    {"name": "narrow_channel_meeting",     "rules": [9, 34],          "n": 28},
    {"name": "tss_crossing",               "rules": [10],             "n": 28},
    {"name": "restricted_visibility",      "rules": [19, 35, 7],      "n": 32},
    {"name": "vessel_not_under_command",   "rules": [18, 3],          "n": 26},
    {"name": "restricted_manoeuvre",       "rules": [18, 3],          "n": 26},
    {"name": "constrained_by_draught",     "rules": [18, 3],          "n": 24},
    {"name": "fishing_vessel_encounter",   "rules": [18],             "n": 24},
    {"name": "sailing_vessel_encounter",   "rules": [12, 18],         "n": 24},
    {"name": "anchored_vessel_hail",       "rules": [9, 2],           "n": 24},
    {"name": "vts_reporting",              "rules": [10, 2],          "n": 26},
    {"name": "distress_relay_collision",   "rules": [2, 8],           "n": 26},
    {"name": "multi_vessel_close_quarters","rules": [7, 8],           "n": 28},
    {"name": "give_way_not_responding",    "rules": [17, 16],         "n": 30},
]
# Sum ≈ 500 (adjust the last category if the total drifts)

REGIONS = [
    "English Channel / Dover Strait (CNIS)", "Singapore Strait / TSS",
    "San Francisco Bay approaches", "Rotterdam / Europoort approaches",
    "Houston Ship Channel", "Strait of Gibraltar",
    "Panama Canal approaches", "open ocean, mid-Atlantic",
    "Malacca Strait", "New York Harbor / Ambrose Channel",
    "Suez Canal northern approach", "Great Barrier Reef inner route",
    "Puget Sound / Seattle approaches", "Osaka Bay",
    "North Sea, off the Dutch coast", "Bosphorus Strait",
]

VESSEL_TYPES = [
    "container ship", "crude oil tanker", "LNG carrier", "bulk carrier",
    "car carrier (RoRo)", "cruise ship", "general cargo vessel",
    "tug with tow", "fishing trawler", "sailing yacht", "pilot vessel",
    "passenger ferry", "offshore supply vessel", "research vessel",
    "naval auxiliary", "coastal cargo vessel",
]

RADIO_CALL_FORMAT_EXAMPLE = (
    'Correct VHF call format -- ALWAYS call the OTHER station first, then identify yourself: '
    '"[TARGET VESSEL NAME], [TARGET VESSEL NAME], THIS IS [OWN VESSEL NAME], [OWN VESSEL NAME], '
    'OVER." e.g. own vessel MV Atlantic Horizon hailing target MV Mediterranean Explorer must '
    'transmit: "MEDITERRANEAN EXPLORER, MEDITERRANEAN EXPLORER, THIS IS ATLANTIC HORIZON, '
    'ATLANTIC HORIZON, OVER." NEVER have a vessel call out its own name as the station being '
    'hailed -- that is a critical, disqualifying error.'
)

SYSTEM_PROMPT = """You are a senior maritime examiner and VHF radio-procedure instructor \
writing realistic training/evaluation scenarios for an AI "Auto Pilot" collision-avoidance \
agent aboard a ship. The agent must be tested on THREE things simultaneously:
  1. Correct COLREG rule application (the rule text is given to you -- ground your scenario in it).
  2. Correct, realistic VHF procedure: which channel to hail on (normally Channel 16), which \
     working channel to move to (varies by region -- pick something plausible for the stated \
     region and say so is a regional convention), correct power/settings, and correct radio \
     phrasing (station identification, "THIS IS", "OVER", proword usage). """ + RADIO_CALL_FORMAT_EXAMPLE + """ \
     Note explicitly when appropriate that VHF agreement does NOT override the COLREG \
     give-way/stand-on rules or sound/light signals -- a vessel must never rely on a VHF \
     "arrangement" to breach Rule 8 duties, and if in doubt the standard COLREG rule action \
     still applies.
  3. A concrete, fluent, natural-language gold answer -- NOT a mechanical list of labelled \
     fields. Write it the way a real, well-trained officer would explain their actions: \
     flowing prose that includes the actual radio call verbatim (in quotes, in the correct \
     "target, target, THIS IS own, own, OVER" format) followed by the manoeuvre and the \
     COLREG justification.

Return STRICT JSON: a list of exactly {batch_size} objects, each with this schema:
{{
  "category": "{category}",
  "colreg_rules": ["Rule X", ...],
  "region": "one of the suggested regions or a very similar plausible one",
  "own_vessel": "short description: name, type, size, course/speed",
  "target_vessel": "short description: name, type, bearing, range, aspect, closing or not",
  "scenario": "2-4 sentence situational briefing a navigator would receive",
  "vhf_channel": {{"hailing": "16", "working": "<plausible working channel for the region>", \
"settings": "e.g. switch to low power 1W once contact established, maintain Ch16 watch", \
"note": "one sentence noting this is a regional convention to verify locally"}},
  "question": "As the Auto Pilot aboard <own_vessel name>, what do you transmit over VHF and \
what collision-avoidance action do you take, and why?",
  "expected_points": ["4 to 7 atomic facts a fully correct answer must contain -- mix of the \
COLREG action, the channel/procedure, and the radio phrasing"],
  "gold_answer": "a complete, fluent 4-8 sentence answer, in prose, that includes the ACTUAL \
radio transmission in quotation marks and then explains the manoeuvre with rule citation"
}}

Make the {batch_size} scenarios in this batch clearly DIFFERENT from one another: vary vessel \
types, regions, bearings/ranges, and whether risk of collision is marginal or clear-cut. \
Do not include markdown or commentary outside the JSON array."""


def load_colreg_rules() -> dict[int, dict]:
    records = json.loads(COLREG_FILE.read_text(encoding="utf-8"))
    return {r["rule"]: r for r in records}


def user_prompt(category: dict, rules_by_num: dict[int, dict], batch_size: int) -> str:
    rule_texts = []
    for rn in category["rules"]:
        r = rules_by_num.get(rn)
        if r:
            rule_texts.append(f"Rule {rn} ({r['title']}): {r['text']}")
    regions_hint = ", ".join(random.sample(REGIONS, k=min(4, len(REGIONS))))
    vessels_hint = ", ".join(random.sample(VESSEL_TYPES, k=min(5, len(VESSEL_TYPES))))
    return (
        f"Encounter category: {category['name']}\n\n"
        f"Relevant COLREG rule text (ground every scenario in this):\n"
        + "\n\n".join(rule_texts)
        + f"\n\nSuggested regions to draw from (pick different ones per scenario): {regions_hint}"
        f"\nSuggested vessel types to draw from (vary them): {vessels_hint}"
        f"\n\nGenerate {batch_size} distinct scenarios now, as a JSON array."
    )


def parse_batch(raw: str) -> list[dict] | None:
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    if isinstance(obj, dict):
        # some models wrap the array in {"scenarios": [...]}
        for v in obj.values():
            if isinstance(v, list):
                obj = v
                break
    return obj if isinstance(obj, list) else None


def generate_batch(client: OpenAI, category: dict, rules_by_num: dict, batch_size: int) -> list[dict]:
    sys_msg = SYSTEM_PROMPT.format(category=category["name"], batch_size=batch_size)
    usr_msg = user_prompt(category, rules_by_num, batch_size)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.9,
        response_format={"type": "json_object"} if False else None,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": usr_msg},
        ],
        max_tokens=3500,
    )
    raw = resp.choices[0].message.content or ""
    items = parse_batch(raw)
    return items or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="generate ~10 records only, for a quick quality check")
    ap.add_argument("--resume", action="store_true", help="keep existing records, only top up shortfalls per category")
    args = ap.parse_args()

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing from .env")
    client = OpenAI()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules_by_num = load_colreg_rules()
    print(f"Loaded {len(rules_by_num)} COLREG rules from {COLREG_FILE.name}")

    existing: list[dict] = []
    if args.resume and OUT_FILE.exists():
        existing = json.loads(OUT_FILE.read_text(encoding="utf-8"))
        print(f"Resuming: {len(existing)} records already on disk")

    categories = CATEGORIES
    if args.smoke:
        categories = [{**c, "n": 2} for c in CATEGORIES[:5]]  # ~10 records

    existing_by_cat: dict[str, int] = {}
    for r in existing:
        existing_by_cat[r.get("category", "?")] = existing_by_cat.get(r.get("category", "?"), 0) + 1

    jobs: list[tuple[dict, int]] = []  # (category, batch_size_for_this_call)
    for cat in categories:
        remaining = cat["n"] - existing_by_cat.get(cat["name"], 0)
        while remaining > 0:
            bs = min(BATCH_SIZE, remaining)
            jobs.append((cat, bs))
            remaining -= bs

    print(f"Categories: {len(categories)}   LLM batch calls queued: {len(jobs)}   "
          f"target new records: {sum(bs for _, bs in jobs)}")

    all_records: list[dict] = list(existing)

    def run_jobs(job_list: list[tuple[dict, int]]) -> None:
        t0 = time.time()
        done_calls = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(generate_batch, client, cat, rules_by_num, bs): (cat, bs)
                       for cat, bs in job_list}
            for fut in as_completed(futures):
                cat, bs = futures[fut]
                try:
                    items = fut.result()
                except Exception as e:
                    print(f"  [error] {cat['name']}: {e}")
                    items = []
                for it in items:
                    it["category"] = it.get("category") or cat["name"]
                    all_records.append(it)
                done_calls += 1
                if done_calls % 5 == 0 or done_calls == len(job_list):
                    rate = done_calls / (time.time() - t0)
                    eta = (len(job_list) - done_calls) / rate if rate else 0
                    print(f"  batches {done_calls}/{len(job_list)}  records so far={len(all_records)}  "
                          f"ETA={eta:.0f}s", flush=True)

    run_jobs(jobs)

    # ── Retry pass: top up any category that fell short (failed/empty batches) ──
    for attempt in range(3):
        counts_now: dict[str, int] = {}
        for r in all_records:
            counts_now[r.get("category", "?")] = counts_now.get(r.get("category", "?"), 0) + 1
        retry_jobs: list[tuple[dict, int]] = []
        for cat in categories:
            shortfall = cat["n"] - counts_now.get(cat["name"], 0)
            while shortfall > 0:
                bs = min(BATCH_SIZE, shortfall)
                retry_jobs.append((cat, bs))
                shortfall -= bs
        if not retry_jobs:
            break
        print(f"\nRetry pass {attempt+1}: topping up {sum(bs for _, bs in retry_jobs)} "
              f"records across {len(retry_jobs)} batches...")
        run_jobs(retry_jobs)

    # ── Assign stable sequential ids ──────────────────────────────────────
    for i, r in enumerate(all_records, 1):
        r["id"] = f"vhfcol_{i:05d}"

    OUT_FILE.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}  ({OUT_FILE.stat().st_size/1024:.1f} KB)  records={len(all_records)}")

    # ── Sanity check: flag records where own vessel appears to hail itself ─
    def _own_name(rec: dict) -> str:
        return (rec.get("own_vessel", "").split(",")[0]
                .replace("MV ", "").replace("M/V ", "").strip().upper())

    suspect = 0
    for r in all_records:
        own = _own_name(r)
        if not own:
            continue
        m = re.search(r'"([^"]{5,200})"', r.get("gold_answer", ""))
        if m and own and m.group(1).upper().startswith(own):
            suspect += 1
    if suspect:
        print(f"\n⚠️  {suspect}/{len(all_records)} records may have the vessel "
              f"self-hailing itself (quoted call starts with own-vessel name) — spot-check these.")

    # ── Report per-category counts ────────────────────────────────────────
    from collections import Counter
    counts = Counter(r.get("category", "?") for r in all_records)
    print("\nPer-category counts:")
    for name, n in sorted(counts.items()):
        print(f"  {name:<28} {n:>3}")

    # ── Sample ────────────────────────────────────────────────────────────
    if all_records:
        ex = all_records[0]
        print("\n=== Sample record ===")
        print(json.dumps(ex, indent=2, ensure_ascii=False)[:1800])


if __name__ == "__main__":
    main()
