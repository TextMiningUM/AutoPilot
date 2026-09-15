"""
================================================================================
build_vhf_conversations.py — multi-turn VHF collision-avoidance conversation
training data (incoming AND self-initiated)
================================================================================

WHY THIS EXISTS
---------------
Existing real reference material (VHFPro.txt etc.) already gives the model good
SMCP-style dialogue for VTS reporting, pilotage, cargo ops and MAYDAY/PAN-PAN --
those flow through the normal § 8-12 pipeline as ordinary source documents.

What is MISSING is ship-to-ship COLLISION-AVOIDANCE negotiation dialogue: the
actual back-and-forth a vessel has when hailing (or being hailed by) another
vessel to confirm passing arrangements in a head-on / crossing / overtaking /
narrow-channel situation, while still correctly applying COLREG give-way /
stand-on duties (VHF agreement never overrides the Rules).

This script generates MULTI-TURN conversations in BOTH directions:
  * "incoming"  -- another vessel/VTS hails the Auto Pilot's own ship first;
                   the Auto Pilot must respond correctly and keep complying
                   with COLREG throughout the exchange.
  * "outgoing"  -- the Auto Pilot itself detects a risk-of-collision situation
                   and must INITIATE the call (e.g. as give-way vessel stating
                   intentions, or as stand-on vessel querying an unresponsive
                   give-way vessel per Rule 17).

Grounded in the same COLREG rule text used for vhf_colreg_scenarios.json, and
reusing its region/vessel-type pools for consistency.

Output: Data/VHF/VHF_Agents_Training/vhf_conversations.jsonl
Each line: {"id", "direction", "category", "colreg_rules", "region",
            "own_vessel", "target_vessel", "messages": [system, user, assistant, ...]}

`messages` is TRL-SFTTrainer-ready: a system turn describing the Auto Pilot's role,
then alternating user (incoming VHF traffic / situational radar picture) and
assistant (the Auto Pilot's own transmissions + brief COLREG-justified action)
turns -- 2-3 assistant turns per conversation, natural fluent radio prose (not
label:value dumps).

USAGE
-----
    python build_vhf_conversations.py --smoke     # ~8 conversations, quick check
    python build_vhf_conversations.py             # full run (~360 conversations)
    python build_vhf_conversations.py --resume    # top up shortfalls only
"""
from __future__ import annotations
import os, json, argparse, random, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from build_vhf_colreg_scenarios import load_colreg_rules, REGIONS, VESSEL_TYPES
from core import AgentPaths, load_env

paths = AgentPaths.vhf()
W = paths.workspace
OUT_DIR  = paths.cache_dir
OUT_FILE = OUT_DIR / "vhf_conversations.jsonl"

MODEL       = "gpt-4o-mini"
BATCH_SIZE  = 4     # conversations per LLM call
MAX_WORKERS = 6

# Same 18 encounter categories as vhf_colreg_scenarios.json, generated in BOTH
# directions (incoming / outgoing). Counts are per-direction, per-category.
CATEGORIES: list[dict] = [
    {"name": "head_on",                    "rules": [14, 34]},
    {"name": "crossing_give_way",          "rules": [15, 16]},
    {"name": "crossing_stand_on",          "rules": [15, 17]},
    {"name": "overtaking_open_water",      "rules": [13, 16]},
    {"name": "overtaking_narrow_channel",  "rules": [9, 13, 34]},
    {"name": "narrow_channel_meeting",     "rules": [9, 34]},
    {"name": "tss_crossing",               "rules": [10]},
    {"name": "restricted_visibility",      "rules": [19, 35, 7]},
    {"name": "vessel_not_under_command",   "rules": [18, 3]},
    {"name": "restricted_manoeuvre",       "rules": [18, 3]},
    {"name": "constrained_by_draught",     "rules": [18, 3]},
    {"name": "fishing_vessel_encounter",   "rules": [18]},
    {"name": "sailing_vessel_encounter",   "rules": [12, 18]},
    {"name": "anchored_vessel_hail",       "rules": [9, 2]},
    {"name": "vts_reporting",              "rules": [10, 2]},
    {"name": "distress_relay_collision",   "rules": [2, 8]},
    {"name": "multi_vessel_close_quarters","rules": [7, 8]},
    {"name": "give_way_not_responding",    "rules": [17, 16]},
]
PER_CATEGORY_PER_DIRECTION = 10   # 18 categories x 2 directions x 10 = 360

DIRECTIONS = ["incoming", "outgoing"]

RADIO_CALL_FORMAT_EXAMPLE = (
    'Correct VHF call format -- ALWAYS call the OTHER station first, then identify yourself: '
    '"[OTHER STATION NAME], [OTHER STATION NAME], THIS IS [OWN VESSEL NAME], [OWN VESSEL NAME], '
    'OVER." NEVER have a vessel call out its own name as the station being hailed -- that is a '
    'critical, disqualifying error.'
)

SYSTEM_PROMPT = """You are a senior maritime examiner and VHF radio-procedure instructor \
writing MULTI-TURN training conversations for an AI "Auto Pilot" collision-avoidance agent \
aboard a ship. You are generating SFT training data, so realism and fluent natural language \
matter enormously -- write like real, well-trained officers actually speak on the radio, \
never as mechanical label:value lists.

Direction for this batch: "{direction}".
  * If "incoming": another vessel or VTS station hails the Auto Pilot's own ship FIRST. The \
    Auto Pilot must respond correctly, establish/confirm the situation, and conduct the \
    exchange while still fully complying with COLREG give-way/stand-on duties -- a VHF \
    agreement never overrides the Rules; if in doubt the standard COLREG action still applies.
    CRITICAL: in the opening line the OTHER vessel is the CALLER and the own vessel is the \
    STATION BEING CALLED. The correct format is "[OWN VESSEL NAME], [OWN VESSEL NAME], THIS IS \
    [OTHER VESSEL NAME], [OTHER VESSEL NAME], OVER." -- own vessel's name comes FIRST (being \
    called), other vessel's name comes SECOND after "THIS IS" (the caller). Getting this \
    backwards is a critical, disqualifying error.
  * If "outgoing": the Auto Pilot itself detects the risk-of-collision situation from its own \
    radar/lookout picture and INITIATES the call -- e.g. as give-way vessel stating its \
    intended action, or as stand-on vessel querying an unresponsive give-way vessel under \
    Rule 17, or requesting confirmation of a passing arrangement in a narrow channel.

Each conversation must have 2 to 3 assistant turns (the Auto Pilot's own transmissions) \
alternating with situational/incoming-traffic turns, and must:
  1. Ground the collision-avoidance action in the given COLREG rule text.
  2. Use correct, realistic VHF procedure: Channel 16 hailing, a plausible regional working \
     channel, correct power/settings, and correct radio phrasing. """ + RADIO_CALL_FORMAT_EXAMPLE + """
  3. Read like an authentic radio exchange -- natural spoken cadence, not a form to fill in.

Return STRICT JSON: a list of exactly {batch_size} objects, each with this schema:
{{
  "category": "{category}",
  "colreg_rules": ["Rule X", ...],
  "region": "one of the suggested regions or a very similar plausible one",
  "own_vessel": "short description: name, type, size, course/speed",
  "target_vessel": "short description: name, type, bearing, range, aspect, closing or not",
  "vhf_channel": {{"hailing": "16", "working": "<plausible working channel>", \
"note": "one short sentence noting regional convention"}},
  "messages": [
    {{"role": "system", "content": "You are the Auto Pilot, the automated VHF radio watch-keeper \
aboard <own_vessel name>. Conduct correct, COLREG-compliant VHF communication at all times. \
Only ever transmit as <own_vessel name>; never impersonate or speak for another station."}},
    {{"role": "user", "content": "For 'incoming': the other station's opening transmission, \
verbatim, framed as '[Incoming VHF, Channel 16] \\"...\\"'. For 'outgoing': a situational \
briefing framed as '[Radar/lookout picture] ...' describing bearing/range/CPA and NOT yet any \
radio traffic."}},
    {{"role": "assistant", "content": "The Auto Pilot's actual transmission in quotes, in the \
correct call format, followed by 1-2 sentences on the COLREG-justified action taken."}},
    {{"role": "user", "content": "[Incoming VHF, Channel <working>] the other station's reply, verbatim, in quotes"}},
    {{"role": "assistant", "content": "The Auto Pilot's next transmission + any follow-up manoeuvre, with rule citation"}},
    ... (2-3 assistant turns total)
  ]
}}

Make the {batch_size} conversations in this batch clearly DIFFERENT from one another: vary \
vessel types, regions, bearings/ranges, and how cooperative/responsive the other station is \
(some should show the other vessel NOT responding or NOT taking correct action, forcing the \
Auto Pilot to escalate per Rule 17 or sound the danger signal). \
Do not include markdown or commentary outside the JSON array."""


def user_prompt(category: dict, rules_by_num: dict[int, dict], direction: str, batch_size: int) -> str:
    rule_texts = []
    for rn in category["rules"]:
        r = rules_by_num.get(rn)
        if r:
            rule_texts.append(f"Rule {rn} ({r['title']}): {r['text']}")
    regions_hint = ", ".join(random.sample(REGIONS, k=min(4, len(REGIONS))))
    vessels_hint = ", ".join(random.sample(VESSEL_TYPES, k=min(5, len(VESSEL_TYPES))))
    return (
        f"Encounter category: {category['name']}\nDirection: {direction}\n\n"
        f"Relevant COLREG rule text (ground every conversation in this):\n"
        + "\n\n".join(rule_texts)
        + f"\n\nSuggested regions to draw from (pick different ones per conversation): {regions_hint}"
        f"\nSuggested vessel types to draw from (vary them): {vessels_hint}"
        f"\n\nGenerate {batch_size} distinct multi-turn conversations now, as a JSON array."
    )


def _own_name(rec: dict) -> str:
    return (rec.get("own_vessel", "").split(",")[0]
            .replace("MV ", "").replace("M/V ", "").strip().upper())


def has_reversed_opening_call(rec: dict) -> bool:
    """True if an 'incoming' conversation's opening line calls the OTHER vessel
    instead of being addressed TO the own vessel (own name should be the one
    being called, i.e. appear BEFORE 'THIS IS', not after it)."""
    if rec.get("direction") != "incoming":
        return False
    own = _own_name(rec)
    if not own:
        return False
    msgs = rec.get("messages", [])
    first_user = next((m for m in msgs if m.get("role") == "user"), None)
    if not first_user:
        return False
    m = re.search(r'THIS IS ([A-Z][A-Z ]{2,40})', first_user["content"].upper())
    return bool(m and own in m.group(1))


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
        for v in obj.values():
            if isinstance(v, list):
                obj = v
                break
    return obj if isinstance(obj, list) else None


def generate_batch(client: OpenAI, category: dict, rules_by_num: dict, direction: str, batch_size: int) -> list[dict]:
    sys_msg = SYSTEM_PROMPT.format(category=category["name"], direction=direction, batch_size=batch_size)
    usr_msg = user_prompt(category, rules_by_num, direction, batch_size)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.9,
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
    ap.add_argument("--smoke", action="store_true", help="generate ~8 conversations only")
    ap.add_argument("--resume", action="store_true", help="keep existing, only top up shortfalls")
    args = ap.parse_args()

    load_env(W / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing from .env")
    client = OpenAI()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules_by_num = load_colreg_rules()
    print(f"Loaded {len(rules_by_num)} COLREG rules")

    existing: list[dict] = []
    if args.resume and OUT_FILE.exists():
        with OUT_FILE.open("r", encoding="utf-8") as f:
            existing = [json.loads(l) for l in f if l.strip()]
        print(f"Resuming: {len(existing)} conversations already on disk")

    categories = CATEGORIES
    per_cat = PER_CATEGORY_PER_DIRECTION
    if args.smoke:
        categories = CATEGORIES[:4]
        per_cat = 1

    existing_by_key: dict[tuple[str, str], int] = {}
    for r in existing:
        key = (r.get("category", "?"), r.get("direction", "?"))
        existing_by_key[key] = existing_by_key.get(key, 0) + 1

    jobs: list[tuple[dict, str, int]] = []  # (category, direction, batch_size)
    for cat in categories:
        for direction in DIRECTIONS:
            remaining = per_cat - existing_by_key.get((cat["name"], direction), 0)
            while remaining > 0:
                bs = min(BATCH_SIZE, remaining)
                jobs.append((cat, direction, bs))
                remaining -= bs

    target_new = sum(bs for _, _, bs in jobs)
    print(f"Categories: {len(categories)} x {len(DIRECTIONS)} directions   "
          f"LLM batch calls queued: {len(jobs)}   target new conversations: {target_new}")

    all_records: list[dict] = list(existing)

    def run_jobs(job_list: list[tuple[dict, str, int]]) -> None:
        t0 = time.time()
        done_calls = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(generate_batch, client, cat, rules_by_num, direction, bs): (cat, direction, bs)
                       for cat, direction, bs in job_list}
            for fut in as_completed(futures):
                cat, direction, bs = futures[fut]
                try:
                    items = fut.result()
                except Exception as e:
                    print(f"  [error] {cat['name']}/{direction}: {e}")
                    items = []
                for it in items:
                    it["category"]  = it.get("category") or cat["name"]
                    it["direction"] = direction
                    all_records.append(it)
                done_calls += 1
                if done_calls % 5 == 0 or done_calls == len(job_list):
                    rate = done_calls / (time.time() - t0)
                    eta = (len(job_list) - done_calls) / rate if rate else 0
                    print(f"  batches {done_calls}/{len(job_list)}  conversations so far={len(all_records)}  "
                          f"ETA={eta:.0f}s", flush=True)

    run_jobs(jobs)

    # ── Retry pass: top up shortfalls AND drop+regenerate quality-flagged records ──
    for attempt in range(4):
        before = len(all_records)
        all_records[:] = [r for r in all_records if not has_reversed_opening_call(r)]
        dropped = before - len(all_records)
        if dropped:
            print(f"Dropped {dropped} conversations with a reversed opening call (will regenerate)")

        counts_now: dict[tuple[str, str], int] = {}
        for r in all_records:
            key = (r.get("category", "?"), r.get("direction", "?"))
            counts_now[key] = counts_now.get(key, 0) + 1
        retry_jobs: list[tuple[dict, str, int]] = []
        for cat in categories:
            for direction in DIRECTIONS:
                shortfall = per_cat - counts_now.get((cat["name"], direction), 0)
                while shortfall > 0:
                    bs = min(BATCH_SIZE, shortfall)
                    retry_jobs.append((cat, direction, bs))
                    shortfall -= bs
        if not retry_jobs:
            break
        print(f"\nRetry pass {attempt+1}: topping up {sum(bs for _, _, bs in retry_jobs)} "
              f"conversations across {len(retry_jobs)} batches...")
        run_jobs(retry_jobs)

    # ── Final safety filter (in case the last retry pass still had strays) ──
    all_records[:] = [r for r in all_records if not has_reversed_opening_call(r)]

    # ── Assign stable ids, validate radio-call direction, write JSONL ────
    suspect = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for i, r in enumerate(all_records, 1):
            r["id"] = f"vhfconv_{i:05d}"
            own = _own_name(r)
            for m in r.get("messages", []):
                if m.get("role") != "assistant":
                    continue
                mm = re.search(r'"([^"]{5,200})"', m.get("content", ""))
                if mm and own and mm.group(1).upper().startswith(own):
                    suspect += 1
                    break
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {OUT_FILE}  ({OUT_FILE.stat().st_size/1024:.1f} KB)  conversations={len(all_records)}")
    if suspect:
        print(f"⚠️  {suspect}/{len(all_records)} conversations may have a self-hailing error — spot-check these.")

    from collections import Counter
    counts = Counter((r.get("category", "?"), r.get("direction", "?")) for r in all_records)
    print("\nPer-category/direction counts:")
    for (name, direction), n in sorted(counts.items()):
        print(f"  {name:<28} {direction:<9} {n:>3}")

    if all_records:
        print("\n=== Sample conversation ===")
        print(json.dumps(all_records[0], indent=2, ensure_ascii=False)[:2200])


if __name__ == "__main__":
    main()
