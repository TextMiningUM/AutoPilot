"""Chief Engineer known-issues reasoning-trace extraction (2026-10-03).

Builds ``Data/ChiefEngineer/ChiefEngineer_Agents_Training/chief_engineer_known_issues_traces.jsonl``
-- real + LLM-synthesized "known problem / cause / corrective action" entries for the MAN
B&W S50ME-C10.7 / S60ME-C10.7 two-stroke engines (design_chief_engineer.md's reference
installation), from THREE provenance tiers:

  1. manual_extract  -- real paragraphs pulled straight out of the 2 already-acquired
                        Project Guide PDFs (chapters 03/07-18: Turbocharger, Fuel,
                        Lubricating oil, Cylinder lubrication, cooling water, starting/
                        scavenge air, exhaust gas, engine control, vibration, monitoring/
                        alarms), filtered to paragraphs that actually describe a failure
                        mode / limit / corrective action (keyword-density screen, same
                        style as pipeline/ingest/screen_incidents.py). Every row carries a
                        real page citation.
  2. web_sourced     -- real, dated loss-prevention articles (Gard P&I club "Insights"),
                        fetched and read this session, each with its real URL kept as
                        citation. Small (a handful of articles) -- genuinely-documented
                        engine-model-agnostic known issues are not available in the
                        thousands on the open web (see design_chief_engineer.md sec 7).
  3. llm_synthesized -- gpt-4o-mini generates MANY distinct, realistic problem/cause/
                        corrective-action entries per real engine system (grounded by
                        feeding it that system's own REAL manual_extract paragraphs as
                        context), explicitly instructed NOT to invent fake specific
                        incidents (no vessel names/dates) -- general domain-knowledge
                        synthesis, not a citation. This is the volume driver (hybrid
                        approach, user-approved 2026-10-03) -- every row's `provenance`
                        field says so honestly, never disguised as a real case.

Output schema REUSES the VHF/OOW/incident reasoning-trace field names (situation/trigger/
procedures/constraints/warnings/outcomes/key_facts/question_seeds, chunk_id/source_file/
chapter_title/chunk_concepts) so build_sft.py/build_multihop.py/build_rlhf.py/
build_reflection.py can consume these traces with NO code changes (same compatibility
pattern as extract_incident_reasoning.py) -- this is what makes the file usable as BOTH a
RAG-chunk source (the `trace.situation`/`key_facts`/`procedures` text) AND reasoning-trace
training data directly, per the user's own framing of this task. Channels/prowords_used
are kept as empty lists (not applicable here, present only for schema compatibility).

Safe to run LOCALLY (OpenAI API calls only, no GPU/model loading) -- same category as
extract_incident_reasoning.py per the project's local/cloud split.

Run with: python -m pipeline.track1.extract_chief_engineer_known_issues
          python -m pipeline.track1.extract_chief_engineer_known_issues --limit 10
          python -m pipeline.track1.extract_chief_engineer_known_issues --synth-per-system 40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pdfplumber
import pymupdf
from openai import OpenAI

from core import AgentPaths, load_env
from core.text_segmentation import join_hyphenated_linebreaks

paths = AgentPaths.chief_engineer()
W = paths.workspace
CACHE = paths.cache_dir
MANUALS_DIR = paths.source_dir / "man_project_guides"
OUT_FILE = CACHE / "chief_engineer_known_issues_traces.jsonl"

MODEL = "gpt-4o-mini"
MAX_WORKERS = 4

# ── § 1  Manual chapters worth mining (chapter-title substring -> engine system tag) ───
# Matched against each PDF's own get_toc() top-level (level==1) chapter titles -- NOT
# hardcoded page numbers, since S50/S60's chapters start at different pages.
MANUAL_CHAPTERS = {
    "03 Turbocharger selection & exhaust gas bypass": "turbocharger",
    "07 Fuel": "fuel_oil",
    "08 Lubricating oil": "lubricating_oil",
    "09 Cylinder lubrication": "cylinder_lubrication",
    "10 Piston rod stuffing box drain oil": "piston_rod_stuffing_box",
    "11 Low-temperature cooling water": "cooling_water",
    "12 High-temperature cooling water": "cooling_water",
    "13 Starting and control air": "starting_control_air",
    "14 Scavenge air": "scavenge_air",
    "15 Exhaust gas": "exhaust_gas",
    "16 Engine control system": "engine_control_system",
    "17 Vibration aspects": "vibration",
    "18 Monitoring systems and instrumentation": "monitoring_and_alarms",
}

# Paragraphs must hit at least this many distinct problem-indicating keywords to be kept
# as a manual_extract seed (screens out pure spec/dimension prose with no failure-mode content).
PROBLEM_KEYWORDS = [
    "alarm", "wear", "contaminat", "corros", "crack", "leak", "fail", "damage",
    "exceed", "deviation", "prevent", "risk", "inspect", "replace", "clean",
    "drain", "monitor", "trip", "shutdown", "shut down", "slowdown", "slow down",
    "scuffing", "fire", "explosion", "fault", "deposit", "erosion", "fatigue",
    "overheat", "blockage", "fouling", "vibration", "misalignment", "seize",
]
MIN_PARAGRAPH_CHARS = 200
MAX_SEEDS_PER_SECTION = 10

CONCEPT_KEYWORDS: dict[str, str] = {
    "alarm": "alarm", "slowdown": "alarm", "slow down": "alarm", "shutdown": "alarm", "shut down": "alarm",
    "bearing": "bearing_wear", "crosshead": "bearing_wear", "crankcase": "crankcase",
    "oil mist": "crankcase", "cylinder liner": "cylinder_liner", "piston ring": "cylinder_liner",
    "scuffing": "cylinder_liner", "exhaust valve": "exhaust_valve", "turbocharger": "turbocharger",
    "surge": "turbocharger", "scavenge": "scavenge_air", "lubricating oil": "lubricating_oil",
    "cylinder lubrication": "cylinder_lubrication", "alpha": "cylinder_lubrication",
    "fuel oil": "fuel_oil", "injector": "fuel_oil", "fiva": "fuel_oil", "cooling water": "cooling_water",
    "corrosion": "corrosion", "contaminat": "contamination", "water in": "contamination",
    "starting air": "starting_air", "control air": "starting_air", "vibration": "vibration",
    "stuffing box": "piston_rod_stuffing_box", "wear monitoring": "condition_monitoring",
    "condition monitoring": "condition_monitoring", "cocos": "condition_monitoring",
}


def tag_concepts(text: str) -> list[str]:
    """Cheap keyword-hit concept tagger (mirrors build_vhf_json.py's tag_text() in spirit,
    own small vocabulary -- Chief Engineer has no shared CONCEPT_KEYWORDS dict yet)."""
    low = text.lower()
    return sorted({tag for kw, tag in CONCEPT_KEYWORDS.items() if kw in low})


def problem_score(text: str) -> int:
    low = text.lower()
    return sum(1 for kw in PROBLEM_KEYWORDS if kw in low)


# ── § 2  Manual paragraph extraction ────────────────────────────────────────────────────
def _toc_section_ranges(doc: pymupdf.Document, chapter_title: str) -> list[tuple[str, int, int]]:
    """Return [(subsection_title, start_page_0idx, end_page_0idx_exclusive), ...] for every
    level-2 TOC entry under the given level-1 chapter title."""
    toc = doc.get_toc()
    top = [t for t in toc if t[0] == 1]
    try:
        idx = next(i for i, t in enumerate(top) if t[1] == chapter_title)
    except StopIteration:
        return []
    chapter_start = top[idx][2]
    chapter_end = top[idx + 1][2] if idx + 1 < len(top) else doc.page_count + 1
    subs = [t for t in toc if t[0] == 2 and chapter_start <= t[2] < chapter_end]
    ranges = []
    for i, s in enumerate(subs):
        start = s[2] - 1  # TOC pages are 1-indexed
        end = (subs[i + 1][2] - 1) if i + 1 < len(subs) else chapter_end - 1
        ranges.append((s[1], start, max(end, start + 1)))
    return ranges


def extract_manual_seeds(pdf_path: Path, engine_model: str) -> list[dict]:
    """Walk MANUAL_CHAPTERS, pull real paragraphs scoring >=2 on PROBLEM_KEYWORDS."""
    seeds: list[dict] = []
    doc = pymupdf.open(pdf_path)
    with pdfplumber.open(pdf_path) as plumber_doc:
        for chapter_title, system in MANUAL_CHAPTERS.items():
            for sub_title, start, end in _toc_section_ranges(doc, chapter_title):
                raw_pages = []
                for pno in range(start, min(end, len(plumber_doc.pages))):
                    raw_pages.append((pno + 1, plumber_doc.pages[pno].extract_text() or ""))
                full_text = "\n".join(t for _, t in raw_pages)
                full_text = join_hyphenated_linebreaks(full_text)
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
                scored = [(problem_score(p), p) for p in paragraphs if len(p) >= MIN_PARAGRAPH_CHARS]
                scored = [sp for sp in scored if sp[0] >= 2]
                scored.sort(key=lambda sp: -sp[0])
                for i, (score, para) in enumerate(scored[:MAX_SEEDS_PER_SECTION]):
                    page_guess = start + 1  # section-level citation (good enough; exact
                    # per-paragraph page would need per-page paragraph tracking)
                    seeds.append({
                        "document_id": f"CE-MAN-{engine_model}-{re.sub(r'[^A-Za-z0-9]+', '_', sub_title)[:40]}-{i}",
                        "source_file": pdf_path.name,
                        "chapter_title": sub_title,
                        "provenance": "manual_extract",
                        "mode": "extract",
                        "engine_model": engine_model,
                        "system": system,
                        "page": page_guess,
                        "url": None,
                        "seed_text": para,
                    })
    doc.close()
    return seeds


# ── § 3  Real web-sourced seeds (fetched + read this session, see design_chief_engineer.md sec 7) ──
WEB_SOURCED_SEEDS = [
    {
        "document_id": "CE-WEB-gard-bearing-failures",
        "source_file": "gard.no/en/insights/bearing-failures-when-normal-readings-hide-the-risk",
        "chapter_title": "Bearing failures: When normal readings hide the risk (Gard P&I loss prevention)",
        "provenance": "web_sourced",
        "mode": "extract",
        "engine_model": "MAN_BW_two_stroke_general",
        "system": "bearing_condition_monitoring",
        "page": None,
        "url": "https://www.gard.no/en/insights/bearing-failures-when-normal-readings-hide-the-risk",
        "seed_text": (
            "A bearing failure on a container vessel caused rapid heat generation, oil mist "
            "formation, crankcase overpressure and machinery damage, seriously burning a crew "
            "member; the vessel had to be towed to port for repairs. The immediate cause was "
            "failure of the aft bearing on the upper intermediate gear: the bearing shell failed, "
            "metal-on-metal contact occurred, and rotation of the shell closed the oil inlet bore, "
            "cutting lubrication to the bearing. With no oil supply, friction generated a hotspot "
            "and catastrophic bearing damage. The resulting flammable oil mist formed in an upper "
            "crankcase area NOT protected by oil mist detectors; when it ignited, the crankcase "
            "over-pressure activated the relief valves and eighteen engine-room doors buckled off "
            "their hinges. The incident developed in seconds: abnormal oil-mist-detector, "
            "main-engine-slowdown, engine-room-fire and firefighting-system alarms were all "
            "recorded within four seconds of each other. Critically, standard pressure and "
            "temperature readings had remained completely normal right up until the failure -- "
            "the investigation found no evidence of poor maintenance, oil contamination, "
            "restricted oil supply, or misalignment beforehand. Recommendation: effective "
            "condition monitoring must combine temperature-trend measurement with oil mist "
            "detection, lubrication-system data, vibration monitoring, alignment checks and "
            "physical inspection -- a local bearing failure can develop into a major incident "
            "before conventional monitoring alone gives a clear warning. Monitoring data (alarm "
            "history, load, rpm, lubricating-oil pressure/temperature with timestamps) should be "
            "retained for future incident investigation."
        ),
    },
    {
        "document_id": "CE-WEB-gard-aux-engine-overspeed",
        "source_file": "gard.no/en/insights/preventing-auxiliary-engine-overspeed",
        "chapter_title": "Preventing auxiliary engine overspeed (Gard P&I loss prevention)",
        "provenance": "web_sourced",
        "mode": "extract",
        "engine_model": "MAN_BW_two_stroke_general",
        "system": "engine_control_system",
        "page": None,
        "url": "https://www.gard.no/en/insights/preventing-auxiliary-engine-overspeed",
        "seed_text": (
            "Auxiliary engine overspeed can occur within seconds from a simple setup error or a "
            "minor mechanical fault, and the resulting damage is severe: in reviewed claims, "
            "direct repair costs averaged about USD 250,000 with repair periods of 22-90 days. "
            "Known causes: (1) Mechanical stop transmission as a weak link -- in two cases, "
            "shutdown signals and overspeed alarms activated correctly, but faults in the "
            "mechanical linkage prevented the fuel rack from actually moving to zero, so the "
            "engines kept running until the crew stopped fuel supply manually; both caused "
            "extensive damage to bearings, valve gear and crankshaft components. Regular torque "
            "checks of clamps, inspection of pins/flexible links, and blue-paste contact testing "
            "could have prevented these failures. (2) Verification gap at commissioning -- in one "
            "case a new actuator was installed with the wrong rotation setting; neither the "
            "service engineers nor the crew verified rotation direction and safety functions "
            "before starting, causing immediate overspeed and major damage. (3) Lost speed signal "
            "-- damaged speed pickups came into contact with the flywheel, the engine lost its "
            "speed signal, the overspeed protection system could not initiate shutdown, and the "
            "emergency stop valve also failed because deteriorated O-rings had immobilised its "
            "internal components, preventing control air from moving the fuel rack to zero. "
            "Recommendation: after any control/fuel/governor/actuator work, run a local emergency "
            "stop function test (apply air to the stop cylinder and visually confirm the fuel "
            "rack goes to zero on every pump); make verification of serviced parts a checklist "
            "item before start-up; perform weekly visual linkage checks and monthly torque/"
            "contact verifications, paired with a monthly test that the rack is drawn to zero; "
            "maintain detailed logs of torque values, blue-paste photos, rotation-alignment "
            "images, overspeed-test sheets and hardness/runout results."
        ),
    },
]


# Fallback 1-line grounding description for a system with zero manual_extract hits (its
# manual chapter is short/dimension-heavy, e.g. nothing scored >=2 PROBLEM_KEYWORDS) -- still
# a real, well-documented MAN B&W two-stroke failure area, so worth a synthesis anchor anyway.
FALLBACK_SYSTEM_CONTEXT = {
    "turbocharger": "The turbocharger compresses scavenge air using exhaust gas energy; selection "
                    "and exhaust-gas-bypass sizing is covered in Project Guide chapter 03.",
    "piston_rod_stuffing_box": "The piston rod stuffing box (chapter 10) separates the crankcase "
                                "from the scavenge air space and drains piston-rod cooling/scraper-ring oil.",
}

# Items-per-call kept conservative so gpt-4o-mini's response stays well inside its output-token
# budget (each trace object is ~250-400 tokens; 25 items/call is a safe margin against truncation).
MAX_ITEMS_PER_SYNTH_CALL = 25


def build_all_seeds(synth_per_system: int, synth_passes: int) -> list[dict]:
    seeds: list[dict] = []
    for pdf_name, engine_model in [
        ("S50ME-C10.7_project_guide.pdf", "S50ME-C10.7"),
        ("S60ME-C10.7_project_guide.pdf", "S60ME-C10.7"),
    ]:
        pdf_path = MANUALS_DIR / pdf_name
        if pdf_path.exists():
            seeds.extend(extract_manual_seeds(pdf_path, engine_model))
        else:
            print(f"  [warn] missing {pdf_path}", file=sys.stderr)
    seeds.extend(WEB_SOURCED_SEEDS)

    # Synthesis anchors: one per real engine system, grounded with that system's own real
    # manual_extract paragraphs (dedup across the 2 engines, cap context size). Split into
    # multiple smaller passes per system for 2 reasons: (1) stay under MAX_ITEMS_PER_SYNTH_CALL
    # per API call, (2) a fresh higher-temperature call per pass gives more distinct failure
    # modes than asking for everything in one shot -- downstream SFT dedup (DEDUP_THRESH) is
    # what catches any near-duplicate items across passes, not an explicit avoid-list here.
    by_system: dict[str, list[str]] = {}
    for s in seeds:
        if s["provenance"] == "manual_extract":
            by_system.setdefault(s["system"], []).append(s["seed_text"])
    all_systems = set(MANUAL_CHAPTERS.values()) | set(by_system)
    per_call = min(synth_per_system, MAX_ITEMS_PER_SYNTH_CALL)
    for system in sorted(all_systems):
        paras = by_system.get(system) or [FALLBACK_SYSTEM_CONTEXT.get(system, system.replace("_", " "))]
        context = "\n\n---\n\n".join(paras[:6])
        for p in range(synth_passes):
            seeds.append({
                "document_id": f"CE-SYN-{system}-p{p}",
                "source_file": "llm_synthesized",
                "chapter_title": f"{system.replace('_', ' ').title()} -- synthesized domain knowledge",
                "provenance": "llm_synthesized",
                "mode": "synthesize",
                "engine_model": "MAN_BW_two_stroke_general",
                "system": system,
                "page": None,
                "url": None,
                "seed_text": context,
                "n_items": per_call,
                "pass": p,
            })
    return seeds


# ── § 4  LLM extraction / synthesis ─────────────────────────────────────────────────────
TRACE_SCHEMA_BLOCK = """{
  "situation": "2-3 sentence neutral description of the problem/failure mode",
  "trigger": "the symptom or alarm that first indicates this problem, or null",
  "procedures": [
    {"step": 1, "action": "corrective action to take", "why": "why this action addresses the root cause"}
  ],
  "constraints": ["operating limit / class or manufacturer requirement relevant to this problem"],
  "prowords_used": [],
  "channels": [],
  "regulations": ["classification-society or manual requirement cited, if any"],
  "warnings": ["safety or escalation warning, e.g. what happens if ignored"],
  "outcomes": ["consequence if the problem is NOT corrected in time", "consequence/benefit if corrected properly"],
  "key_facts": ["standalone factual statement an engineer would need (parameter, limit, component)"],
  "question_seeds": [
    {"angle": "what|when|how|why|which|who", "text": "realistic duty-engineer troubleshooting question about THIS problem"}
  ],
  "known_issue": {
    "system": "engine system this belongs to",
    "severity": "minor|moderate|serious|catastrophic",
    "failure_mode": "short technical name for the failure mode",
    "detection_method": "how this is normally detected (sensor/inspection/alarm)"
  }
}"""

EXTRACT_SYSTEM_PROMPT = f"""You are extracting known-problem/corrective-action reasoning for a Chief \
Engineer AI assistant from a real excerpt (an engine manual section or a P&I club loss-prevention \
article) about MAN B&W two-stroke marine diesel engines. Extract everything from the excerpt only \
-- no external knowledge, no invented specifics beyond what the excerpt states.

Return STRICT JSON with this exact schema:
{TRACE_SCHEMA_BLOCK}

Rules:
- All top-level fields are required. Use [] for lists that don't apply, null only for "trigger".
- "known_issue" is required, every sub-field required.
- procedures / key_facts must be grounded in the excerpt text, do not invent detail.
- question_seeds: 2-4 realistic questions a duty engineer might ask about this problem.
- Do not include markdown, code fences, or commentary outside the JSON.
- If the excerpt contains NO real failure-mode/limit/corrective-action content (pure spec/dimension \
table prose), return {{"skip": true, "reason": "..."}}"""

SYNTHESIZE_SYSTEM_PROMPT = f"""You are generating a bank of known-problem/corrective-action training \
examples for a Chief Engineer AI assistant, for the system named by the user, on MAN B&W two-stroke \
marine diesel engines (S-series, e.g. S50ME-C10.7 / S60ME-C10.7). You are given real excerpts from \
that system's own manual chapter as grounding context (real limits/components/terminology) -- use \
them to stay technically accurate, but DO NOT copy them verbatim and DO NOT invent a specific \
documented incident (no fake vessel names, dates, or claimed citations). This is general synthesized \
engineering domain knowledge, not a real case -- each entry must read as a genuinely distinct, \
realistic problem (different failure modes, different severities, different detection methods), not \
reworded repeats of the same issue.

Return STRICT JSON: {{"items": [<N distinct trace objects>]}}, each trace object matching exactly:
{TRACE_SCHEMA_BLOCK}

Rules: same field rules as above, but never use the skip mechanism here -- always produce the \
requested number of distinct items."""


def user_prompt_extract(seed: dict) -> str:
    return (
        f"Source: {seed['source_file']} / {seed['chapter_title']}\n"
        f"Engine model: {seed['engine_model']}\n"
        f"System: {seed['system']}\n\n"
        f"Excerpt:\n\"\"\"\n{seed['seed_text']}\n\"\"\""
    )


def user_prompt_synthesize(seed: dict) -> str:
    return (
        f"System: {seed['system'].replace('_', ' ')}\n"
        f"Engine family: MAN B&W two-stroke S-series (S50ME-C10.7 / S60ME-C10.7)\n"
        f"Generate exactly {seed['n_items']} distinct items.\n\n"
        f"Grounding excerpts from the real manual (for technical accuracy only):\n\"\"\"\n{seed['seed_text']}\n\"\"\""
    )


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


def process_seed(client: OpenAI, seed: dict) -> tuple[str, dict | list | None, str | None]:
    is_synth = seed["mode"] == "synthesize"
    # Vary temperature per pass (0.55/0.65/0.75/...) so repeated passes over the same system
    # sample genuinely different failure modes rather than near-duplicates of pass 0.
    temperature = round(0.55 + 0.1 * seed.get("pass", 0), 2) if is_synth else 0.3
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=min(temperature, 0.95),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT if is_synth else EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt_synthesize(seed) if is_synth else user_prompt_extract(seed)},
            ],
            max_tokens=12000 if is_synth else 1200,
        )
        raw = resp.choices[0].message.content or ""
        obj = parse_response(raw)
        if obj is None:
            return seed["document_id"], None, "parse-failure"
        return seed["document_id"], obj, None
    except Exception as e:
        return seed["document_id"], None, str(e)[:200]


def load_done_ids(path: Path) -> set[str]:
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N seeds (smoke-testing).")
    ap.add_argument("--synth-per-system", type=int, default=25,
                     help="How many LLM-synthesized items to request PER PASS per engine system (default 25, capped at MAX_ITEMS_PER_SYNTH_CALL).")
    ap.add_argument("--synth-passes", type=int, default=2,
                     help="How many independent synthesis passes (batches) per engine system (default 2) -- more passes = more volume/diversity.")
    args = ap.parse_args()

    load_env(W / ".env")
    client = OpenAI()

    CACHE.mkdir(parents=True, exist_ok=True)
    seeds = build_all_seeds(args.synth_per_system, args.synth_passes)
    if args.limit:
        seeds = seeds[: args.limit]
    by_prov = {}
    for s in seeds:
        by_prov[s["provenance"]] = by_prov.get(s["provenance"], 0) + 1
    print(f"Seeds built: {len(seeds)}  ({by_prov})")

    done = load_done_ids(OUT_FILE)
    # For synth seeds, "done" means the whole batch already written (checked via base id).
    todo = [s for s in seeds if s["document_id"] not in done]
    print(f"Already done: {len(done)}   Todo: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Extracting with {MODEL}, workers={MAX_WORKERS}...")
    t0 = time.time()
    errs = skipped = written = 0
    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_seed, client, s): s for s in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                did, obj, err = fut.result()
                seed = next(s for s in todo if s["document_id"] == did)
                if err or obj is None:
                    errs += 1
                    f_out.write(json.dumps({
                        "chunk_id": did, "source_file": seed["source_file"],
                        "error": err or "empty", "trace": None,
                    }) + "\n")
                    continue
                if seed["mode"] == "extract":
                    if obj.get("skip"):
                        skipped += 1
                        f_out.write(json.dumps({
                            "chunk_id": did, "source_file": seed["source_file"],
                            "skip": True, "reason": obj.get("reason", ""), "trace": None,
                        }) + "\n")
                        continue
                    row = {
                        "chunk_id": did,
                        "source_file": seed["source_file"],
                        "chapter_title": seed["chapter_title"],
                        "chunk_concepts": tag_concepts(seed["seed_text"]),
                        "provenance": seed["provenance"],
                        "engine_model": seed["engine_model"],
                        "page": seed["page"],
                        "url": seed["url"],
                        "trace": obj,
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                else:
                    items = obj.get("items") if isinstance(obj, dict) else obj
                    if not isinstance(items, list):
                        errs += 1
                        f_out.write(json.dumps({
                            "chunk_id": did, "source_file": seed["source_file"],
                            "error": "bad-items-shape", "trace": None,
                        }) + "\n")
                        continue
                    for j, item in enumerate(items):
                        row = {
                            "chunk_id": f"{did}-{j:03d}",
                            "source_file": seed["source_file"],
                            "chapter_title": seed["chapter_title"],
                            "chunk_concepts": tag_concepts(seed["seed_text"] + " " + json.dumps(item)),
                            "provenance": seed["provenance"],
                            "engine_model": seed["engine_model"],
                            "page": seed["page"],
                            "url": seed["url"],
                            "trace": item,
                        }
                        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                    # sentinel so a completed batch isn't re-requested on resume
                    f_out.write(json.dumps({"chunk_id": did, "batch_complete": True, "n": len(items)}) + "\n")
                if i % 5 == 0 or i == len(todo):
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)}  ({rate:.2f}/s)  written={written} skipped={skipped} errs={errs}")

    print(f"\nDone in {time.time()-t0:.0f}s. written={written} skipped={skipped} errs={errs}")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
