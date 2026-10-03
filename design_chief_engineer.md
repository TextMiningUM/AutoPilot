# Design Note: Chief Engineer — Conditional/Predictive Maintenance + Malfunction-Troubleshooting Chatbot (Track 4)

**Status:** DESIGN / RESEARCH only — nothing implemented yet, per explicit user instruction (2026-10-03). This is the fourth domain alongside VHF (mature), OOW (actively being built) and Captain (design-only). `Data/ChiefEngineer/` already exists as an empty, reserved folder (noted in `design_captain_missions.md` §2.5/§2.6 as "anticipated but never built").

This domain is **not** a mission-command or navigation agent like Captain/OOW — it is a **technical/engineering-department** agent with two explicitly requested, distinct parts:

1. **Conditional / predictive maintenance** — a (mostly non-LLM) condition-monitoring and remaining-useful-life layer that watches machinery parameters and produces maintenance recommendations and operational limits.
2. **A malfunction-troubleshooting chatbot** — a RAG + KG + reranking + PG retrieval system grounded in real engine/installation manuals *and* technical drawings, so a duty engineer (or the Captain agent) can ask "what do I do about X" during a fault and get an answer grounded in the actual manual/diagram for *that* installation.

## Table of contents

- [1. Chief Engineer job description](#sec-1)
  - [1.1 Legal/regulatory basis](#sec-1-1)
  - [1.2 Core responsibilities](#sec-1-2)
  - [1.3 Engine department rank structure](#sec-1-3)
  - [1.4 Relationship to the existing Captain/OOW design](#sec-1-4)
- [2. Two-part scope (as requested)](#sec-2)
- [3. Part A — Conditional / predictive maintenance](#sec-3)
  - [3.1 Terminology: PM vs CBM vs PdM](#sec-3-1)
  - [3.2 Machinery/systems scope](#sec-3-2)
  - [3.3 Condition parameters per system](#sec-3-3)
  - [3.4 Architecture: deterministic condition layer, LLM only for explanation](#sec-3-4)
  - [3.5 Data model](#sec-3-5)
  - [3.6 Relationship to Captain's existing `EngineStatus`/`EngineFailureContext`](#sec-3-6)
  - [3.7 Evaluation](#sec-3-7)
  - [3.8 Synthetic incident/casualty history (for RAG + ground truth)](#sec-3-8)
- [4. Part B — Malfunction-troubleshooting chatbot (RAG+KG+Reranking+PG)](#sec-4)
  - [4.1 Choosing "a particular" reference installation](#sec-4-1)
  - [4.2 Two-track data split](#sec-4-2)
  - [4.3 The new problem: ingesting technical drawings, not just text](#sec-4-3)
  - [4.4 KG design: fault→cause→system→corrective-action graph](#sec-4-4)
  - [4.5 PG design: procedures (startup/shutdown/alarm response/PMS jobs)](#sec-4-5)
  - [4.6 Reranking](#sec-4-6)
  - [4.7 Example interaction](#sec-4-7)
  - [4.8 Evaluation](#sec-4-8)
- [5. `Data/ChiefEngineer/` folder structure](#sec-5)
- [6. Exhaustive malfunction taxonomy (troubleshooting categories)](#sec-6)
- [7. Documentation/manual/drawing sourcing — research status](#sec-7)
- [8. Simulator representation — Engine Room Dashboard](#sec-8)
  - [8.1 Chain of command: Chief Engineer → Captain, not OOW](#sec-8-1)
  - [8.2 Dashboard panel](#sec-8-2)
- [9. Decisions (resolved 2026-10-03) and remaining open items](#sec-9)

---

<a id="sec-1"></a>
## 1. Chief Engineer job description

<a id="sec-1-1"></a>
### 1.1 Legal/regulatory basis (real regimes, not invented)

- **STCW Convention**, Chapter III ("Standards regarding the engine department"): Regulation III/2 sets the certification standard for **Chief Engineer Officer** and Second Engineer Officer on ships powered by main propulsion machinery of 3,000 kW propulsion power or more; III/3 covers smaller vessels (750–3,000 kW); III/1 covers the **Engineering Officer of the Watch (EOOW)** — the engine-room equivalent of the OOW, standing watch in a manned engine room or holding "Unmanned Machinery Space" (UMS) on-call duty where the engine room is periodically unattended.
- **ISM Code** (SOLAS Ch. IX): the Chief Engineer is the Master's direct technical counterpart under the ship's Safety Management System — responsible for implementing the planned maintenance system (PMS), reporting non-conformities, and (per `design_captain_missions.md` §2.5) the one who actually declares operational limits like "do not exceed X kn" after a technical fault.
- **SOLAS Ch. II-1** (Construction — machinery and electrical installations): mandates the provision/testing of essential auxiliary machinery, emergency power, steering gear redundancy, and periodic machinery surveys — the technical backbone of what a classification society audits.
- **MARPOL Annex VI**: engine-room emissions compliance (NOx Technical Code, SOx scrubber/fuel-sulphur compliance, EEXI/CII) — the Chief Engineer is personally responsible for the Engine International Air Pollution Prevention (EIAPP) certificate's continued validity and the Ship Energy Efficiency Management Plan (SEEMP) in practice.
- **Classification society rules** (ABS/DNV/Lloyd's Register/ClassNK, etc.): define survey intervals, continuous machinery survey (CMS) schemes, and condition-based "machinery in continuous survey" alternatives to fixed 5-year special surveys — this is the real-world regulatory anchor for "conditional maintenance" as a recognised alternative to pure calendar-based maintenance, not a project invention.
- **MLC 2006**: engine department rest-hour limits apply identically to engineers (STCW A-VIII/1) — relevant to any future "fatigue" cross-link with the Captain domain's own rest-hour ledger (`design_captain_missions.md` §13.A.3).

<a id="sec-1-2"></a>
### 1.2 Core responsibilities (for the agent persona)

1. **Keep propulsion and essential auxiliary machinery operational and safe** — main engine(s), generators, boilers, steering gear, pumps, purifiers, compressors, refrigeration, within manufacturer and class limits.
2. **Plan and execute maintenance** — via the ship's Planned Maintenance System (PMS): calendar/running-hour-based jobs (the historical baseline) *and*, where fitted, condition-based jobs triggered by sensor trend data (vibration, oil analysis, temperature/pressure deviation) — this is exactly Part A below.
3. **Diagnose and respond to malfunctions** — using manufacturer manuals, class/flag-state guidance, and the ship's own drawings (P&ID, GA, wiring diagrams) to find root cause and corrective action under time pressure — this is exactly Part B below.
4. **Report to the Master/Captain** — machinery status, operating restrictions, and any fault with safety implications (ties directly into the existing Captain design's "calling the master" protocol, `design_captain_missions.md` §3.1, and the already-defined `EngineStatus` fact container, §3.6 below).
5. **Manage spares, bunkers, and consumables** — fuel/lube oil quality and quantity, spare parts inventory against class-required minimum holdings.
6. **Supervise the engine department** — Second/Third/Fourth Engineers, Electro-Technical Officer, motormen/oilers/wipers (mirrors the deck department's rank structure already documented for OOW/Captain).
7. **Record-keeping** — Engine Room Log Book (legal document, same status as the deck log), Oil Record Book (MARPOL), PMS job completion records — auditable by Port State Control and class surveyors.

<a id="sec-1-3"></a>
### 1.3 Engine department rank structure (for completeness, not separate agents — see §1.4)

Chief Engineer → Second Engineer (deputy, day-to-day PMS/watch schedule) → Third Engineer (typically generators/boilers) → Fourth Engineer (typically purifiers/auxiliary pumps) → Electro-Technical Officer (ETO, electrical/automation/UMS alarm systems) → Motorman/Oiler/Wiper (ratings). As with OOW's subordinates (`design_captain_missions.md` §2.6), these are **not** proposed as separate fine-tuned agents — their reports are structured facts (sensor readings, job-completion records) feeding the Chief Engineer agent, same "state computed facts directly" convention used throughout this project.

<a id="sec-1-4"></a>
### 1.4 Relationship to the existing Captain/OOW design

`design_captain_missions.md` §2.5/§2.6 already anticipated a Chief Engineer role, but explicitly scoped it down to "a subordinate fact container (`EngineStatus`), never a separate fine-tuned agent" — a deliberate, documented decision at the time. **This note supersedes that scope reduction**, per the user's explicit request to build Chief Engineer as its own domain (own `AgentPaths(domain="ChiefEngineer", ...)`, own RAG/KG/PG corpus, own eventual fine-tune). The existing `EngineStatus`/`EngineFailureContext` dataclasses (`pipeline/captain_types.py`, `pipeline/captain_agent_spec.py`) are **not discarded** — they become the formal *output interface* this new domain feeds into: Part A's condition-monitoring layer is what should *actually compute* `EngineStatus.max_speed_kn`/`fault` instead of that field being hand-set by a scenario script, and Part B's chatbot is what the Captain (or a human duty engineer) consults when `fault` is non-null. This keeps the "pipeline-never-depends-on-Basic-Simulator/app" rule intact (per `pipeline/captain_agent_spec.py`'s own `EngineFailureContext` docstring) — the Chief Engineer domain's pipeline output is plain data, and only `app/`-level glue code (mirroring `app/captain_skeleton.py`) would wire it into the simulator.

---

<a id="sec-2"></a>
## 2. Two-part scope (as requested)

| | Part A — Conditional/predictive maintenance | Part B — Malfunction-troubleshooting chatbot |
|---|---|---|
| **Question answered** | "Does component X need maintenance *now*, and what operating limit follows from its current condition?" | "Given this fault/symptom, what do the manuals and drawings for *this* installation say to do?" |
| **Nature** | Mostly deterministic/statistical (condition monitoring, trend/threshold, RUL estimation) — an LLM is optional and only for turning a computed verdict into readable prose, never for the verdict itself (same "facts-only" principle as OOW/Captain). | Fundamentally a retrieval problem — RAG (text) + KG (fault→cause→system graph) + reranking (cross-encoder) + PG (procedures) over manuals and drawings, with an LLM for the conversational layer. |
| **Input** | Sensor/parameter time series (real or synthetic): temperature, pressure, vibration, oil-analysis results, running hours. | A free-text question or a structured fault code/alarm ID, optionally plus Part A's current condition verdict as context. |
| **Output** | A maintenance recommendation + severity + (if applicable) an operating limit (`EngineStatus`-shaped). | A grounded answer citing the specific manual section/drawing, plus (if applicable) a procedure (PG) to follow. |
| **Training-data style** | Not a conversational dataset — closer to OOW's `oracle_planner.py` (a deterministic baseline) than to SFT/DPO dialogue data. | Same two-track convention as VHF/OOW (Track 1 rules-from-manuals + Track 2 conversational malfunction dialogues), SFT/DPO/Reflection eventually. |

Both parts share one corpus (the manuals/drawings, §4) and one domain folder (§5) — Part A's thresholds and failure-mode knowledge should themselves be *sourced from* the same manuals Part B retrieves from, not re-invented independently (e.g. a vibration alarm limit stated in a manufacturer manual is both a Part A threshold *and* a Part B citeable fact).

---

<a id="sec-3"></a>
## 3. Part A — Conditional / predictive maintenance

<a id="sec-3-1"></a>
### 3.1 Terminology: PM vs CBM vs PdM (grounded, not invented)

- **Preventive/planned maintenance (PM)**: fixed calendar or running-hour intervals regardless of actual condition — the historical baseline on most ships' PMS.
- **Condition-based maintenance (CBM)**: maintenance triggered when a *measured* condition parameter crosses a defined threshold (ISO 17359 "Condition monitoring and diagnostics of machines — General guidelines" is the real international standard governing this; ISO 13374 covers data processing/communication/presentation; ISO 14830-1 covers tribology/oil-analysis-based monitoring specifically).
- **Predictive maintenance (PdM)**: a further step — using trend data (not just a single threshold crossing) to *forecast* a remaining-useful-life (RUL) and schedule maintenance just before failure, maximising component life while still avoiding unplanned breakdown. This is the real distinction the user's "conditional predictive maintenance" phrase points at: not just "alarm fired, go fix it" but "trend says bearing wear will cross the limit in ~40 running hours, schedule it at the next port."
- Real marine-specific condition-monitoring techniques, all grounded and already standardised (not project inventions): vibration analysis (rotating equipment — main/aux engines, pumps, turbochargers), used-oil/wear-particle analysis (iron/aluminium/chrome content indicates specific component wear, e.g. liner/piston/ring wear), infrared thermography (electrical contacts, bearings), performance trending (fuel rack position vs load, exhaust gas temperature spread across cylinders — a classic two-stroke diagnostic), and model-based electrical signature analysis for motors/generators.

<a id="sec-3-2"></a>
### 3.2 Machinery/systems scope (v1, not exhaustive — mirrors §6's malfunction taxonomy)

1. Main propulsion engine (two-stroke slow-speed crosshead *or* four-stroke medium-speed trunk — pick one as the reference installation, §4.1)
2. Auxiliary generators (diesel or shaft-driven)
3. Boilers (auxiliary + exhaust-gas economiser, if fitted)
4. Fuel oil system (purification, viscosity/temperature control, injection)
5. Lube oil system (main engine + auxiliary, including centrifugal purifiers)
6. Cooling water systems (sea water + fresh water, central cooling)
7. Compressed air system (starting air, control air)
8. Steering gear (hydraulic power units — shared interest with OOW/Captain's existing "steering gear failure" brown-envelope category)
9. Pumps (bilge, ballast, fire, general service)
10. Electrical generation/distribution (switchboard, automation/alarm monitoring system)
11. Refrigeration/HVAC (reefer cargo plant where relevant, same category as `design_captain_missions.md` §4.C's "refrigeration failure")
12. Exhaust gas system / turbocharger

<a id="sec-3-3"></a>
### 3.3 Condition parameters per system (illustrative, not exhaustive)

| System | Parameters monitored | Degradation mode | Real technique |
|---|---|---|---|
| Main engine | cylinder exhaust gas temp spread, cylinder pressure (Pmax/Pcomp), crankcase oil mist, vibration | liner/piston/ring wear, injector fouling, turbocharger surge | performance trending + used-oil analysis + vibration |
| Generators | load-sharing deviation, winding temperature, vibration, insulation resistance | bearing wear, insulation degradation | vibration + thermography + electrical signature analysis |
| Lube oil | viscosity, water content, TBN (total base number), wear-metal ppm (Fe/Al/Cr/Cu) | additive depletion, abnormal wear, water/fuel contamination | oil analysis (the single most mature CBM technique on ships) |
| Pumps/compressors | vibration spectrum, bearing temperature, flow/head vs design curve | bearing wear, impeller erosion, seal degradation | vibration analysis + performance trending |
| Steering gear | hydraulic pressure, response time, oil condition | seal wear, pump wear | performance trending + oil analysis |
| Boilers | feed water chemistry (TDS, pH, oxygen), exhaust gas side temperature | scaling, tube fouling, corrosion | chemical trending + thermography |

<a id="sec-3-4"></a>
### 3.4 Architecture: deterministic condition layer, LLM only for explanation

Following this project's established "state computed facts directly, don't trust the model to re-derive them" convention (`basic_simulator.md`, carried into Captain's facts-only pivot, `design_captain_missions.md` §13.B.5):

1. **Condition-evaluation functions** (pure, deterministic, per system/parameter) — threshold check (hard limit from the manual/class rule) + a simple trend/slope estimate over the last N readings (linear regression or exponential-weighted trend is enough for v1; no need for a neural RUL model at this stage) → a severity tier (`nominal` / `watch` / `warning` / `critical`), mirroring the Captain domain's own severity vocabulary (`minor`/`moderate`/`serious`/`catastrophic`) for consistency across domains.
2. **RUL estimate** (hours until the trend crosses the hard limit, where a trend exists) — a simple, explainable linear extrapolation, not a black-box forecaster, so every number is traceable back to the manual's stated limit and the observed trend — consistent with this project's "determinism by construction" and "no untested ML surprises" conventions.
3. **Maintenance recommendation** — a deterministic lookup (severity + system) → recommended action + urgency, exactly analogous to Captain's `PROCEDURE_LIBRARY` (`pipeline/captain_agent_spec.py`) mandatory/shield/decision-layer split: a `critical` severity is a **shield-level** hard operating-limit cap (e.g. force `EngineStatus.max_speed_kn` down), not something an LLM is ever allowed to override.
4. **LLM layer (optional, additive)** — takes the condition verdict + RUL + recommendation and, grounded via Part B's RAG corpus, writes the actual work-order ticket text / explains *why* (citing the real manual section the limit came from) — this is where Part A and Part B connect: Part A never invents a limit, it looks the limit up from the same ingested manual Part B retrieves from.

<a id="sec-3-5"></a>
### 3.5 Data model (sketch, not final — same status as Captain's §15.1 "finalise while coding" position)

```
ConditionReading:  component_id, parameter, value, unit, timestamp_s
ConditionLimit:    component_id, parameter, warning_threshold, critical_threshold, source_citation
ConditionVerdict:  component_id, severity, trend_slope, rul_hours | None, limiting_parameter
MaintenanceRecommendation: component_id, severity, action, urgency, due_by, source_citation
```
`source_citation` on both `ConditionLimit` and `MaintenanceRecommendation` is mandatory — every number must trace back to a real ingested manual/class-rule chunk id, mirroring every other domain's RAG-grounding discipline in this project.

<a id="sec-3-6"></a>
### 3.6 Relationship to Captain's existing `EngineStatus`/`EngineFailureContext`

`EngineStatus(max_speed_kn, fault, reported_at)` and `EngineFailureContext` (`pipeline/captain_agent_spec.py`) already exist and are consumed by the Captain's engine-failure decision layer (`candidates_engine_failure`/`rollout_engine_failure`). Part A's job is to make `EngineStatus` a **computed output** of this new domain instead of a scenario-authored constant: a `ConditionVerdict` reaching `critical` severity is exactly what should populate `EngineStatus.fault` and cap `max_speed_kn`, which the Captain's existing shield/decision layer already knows how to consume **unchanged** — no changes needed to `pipeline/captain_agent_spec.py` itself, only a new upstream producer of the same dataclass.

<a id="sec-3-7"></a>
### 3.7 Evaluation

- **Threshold/trend correctness**: a held-out set of synthetic degradation traces (seeded, deterministic generation — same "determinism by construction" rule as `pipeline/nomoto.py`'s `SHIP_PROFILES`/`sample_ship_profile()`) with a known ground-truth failure point; score = how close the predicted RUL is to the true remaining time, and whether severity tiers fire at the correct point (neither too early — false alarms/"crying wolf", directly reusing Captain's §13.A.5 evidence/reliability framing — nor too late).
- **Citation accuracy**: every `source_citation` must resolve to a real chunk in the ingested corpus (a cheap, fully automatable check — no LLM needed to verify this axis).
- Mirrors Captain's own two-hard-gate-plus-weighted-axis evaluator pattern (`design_captain_missions.md` §10) rather than inventing a new evaluation philosophy.

<a id="sec-3-8"></a>
### 3.8 Synthetic incident/casualty history (for RAG + ground truth) — DECIDED 2026-10-03

User confirmed: generate **synthetic sensor degradation traces** (§3.7, already deterministic/seeded) **and** a synthetic **incident/casualty history** for the reference vessel's own engine room — i.e. a fictional but structurally realistic "this ship's own past" (e.g. "in 2024, cylinder 4's exhaust temperature trended up over 3 weeks before a liner failure at X running hours") rather than only relying on third-party CHIRP/MAIB narratives (§7). This mirrors the project's existing fictional-but-structurally-real convention (Imazu/RND missions' hand-authored geometry, `design_captain_missions.md` §14.1 Tier 0) applied to maintenance history instead of navigation scenarios:

- Each synthetic incident is generated **from** a `ConditionVerdict`/`MaintenanceRecommendation` trace that was deliberately *not* acted on in time (or was acted on correctly) — i.e. the incident generator and the condition-evaluation functions (§3.4) share the same underlying degradation model, so a synthetic incident's "what should have been done" is always consistent with what Part A would compute today, not an independently-invented storyline.
  - **Correct-response incidents** (maintenance caught the trend in time) become Track 1 RAG content ("in a prior case, the second engineer correctly scheduled a bearing replacement after noticing...") and Part A/§3.7's RUL-accuracy ground truth.
  - **Missed/late-response incidents** (the trend was ignored or misread) become DPO-rejected examples (mirrors Captain's §13.C.10 category split) and Track 2 malfunction-troubleshooting training data — a dialogue where the duty engineer has to diagnose a fault that a synthetic incident shows *should* have been caught earlier.
- Seeded and deterministic (one seed per incident, reproducible), same rule as every other generator in this project (`pipeline/nomoto.py`'s `sample_ship_profile()`, `generate_random_imazu_missions.py`, etc.) — never a bare unseeded random draw.
- Feeds **both** halves of this domain at once: Part A gets a ground-truth RUL-prediction benchmark (§3.7) for free, and Part B's RAG/Track 2 corpus gets realistic "has this happened before on this ship" content without waiting on real-world incident-report acquisition (§7) to finish.

---

<a id="sec-4"></a>
## 4. Part B — Malfunction-troubleshooting chatbot (RAG+KG+Reranking+PG)

<a id="sec-4-1"></a>
### 4.1 Choosing "a particular" reference installation — DECIDED 2026-10-03

**User confirmed: use the SAME reference vessel already established for OOW/Captain** — Sawada et al. (2021)'s 106 m cargo vessel (`pipeline/nomoto.py`'s `SHIP_PROFILES["sawada2021"]`), for cross-domain consistency (one ship, four agent domains). This replaces the earlier "start with a smaller auxiliary generator" recommendation as the primary scope:

- **Engine**: the main propulsion plant implied by that vessel class — a **slow-speed two-stroke crosshead main engine** (e.g. a MAN B&W S-series type, the real engine family that powers vessels of this size/class) — is the v1 reference installation, not the auxiliary generator. The auxiliary generator (a medium-speed four-stroke trunk type, e.g. MAN 32/40 or Wärtsilä 32-class) remains a natural, smaller v2/parallel addition once the main-engine corpus exists, since generator sets are genuinely simpler systems and share much of the same fuel/lube-oil/cooling infrastructure.
- **Drawings**: per §4.3's decision, a real, vessel-specific GA/P&ID for Sawada's own academic reference vessel does not exist publicly — the drawings acquired (§4.3) are honestly real but from OTHER real vessels (and are labelled as such in their own manifest for provenance), used purely as **illustrative stand-ins** "presumed" to belong to the reference vessel, exactly the convention the user specified.

<a id="sec-4-2"></a>
### 4.2 Two-track data split (mirrors VHF/OOW's established convention)

| | Track 1 — Manual/drawing knowledge | Track 2 — Conversational malfunction troubleshooting |
|---|---|---|
| Eval data (held out) | `chiefengineer_gold_answers.json` (hand-authored Q&A against the manuals) | `chiefengineer_malfunction_scenarios.json` (realistic fault-diagnosis dialogues with a known correct procedure) |
| Training data | RAG chunks + KG (built from manuals/drawings directly, §4.3/§4.4) | SFT/DPO/Reflection mined from troubleshooting dialogues (reuses `extract_*_reasoning.py`-style extraction, same schema fields required: `chunk_id`/`source_file`/`chapter_title`/`chunk_concepts`, per `pipeline-notes.md`'s documented compatibility requirement) |
| Eval script | `eval_finetuned.py`-equivalent (reused unchanged, same pattern as every other domain) | `eval_colreg_scenarios.py`-equivalent, renamed for this domain |

<a id="sec-4-3"></a>
### 4.3 The new problem: ingesting technical drawings, not just text

Every existing domain's pipeline (`build_*_json.py` → `build_rag.py` → `build_kg.py`) is **text-only**. Drawings (P&ID, GA, wiring/one-line diagrams, cross-section views) are the first genuinely new ingestion capability this domain needs — flagged honestly as *not solved by any existing script*, not glossed over:

- **What a drawing contributes**: equipment tags (e.g. "P-101", "V-205"), system boundaries, flow direction, and cross-references the text manual assumes the reader can see (e.g. "refer to Fig. 4.2 valve lineup").
- **Proposed approach (design only, not built)**: a new `build_drawing_index.py` that (a) OCRs/extracts text labels and tag numbers from each drawing image/PDF page, (b) uses a vision-capable model (same family already used for this project's OpenAI-API reasoning-trace extraction, e.g. GPT-4o/4o-mini with image input) to produce a structured caption (equipment list + what the drawing shows + which system it belongs to), and (c) stores the drawing as a retrievable "chunk" whose embedding is computed from the caption text (same `core/embedding.py` embedder as every other chunk, so it participates in the *same* RAG index without a separate retrieval path) while the original image/PDF page is kept as a referenceable artifact the chatbot can point the user to ("see Drawing GA-04, Fig 3").
- **KG linkage**: each OCR'd equipment tag becomes a KG node, linked to any text-manual sections that also reference that tag — this is what lets a question like "what does P-101 do and where is it" be answered by combining a text citation and a drawing citation in one answer.
- This is explicitly the highest-uncertainty new piece of this whole design and should be prototyped on a *small* sample (2–3 drawings) before committing to a full pipeline, mirroring this project's own "walking skeleton first" discipline.

**Sourcing — DECIDED 2026-10-03 and a first batch already acquired.** Per the user's explicit instruction, real vessel-specific drawings for Sawada's academic reference vessel don't exist publicly — so this domain **downloads real, public-domain engine/ship-system drawings from whatever vessels are actually available, and presumes/treats them as if they belong to our reference vessel/engine**, purely for illustration (exactly the same "fictional but structurally real" spirit as the Imazu/RND missions' hand-authored geometry, just applied to borrowed-but-real drawings instead of invented ones). A first batch of 5 real drawings was collected this session (one-off manual download, not yet a reusable scraper script — same posture as Captain's very first geo-data acquisition pass before `build_captain_legal_corpus.py` existed):

| File | Category | Real source vessel (for honesty/provenance — never hide this when citing) |
|---|---|---|
| `uss_eltanin_electrical_wiring_diagram.jpg` | electrical | USS Eltanin (T-AK-270) — propulsion electrical circuits schematic |
| `uss_cimarron_diesel_generator_service_piping.jpg` | oil/fuel | USS Cimarron (AO-22) — diesel generator engine service piping |
| `uss_s3_s9_lubricating_oil_piping_diagram.jpg` | oil | USS S-3 to USS S-9 submarine class — lubricating oil piping |
| `uss_cimarron_deck_service_piping.jpg` | water | USS Cimarron (AO-22) — deck service piping |
| `uss_cimarron_cargo_piping_fo_jp5_avgas.jpg` | gas/fuel | USS Cimarron (AO-22) — cargo piping & tank vents (F.O./JP-5/avgas) |

All 5 are real US Navy engineering drawings, originally held by NARA (National Archives), public domain as US federal government works, sourced via Wikimedia Commons (`Data/ChiefEngineer/ChiefEngineerManuals/Drawings/manifest.json` records each file's real title/source vessel/URL/licence — the manifest's own `_comment` states explicitly that these are illustrative stand-ins, not genuine Sawada-vessel drawings, so this is never misrepresented downstream). **Still missing** (no Commons/NARA hit found this pass, flagged for a future search pass): a **hydraulic** diagram (e.g. steering-gear power unit) and a **pneumatic/compressed-air** diagram (starting air/control air) — the 2 categories the user asked for that aren't yet covered.

### 4.3.1 Known-problem/corrective-action reasoning traces — FIRST REAL DATA BUILT 2026-10-03

User asked to search for known problems/solutions for the 2 acquired MAN B&W S50ME-C10.7/
S60ME-C10.7 Project Guides and turn them into chunks usable as both RAG content and reasoning
traces. New script `pipeline/track1/extract_chief_engineer_known_issues.py` (mirrors
`extract_incident_reasoning.py`'s schema/pattern exactly — same `chunk_id`/`source_file`/
`chapter_title`/`chunk_concepts` + nested `trace` shape, so it needs zero changes to be consumed
by `build_sft.py`/`build_multihop.py`/`build_rlhf.py`/`build_reflection.py` once Chief
Engineer's own training stage is wired up) produced **706 real trace rows** (hybrid approach,
user-approved — see the chat session, not re-litigated here), each with an honest `provenance`
tag, in `Data/ChiefEngineer/ChiefEngineer_Agents_Training/chief_engineer_known_issues_traces.jsonl`:

- **49 `manual_extract`** — real paragraphs pulled straight out of the 2 Project Guides
  (chapters 03/07–18), keyword-screened for actual failure-mode/limit/corrective-action content
  (not spec/dimension prose), each with a real page citation.
- **2 `web_sourced`** — real, dated Gard P&I club loss-prevention articles (bearing failures/
  crankcase deflagration; auxiliary engine overspeed), each with its real URL kept as citation.
- **655 `llm_synthesized`** — gpt-4o-mini-generated, distinct problem/cause/corrective-action
  entries per real engine system (turbocharger, fuel oil, lubricating oil, cylinder lubrication,
  piston rod stuffing box, cooling water, starting/control air, scavenge air, exhaust gas, engine
  control system, vibration, monitoring/alarms), grounded by feeding the model that system's own
  real manual excerpts as context, explicitly instructed never to invent a fake specific incident
  (no vessel names/dates) — honestly labeled general domain-knowledge synthesis, not a citation.

This is the project's established "literal thousands of genuinely distinct documented known
issues for one specific engine model don't exist on the open web" finding, resolved via the
user-approved hybrid approach rather than inflating volume by mislabeling synthetic content as
real. Not yet wired into `build_rag.py`'s embedding/retrieval step or any of the Track-1 SFT/DPO/
Reflection builders — this file is ready-shaped for both, but AgentPaths(domain="ChiefEngineer")
has no RAG/KG-building call sequence wired up yet (added `AgentPaths.chief_engineer()` in
`core/paths.py` this pass, nothing downstream run yet).

### 4.3.2 Cross-model (GPT vs. Claude) agreement filter on the 655 `llm_synthesized` rows

User asked to independently re-answer the same 655 synthesis prompts with Claude (the user's own
Anthropic key, model `claude-sonnet-5`) and keep only the GPT items that a second, independent
model agrees with — a real second opinion, not just a volume-booster. New script
`pipeline/track1/extract_chief_engineer_known_issues_crossval.py` (reuses `build_all_seeds()`/
`SYNTHESIZE_SYSTEM_PROMPT`/`user_prompt_synthesize`/`parse_response` from
`extract_chief_engineer_known_issues.py` unchanged — only swaps which API answers):

1. **generate** — re-ran the same 36 synthesis seeds through Claude, producing **721 independent
   items** in `chief_engineer_known_issues_traces_claude.jsonl` (1 seed errored out of 36).
   Needed 2 real fixes specific to `claude-sonnet-5` (both logged for future reuse): (a) this
   model returns an extended-thinking block ahead of the actual text block in `resp.content`,
   so `resp.content[0]` is NOT reliably the answer (every older Claude model used elsewhere in
   this repo puts the answer at `content[0]`) — fixed by scanning for the first `type=="text"`
   block instead of assuming position 0; (b) a non-streaming call truncated silently at
   `max_tokens=16000` even for modest item counts (confirmed empirically: a 15-item request hit
   the cap and produced unparseable JSON), and the API outright REJECTS `max_tokens` high enough
   to avoid this on a non-streaming call ("Streaming is required for operations that may take
   longer than 10 minutes") — fixed by switching to `client.messages.stream()` with
   `max_tokens=32000`.
2. **crossval** — embeds every GPT item and every Claude item (same `BAAI/bge-large-en-v1.5`
   embedder as the rest of this project) and keeps only GPT items with a sufficiently similar
   Claude-generated counterpart. FIRST ATTEMPT had a real bug (caught via the measured
   distribution, not assumed): restricting candidates to "same system" compared GPT's free-text
   `known_issue.system` field (e.g. "low-temperature cooling water system") against Claude's
   code-level slug (e.g. `cooling_water`) — these essentially never matched as exact strings,
   so ~98% of GPT items had zero candidates (p95 similarity measured at 0.000, the tell). FIXED
   by dropping the same-system restriction entirely (a full 655x721 cosine-similarity search is
   trivial at this scale, and each item's real content -- situation + failure mode + corrective
   actions -- is specific enough that genuine cross-system false matches aren't a real risk).
   Re-measured distribution: p5=0.720, p50=0.790, p75=0.825, p95=0.871, mean=0.793 -- a
   believable range for independently-generated-but-topically-related content (not near-
   duplicates, which would cluster near DEDUP_THRESH~0.94). At the default threshold 0.80,
   **284/655 (43.4%) GPT items** have a Claude-agreed counterpart -- written to
   `chief_engineer_known_issues_traces_agreed.jsonl`, each row annotated with
   `cross_validated_by`/`agreement_similarity`/`matched_claude_chunk_id`. Threshold is an
   explicit starting point (same "illustrative, tunable" status as every other threshold in
   `core/embedding.py`), not yet independently calibrated.

<a id="sec-4-4"></a>
### 4.4 KG design: fault → cause → system → corrective-action graph

Reuses the existing `pipeline/ingest/build_kg.py` machinery (concept-tagging + edges), but the *concept vocabulary* for this domain is fault-centric rather than COLREG-rule-centric: `CONCEPT_KEYWORDS`-equivalent entries for symptom terms (e.g. "high exhaust temperature", "low lube oil pressure", "excessive vibration"), linked to cause nodes (e.g. "fouled injector", "worn bearing"), linked to corrective-action nodes (sourced from the manual's own troubleshooting/fault-finding tables — most real engine manuals already contain a tabular fault-finding section, which is an unusually clean source for this exact graph structure, arguably easier to mine than COLREG rule text was).

<a id="sec-4-5"></a>
### 4.5 PG design: procedures (startup/shutdown/alarm response/PMS jobs)

Reuses `pipeline/ingest/build_pg.py`'s procedural-graph machinery unchanged — engine manuals are *inherently* procedural (step-by-step startup sequences, shutdown sequences, alarm-response flowcharts, PMS job cards) in a way that's an even more natural fit for a procedural graph than COLREG rule text was for OOW. This is the strongest argument for reusing the existing 4-script pipeline (`build_rag`/`build_kg`/`build_pg` + reranker) unchanged rather than inventing new retrieval architecture for this domain, consistent with the project's hard "never fork a second copy of a pipeline script for a new domain" rule.

<a id="sec-4-6"></a>
### 4.6 Reranking

Reuses the existing cross-encoder reranking mechanism already wired into `Basic Simulator/app/agents.py`'s `rerank_hits()` (retrieve a wide pool via `kg_retrieve()`, then rerank down to k) — no new reranking architecture needed, only a new domain-scoped index (`AgentPaths(domain="ChiefEngineer", ...)`'s own `cache_dir`) for it to run against.

<a id="sec-4-7"></a>
### 4.7 Example interaction (illustrative target, not yet buildable)

> **User/Captain**: "Cylinder 3 exhaust gas temperature is 40°C above the mean of the other cylinders, and we're getting a high lube-oil-mist alarm. What's going on and what do I do?"
>
> **Chatbot (target behaviour)**: retrieves the manual's fault-finding table entry for "exhaust gas temperature deviation, single cylinder" (cites section/page), cross-references the KG's linked causes (fouled injector / exhaust valve leakage / piston ring wear), retrieves the PG's "high lube oil mist — emergency shutdown procedure" (a safety-critical procedure, cited verbatim where safety-critical), and — if Part A's condition layer is live — folds in the already-computed severity/RUL verdict for cylinder 3 rather than re-deriving it from scratch.

<a id="sec-4-8"></a>
### 4.8 Evaluation

Same two-track pattern as VHF/OOW: Track 1 scored by the existing RAGAS-style metrics already proven on VHF (`SemSim`/`ClaimPrec`/`ClaimRec`/`CorpusGrounded`, see `ablation_scored_pilot_n40.jsonl`), Track 2 scored by a composite evaluator in the same two-hard-gate-plus-weighted-axes style as Captain/OOW (safety-critical-procedure-followed as a hard gate, root-cause-identification-accuracy and citation-accuracy as weighted axes).

---

<a id="sec-5"></a>
## 5. `Data/ChiefEngineer/` folder structure

Mirrors the `AgentPaths` convention exactly (`core/paths.py`) — a new `AgentPaths.chief_engineer()` classmethod would be added alongside `.vhf()`/`.oow()` when coding starts:

```
Data/ChiefEngineer/
├── ChiefEngineerManuals/        ← source_dir: the reference installation's manuals (text)
│   └── Drawings/                ← P&ID/GA/wiring diagrams (new artifact type, §4.3)
├── ChiefEngineer_Eval/          ← held-out, NEVER used for training
│   ├── chiefengineer_gold_answers.json
│   └── chiefengineer_malfunction_scenarios.json
├── ChiefEngineer_JSON/          ← §8-style per-document JSON (text) + drawing-caption JSON
└── ChiefEngineer_Agents_Training/  ← RAG chunks/embeddings, KG, PG, SFT/DPO/Reflection jsonl
```

`_models/ChiefEngineer/` for any eventual fine-tuned artefacts, following the exact same `models_root`/`domain_models_dir` convention already shared by VHF/OOW.

---

<a id="sec-6"></a>
## 6. Exhaustive malfunction taxonomy (troubleshooting categories, v1 starting list)

Mirrors the structure of Captain's §4 brown-envelope taxonomy, scoped to engine-department technical faults (overlaps intentionally with Captain's §4.C "Technical/mechanical" category — this is the detailed, engineer's-eye-view expansion of that same category, not a competing taxonomy):

<details>
<summary><b>A. Main propulsion</b></summary>

- Cylinder exhaust gas temperature deviation (single or multiple cylinders)
- Abnormal cylinder pressure (Pmax/Pcomp out of range)
- Turbocharger surge/fouling
- Scavenge fire
- Piston ring/liner wear beyond limit
- Crankcase oil mist / crankcase explosion risk
- Fuel injector fouling/dribbling
- Excessive vibration (misalignment, torsional)
- Low starting air pressure
</details>

<details>
<summary><b>B. Electrical generation</b></summary>

- Generator overload / load-sharing failure
- Insulation resistance drop
- Blackout (total loss of electrical power)
- Automatic voltage regulator (AVR) fault
- Switchboard/breaker trip
</details>

<details>
<summary><b>C. Fuel and lube oil systems</b></summary>

- Purifier bowl fault / sludge discharge failure
- Fuel contamination (water, catalytic fines)
- Lube oil low pressure
- Lube oil contamination (water, fuel dilution, wear-metal spike)
- Viscosity control fault (heater/cooler malfunction)
</details>

<details>
<summary><b>D. Cooling and compressed air</b></summary>

- Sea water pump failure / strainer blockage
- Fresh water cooling leak
- Central cooler fouling
- Starting/control air compressor failure
- Air receiver safety valve/relief fault
</details>

<details>
<summary><b>E. Steering and auxiliary machinery</b></summary>

- Steering gear hydraulic failure (shared with Captain's existing brown-envelope category)
- Bilge/ballast pump failure
- Fire pump failure
- Boiler low water / flame failure
- Economiser soot fire
</details>

<details>
<summary><b>F. Automation / alarm monitoring system</b></summary>

- False alarm (single-sensor, needs corroboration — direct analogue to Captain's §13.A.5 evidence/reliability framing)
- Alarm system failure (loss of monitoring)
- UMS (unmanned machinery space) watch-transfer fault
</details>

This list is explicitly a **starting point** — once a reference installation is chosen (§4.1) and its manual's own fault-finding tables are ingested, the real taxonomy should be derived from that manual directly rather than only from this hand-authored list, exactly as Captain's taxonomy was later supplemented by real CHIRP/MAIB content.

---

<a id="sec-7"></a>
## 7. Documentation/manual/drawing sourcing — research status

A first real acquisition pass ran 2026-10-03 via a new script, `pipeline/ingest/build_chief_engineer_corpus.py` (mirrors `build_captain_legal_corpus.py`'s raw-acquisition-only posture exactly — no parsing/chunking yet). 21 real files now sit in `Data/ChiefEngineer/ChiefEngineerManuals/` (+5 drawings from the earlier pass, §4.3), all provenance-tracked in `manifest.json`:

| Source | Content | Status |
|---|---|---|
| **US Navy NAVEDTRA training manuals** (Engineman 1&C, Engineman 2, Fireman, Machinery Repairman, Fluid Power, Blueprint Reading and Sketching, Basic Machines, Tools and Their Uses, Principles of Naval Engineering, Joint Oil Analysis Program Manual) | Real, detailed, illustrated engine-room/machinery training text — the oil-analysis manual is a direct CBM/Part-A source | **ACQUIRED 2026-10-03** — 10 PDFs downloaded from `legacy.maritime.org` (US federal government works, public domain). Key technical finding: the site's WAF 403s every bare scripted request (confirmed via both PowerShell `Invoke-WebRequest` and Python `urllib`, even with a full realistic browser header set) but accepts the identical request once a `Referer` header pointing at its own `/doc/index.php` manuals index is added — this is what makes scripted acquisition work here, now the script's default for every maritime.org fetch. |
| **CHIRP Maritime Feedback Newsletters** (`Data/MarineNewsLetters/`) | Real engine-room near-miss/incident narratives | **Already in repo, already parsed** (`build_chirp_json.py`) — the same ~427 currently-unused-for-OOW articles flagged in `design_captain_missions.md`'s Captain research are a direct, zero-new-scraping source for Track 2 malfunction dialogues once filtered for engine-department relevance (a new filter, not the existing COLREG-concept one) — filter not yet built. |
| **MAIB/NTSB/TSB/ATSB incident reports** (`Data/OOW/OOW_Incidents/`, `Data/Captain/Processed Leo/`) | Real investigated machinery-failure casualties (fire, flooding, blackout, grounding-from-propulsion-loss) | **Already in repo** — needs a machinery-relevance filter (different from the existing collision-regex filter used for the Processed-Leo MAIB set) — filter not yet built. |
| **MARPOL Annex VI implementing legislation** (`Data/Captain/Legal_Reference/uk_legislation/`) | Already-acquired UK SI for Annex VI (air pollution) | **COPIED 2026-10-03** (not re-downloaded, per the user's own instruction to reuse already-acquired data) into `Data/ChiefEngineer/ChiefEngineerManuals/legal_reference/uk_marpol_annex_vi_air_pollution.xml`. Same legislation.gov.uk `CommentaryRef key-<32 hex>` false-positive-secret pattern already documented for Captain recurs here (confirmed via grep) — added the same narrow, documented `.gitignore` exception (`Data/ChiefEngineer/ChiefEngineerManuals/legal_reference/*.xml`), `manifest.json` stays tracked. |
| **Classification society rules (ABS/DNV/LR) — machinery sections** | Survey/CBM scheme requirements | **CHECKED 2026-10-03, CONFIRMED GATED** — DNV's own "Rules and Standards Explorer" and ABS's "MyFreedom" client portal both require account registration/subscription, same class of blocker as IMO's IMODOCS wall already documented for Captain. Not pursued further. |
| **Manufacturer "Project Guides"** (MAN Energy Solutions, now rebranded **Everllence**) | Real engine specifications, systems diagrams | **ACQUIRED 2026-10-03** — the earlier "unverified" finding was from a stale pre-rebrand man-es.com URL; the real current site (`everllence.com/marine/products/planning-tools-and-downloads/project-guides`) publishes genuine "Complete Project Guide" PDFs with **no login wall at all**. Downloaded 2 real two-stroke guides directly matching this design's own MAN B&W S-series reference-engine choice (§4.1): `S50ME-C10.7_project_guide.pdf` (31.4 MB) and `S60ME-C10.7_project_guide.pdf` (29.5 MB). WinGD/Wärtsilä equivalents not checked this pass. |
| **Reeds Marine Engineering / Pounder's Marine Diesel Engines / Marine Engineering study guides** | The real, industry-standard engineer training textbooks | **Commercially sold and copyrighted** — same explicit guardrail already documented for officer-training textbooks in Captain's research (`design_captain_missions.md` §5.4): **do not scrape without a licence.** |
| **Wikipedia** (Diesel engine, Marine propulsion, Marine engineering, Condition monitoring, Predictive maintenance, Turbocharger, Crankshaft, Marine steam engine articles) | General background, CC BY-SA 4.0 | **ACQUIRED 2026-10-03** — all 8 articles' plaintext extracts pulled via Wikipedia's own public API, fine for training with attribution, not for verbatim redistribution. |
| **Technical drawings (P&ID/GA) for a specific reference installation** | Needed for §4.3 | **Still partially open** — the 5 drawings from the earlier pass (electrical/oil×2/water/gas-fuel) are unchanged; hydraulic and pneumatic/compressed-air were RE-SEARCHED this pass (several new Wikimedia Commons MediaSearch term variants: "steering gear diagram NARA", "air compressor piping diagram NARA", etc.) and still returned zero hits — stopped per the project's "don't brute-force a blocked approach" rule, remains the single biggest open gap in §4.3/§9. |

---

<a id="sec-8"></a>
## 8. Simulator representation — Engine Room Dashboard

<a id="sec-8-1"></a>
### 8.1 Chain of command: Chief Engineer → Captain, not OOW — DECIDED 2026-10-03

User asked directly whether OOW should also receive the Chief Engineer's dashboard data, or whether Captain alone is the right interface — answered using the chain of command this project has **already established and committed to**, not a new invention: `design_captain_missions.md` §2.5 already states "the Captain's instruction to the OOW in such a case is really a **relay** of the Chief Engineer's own technical constraint, escalated through the Captain, not a routine bridge matter," and §3.3 already has the exact mechanism for this — a Captain-issued instruction (e.g. a speed cap) renders as an additional constraint line in the OOW's prompt, same as any other Captain instruction. **Decision: the Chief Engineer agent communicates to the Captain only.** OOW never talks to the Chief Engineer directly and never sees raw `EngineStatus`/condition-monitoring data — it only ever sees the *consequence* (a speed/course limit) via the same `captain_speed_cap_kn`-style constraint field Captain already uses to constrain OOW (`design_captain_missions.md` §15.2). Reasons this is the right call, not just the simpler one:

1. **No new wiring needed.** This is exactly the interface `EngineFailureContext`/`candidates_engine_failure` (`pipeline/captain_agent_spec.py`) already assumes — Part A's job is only to become a real *producer* of `EngineStatus` (§3.6), the *consumer* side (Captain → OOW relay) is already fully designed.
2. **Matches real bridge/engine-room doctrine.** OOW's job is course/speed/collision-avoidance execution; technical machinery judgement (how serious is this fault, what's the safe operating envelope) is explicitly the Chief Engineer's and Captain's domain, not the OOW's — giving OOW raw engine telemetry would blur a chain-of-command boundary this project has deliberately kept sharp (the same reasoning already used to justify why Captain and OOW are two separate models in the first place, `design_captain_missions.md`'s "why not one model for everything" table).
3. **Keeps OOW's prompt facts-only and uncluttered** — OOW already receives a lot of COLREG/geometry facts each step; adding raw machinery data it can't act on directly (it can only ever receive a *speed cap*, same end effect as today) would add noise without adding capability.

<a id="sec-8-2"></a>
### 8.2 Dashboard panel

Mirrors Captain's own proposed UI pattern (`design_captain_missions.md` §11.4's Captain panel: gray-box `st.container`, checkpoint-scrubbing, "Inspect moment"/"Show details") rather than inventing new UI conventions — an **Engine Room panel** sits alongside (not above/below in strict hierarchy, but a peer department view) the existing Captain panel:

- **Always-visible status strip**: per-system severity tier (nominal/watch/warning/critical, §3.4), current `EngineStatus.max_speed_kn` cap if active, and a running "time since last PMS job" indicator — mirrors Captain's own persistent "cockpit" status strip.
- **Condition detail view**: selecting a system (e.g. "Main engine — Cylinder 3") shows its current `ConditionReading` trend plot, the `ConditionLimit` it's being measured against (with `source_citation` link into the ingested manual), and the live `MaintenanceRecommendation` if any.
- **Chatbot panel** (Part B): a free-text query box for "ask the Chief Engineer" — available to the Captain (via the simulator's own operator, standing in for the human player), not exposed as an OOW-facing control, consistent with §8.1.
- **Incident log**: the synthetic incident history (§3.8) rendered as a simple timeline, same "Mission Log" unified-timeline spirit as Captain's §11.5, scoped to this department.

---

<a id="sec-9"></a>
## 9. Decisions (resolved 2026-10-03) and remaining open items

All 5 of the originally-open questions from this design note's first pass are now resolved by direct user instruction:

1. **Reference installation** — DECIDED: use the Sawada et al. (2021) vessel already shared by OOW/Captain (§4.1), not a separate smaller engine. The main propulsion engine (two-stroke, MAN B&W S-series-class) is the v1 scope; the auxiliary generator is a natural v2/parallel addition, not the v1 starting point as originally recommended.
2. **Drawing sourcing** — DECIDED: download real, public-domain drawings from whatever vessels/engines are actually findable (electrical/hydraulic/pneumatic/water/gas/oil categories), and *presume*/treat them as the reference vessel's own for illustration — never invent drawings from scratch, and never hide the real provenance in the manifest (§4.3). A first batch of 5 (electrical, oil ×2, water, gas/fuel) is already downloaded into `Data/ChiefEngineer/ChiefEngineerManuals/Drawings/`; hydraulic and pneumatic/compressed-air remain unfound, flagged for a future search pass.
3. **Fine-tuning scope** — DECIDED: YES, Chief Engineer gets its own QLoRA SFT/DPO(+Reflection) fine-tune, in parallel with VHF/OOW/Captain's existing per-domain fine-tunes (own `AgentPaths(domain="ChiefEngineer", ...)`, own training-data builders) — **not** a pure retrieval-only system. Per the user's explicit instruction, wherever an existing pipeline script/mechanism from another domain already does the needed job, **copy/reuse it directly** rather than redesigning — this is already this whole design's default posture (§4.4/§4.5/§4.6 all reuse existing scripts unchanged) and is now confirmed as the intended approach for the training stages too (reuse `build_sft.py`/`build_rlhf.py`/`build_reflection.py`/`build_multihop.py`/`train_sft.py`/`train_dpo.py`/`train_reflection.py` unmodified, parameterised by the new domain, exactly as OOW and Captain already do/plan to do).
4. **Part A's sensor data** — DECIDED: generate **synthetic** data for both halves — synthetic degradation traces (§3.7, already planned) **and** a synthetic incident/casualty history for the reference vessel's own engine room, also feeding Part B's RAG corpus (§3.8, new subsection added this pass). Not yet wired into `app/mission_sim.py`'s live clock — that remains a later integration step once the walking-skeleton condition-evaluation functions (§3.4) exist, same "don't build the UI/live-wiring before the backend logic works" sequencing Captain's own design already adopted (`design_captain_missions.md` §15).
5. **Where Part A's output surfaces** — DECIDED: both. `EngineStatus` continues to feed the existing Captain pipeline unchanged (§3.6), **and** a new standalone Engine Room Dashboard is added (§8), mirroring Captain's own simulator-UI proposal. The dashboard talks to the **Captain**, not the OOW (§8.1) — this also resolves the user's own follow-up question about whether OOW needs this data (it does not; OOW only ever receives the already-planned `captain_speed_cap_kn`-style relayed constraint).

**Remaining genuinely open items** (not resolved by this pass, carried forward):
- Final sourcing of hydraulic and pneumatic/compressed-air reference drawings (§4.3) — re-searched 2026-10-03, still not found.
- WinGD / Wärtsilä project-guide equivalents not checked (MAN Energy Solutions/Everllence's own guides ARE now acquired, §7).
- Exact machinery-relevance filter for mining the existing MAIB/NTSB/TSB/ATSB/CHIRP corpora (§7) — not yet designed, same status as before this pass.
- No parsing/chunking/RAG-building has been done on the 21 newly-acquired files — same "raw acquisition only" posture as `build_captain_legal_corpus.py`, deciding what enters RAG/KG/PG is a separate later step.
- No code has been written for either Part A or Part B's actual condition-monitoring/retrieval logic — this remains a design/research document only for that part, per the user's original instruction. (The new `pipeline/ingest/build_chief_engineer_corpus.py` is acquisition tooling, not Part A/B logic.)

## 2026-10-03 (same day, later): first real corpus acquisition pass — 21 files
User asked to download/scrape "as much as possible" of everything listed in §7. Built
`pipeline/ingest/build_chief_engineer_corpus.py` (raw acquisition only, mirrors
`build_captain_legal_corpus.py`) and ran it successfully: 10 NAVEDTRA manuals, 2 real
MAN B&W S-series Project Guide PDFs, 1 copied MARPOL Annex VI UK SI, 8 Wikipedia
background articles — all in `Data/ChiefEngineer/ChiefEngineerManuals/`, provenance in
`manifest.json`. Checked and ruled out DNV/ABS (subscription-gated) and re-searched
hydraulic/pneumatic drawings (still not found). Added a `.gitignore` exception for the
copied MARPOL XML (same legislation.gov.uk false-positive secret pattern as Captain's
corpus). Full detail in §7's table above.
