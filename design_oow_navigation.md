# Design Note: Officer of the Watch (OOW) — Reverse-Engineered Architecture (Track 2)

**Status:** DOCUMENTATION of an EXISTING, actively-built system — unlike `design_captain_missions.md` (Track 3, design-only) and `design_chief_engineer.md` (Track 4, design-only), OOW already has a large amount of real code, real training data, and real evaluation results. This note was written 2026-10-03 by reverse-engineering the current state of `pipeline/oow_agent_spec.py`, `Basic Simulator/app/*.py`, the `pipeline/track1/`/`pipeline/track2/`/`pipeline/train/` scripts, and `Data/OOW/` — it documents **what is actually built today**, at the same level of detail as the Captain design note, so OOW has a single canonical architecture reference instead of only scattered session notes (`pipeline-notes.md`, `basic_simulator.md`). Where the system is still evolving or has known open bugs, this is stated explicitly (§11) rather than glossed over.

## Table of contents

- [1. OOW job description](#sec-1)
  - [1.1 Legal/regulatory basis](#sec-1-1)
  - [1.2 Core responsibilities (as implemented)](#sec-1-2)
  - [1.3 Relationship to Captain/Chief Engineer](#sec-1-3)
- [2. Two-track architecture](#sec-2)
- [3. Ship dynamics: Nomoto physics](#sec-3)
  - [3.1 Two kinematics models](#sec-3-1)
  - [3.2 Ship profiles (ship-dynamics-agnostic training)](#sec-3-2)
- [4. The decision task: prompt, schema, risk model](#sec-4)
  - [4.1 `SYSTEM_OOW_AGENT` — the live system prompt](#sec-4-1)
  - [4.2 The JSON action schema](#sec-4-2)
  - [4.3 The risk model: `real_risk()`/`risk_band()`/risk horizon](#sec-4-3)
  - [4.4 `classify_encounter()`/`classify_rules()` — COLREG ground truth](#sec-4-4)
  - [4.5 Adaptive decision cadence](#sec-4-5)
  - [4.6 Per-row sampled training variables](#sec-4-6)
- [5. Retrieval stack: RAG + KG + Reranking + PG](#sec-5)
  - [5.1 The 14 model configs (`v0_base`...`v11`)](#sec-5-1)
  - [5.2 RAG + KG retrieval](#sec-5-2)
  - [5.3 Reranking](#sec-5-3)
  - [5.4 Procedural Graphs (PG)](#sec-5-4)
  - [5.5 Corpus scoping (`colreg_only`)](#sec-5-5)
- [6. Deterministic baselines + the oracle planner](#sec-6)
  - [6.1 The 6 baselines](#sec-6-1)
  - [6.2 `oracle_planner.py`](#sec-6-2)
- [7. Data pipeline](#sec-7)
  - [7.1 `Data/OOW/` folder structure](#sec-7-1)
  - [7.2 Ingestion](#sec-7-2)
  - [7.3 Track 1 builders (rules & knowledge)](#sec-7-3)
  - [7.4 Track 2 builders (conversational/mission compliance)](#sec-7-4)
  - [7.5 Mission/scenario generation](#sec-7-5)
- [8. Training pipeline](#sec-8)
  - [8.1 SFT/DPO/Reflection](#sec-8-1)
  - [8.2 Model variants registry](#sec-8-2)
- [9. Evaluation](#sec-9)
  - [9.1 `evaluate_run.py` — the composite scorer](#sec-9-1)
  - [9.2 `measurement.py` — per-decision quality checks](#sec-9-2)
  - [9.3 `audit_runs.py` — aggregate auditing](#sec-9-3)
  - [9.4 Track 2 Q&A-style eval](#sec-9-4)
- [10. Simulator (Basic Simulator/app/)](#sec-10)
- [11. Known open issues / research threads](#sec-11)
- [12. Relationship to Captain and Chief Engineer](#sec-12)

---

<a id="sec-1"></a>
## 1. OOW job description

<a id="sec-1-1"></a>
### 1.1 Legal/regulatory basis

Already established in `design_captain_missions.md` §2.1 (written when the Captain layer needed OOW clearly defined as the layer it oversees) — not re-derived here, just restated for this doc's own completeness:

- **STCW Code Section A-VIII/2, Part 3-1**: primary responsibility for safe navigation during the watch, on the Master's behalf.
- **COLREG Rule 5**: maintain a proper look-out "by sight and hearing, as well as by all available means" — continuously assess risk of collision and take timely, substantial, COLREG-compliant action. **This is the exact competency this whole domain trains and evaluates.**
- **SOLAS V/34**: execute the pre-approved passage plan; deviations require a new instruction (Captain) or an unavoidable COLREG manoeuvre.
- **ICS Bridge Procedures Guide**: the de-facto industry-standard bridge-team manual most Safety Management Systems are built on — grounds the subordinate-role structure (§2.2–§2.4 of the Captain doc: Look-out, Helmsman, QMOW).

<a id="sec-1-2"></a>
### 1.2 Core responsibilities (as implemented)

The live system (`pipeline/oow_agent_spec.py`'s `SYSTEM_OOW_AGENT`, §4.1) reduces this legal mandate to exactly 3 **mission requirements**, explicitly ranked by a **priority order**:

1. Every contact's CPA stays at or above the mission's safe passing distance at all times.
2. Every manoeuvre taken while a real collision risk exists complies with COLREG.
3. The mission goal is reached.

Priority order when these pull in different directions: **(1) collision avoidance first, (2) mission progress second** — never reversed. This is deliberately the *entire* explicit policy stated to the model; everything else (which rule applies, how to resolve it) is left to be derived from stated facts + retrieved COLREG text + learned weights, not hand-written imperative rules (§4.1's "facts-only" design history).

<a id="sec-1-3"></a>
### 1.3 Relationship to Captain/Chief Engineer

OOW is the **execution layer** in this project's 3-tier command architecture (`design_captain_missions.md`'s "why not one model for everything" table): Captain plans/intervenes at mission scale (minutes–hours, brown-envelope events), OOW executes tactically at encounter scale (30–200s decisions), Chief Engineer reports machinery condition upward through Captain only (never directly to OOW — `design_chief_engineer.md` §8.1). Concretely: a Captain instruction (e.g. a speed cap after an engine fault) renders as **one additional constraint line in OOW's existing prompt** — the exact same mechanism already used for `VesselConstraints` (§4.1), so no new prompt-injection architecture is needed when Captain eventually gets built.

---

<a id="sec-2"></a>
## 2. Two-track architecture

OOW follows the same two-track convention established by VHF (`.github/copilot-instructions.md`):

| | Track 1 — Rules & knowledge | Track 2 — Conversational/mission compliance |
|---|---|---|
| **Question** | Does the model know COLREG, real incident lessons? | Can the model actually sail a mission, making live turn/speed decisions under real geometry? |
| **Eval data (held out)** | `Data/OOW/OOW_Eval/gold.json` (COLREG Q&A, VHF-gold-answers-style) | `Data/OOW/OOW_Eval/scenario.json` + the live mission corpus itself (Imazu/UM/RND/IMP — see §7.5), scored via `evaluate_run.py` |
| **Training data sources** | COLREG text (`build_oow_json.py`), real incident reports (MAIB/NTSB, `screen_incidents.py`→`extract_incident_reasoning.py`), CHIRP newsletters | Synthetic MOOS-derived scenario traces (`build_oow_scenarios.py`), Leo's 7928-state MOOS-trajectory dataset (`build_oow_scenarios_leo.py`), oracle-planner-labeled RND rollouts, DAgger-mined pairs from real model mistakes |
| **Training-row style** | `oow_sft_{direct,cot,rag}.jsonl`, `oow_multihop.jsonl`, `oow_dpo_pairs.jsonl`, `oow_reflection.jsonl`, `oow_pg_sft.jsonl` | `oow_scenario_*_nomoto.jsonl` family, `oow_outcome_dpo_pairs.jsonl`, `oow_dagger_dpo_pairs.jsonl`, `oow_measurement_dpo_pairs.jsonl` |

Both tracks feed the **same** QLoRA SFT→DPO(→Reflection) fine-tune (§8), kept in separate files so each competency's contribution stays traceable — identical convention to VHF.

---

<a id="sec-3"></a>
## 3. Ship dynamics: Nomoto physics

<a id="sec-3-1"></a>
### 3.1 Two kinematics models

`Basic Simulator/app/simulation.py`'s `VesselConstraints.kinematics_model` selects between:

- **`"kinematics"`** (legacy default): a flat turn-rate slew (`turn_rate_deg_s`, default 3.0 deg/s) — own-ship's heading rate-limits toward a commanded `target_heading` every step, no inertia/lag modelling.
- **`"nomoto"` / `"nomoto_v2"`** (opt-in, physically realistic): Sawada et al. (2021)'s 2nd-order Nomoto + rudder-servo model (`pipeline/nomoto.py`) — a real yaw-rate lag (T=50s) *and* rudder-servo lag (T_E=2.5s) stacked, so a commanded turn **builds up gradually** rather than at a constant rate. A 60° turn takes ~155s under Nomoto vs ~20s under the legacy model (~7-8× slower) — this is the single biggest physical-realism upgrade OOW has had, and genuinely changes which deterministic baselines succeed (§6.1, §11).
- `"nomoto"` vs `"nomoto_v2"` are **the same physics**, different labels only — `"nomoto_v2"` exists so post-fix runs (after the `target_heading`-vs-`own.heading` goal-course-check bug, §11) are never confused with pre-fix archived data.

<a id="sec-3-2"></a>
### 3.2 Ship profiles (ship-dynamics-agnostic training)

`pipeline/nomoto.py`'s `SHIP_PROFILES` holds 3 real, literature-cited Nomoto parameter sets (never invented numbers):

| Profile | Source | Scale | Training weight |
|---|---|---|---|
| `sawada2021` | Sawada, Sato & Majima (2021), *J. Mar. Sci. Technol.* 26:509–524 | L=106m cargo vessel (this project's primary reference ship — also the one `design_chief_engineer.md` adopted) | 0.7 |
| `yukun2023_large` | Wang et al. (2023), *JMSE* 11(11):2101 ("YU KUN" training vessel) | L=105m, same scale as Sawada's but markedly different K/T | 0.3 |
| `xie2023_small` | Xie et al. (2023), *JMSE* 11(2):273, Table 3 | L=52.5m, smaller/faster-responding | **0.0 (held out)** |

`xie2023_small` is **deliberately excluded from training** (`TRAINING_PROFILE_WEIGHTS`) and reserved as `HELD_OUT_EVAL_PROFILE` — a genuine unseen-ship-dynamics generalisation test, not just in-distribution recall. `pipeline/oow_agent_spec.py`'s `sample_ship_profile(row_id, allow_held_out=False)` draws deterministically per training row (seeded from the row id) so the model learns to **reason from stated manoeuvre-time facts**, not memorise one ship's response.

---

<a id="sec-4"></a>
## 4. The decision task: prompt, schema, risk model

`pipeline/oow_agent_spec.py` is the **single source of truth** for the task format — imported by both the live simulator (`Basic Simulator/app/agents.py`) and every Track-2 training-data generator, specifically so train and eval can never silently drift onto different tasks (a real, named architectural principle in this codebase, "RAG-rebuild-v2 plan, Fase B2").

<a id="sec-4-1"></a>
### 4.1 `SYSTEM_OOW_AGENT` — the live system prompt

Current version (`PROMPT_VERSION = "2026-09-24-uncapped-turn-adaptive-cadence"`), reproduced here as the canonical reference:

> *"You are the navigator on a large commercial vessel. Decide the next helm order."* — states the 3 mission requirements + 2-step priority order (§1.2), then 4 **facts given to you** (units convention, signed relative bearing, CPA/TCPA definition including the "already past closest point" disambiguation, and that GOAL COURSE CHECK's action/degrees are pre-computed and must never be re-derived), then a 5-step **decision procedure**: (1) check every contact against safe distance/risk horizon, (2) if any real risk exists, pick the ONE COLREG-compliant action — overrides everything below, (3) otherwise follow GOAL COURSE CHECK exactly, (4) never zigzag, (5) speed up instead of holding course once on the goal bearing below rated speed.

This is deliberately a **facts-only** prompt — the single most important architectural decision in OOW's history (2026-09-24 "architecture pivot", `basic_simulator.md`): an earlier, more prescriptive version ("you MUST...", "do NOT answer X...") was found to risk teaching the model to pattern-match trigger phrases instead of genuinely reasoning, and was deliberately stripped back down. Judgement/timing nuance is now **only** acquired via (a) RAG/PG-retrieved real COLREG text (§5), (b) SFT/DPO/Reflection training data (§8) — never a hand-written prompt rule. This is the same principle later explicitly carried into the Captain design (`design_captain_missions.md` §13.B.5) and the Chief Engineer design's Part A (`design_chief_engineer.md` §3.4).

<a id="sec-4-2"></a>
### 4.2 The JSON action schema

```json
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": "<float, only for turn_left/turn_right>",
 "encounter_rule": "<'Rule <N>' or 'none'>",
 "conduct_rule": "<'Rule <N>' or 'none'>",
 "reasoning": "<one or two sentences>"}
```
`degrees` is **uncapped** (2026-09-24 change) — any size turn order is accepted, it just takes longer to physically complete under whichever kinematics model is active (§3.1); `max_rudder_angle_deg` is no longer a per-command cap, only the reference angle risk-horizon sizing uses. `validate_action_json()` (same module) is the one shared schema validator used by the live parser (`_parse_json_action()` in `agents.py`), every Track-2 training-data generator, and the deterministic baselines (§6.1) — so a schema violation can never be defined differently in two places.

<a id="sec-4-3"></a>
### 4.3 The risk model: `real_risk()`/`risk_band()`/risk horizon

`real_risk(cpa_m, tcpa_s, safe_distance_m, risk_horizon_s)` is the **one** definition of "does this contact pose a real collision risk right now" used everywhere (live prompt, training labelers, the auditor's ground truth) — requires **both** CPA below the mission's safe distance **and** TCPA within `[0, risk_horizon_s)`. `risk_band()` classifies every contact into one of 4 bands: `"passed"` (TCPA<0, already diverging), `"safe"` (CPA≥safe distance), `"acute"` (`real_risk()`==True, action mandatory), `"early"` (a real encounter not yet mandating action — identify the rule now, don't ignore it, but don't act with `stop` either). The risk horizon itself is **not** a fixed constant — `derive_risk_horizon_s()` derives it per-mission from how long own-ship actually takes to open the safe distance by turning at its reference angle (a real physics-derived number, `RISK_HORIZON_K=3.5` safety multiple on top of the bare manoeuvre time), with a Nomoto-aware sibling `derive_risk_horizon_s_nomoto()` that uses the real simulated turn time instead of an analytic instant-turn estimate.

<a id="sec-4-4"></a>
### 4.4 `classify_encounter()`/`classify_rules()` — COLREG ground truth

`classify_rules(role, action, restricted_visibility=False) -> (encounter_rule, conduct_rule)` is the one ground-truth mapping from (encounter role, action taken) to the correct rule citation pair — `role` is one of `mutual` (head-on, Rule 14), `give_way`/`stand_on` (crossing, Rule 15), `overtaking_give_way`/`overtaking_stand_on` (Rule 13), `stationary` (a real-risk non-vessel object — Rule 8 only, never a COLREG encounter rule), or no real risk at all. This single mapping replaced an earlier, conflated `rule_applied` field that mislabelled every emergency stop as Rule 17 regardless of role (Rule 8 is the correct citation for a give-way vessel's own emergency stop; 17(b) is exclusively the stand-on vessel's).

<a id="sec-4-5"></a>
### 4.5 Adaptive decision cadence

`live_decision_interval()` (`app/narrate.py`) recomputes the gap until the next LLM call **at every checkpoint** from the live encounter state — 10/15/20 simulation steps depending on the closest contact's TCPA bucket (<250s/<600s/otherwise), floored to 3 steps the moment any contact enters real acute risk. This replaced an earlier "inversely adaptive" bug where the cadence was computed once from mission start and never tightened even as a real risk became imminent. Capped by `transit_step_cap()` so a fast/short mission (Imazu-scale, ~20 m/s) never ends up with fewer than ~5 decision points across its whole transit.

<a id="sec-4-6"></a>
### 4.6 Per-row sampled training variables

Three training-data constants are **sampled per row**, not fixed, so the model never learns a shortcut against one hardcoded number: `safe_distance_m` (300/400/500/750/926m, weighted toward 500), `max_turn_deg` (20/25/30/35°, weighted toward 30), and a `risk_horizon_multiplier` (0.6–1.6× the geometry-derived default). `decision_interval_s` is also sampled (30/100/150/200s) to match the live adaptive cadence's own range. `sample_row_limits()`/`sample_row_limits_nomoto()` (the Nomoto-aware sibling, also sampling a ship profile, §3.2) are both **deterministic per row id** (seeded via SHA-256 of the row id) — the same row always regenerates identically.

---

<a id="sec-5"></a>
## 5. Retrieval stack: RAG + KG + Reranking + PG

<a id="sec-5-1"></a>
### 5.1 The 14 model configs (`v0_base`...`v11`)

`Basic Simulator/app/agents.py`'s `MODEL_CONFIGS`/`_CONFIG_SPECS` define every prompt-ingredient combination the live simulator (and the offline sweep tooling) can run:

| Config | rag | rerank | cot | pg | corpus |
|---|---|---|---|---|---|
| `bare_qwen` | – | – | – | – | zero-extra-framing ablation floor (own separate minimal prompt) |
| `v0_base` | no | – | no | – | OOW framing only, no retrieval |
| `v1_rag`...`v6_pg_scenario` | varies | no | varies | varies | **archived ablation arm** — 147 runs depend on these exact definitions, never redefine |
| `v7_super_rag` | yes | yes | no | – | full retrieval stack (dense+KG+rerank), full corpus |
| `v8_super_cot_pg` | no | – | yes | scenario+incident | CoT + PG, no retrieval |
| `v9_super_all` | yes | yes | yes | scenario+incident | v7+v8 combined |
| `v10_super_colreg_rag` | yes | yes | no | – | **same retrieval stack, scoped to the COLREG-only index** (123 chunks vs 3870 full) |
| `v11_super_colreg_rag_cot` | yes | yes | yes | scenario+incident | v10 + CoT/PG |

`v10`/`v11`'s `corpus="colreg_only"` exists because the full-corpus dense index was found to systematically out-score real rule text with `leo_moos_cases`/incident chunks regardless of real relevance (§11) — scoping to just `colreg_consolidated_2018`+`simple_colreg` also matches the reranker's own training distribution (its 76 training pairs were ALL colreg-chunk pairs, never a `leo_moos_cases` candidate).

<a id="sec-5-2"></a>
### 5.2 RAG + KG retrieval

`pipeline/ingest/build_kg.py`'s `kg_retrieve()` — dense cosine-similarity retrieval (bge-base-en-v1.5, `core/embedding.py`) plus KG concept-boosting (keyword→chunk-id lookup for COLREG vocabulary) over whichever corpus a config selects. `max_per_document` (currently `RAG_MAX_PER_DOCUMENT=1` in `agents.py`) caps how many chunks from the SAME source document can appear in one retrieval pool — added to fight a real, measured degenerate-retrieval bug (§11) where 2 fixed MOOS-case chunks dominated almost every query.

<a id="sec-5-3"></a>
### 5.3 Reranking

A fine-tuned cross-encoder (`pipeline/train/train_reranker.py`, `oow_reranker/`, ~22M params, CPU) reranks a wider dense pool (`dense_n`, default 40) down to the final `k` (default 6) for every `rerank=True` config. Trained on 76 (situation-query, single-Rule-N-chunk) pairs — exclusively COLREG-text pairs, which is exactly why `v10`/`v11`'s `colreg_only` corpus scoping brings inference back in line with its own training distribution.

<a id="sec-5-4"></a>
### 5.4 Procedural Graphs (PG)

`pipeline/ingest/pg_guidance.py`'s `ProceduralGraph`/`render_guidance()` — a graph mined from incident-report and Track-2 scenario reasoning traces, encoding real procedural sequences ("assess the fused contact picture → determine own-ship's role → execute the action"). 3 separate graphs exist (`merged`, `incident`, `scenario`) plus a combined `scenario+incident` load used by v8/v9/v11. Retrieval into this graph uses its own **separate query**, `_pg_match_query()` — built from `classify_encounter()`'s own already-computed ground truth (never shown to the model directly, exactly the same retrieval-only use of ground truth RAG's own query already makes) — NOT the raw numeric situation report, since the graph's nodes are mined from rule/encounter vocabulary that a deliberately-numeric-only situation report never contains (a real, previously-fixed bug: every mission got the identical generic fallback guidance before this fix).

<a id="sec-5-5"></a>
### 5.5 Corpus scoping (`colreg_only`)

A separate, pre-built, much smaller retrieval index (`oow_rag_chunks_colreg_only.json` + embeddings/ids/KG, 123 chunks) filtered from the full corpus (no re-embedding needed) — exists purely to let v10/v11 retrieve from COLREG rule text only, bypassing the full corpus's `leo_moos_cases`/incident-report chunks entirely.

---

<a id="sec-6"></a>
## 6. Deterministic baselines + the oracle planner

<a id="sec-6-1"></a>
### 6.1 The 6 baselines

`Basic Simulator/app/baselines/` — every baseline shares `decide(mission, own, targets, constraints) -> (decision, debug)`, the same schema `ask_oow()` returns, validated by the same `validate_action_json()`:

| Baseline | Method | Citation |
|---|---|---|
| `baseline_ruletree` | Rule-based COLREG decision tree | IMO COLREGS (1972) itself |
| `baseline_vo` | Velocity Obstacle / Collision Cone | Fiorini & Shiller (1998), *IJRR* 17(7) |
| `baseline_apf` | Artificial Potential Field | Khatib (1986), *IJRR* 5(1) |
| `baseline_dwa` | Dynamic Window Approach | Fox, Burgard & Thrun (1997), *IEEE RAM* 4(1) |
| `baseline_mpc` | Model Predictive Control, rollout-based | Garcia, Prett & Morari (1989), *Automatica* 25(3) |
| `baseline_sawada` | Sawada et al. (2021)'s own CRI-based conventional method | best-effort reconstruction, *J. Mar. Sci. Technol.* 26(2) |

These exist for 2 purposes: (a) a non-LLM comparison point for every LLM config, (b) raw material for the oracle planner's own candidate generation (§6.2). A real falsification study (`Docs/nomoto_dynamics_design_and_verification.md` §12) found Nomoto physics (§3.1) alone (no adversarial manoeuvring needed) breaks `vo`/`dwa`/`mpc` outright at the mildest severity setting, `apf`/`ruletree` at moderate-to-high severity, and only `baseline_sawada` resisted every attempted break — attributed to it being the only baseline that scales its avoidance-turn *magnitude* continuously with risk rather than assuming an instantly-achievable candidate heading.

<a id="sec-6-2"></a>
### 6.2 `oracle_planner.py`

`Basic Simulator/app/oracle_planner.py`'s `plan()` — the project's actual **ground-truth labeling mechanism** for Track 2 data mining (§7.4), not a live-simulator agent. Candidates = the 6 baselines' own first-step turn choice **plus** a coarse ±90°/15° grid, each rolled forward 6×10s under **real Nomoto physics** (not any baseline's own simplified rollout model) using the same cost weights as `baseline_mpc` (`W_GOAL=2.0`, `W_CLEARANCE=5.0`, `W_EFFORT=0.02`). The key addition beyond every baseline: a **hard COLREG rule-legality gate**, `required_direction()`, runs *before* cost-ranking — a candidate violating the mandatory direction for the decisive contact's role/band is excluded outright, never merely cost-penalised. `required_direction()` correctly returns `"hold"` (not "either direction, pick cheaper") for a stand-on vessel not yet past its Rule 17(a)(ii)/(b) deadline — directly targeting the dominant real failure mode found in the 2026-09-29 audit (`B_17c`, "stand-on vessel acted too early").

---

<a id="sec-7"></a>
## 7. Data pipeline

<a id="sec-7-1"></a>
### 7.1 `Data/OOW/` folder structure

```
Data/OOW/
├── OOW_Protocols/              ← source_dir: COLREG-Consolidated-2018.pdf, simple_colreg.json,
│                                   free_radar_workbook.pdf, nav_maths_drills.json, compass/CPA/bearing refs
├── OOW_Eval/                   ← held out, NEVER used for training
│   ├── gold.json                   (Track 1 Q&A)
│   └── scenario.json               (Track 2 scenarios)
├── OOW_JSON/                   ← per-document JSON (§8-style ingest output)
├── OOW_Incidents/              ← raw MAIB/NTSB accident-report PDFs (~806 files)
├── OOW_Agents_Training/        ← RAG chunks/embeddings, KG, PG, SFT/DPO/Reflection jsonl (the bulk of generated data)
├── OOW_Scenarios_Leo/          ← Leo's own raw MOOS-trajectory source data
├── OOW_Data Generation Scripts/← (historical/manual) scenario-authoring scripts
└── OOW_MOOS_Integration/       ← explicitly "later phase", not wired into any pipeline today
```

<a id="sec-7-2"></a>
### 7.2 Ingestion

- `pipeline/ingest/build_oow_json.py` — a bespoke PART/RULE/ANNEX regex parser for the COLREG text, plus `build_extra_docs()` for everything else in `OOW_Protocols/` (generic markdown heading splitter for prose refs, a doc-specific radar-workbook lesson splitter, and `nav_maths_drills.json`'s own already-structured Q&A ingested directly as `"qa"`-type sections).
- `pipeline/ingest/build_chirp_json.py` — parses all CHIRP Maritime Feedback newsletters (`Data/MarineNewsLetters/`) into the same chunk schema; only the subset passing a COLREG-concept filter is currently used for OOW training (the rest — ~70% — is the exact pool `design_chief_engineer.md` §7 flagged as reusable for a different, machinery-relevance filter).
- `pipeline/ingest/screen_incidents.py` → `pipeline/ingest/build_incident_excerpts.py` → `pipeline/track1/extract_incident_reasoning.py` — the 3-stage real-accident-report pipeline (keyword-density screening, opening-summary + Analysis/Conclusions excerpting, then a GPT-4o-mini reasoning-trace extraction with an additive `incident` sub-object: vessels/roles, fault_attribution, actual_actions_taken vs. correct procedures).

<a id="sec-7-3"></a>
### 7.3 Track 1 builders (rules & knowledge)

Shared with VHF, unchanged scripts, just domain-parameterised: `build_rag.py` (chunks+embeddings), `build_kg.py` (concept graph), `pipeline/track1/build_sft.py`/`build_rlhf.py`/`build_reflection.py`/`build_multihop.py` (deterministic, reused verbatim across VHF/OOW/incident-sourced traces), `pipeline/ingest/build_pg.py` (procedural graph) → `build_pg_sft.py` (step-order Q&A).

<a id="sec-7-4"></a>
### 7.4 Track 2 builders (conversational/mission compliance)

| Script | Source | Produces |
|---|---|---|
| `build_oow_scenarios.py` | Deterministic synthetic MOOS-style scenarios | `oow_scenario_sft_{direct,cot}_nomoto.jsonl`, `oow_scenario_dpo_pairs_nomoto.jsonl`, reflection rows |
| `build_oow_scenarios_leo.py` | Leo's 7928-state real MOOS-trajectory dataset | `oow_scenario_Leo_*_nomoto.jsonl` (capped at 15% of the non-Leo Track-2 pool at load time, §8.1) |
| `build_oow_scenarios_rnd.py` | The random-geometry RND01–60 pool (§7.5), rolled out via `oracle_planner.plan()` | `oow_scenario_RND_*_nomoto.jsonl` (+ `_llc` variant using an opt-in low-level controller to fill gaps between oracle decision points — measurably fixes missions that otherwise collide outright) |
| `build_outcome_dpo.py` / `build_outcome_reflection.py` | **Verdict-driven mining**: every checkpoint of a run whose actual outcome was a collision/CPA-violation/goal-not-reached, where the model's action didn't match the independently-recomputed correct one | `oow_outcome_dpo_pairs.jsonl`/`oow_outcome_reflection.jsonl` — also includes a dedicated `rule13_mislabel_events()` category (counter-examples for a measured Rule-13-over-citation bias, §11) |
| `build_measurement_dpo.py` / `build_measurement_reflection.py` | Per-decision reasoning-quality checks (`app/measurement.py` Checks A/B/C), independent of whether the mission ultimately failed | `oow_measurement_dpo_pairs.jsonl` |
| `build_dagger_dpo.py` | **DAgger mining**: replays already-recorded closed-loop LLM run logs, recomputes `oracle_planner.plan()` at every visited checkpoint, mines a pair wherever they disagree | `oow_dagger_dpo_pairs.jsonl` — the only source that mines mistakes from states the model *itself* actually visits, not a fixed generator's rollout |

All Track-2 generators gate on `PROMPT_VERSION` (and a byte-exact prompt hash for stricter audit aggregation) so a run generated under a superseded prompt architecture is never silently mined as if it were current.

<a id="sec-7-5"></a>
### 7.5 Mission/scenario generation

| Family | Generator | Count | Purpose |
|---|---|---|---|
| Imazu | `generate_imazu_missions.py` | 22 (2 permanently removed as exact duplicates) | The original academic COLREG benchmark set |
| UM | `generate_um_rescaled_missions.py` | ~13 | Imazu-geometry rescaled to Sawada-scale (12kt) speeds |
| RND | `generate_random_imazu_missions.py` | 60 (RND01–60) | Parametrised random geometry, 4 weighted `failure_category` tags (stand-on-timing pressure, multi-target crossing RAG risk, CoT time-pressure, baseline-mixed) targeting the 3 worst-measured failure modes from a real audit; ~15% held out by a deterministic index-based flag |
| IMP | `generate_impossible_missions.py` | 10 (IMP01–10) | Hand-authored "brown-envelope"-style adversarial scenarios (wrong-way give-way, simultaneous stand-on/give-way conflict, in-extremis, a blocked escape route, etc.), several using scripted `target_maneuvers` |

All 4 families are plain JSON under `Basic Simulator/Data/missions/`, loaded by `app/missions.py`.

---

<a id="sec-8"></a>
## 8. Training pipeline

<a id="sec-8-1"></a>
### 8.1 SFT/DPO/Reflection

Identical 3-stage QLoRA pipeline to VHF (`pipeline/train/train_sft.py`→`train_dpo.py`→`train_reflection.py`→`merge_adapter.py`), parameterised by `AgentPaths.from_env()`/`AUTOPILOT_DOMAIN=OOW`. `train_sft.py`'s OOW file list spans both tracks (§2) plus the Nomoto-aware Track-2 family exclusively — the legacy (non-Nomoto) scenario/Leo siblings were **deliberately dropped from the mix** (2026-09-28) since Nomoto is now the only ship-dynamics model anything downstream actually exercises. Leo's own 7928-row pool is capped at **15% of the non-Leo Track-2 row count** (computed dynamically, not a stale fixed number) so it can never dominate the mix the way it once did (69% of all SFT rows, pre-fix).

Reflection is trained but **not currently deployed**: the Draft/Critique/Refined format-leak bug was fixed (2026-09-25), but the resulting model still measurably underperforms SFT+DPO alone on real mission runs (collided on cases SFT+DPO alone reached goal on) — not registered as a usable `model_variants.py` entry until this is addressed.

<a id="sec-8-2"></a>
### 8.2 Model variants registry

`Basic Simulator/app/model_variants.py`'s `MODEL_VARIANTS` — a short, filesystem-safe id → `--weights` string + label mapping (needed because raw weights strings like `"MERGED:<dir>"` contain characters illegal in Windows filenames). Registered variants as of this note: `qwen_base` (untuned), `qwen_sftdpo` (the main comparison baseline, quantization-matched merge), `qwen_sftdpo_nomoto` (same stages, trained on a mix of legacy+Nomoto Track-2 data), `qwen_sftdpo_nomoto_v2` (adds oracle-planner RND rollouts + expanded DAgger mining + a small RFT self-consistency set).

A real, documented pitfall (shared with VHF): QLoRA adapters **must** be merged onto a base model loaded in the **same 4-bit NF4 quantization** used during training — merging onto full-precision bf16 silently miscalibrates every adapter (symptom: normal general chat ability, 100% format-task failure) — `merge_adapter.py` already does this correctly.

---

<a id="sec-9"></a>
## 9. Evaluation

<a id="sec-9-1"></a>
### 9.1 `evaluate_run.py` — the composite scorer

`Basic Simulator/Evaluation Functions/evaluate_run.py` — loosely follows Woerner, Benjamin, Novitzky & Leonard's 4-axis COLREG-compliance-metrics framework (*Autonomous Robots*, 2019), extended with 3 more of this project's own:

- **Safety** (hard gate: any literal collision → composite=0, not just a weighted penalty; otherwise a continuous `min(1, worst_cpa/safe_distance)` term).
- **Compliance** — split (2026-09-24) into **manoeuvre-category** codes (was the physical action actually safe+COLREG-correct) and **explanation-category** codes (did the model's self-reported rule citation match ground truth) — kept deliberately separate after the old blended score was found to be dominated by citation noise, not real physical-safety violations.
- **Temporal/spatial efficiency** — vs. a straight-line baseline; both floored at 0 (not undefined) if the mission was never reached.
- **Manoeuvre count** — counts only genuine *contradictions* (a turn reversing the last commanded direction), grounded directly in Rule 8(b)'s own "a succession of small alterations... should be avoided" — normalised by this run's own decision-opportunity count and contact count, not an absolute cap.
- **Smoothness** — heading-rate/speed-rate control-effort penalty, deliberately kept purely trajectory-based (a real, engine-dependent physical fact under Nomoto, not an artifact).
- **Explanation** (2026-09-29, now weighted into composite at 0.10) — citation-accuracy only, not narrative richness.

Current `DEFAULT_WEIGHTS`: safety 0.35, compliance 0.20, temporal 0.10, spatial 0.10, manoeuvre 0.10, smoothness 0.05, explanation 0.10. Verdict is `"FAIL — collision occurred"` / `"PASS_WITH_CPA_VIOLATION"` (a deliberate 2026-09-23 addition — a min-separation breach without a literal collision is no longer a bare, misleading "PASS") / `"PASS"` / `"FAIL — did not reach the goal"`.

<a id="sec-9-2"></a>
### 9.2 `measurement.py` — per-decision quality checks

`app/measurement.py`'s `measure_decision_quality()` — per-checkpoint Checks A (fabricated risk — a rule cited with no real encounter), B (wrong direction), C (turn order beyond a sanity bound, now a fixed 120° cap, no longer tied to `max_rudder_angle_deg` since turns are uncapped), D (no action when one was required). Shared, not duplicated, by the live Agent panel, `audit_runs.py`, and the measurement-driven DPO/reflection miners.

<a id="sec-9-3"></a>
### 9.3 `audit_runs.py` — aggregate auditing

`_analysis/audit_runs.py` — a deterministic, read-only, no-LLM auditor over a whole tagged batch of run logs: reuses `measurement.py`/`oow_agent_spec.py`/`narrate.py` directly (never duplicates their logic), adds its own auditor-specific codes (`E_role_fabrication`, `E_encounter_mismatch`, `E_unclassified_encounter`, `B_17c` stand-on-acted-too-early, `P_wrong_side_pass`/`P_port_toward_contact`), and can export any filtered finding set straight to DPO-rejected candidate files. Wired into `sweep_dashboard.py`'s own "Audit" tab.

<a id="sec-9-4"></a>
### 9.4 Track 2 Q&A-style eval

Mirrors VHF's `eval_colreg_scenarios.py` pattern (Track 2 holds its own scenario file, `OOW_Eval/scenario.json`, scored on procedure/compliance rather than recall) — same composite philosophy (behavioural graders weighted heaviest, an LLM judge for whether the stated action actually complies with the cited rule(s)), parameterised for OOW's own scenario schema.

---

<a id="sec-10"></a>
## 10. Simulator (Basic Simulator/app/)

A Streamlit app (`streamlit_app.py`) exercising the same live `ask_oow()`/baselines/oracle against hand-picked or precomputed missions: Scenario Preview (unavoided collision-course path), Manual helm (D-pad), Play Agent Mission (native Plotly animation over a precomputed run log — chosen over a rerun-driven scrub UI after measuring it eliminates server-side flicker entirely), and Agent Real-Time (a genuine live model call per click). `run_llm_scenario.py`/`sweep_llm_params.py`/`run_baseline_scenario.py` are the headless CLI equivalents used for batch sweeps (the actual source of every archived `_llm_runs/*.json` file `evaluate_run.py`/`audit_runs.py` score). `sweep_dashboard.py` is a separate, read-only progress/audit viewer — never runs compute itself.

---

<a id="sec-11"></a>
## 11. Known open issues / research threads

Documented honestly (mirrors the Captain/Chief Engineer docs' "open questions" sections) rather than presented as a finished system:

1. **Stale RAG retrieval bug (still open)**: dense embeddings for situation-report-style queries are near-degenerate (pairwise cosine similarity 0.978–0.999 across genuinely different situations) because the template is almost entirely boilerplate phrasing with only numbers changing — a general-purpose embedder structurally cannot discriminate numeric differences in otherwise-near-identical text. Measured consequence: `v7_super_rag` turns only 3% of the time when a real risk is cited (vs. 59% for `v0_base`, no RAG) — bare RAG retrieval measurably biases the model toward imitating a stale "stand-on, hold course" chunk. `v10_super_colreg_rag`'s corpus-scoping (§5.5) mitigates but does not fully solve this; the real fix (a richer/more discriminating retrieval query or strategy) is diagnosed but not yet implemented.
2. **Rule-13 citation bias**: ~80% of non-none `encounter_rule` citations say "Rule 13" but only ~28–33% are actually correct by recomputed geometry (vs. Rule 15's 88.9% citation precision). A prompt-wording fix (de-prioritising the enum listing order) was tried; the user-approved path since has been data-side counter-examples only (`rule13_mislabel_events()`, §7.4) rather than another prompt change — whether this fully resolves the bias is not yet re-measured.
3. **Reflection underperforms** (§8.1) — trained, format-bug-fixed, not deployed.
4. **Two still-unfixed audit findings**: `BLOCKER_1_1` tcpa/rel-bearing mismatches (likely a nearest-timestamp interpolation sensitivity near TCPA sign changes) and a `BLOCKER_1_6_kinematics_violation` suspected to be an audit-side false positive (not yet fully root-caused).
5. **`OOW_MOOS_Integration/`** remains an explicitly-deferred "later phase" — no live MOOS bridge wiring exists today beyond the historical prototype in `Brain Storming/pilot_agents.ipynb`.

---

<a id="sec-12"></a>
## 12. Relationship to Captain and Chief Engineer

OOW is the only one of the 3 domains with a mature, running fine-tune + evaluation loop as of this note — it is the concrete reference implementation both Captain (`design_captain_missions.md`) and Chief Engineer (`design_chief_engineer.md`) deliberately mirror: the two-track data split (§2), the `AgentPaths`-parameterised shared pipeline scripts (§7–§8), the facts-only prompt principle (§4.1), and the composite-evaluator-with-hard-gates pattern (§9.1) are all first proven here before being adopted as this project's standard conventions for every later domain. Captain's `EngineFailureContext`/shield mechanism (§9.1 of the Captain doc) explicitly reuses OOW's own `oracle_planner.py`-style deterministic-baseline-as-ground-truth pattern (§6.2) rather than inventing a new one.
