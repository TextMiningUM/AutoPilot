# Design Note: VHF / External Communications Agent — Reverse-Engineered Architecture + Interface & Multi-Channel Expansion (Track 1, revisited)

**Status:** Part 1 (§1–§6) is **documentation of an existing, mature system** (same posture as `design_oow_navigation.md`) — VHF is the most-built-out domain in this project (`.github/copilot-instructions.md` calls it "most mature"), with a full 25-stage training+compression chain already run. Part 2 (§7–§10) is **new design-only work** (nothing built yet), per the user's explicit 2026-10-03 direction: VHF will eventually get its own simulator **interface**, just like Captain/Chief Engineer/OOW already have or are designed to have, and that interface's job is specifically to **show communication with others** — and the user explicitly wants the agent's own scope broadened beyond radio to **flags and light signals** too, not just VHF.

## Table of contents

- [1. VHF/Communications Officer job description](#sec-1)
  - [1.1 Legal/regulatory basis](#sec-1-1)
  - [1.2 Core responsibilities (as implemented today)](#sec-1-2)
- [2. Two-track architecture (the original, VHF is where this pattern was invented)](#sec-2)
- [3. Data corpus](#sec-3)
  - [3.1 Source documents (~45 files)](#sec-3-1)
  - [3.2 `Data/VHF/` folder structure](#sec-3-2)
- [4. Training + compression pipeline (25 stages, the most complete of any domain)](#sec-4)
  - [4.1 Data-building stages (local, cheap)](#sec-4-1)
  - [4.2 Train → eval → ablation → compress (cloud, `cloud/run_all.sh`)](#sec-4-2)
- [5. Evaluation](#sec-5)
- [6. The `ask_vhf()` prototype — VHF's existing (informal) link to OOW](#sec-6)
- [7. Broadening scope: VHF as one of THREE external-communication channels](#sec-7)
  - [7.1 Why broaden now](#sec-7-1)
  - [7.2 Channel 1 — VHF radio (existing)](#sec-7-2)
  - [7.3 Channel 2 — Flag signals (International Code of Signals) — NEW](#sec-7-3)
  - [7.4 Channel 3 — Light & sound signals (COLREG Part D / Annex I/III) — NEW](#sec-7-4)
  - [7.5 One agent, three channels — how this fits the existing architecture](#sec-7-5)
- [8. Communication protocol: who talks to the Comms agent, and why](#sec-8)
  - [8.1 OOW → Comms](#sec-8-1)
  - [8.2 Captain → Comms](#sec-8-2)
  - [8.3 Chief Engineer → Comms (indirect, via Captain)](#sec-8-3)
  - [8.4 External party → Comms (inbound)](#sec-8-4)
- [9. Simulator interface design](#sec-9)
  - [9.1 Panel layout](#sec-9-1)
  - [9.2 Chain-of-command visibility](#sec-9-2)
- [10. Data sourcing for flags/light signals — research status](#sec-10)
- [11. Open questions](#sec-11)

---

<a id="sec-1"></a>
## 1. VHF/Communications Officer job description

<a id="sec-1-1"></a>
### 1.1 Legal/regulatory basis

- **ITU Radio Regulations** + **GMDSS** (Global Maritime Distress and Safety System): govern VHF channel allocation, DSC (Digital Selective Calling), distress/urgency/safety priority (MAYDAY/PAN PAN/SÉCURITÉ), watchkeeping duty — the existing corpus's primary regulatory anchor (§3.1).
- **IMO Standard Marine Communication Phrases (SMCP)**: the standardised phraseology this agent is trained to use, explicitly designed to be understood regardless of operator native language.
- **COLREG Rule 34/35** (manoeuvring and warning signals, sound signals in restricted visibility) **and Annex I/III/IV** (position/technical details of lights and shapes, technical details of sound signal appliances, distress signals) — **already exist in this project's OOW COLREG corpus** (`colreg_consolidated_2018.json`, "Part D — Sound and Light Signals") but are not currently *owned or presented* by the VHF agent — this is the real regulatory basis for §7.4's proposed broadening.
- **International Code of Signals (INTERCO)**, IMO: the real, standing international standard for flag-hoist signalling between ships and ship-to-shore when radio isn't used/available/trusted — the real regulatory basis for §7.3's proposed broadening. Not yet sourced in this repo (§10).
- **COLREG Rule 2 (general prudence)**: VHF/flag/light arrangements may assist mutual understanding but **never override the COLREG-required action** — an already-encoded, tested principle in this project's own COLREG-judge prompt (`eval_colreg_scenarios.py`'s `COLREG_JUDGE_PROMPT` explicitly penalises "an answer that relies on VHF agreement INSTEAD of the COLREG-required action"). This is the single most important cross-cutting rule for whichever channel the Comms agent uses — a flag hoist or light signal is exactly as subordinate to COLREG as a VHF call already is.

<a id="sec-1-2"></a>
### 1.2 Core responsibilities (as implemented today)

Scoped, as built, to **VHF radio only**:
1. Conduct distress/urgency/safety/routine voice traffic per GMDSS/SMCP procedure (hailing sequence, channel selection/handover, call format).
2. Conduct a COLREG-adjacent VHF exchange during a collision-avoidance encounter — **draft the transmission that reflects/confirms an already-COLREG-compliant manoeuvre**, never a substitute for it (§1.1).
3. Operate DSC (Digital Selective Calling) alerting/acknowledgement.
4. Answer general VHF/GMDSS procedural and regulatory knowledge questions (Track 1, §2).

---

<a id="sec-2"></a>
## 2. Two-track architecture (the original, VHF is where this pattern was invented)

VHF is the domain `.github/copilot-instructions.md`'s two-track convention was first built for — OOW (`design_oow_navigation.md` §2) and the still-design-only Captain/Chief Engineer docs all explicitly copy this exact split:

| | Track 1 — Rules & knowledge | Track 2 — Conversational compliance |
|---|---|---|
| **Eval data (held out)** | `vhf_gold_answers.json` (540 SRC-exam-style Q&A) | `vhf_colreg_scenarios.json` (498 collision-avoidance VHF scenarios) |
| **Training data** | `vhf_sft_{direct,cot,rag}.jsonl`, `vhf_multihop.jsonl`, `vhf_dpo_pairs.jsonl`, `vhf_reflection.jsonl`, `vhf_pg_sft.jsonl` | `vhf_conversations.jsonl` (raw dialogues) → mined via `extract_conversation_reasoning.py` into `vhf_colreg_sft_*.jsonl`/`vhf_colreg_multihop.jsonl`/`vhf_colreg_dpo_pairs.jsonl`/`vhf_colreg_reflection.jsonl` |
| **Eval script** | `pipeline/eval/eval_finetuned.py` ("does the model know VHF/GMDSS rules?") | `pipeline/eval/eval_colreg_scenarios.py` ("can the model actually conduct a correct, compliant VHF exchange during a real collision-avoidance situation?") |

Both tracks feed the **same** QLoRA SFT→DPO→Reflection fine-tune, kept in separate files for per-competency traceability.

---

<a id="sec-3"></a>
## 3. Data corpus

<a id="sec-3-1"></a>
### 3.1 Source documents (~45 files)

`Data/VHF/VHFProtocol/` — unlike OOW's single COLREG convention text, this is ~45 **heterogeneous** documents with no fixed structure (guides, regulations, procedure cards, cheatsheets), ingested via `pipeline/ingest/build_vhf_json.py`'s generic heading-detection + hand-curated per-file `(source_type, publisher, language)` classification rather than a bespoke parser. Real sources include: `IMO standard marine comms phrases.pdf` (SMCP itself), `CEPT Regulations.pdf`, `Channel Listing.pdf` (ITU/CEPT), `GMDSS VHF DSC procedures for small boat users - GOV.UK.pdf`, `mgn375.pdf`/`mgn_324.pdf` (UK MCA Marine Guidance Notes), `R-REC-M.489-2...pdf` (an actual ITU-R Recommendation), `DSC alert flow chart.pdf`, `Basic MAYDAY Call.pdf`, `NATO_Phonetic_Alphabet.txt`, plus several informal guides/blogs/cheatsheets (`BoatUS.txt`, `CompleteGuideVHF.txt`, etc.) that round out plain-language coverage alongside the formal regulatory texts.

<a id="sec-3-2"></a>
### 3.2 `Data/VHF/` folder structure

```
Data/VHF/
├── VHFProtocol/            ← source_dir, ~45 heterogeneous source documents (§3.1)
├── VHF_Eval/                ← held out: vhf_gold_answers.json, vhf_colreg_scenarios.json
├── VHF_JSON/                ← per-document parsed JSON (build_vhf_json.py output)
└── VHF_Agents_Training/     ← RAG chunks/embeddings/KG/PG/reranker pairs, every SFT/DPO/Reflection/multihop
                                jsonl for both tracks, ablation/eval/probe output, overnight_logs/
```

---

<a id="sec-4"></a>
## 4. Training + compression pipeline (25 stages, the most complete of any domain)

<a id="sec-4-1"></a>
### 4.1 Data-building stages (local, cheap)

Mirrors the VHF_Agent_Training_Pipeline.ipynb's own §2–§9 progression (ingest → `build_rag.py`/`build_kg.py` → `pipeline/track1/extract_reasoning.py` reasoning-trace mining → deterministic `build_sft.py`/`build_rlhf.py`/`build_reflection.py`/`build_multihop.py` → `build_pg.py`/`build_pg_sft.py`) — all safe to run locally per the project's local/cloud split, no GPU needed.

<a id="sec-4-2"></a>
### 4.2 Train → eval → ablation → compress (cloud, `cloud/run_all.sh`)

The **only** domain with a complete, already-run model-compression chain (25 numbered stages) — reproduced here as the canonical reference since no other doc has it written out fully:

| Stages | What |
|---|---|
| 1–5 | Mine agentic Track-2 training data from the 360 training conversations (`extract_conversation_reasoning.py` → SFT/DPO/Reflection/multihop builders, scoped so the held-out `vhf_colreg_scenarios.json` is never touched) |
| 6–9 | SFT → DPO → Reflection → merge (`train_sft.py`/`train_dpo.py`/`train_reflection.py`/`merge_adapter.py`) |
| 10–11 | Eval base Qwen3-8B, both tracks |
| 12–14 | Prompt-injection ablation (V0–V4, base Qwen only) |
| 15–16 | Eval VHF-QWEN (the merged fine-tune), both tracks |
| 17–18 | Stage-attribution probes (`probe_dpo.py` — DPO axis win-rates, reflection delta) |
| 19–21 | **AWQ int4 quantization** + eval, both tracks |
| 22 | **Pruning** (`compress_prune.py`) |
| 23–25 | **Distillation** (`compress_distill.py`, merge-final) + eval, both tracks |

Every eval stage runs **both tracks** unconditionally — Track 2 is explicitly "a permanent, first-class part of the pipeline, not an optional extra" (the script's own comment). This compression chain (AWQ/prune/distill) does not yet exist for OOW or any other domain — a natural template to reuse once a domain's base fine-tune is stable enough to be worth compressing.

---

<a id="sec-5"></a>
## 5. Evaluation

Track 1 (`eval_finetuned.py`): embedding-similarity + claim-level metrics against the 540 gold Q&A. Track 2 (`eval_colreg_scenarios.py`): a genuinely different metric set, since this track tests *procedure*, not recall — `SemSim`/`AnsRel`/`Cover` (embedding-based), `ChannelProc` (rule-based: did it hail on 16 AND name a distinct working channel?), `CallFormatOK` (rule-based: calls the OTHER vessel first, not a self-hailing bug caught during data generation), `ColregCorrect` (gpt-4o-mini judge: does the stated action actually comply with the cited rule(s) **and avoid treating the VHF arrangement as overriding COLREG**, §1.1). `COMPOSITE_V2_WEIGHTS` weights `ColregCorrect` heaviest (0.30) — this track cares most about compliance, not fluency.

---

<a id="sec-6"></a>
## 6. The `ask_vhf()` prototype — VHF's existing (informal) link to OOW

`Basic Simulator/Brain Storming/pilot_agents.ipynb` already contains a **working prototype** (calls the Anthropic API directly, not the local fine-tune) of exactly the cross-agent call this design note formalises in §8:

```python
def ask_vhf(oow_decision=None):
    state = _current_state()
    if oow_decision is None:
        oow_decision = ask_oow()
    prompt = (narrate(state) + "\n\nOOW decision: " + json.dumps(oow_decision) +
              "\n\nDraft the radio call for this manoeuvre as the specified JSON object.")
    call = _call_llm(VHF_SYSTEM_PROMPT, prompt)
    return call
```

I.e. VHF's job in this pattern is **never to decide the manoeuvre itself** — it receives OOW's already-made decision and drafts the corresponding external communication. This is the correct, already-validated shape for §8.1's formal protocol; nothing about the prototype's own logic needs to change, only promoting it from a notebook experiment to a real, locally-fine-tuned call site (`Basic Simulator/app/agents.py`-equivalent) is new work.

---

<a id="sec-7"></a>
## 7. Broadening scope: VHF as one of THREE external-communication channels

<a id="sec-7-1"></a>
### 7.1 Why broaden now

User's explicit instruction (2026-10-03): the eventual VHF interface should "show communication with others — VHF and flags and light signals." This is a real, grounded broadening, not scope creep — real merchant bridge practice has always used 3 complementary external-communication channels, and this project's own COLREG corpus already contains the rule text for 2 of them (Rule 34/35 + Annex I/III, currently filed under OOW, never surfaced by the VHF agent). Renaming the domain is **not** proposed — "VHF" stays the project's established name (folder structure, `AgentPaths.vhf()`, every existing script) for continuity; only its **scope** broadens, the same way OOW's own corpus already spans far more than steering decisions.

<a id="sec-7-2"></a>
### 7.2 Channel 1 — VHF radio (existing)

Unchanged — §1–§6 above.

<a id="sec-7-3"></a>
### 7.3 Channel 2 — Flag signals (International Code of Signals) — NEW

- **Real-world basis**: the IMO International Code of Signals (INTERCO) — single-flag urgent/common signals (e.g. flag **O** = "man overboard", flag **N over C** = distress), and multi-flag hoists spelling out coded messages from the Code's own phrase tables. Still mandatory carriage on many vessel classes and the real fallback when radio is unavailable, untrusted, or a visual/silent signal is specifically required (e.g. certain naval/security contexts).
- **Not yet in this repo** — no flag/ICS source document has been ingested (confirmed via grep: zero hits for "International Code of Signals" as a corpus topic, only as passing mentions inside existing COLREG/VHF text). §10 covers sourcing.
- **Training-data shape (proposed, not built)**: same Track 1 (what does flag/hoist X mean) + Track 2 (given this situation, what flag signal — if any — should be hoisted) split already proven for VHF/OOW — no new data-pipeline architecture needed, only a new corpus and (per §7.5) a `channel` field on the existing Track-2 schema.

<a id="sec-7-4"></a>
### 7.4 Channel 3 — Light & sound signals (COLREG Part D / Annex I/III) — NEW

- **Real-world basis**: COLREG Rules 32–37 (sound signal definitions, manoeuvring/warning signals, signals in restricted visibility) + Annex I (position/technical details of lights and shapes — day shapes already appear as OOW gold-answer questions, e.g. "a cylinder shape shown by day" = constrained by draught) + Annex III (technical details of sound signal appliances) + Annex IV (distress signals, including flares/rocket signals, which overlap with VHF's own MAYDAY procedure).
- **Already partially in this repo, but filed under OOW, not VHF**: `colreg_consolidated_2018.json`'s "Part D — Sound and Light Signals" section and the already-generated `vhf_eval` OOW gold-answer rows testing day-shape/sound-signal recall. §7.5 proposes how this crosses domain boundaries cleanly.
- **Scope note**: Morse-lamp signalling (an operator keying short/long light flashes, historically also used for VHF-equivalent plain-text messages ship-to-ship) is a real, still-practiced skill (e.g. NATO exercises, some pilot/VTS stations) — flagged as a real, genuine 3rd sub-channel inside "light signals," not yet further designed here; COLREG's own Rule 34/35 signals (fixed, short/prolonged-blast patterns with a small fixed vocabulary) are the higher-priority, more tractable piece to build first.

<a id="sec-7-5"></a>
### 7.5 One agent, three channels — how this fits the existing architecture

**Proposed (not yet decided/built)**: the Comms agent's existing Track-2 schema gains one new field, `channel: "vhf" | "flag" | "light_sound"`, decided by the agent itself from the situation (e.g. radio silence/casualty → flag or light signal; routine arrangement → VHF) — **not hard-coded by a rule**, matching this project's established "facts-only, let the model/training data carry the judgement" principle (`design_oow_navigation.md` §4.1's own documented architecture pivot). The day-shape/sound-signal rule text currently living in OOW's COLREG corpus does **not** need to be duplicated — it can be referenced/shared exactly the way `pipeline/track1/build_sft.py`/`build_reflection.py` already are literal, unmodified, cross-domain-reused scripts; only the RAG/KG index needs to also ingest it under the VHF domain's own corpus (or retrieve cross-domain, a smaller, cheaper option — an open question, §11). No new agent architecture, retrieval stack, or training pipeline is implied — this is a corpus/schema broadening of the existing, proven VHF machinery, exactly the same "reuse before you rebuild" principle this project applies everywhere else.

---

<a id="sec-8"></a>
## 8. Communication protocol: who talks to the Comms agent, and why

Mirrors `design_captain_missions.md` §3's "Communication protocol between Captain and OOW" structure, extended to include the Comms agent as a 3rd party in the chain of command — grounded in real bridge practice (the OOW/Master decide; the radio operator/duty officer executes the actual transmission; nothing about VHF/flags/lights is ever a decision-making layer of its own).

<a id="sec-8-1"></a>
### 8.1 OOW → Comms

**Already prototyped** (§6): OOW makes a COLREG-compliant manoeuvre decision; the Comms agent is invoked with that decision (never the reverse) and drafts/selects the corresponding external communication — a confirmatory VHF call to the other vessel, a flag hoist, or (rarely, for OOW-scale events) a light signal. The Comms agent's output is observational/confirmatory only — it must never be allowed to imply the manoeuvre is contingent on the other vessel's agreement (§1.1's hard rule, already enforced by the Track 2 judge prompt).

<a id="sec-8-2"></a>
### 8.2 Captain → Comms

Per `design_captain_missions.md`'s own existing wording (§9.1's tools table, written before this note): Captain-level instructions to Comms are explicitly listed as examples of the Captain's output channel — **"transmit PAN PAN"**, **"request tug assistance"**, **"report to VTS"**. These are mission-scale, not encounter-scale, communications: distress/urgency broadcasts, contacting the DPA/VTS/port agent/pilot, and (per `design_chief_engineer.md` §8.1) **relaying** a Chief Engineer operational constraint externally if that ever needs reporting off-ship (e.g. a PAN PAN for an engine casualty) — never a Chief Engineer → Comms call directly.

<a id="sec-8-3"></a>
### 8.3 Chief Engineer → Comms (indirect, via Captain)

No direct channel — consistent with `design_chief_engineer.md` §8.1's own decided chain of command (Chief Engineer reports to Captain only). If an engine casualty needs external reporting (e.g. requesting a tug, declaring a PAN PAN), that is a **Captain** decision relayed to Comms, exactly like §8.2 — Chief Engineer never invokes the Comms agent itself.

<a id="sec-8-4"></a>
### 8.4 External party → Comms (inbound)

Not yet designed in any of this project's docs — real bridge practice also requires handling an **incoming** call/signal (another vessel hailing, a shore station's DSC distress relay, an inbound flag hoist from a pilot vessel) and deciding whether/how it changes OOW's or Captain's own facts (e.g. an inbound VHF arrangement proposal must be evaluated against COLREG, never auto-accepted — §1.1 again). Flagged as a genuine, currently-unaddressed gap for whenever this interface is actually built (§11).

---

<a id="sec-9"></a>
## 9. Simulator interface design

Mirrors Captain's own proposed panel pattern (`design_captain_missions.md` §11.4) and Chief Engineer's Engine Room Dashboard (`design_chief_engineer.md` §8) — the Comms agent's panel is the 3rd of 3 peer department panels alongside Captain's and (once built) Chief Engineer's, all sitting alongside the existing OOW Agent panel.

<a id="sec-9-1"></a>
### 9.1 Panel layout

- **Transmission/signal log** (the panel's primary content, directly answering the user's "show communication with others"): a unified, chronological list of every outbound AND inbound communication — each entry tagged with its **channel** (§7.5: VHF/flag/light-sound), the **triggering decision** it confirms (an OOW manoeuvre, a Captain instruction), and the actual content (transmitted text, which flag(s) were hoisted, which light/sound pattern was made).
- **Channel-specific renderers**: a VHF entry renders as the actual transmission text (quoted, SMCP-style); a flag entry renders the real ICS flag graphics for the hoisted signal (a visual aid, not just a text code); a light/sound entry renders the dot/dash or short/prolonged-blast pattern (e.g. "• • •" for 3 short blasts) plus its COLREG rule citation.
- Same "always re-render into placeholders every run" / native-animation-over-rerun-driven-scrubbing lessons already learned and documented for the OOW Streamlit UI (`design_oow_navigation.md` §10) apply directly here — no new UI-flicker research needed, reuse what already works.

<a id="sec-9-2"></a>
### 9.2 Chain-of-command visibility

Per §8, every logged entry must show **which agent triggered it and why** (an OOW manoeuvre decision, a Captain mission-level instruction) — making the §8 protocol literally visible in the UI, the same design goal already stated for Captain's own Mission Log (`design_captain_missions.md` §11.5: "no new decision-logic needed, only a merge/render step" — this panel is exactly that same pattern, one more layer).

---

<a id="sec-10"></a>
## 10. Data sourcing for flags/light signals — research status

No scraping has been done — a prioritised candidate list, same posture as the sourcing research tables in `design_captain_missions.md` §12/§14.1 and `design_chief_engineer.md` §7:

| Source | Content | Status |
|---|---|---|
| **COLREG Annex I/III/IV + Rules 32–37** | Light/shape positions, sound-signal appliance specs, distress signals | **Already in this repo** (`colreg_consolidated_2018.json`, OOW domain) — zero new acquisition needed, only a cross-domain wiring decision (§7.5). |
| **IMO International Code of Signals (INTERCO)** | The real, standing flag-signal standard | **Likely a paid IMO publication** (same category as the consolidated COLREG edition this project already has via its existing licence/bundle, per `design_captain_missions.md` §12's own research) — check first whether the same channel that supplied `COLREG-Consolidated-2018.pdf` also covers INTERCO. |
| **US Navy / Coast Guard flag-signal & semaphore training manuals** | Real, illustrated, likely public-domain (US federal government works, same status already confirmed for Bowditch/NAVEDTRA-style material in `design_chief_engineer.md` §7) | **Unverified, strong candidate** — not yet searched this session. |
| **Wikipedia** (International Code of Signals, Semaphore, Morse code at sea articles) | General background, CC BY-SA 4.0 | **Free**, same caveats (training-only, attribution needed) as every other Wikipedia use in this project. |

---

<a id="sec-11"></a>
## 11. Open questions

1. **Cross-domain retrieval**: does the broadened Comms agent get its **own** copy of the COLREG light/sound-signal text re-ingested under `Data/VHF/`, or does it retrieve cross-domain from OOW's existing index? The latter is cheaper/no-duplication but has no precedent yet in this project's retrieval architecture (every domain's RAG index today is self-contained).
2. **Fine-tuning scope**: does the broadened agent get a single fine-tune spanning all 3 channels (one model, a `channel` field in its output), or does flag/light-signal competency stay a smaller, separate LoRA adapter layered on top of the existing VHF-QWEN? Not yet decided.
3. **Naming**: keep "VHF" as the project-wide name (folders/`AgentPaths.vhf()`/scripts) despite the broadened scope, or introduce a new label (e.g. "Comms")? This note assumes **keep "VHF"** for continuity, per §7.1's own reasoning, but this hasn't been explicitly confirmed by the user.
4. **§8.4's inbound-communication handling** — genuinely undesigned, flagged, not yet addressed by any existing doc.
5. **Interface build order** — Chief Engineer's dashboard (`design_chief_engineer.md` §8) and Captain's panel (`design_captain_missions.md` §11) are also design-only/unbuilt; this note doesn't propose a priority order among the three, only documents how each would fit together once built.
