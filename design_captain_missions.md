# Design Note: Captain / Mission Layer above VHF + OOW (Track 3)

**Status:** DESIGN / RESEARCH only — nothing implemented yet. Explicit instruction from the user (2026-09-30): continue with functional + technical design and architecture, **no coding yet**.

This follows on from the existing two-track architecture (see `.github/copilot-instructions.md`): a VHF officer and an OOW (+ subordinates) already steer the ship COLREG-compliantly and efficiently from A to B. This note raises that to a third layer: a **Captain** who executes a full **mission** via a series of waypoints, during which "brown envelope" events occur that the captain must manage so the mission goal is still achieved.

## Table of contents

- [Core architecture idea](#core-architecture-idea)
  - [Why not one model for everything](#why-not-one-model)
- [1. Captain job description](#sec-1)
  - [1.1 Legal basis of authority](#sec-1-1)
  - [1.2 Core task description](#sec-1-2)
  - [1.3 Translation into the agent design](#sec-1-3)
- [2. OOW and subordinates: responsibilities](#sec-2)
  - [2.1 Officer of the Watch (OOW)](#sec-2-1)
  - [2.2 Look-out](#sec-2-2)
  - [2.3 Helmsman](#sec-2-3)
  - [2.4 Quartermaster of the Watch (QMOW)](#sec-2-4)
  - [2.5 Chief Engineer / duty engineer](#sec-2-5)
  - [2.6 How this maps onto the existing simulator](#sec-2-6)
- [3. Communication protocol between Captain and OOW](#sec-3)
  - [3.1 Upward: OOW → Captain](#sec-3-1)
  - [3.2 Downward: Captain → OOW](#sec-3-2)
  - [3.3 Mapping onto the agent architecture](#sec-3-3)
- [4. Exhaustive list of "brown envelopes"](#sec-4)
- [5. Where to find example missions / source material](#sec-5)
  - [5.1 Already present in this project](#sec-5-1)
  - [5.2 External, publicly accessible sources](#sec-5-2)
- [6. `Data/Captain/` folder structure & training pipeline](#sec-6)
  - [6.1 Proposed folder layout](#sec-6-1)
  - [6.2 Own training pipeline](#sec-6-2)
  - [6.3 "Experience" — RAG + reranking + KG + PG](#sec-6-3)
- [7. Route & waypoints](#sec-7)
- [8. Mission Order format](#sec-8)
  - [8.1 The six parts](#sec-8-1)
  - [8.2 Example Mission Order](#sec-8-2)
  - [8.3 The response: Mission Progress Report](#sec-8-3)
  - [8.4 Mission State](#sec-8-4)
- [9. Captain goals, planning & reasoning](#sec-9)
  - [9.1 Extended tool/output schema](#sec-9-1)
  - [9.2 Grounding in "experience"](#sec-9-2)
  - [9.3 Where the LLM is — and is not — the right tool](#sec-9-3)
  - [9.4 Regime switching](#sec-9-4)
- [10. Evaluation](#sec-10)
  - [10.1 Proposed axes](#sec-10-1)
  - [10.2 Composite formula](#sec-10-2)
  - [10.3 Starting thresholds](#sec-10-3)
- [11. Simulator representation](#sec-11)
  - [11.1 Four new UI surfaces](#sec-11-1)
  - [11.2 Mission Briefing panel](#sec-11-2)
  - [11.3 Route overlay on the plot](#sec-11-3)
  - [11.4 Captain panel](#sec-11-4)
  - [11.5 Mission Log (unified timeline)](#sec-11-5)
  - [11.6 Data model implication](#sec-11-6)
- [12. Open questions / not yet decided](#sec-12)
- [13. Computational specification — the missing layer](#sec-13)
  - [13.A Blocking](#sec-13-a)
    - [13.A.1 Two-timescale simulator](#sec-13-a-1)
    - [13.A.2 Event model](#sec-13-a-2)
    - [13.A.3 Resource model](#sec-13-a-3)
    - [13.A.4 Procedure library](#sec-13-a-4)
    - [13.A.5 Evidence, ambiguity, and reporter reliability](#sec-13-a-5)
    - [13.A.6 Quantifying "safety margin"](#sec-13-a-6)
    - [13.A.7 Units and time convention](#sec-13-a-7)
    - [13.A.8 The `cost()` function, fully specified](#sec-13-a-8)
    - [13.A.9 One discrete-event scheduler](#sec-13-a-9)
  - [13.B Loose contracts](#sec-13-b)
    - [13.B.5 Mission State — formal schema](#sec-13-b-5)
    - [13.B.6 Captain output schema + precedence rules](#sec-13-b-6)
    - [13.B.7 Trigger monitors](#sec-13-b-7)
    - [13.B.8 Route planner with exclusion zones](#sec-13-b-8)
    - [13.B.9 Event → monitor mapping](#sec-13-b-9)
  - [13.C Data & evaluation](#sec-13-c)
    - [13.C.9 Scenario-generator specification](#sec-13-c-9)
    - [13.C.10 Training-row format](#sec-13-c-10)
    - [13.C.11 Ground-truth functions per evaluation axis](#sec-13-c-11)
    - [13.C.12 Run format & versioning](#sec-13-c-12)
    - [13.C.13 Ground-truth scenario file](#sec-13-c-13)
    - [13.C.14 OOW evaluator encounter mode](#sec-13-c-14)
    - [13.C.15 OOW determinism during Captain data/eval](#sec-13-c-15)
- [14. Approach: walking skeleton first](#sec-14)
  - [14.1 Data readiness, in priority order](#sec-14-1)
- [15. Concrete data structures & UI controls](#sec-15)
  - [15.1 My assessment](#sec-15-1)
  - [15.2 Remaining concrete data-structure specifics](#sec-15-2)
  - [15.3 Minimal debug control set](#sec-15-3)
  - [15.4 Full UI control spec — deferred](#sec-15-4)

---

<a id="core-architecture-idea"></a>
## Core architecture idea (confirmed by the user, 2026-09-30)

| Layer | Status | Role |
|---|---|---|
| **OOW** | existing | Performs the actual manoeuvres/navigation, plans the route/waypoints, enforces COLREG on each leg. |
| **VHF** | existing | External communication per encounter/port/emergency. |
| **Captain** | **new, own LLM** | A **separate, separately-trained Qwen agent** (own RAG + own CoT-style reasoning, the same QLoRA SFT+DPO(+Reflection) approach VHF/OOW already use — **not** a shared model with OOW). Can **overrule** the OOW and helps it with complex decisions arising from a brown envelope: new waypoints/course (e.g. a detour around a severe storm, using a current to make better speed), or new operational limits (e.g. "do not exceed X kn after an engine problem"). |

The captain "uses" the OOW and VHF agents as callable **subordinates/tools**, exactly as in the real chain of command on board — not as loose "advice", but as an instruction that actually changes the OOW layer's waypoint/speed limits.

<a id="why-not-one-model"></a>
### Why not one model for everything

The avoidance decision (OOW) and the mission decision (Captain) differ in timescale, in what "good" means, and in how you verify it:

| | OOW (encounter) | Captain (mission) |
|---|---|---|
| **Timescale** | 30–200 s | minutes to hours |
| **Question** | which order, right now | which plan, which priority, which goal to give up |
| **"Good" means** | CPA ≥ safe distance, COLREG-compliant, on the goal course | goals achieved, resources safeguarded, risk acceptable |
| **Verifiable** | yes — simulator + auditor | partly — goals yes, judgement not fully |
| **Regime** | COLREG | COLREG + SOLAS + emergency procedures + the mission order + (under attack) self-protection |

A single model that has to weigh engines, whales, and Rule 15 in every single decision becomes slow, unreliable, and unmeasurable. Two layers with a sharp interface do not.

---

<a id="sec-1"></a>
## 1. Captain job description

Grounded in real merchant-navy protocols — STCW, SOLAS, ISM Code, MLC 2006, COLREG, national law.

<a id="sec-1-1"></a>
### 1.1 Legal basis of authority (not invented — these are real regimes)

- **STCW Convention** (STCW 78, as amended, Manila 2010): Chapter II governs the certification of Masters and deck officers; the Master must hold the highest certificate of competency (Master Mariner) appropriate to the vessel type and trading area.
- **ISM Code** (International Safety Management Code, SOLAS Ch. IX): explicit Article 5 "Master's responsibility and authority" — the Master has **absolute, unrestricted "overriding authority"** over safety and pollution prevention, and may disregard commercial pressure from the company when exercising it. The company must confirm this in writing in the Safety Management System (SMS). This is the core rule for every "brown envelope" decision: **safety beats schedule/cargo/cost, always, without exception.**
- **SOLAS**: Ch. V (Safety of Navigation) governs the duty to render assistance to vessels in distress (Regulation 33 — comparable to UNCLOS Art. 98), mandates voyage/passage planning, and requires reporting of dangers to navigation. Ch. III (Life-Saving) and Ch. II-2 (fire safety) are the technical frameworks behind many brown-envelope scenarios (fire, rescue, evacuation).
- **MARPOL**: environmental reporting duties (oil spill, ballast water, sewage) — the Master is personally responsible/liable for compliance. MARPOL Annex I Art. 8/Protocol I requires immediate reporting of incidents to the coastal state.
- **MLC 2006** (Maritime Labour Convention): crew rights, rest hours (together with STCW A-VIII/1 hour limits), repatriation, medical care on board — relevant to crew-related brown envelopes (exhaustion, illness, disputes).
- **ISPS Code**: ship security (piracy, armed robbery, terrorist threat, ports with elevated security levels) — the Master is ultimately responsible for/over the Ship Security Officer.
- **COLREG 1972**: the Master remains **ultimately responsible** for navigation even though the OOW performs the actual avoiding manoeuvres — the Master can always intervene/overrule ("master's standing orders" and "night order book").
- **National merchant-shipping law** (e.g. the Dutch Wet Zeevarenden, the UK Merchant Shipping Act 1995, comparable laws elsewhere): confirms the Master as the owner's and cargo's legal representative on board, with disciplinary/command authority over the crew and a duty to record special occurrences (average, death, birth, marriage on board, mutiny, collision, general average).

<a id="sec-1-2"></a>
### 1.2 Core task description (for the agent persona)

1. **Ultimate responsibility for safety**: crew, ship, cargo, environment — in that priority order, above the mission goal.
2. **Mission ownership**: safeguards the underlying purpose of the voyage (rescuing people and delivering them safely, delivering cargo intact and on time, etc.) and weighs that against risk — may and must sacrifice the mission goal for safety, never the reverse.
3. **Delegation + oversight**: issues "standing orders" and "night orders" to the OOW (when to be called, what margins to keep, when VHF contact is mandatory) — intervenes personally on escalation (dense fog, narrow waters, emergency, port arrival/departure).
4. **External coordination**: contact with the company/DPA (Designated Person Ashore, mandatory under the ISM Code), the coastal state/VTS, the port agent, insurers in case of damage, authorities for border crossing/quarantine/customs.
5. **Crisis and emergency management**: activates SAR obligations, muster/evacuation procedures, firefighting, medical emergency procedure (including Radio Medical Advice), security protocol (BMP5 anti-piracy), environmental response (oil spill).
6. **Administration and record-keeping**: the deck log/official log book is a legal document — every brown envelope must be recorded factually, time-stamped, without interpretation; mandatory reporting of accidents/near-misses to the flag state and the company.
7. **People management**: crew welfare, rest hours (fatigue is itself a COLREG/safety risk), conflicts, discipline, morale — especially relevant on long/stressful missions.
8. **Decision principle under uncertainty** (a priority ladder for the agent):
   1. immediate danger to life (person overboard, fire, imminent collision)
   2. ship/environmental safety (leak, damage, instability)
   3. legal/COLREG obligations (assistance to distress, reporting duties)
   4. mission goal (completing a rescue, delivering cargo intact, schedule)
   5. commercial/cost/planning — lowest priority, may **never** override 1–3; explicitly established in ISM Code Art. 5.
9. **Escalation**: when does a brown envelope exceed the Master's own mandate, requiring the company/DPA/coastal state to be engaged (e.g. declaring general average, entering a salvage agreement (Lloyd's Open Form), requesting a place of refuge)?

<a id="sec-1-3"></a>
### 1.3 Translation into the agent design

- The captain agent receives: (a) the mission goal + success criteria (see §8, Mission Order), (b) the current status from OOW (position/route/COLREG status) and VHF (external communication status), (c) an incoming brown-envelope event.
- Output: a decision + reasoning + optionally an instruction back to OOW (e.g. "divert to waypoint X", "reduce speed", "report to VTS") or to VHF (e.g. "transmit PAN PAN", "request tug assistance").
- Evaluation criteria: see §10.

---

<a id="sec-2"></a>
## 2. OOW and subordinates: responsibilities

Grounded in STCW watchkeeping standards, COLREG Rule 5, and the ICS *Bridge Procedures Guide*.

This section exists because the Captain layer only makes sense once the layer it oversees/overrules is itself clearly defined. The bridge team below the Master is a real, standardised chain of responsibility, not this project's invention — grounded in STCW Code Section A-VIII/2 (Watchkeeping), COLREG Rule 5 (look-out), and the International Chamber of Shipping's *Bridge Procedures Guide* (the de-facto industry-standard bridge-team manual most Safety Management Systems are built on). The repo already has empty placeholder folders for 4 of these roles (`Data/Helmsman/`, `Data/LookOut/`, `Data/QuarterMasteroftheWatch/`, `Data/ChiefEngineer/`) — i.e. they were anticipated as future agent roles but never built out.

<a id="sec-2-1"></a>
### 2.1 Officer of the Watch (OOW) — the ship's existing agent

- Primary responsibility for the safe navigation of the ship during the watch, on the Master's behalf (STCW A-VIII/2 Part 3-1).
- Maintains a proper look-out "by sight and hearing, as well as by all available means" (COLREG Rule 5) — in practice, by directing/using the Look-out (§2.2) rather than only looking personally.
- Executes the passage plan (berth-to-berth) that was prepared in advance and **approved by the Master** (SOLAS V/34) — the OOW does not invent the route ad hoc, it follows the approved plan and its pre-agreed margins unless a new instruction (from the Captain, or an unavoidable COLREG manoeuvre) requires a deviation.
- Continuously assesses risk of collision using all available means (radar/ARPA/AIS/visual bearings) and takes timely, substantial COLREG-compliant action — this is exactly the existing OOW-agent competency in this repo.
- Monitors position, track, speed, squat/UKC, weather, and traffic; keeps the deck log; operates/monitors the engine telegraph and any automated systems (autopilot, ECDIS alarms).
- Maintains the external communication watch (VHF Ch. 16/DSC, NAVTEX) or delegates it explicitly — in this project's architecture, this is where the VHF agent is invoked.
- Conducts a full watch **handover/relief briefing** to the next OOW (position, traffic, standing/night orders in force, any pending issues) — a documented, mandatory STCW/ICS practice, not optional courtesy.
- Calls the Master immediately whenever any of the "calling the master" criteria in §3.1 below is met — this single duty is the actual hinge between the OOW layer and the new Captain layer in this design.
- Never leaves the bridge unattended and never hands over the watch to a relieving officer who is unfit to safely carry out the duty (STCW A-VIII/2 Part 3-1, §15).

<a id="sec-2-2"></a>
### 2.2 Look-out

- A **dedicated** duty (COLREG Rule 5): a rating whose only task, especially at night, in restricted visibility, or in congested waters, is to detect other vessels, navigational hazards, persons in the water, and any unusual sight/sound — no other tasks assigned at the same time.
- Reports every sighting to the OOW immediately and precisely (bearing, description, estimated range) — does not decide on manoeuvres itself.
- On a small crew or in daylight/clear/open water, the OOW may act as their own look-out (COLREG Rule 5 allows this only when justified by the prevailing circumstances) — but the **default is a separate person**.

<a id="sec-2-3"></a>
### 2.3 Helmsman

- Steers the course actually **ordered** by the OOW (or the Master during a manoeuvre) — repeats the order back before executing it ("closing the loop", standard bridge practice) and reports back once the ordered course/heading is achieved.
- When the vessel is on autopilot (the normal state on most of a modern merchant voyage), a helmsman may not be physically steering but a competent person must be ready to take the wheel immediately if the OOW orders "hand steering" (mandatory in traffic/restricted visibility/manoeuvring per most SMS manuals and Rule 2/6 good-seamanship practice).
- Immediately reports any failure to respond, sluggish steering, or unexpected behaviour of the steering gear — this is the direct real-world hook for the "steering-gear failure" brown envelope in §4.C.

<a id="sec-2-4"></a>
### 2.4 Quartermaster of the Watch (QMOW)

- A broader bridge-rating role (common on larger merchant/naval-adjacent vessels): assists the OOW with plotting positions, maintaining the deck log, operating navigation lights/sound signals, general bridge housekeeping, and can be assigned as helmsman or look-out depending on the situation and manning level.
- Acts as an additional pair of hands during heightened workload (restricted visibility, pilotage, heavy traffic, an active brown envelope) — a practical bridge resource the OOW can call on before ever needing to wake the Master.

<a id="sec-2-5"></a>
### 2.5 Chief Engineer / duty engineer (a separate department, bridge interface)

- Not part of the bridge team, but the OOW's/Master's essential counterpart for anything mechanical: reports engine/machinery status, available power, fuel state, and any operating restrictions to the bridge.
- Executes speed/manoeuvring orders via the engine controls and reports back when an ordered speed/RPM cannot safely be met.
- Is the one who **actually declares** an operational limit like "do not exceed X kn" after a technical brown envelope (e.g. a propulsion or steering problem) — the Captain's instruction to the OOW in such a case is really a relay of the Chief Engineer's own technical constraint, escalated through the Captain because it changes the mission's operational envelope, not a routine bridge matter.
- Reports immediately, unprompted, any fire, flooding, or major machinery failure — these are Captain-notification triggers regardless of severity (see §3.1).

<a id="sec-2-6"></a>
### 2.6 How this maps onto the existing simulator

`Basic Simulator/app/` already only models OOW + own-ship kinematics; the roles above (§2.2–§2.5) are not separately simulated agents today — they are currently implicit inside the OOW agent's own prompt/behaviour. For the Captain layer, the cleanest near-term approach is **not** to build 4 new LLM agents, but to represent their outputs as **structured facts** fed into the Captain's and OOW's prompts (e.g. "engine: max speed currently 8kn following a reported fault", "look-out: contact bearing 045, unidentified small craft") — consistent with this project's established "state computed facts directly, don't trust the model to re-derive them" convention (see `pipeline-notes.md` / `basic_simulator.md`). Whether any of these ever become their own fine-tuned agents (hence the 4 already-reserved empty `Data/` folders) is a separate, later decision, not needed to build the Captain layer itself.

---

<a id="sec-3"></a>
## 3. Communication protocol between Captain and OOW

**Who asks what, from whom, when, and why.** This section formalises the real bridge communication doctrine (STCW/ICS *Bridge Procedures Guide* "calling the master" practice + "standing orders"/"night order book" practice) into a protocol usable directly as the message-passing contract between the two agents.

<a id="sec-3-1"></a>
### 3.1 Upward: OOW → Captain

| Class | Trigger | Urgency | Examples |
|---|---|---|---|
| **Routine** | Scheduled | None — Master reads, responds only if something looks wrong | Watch handover summary; position/progress report (noon report, waypoint arrival — ETA, fuel/consumption); weather updates on material forecast change |
| **Mandatory ("call the Master")** | Event-triggered | High — must call, not just log | See checklist below |
| **Emergency** | Event-triggered | Immediate, overrides everything | Fire, flooding, collision, person overboard, imminent grounding, piracy/security incident |

The **mandatory "call the Master" checklist** is a real, standard industry practice (every company's SMS/Master's Standing Orders includes a version of this list; phrased here in general, non-company-specific form). It is the single most important artefact for this design since it defines exactly **when the Captain agent must be invoked at all** during a simulated mission:

- Visibility deteriorates, or is forecast to deteriorate, below a stated threshold.
- Any doubt about the ship's position, or failure/erratic behaviour of GPS/ECDIS/gyro/radar.
- Traffic density increases, or any risk-of-collision situation develops that is not clearly resolving in good time under COLREG.
- Machinery, steering gear, or navigational equipment failure or malfunction of any kind.
- Weather deteriorates beyond the forecast, the barometer falls rapidly, or heavy weather is expected.
- Any deviation from the planned track/route becomes necessary for any reason.
- Any alarm that is not immediately understood or resolved.
- Approaching a waypoint, landfall, pilot station, or hazard earlier or later than planned by a material margin.
- Any distress signal, urgency message, or unusual radio traffic is received.
- Entering a restricted manoeuvring area, anchorage, or shoal water.
- Any injury, illness, security concern, stowaway, or crew incident.
- Approaching port limits / pilot boarding / berthing.
- **"When in doubt about anything — call the Master"** is the explicit, literal, universal fallback rule taught in every Bridge Procedures Guide; it deliberately has no precise boundary, by design.

Every entry in this list is a direct, one-to-one hook for a brown-envelope category in §4 — this is not a coincidence, it's why real bridge doctrine and this project's brown-envelope taxonomy line up so cleanly.

<a id="sec-3-2"></a>
### 3.2 Downward: Captain → OOW

| Instrument | Scope/duration | Example | Maps onto (this codebase) |
|---|---|---|---|
| **Standing Orders** | Persistent, whole voyage, until changed | "Call me for any target closer than 3nm on a steady bearing"; "no unattended bridge, ever"; "reduce to half speed automatically in visibility under 2nm" | `constraint_line()` / `VesselConstraints` — a persistent constraint injected into every future OOW prompt/decision |
| **Night Orders** | One night/watch period | "Call me before the pilot station"; "expect fishing fleet after 0200, reduce speed proactively"; "wake me at first sight of the reported ice" | A time-boxed override of Standing Orders, logged in the Night Order Book |
| **Direct real-time instruction** | Active brown envelope, until resolved | New waypoint/route; new speed limit; "hold this course until I say otherwise"; "contact VTS now"; "prepare for heavy weather" | The Captain agent's primary **output** channel; changes the OOW's operating envelope directly (see §3.3) |

<a id="sec-3-3"></a>
### 3.3 Mapping onto the agent architecture

- At every decision epoch, the OOW agent's prompt is built as today (situation report + constraints), **plus** any currently-active Captain instruction rendered as an additional, explicit constraint line — exactly the same mechanism `constraint_line()`/`goal_course_check_line()` already use for physical/COLREG facts (see `basic_simulator.md`'s established "state computed facts directly" convention). No new prompt-injection mechanism needs to be invented — this is additive.
- The Captain agent is invoked (a) at scheduled routine-report points (cheap, low-stakes, mirrors 3.1 Routine), (b) whenever a brown-envelope event fires that matches one of the Mandatory triggers, and (c) immediately and unconditionally for anything Emergency.
- The Captain's output structure builds directly on the already-found `Data/Captain/tools.json` schema (`reroute`/`slow_down`/`stop`/`hold`/`resume`) for OOW-facing instructions, extended with the more strategic, non-OOW-facing actions from §1.2/§1.3 (e.g. contacting the DPA, declaring an emergency, requesting a place of refuge) that fall outside `tools.json`'s current 5-tool scope.
- A captain instruction is "resolved" (returns control fully to the OOW's own standing/night orders) either explicitly (`resume`) or implicitly once its own stated condition is met (e.g. "hold until clear of the TSS") — mirrors how `target_heading`/`VesselConstraints` already persist across steps in `app/simulation.py` until explicitly changed.

---

<a id="sec-4"></a>
## 4. Exhaustive list of "brown envelopes" (mission events)

Each category is a source for procedurally generated events, with a severity level (minor/moderate/serious/catastrophic) and an expected captain-response category. This is intended as an exhaustive **starting list** — to be further supplemented with real incidents from `Data/OOW/OOW_Incidents` and `Data/MarineNewsLetters` (see §6).

<details>
<summary><b>A. Weather &amp; environment</b></summary>

- Severe storm/hurricane/typhoon on the route
- Unexpected dense fog / severely reduced visibility
- Icing on deck/rigging (stability risk)
- Lightning strike (electronics/navigation outage)
- Extreme heat (cargo/cooling risk, crew health)
- Shifting/unexpected current (incl. El Niño/La Niña effects on current patterns, seasonal monsoon reversal)
- Unexpected tidal stream/spring tide in narrow waters
- Rogue wave / abnormal sea state
- Sandstorm near the coast (loss of visibility + engine air-intake risk)
- Volcanic ash cloud (visibility, engine-damage risk)
- Seaquake / tsunami warning
</details>

<details>
<summary><b>B. Navigation &amp; traffic</b></summary>

- Unknown/uncharted obstruction or shoal (silted channel)
- Chart/ENC error discovered
- Aid to navigation out of service or shifted
- Very dense traffic situation / fishing fleet in the fairway
- Congestion in a Traffic Separation Scheme
- Delay at a narrow canal (Suez/Panama/Bosphorus) — waiting time, convoy
- Pilot not available at the agreed time
- Tug not available for berthing
- Anchoring fails / anchor chain parts / ship drags anchor
- Near-grounding due to a navigational error
</details>

<details>
<summary><b>C. Technical/mechanical</b></summary>

- Main engine failure
- Steering gear failure / rudder jammed
- Generator failure / blackout
- Propeller fouled/damaged (fishing net, floating object)
- Radar/ECDIS/GPS failure
- Fire in the engine room
- Ingress of water / hull breach
- Ballast system failure (stability problem)
- Refrigeration failure (reefer cargo spoilage)
- Fuel contamination / wrong bunkers loaded
- Fuel shortage (miscalculation/long detour)
- Anchor windlass failure
</details>

<details>
<summary><b>D. Crew/human</b></summary>

- Medical emergency (heart attack, severe injury)
- Crew member overboard
- Death on board
- Psychological crisis / suicide risk
- Fight / threatened mutiny
- Drunkenness on watch
- Desertion in a port
- Stowaway discovered on board
- Crew shortage due to a visa/repatriation problem
- Fatigue-related error by a key officer (e.g. OOW falling asleep)
- Labour dispute/work stoppage (strike-like)
</details>

<details>
<summary><b>E. Cargo &amp; commercial</b></summary>

- Cargo shift (stability risk)
- Cargo fire (especially containers/dangerous goods — cf. real MAIB/CHIRP cases)
- Cargo damage/spoilage
- Wrong cargo loaded (documentation error)
- Cargo theft (in port or under way)
- Charterer demands a speed increase or skipping a port
- Late/incomplete cargo documentation (customs problem)
- Customs dispute at a port of call
- Cargo owner demands a course change (e.g. market price change)
- Sudden bunker-price shock making the planned route uneconomic
- A general-average situation arises
</details>

<details>
<summary><b>F. Security &amp; geopolitics</b></summary>

- Piracy attack / boarding attempt
- Armed robbery in port
- War threat / an area suddenly designated high-risk
- Sanctions-related problem with a port of call/cargo
- Blockade of a strait/canal
- Mine threat (naval conflict zone)
- Cyberattack on ship systems (navigation/cargo/communication)
- GPS jamming/spoofing
- Hostile naval vessel intercepts and demands the ship stop
- Insurgent/militia activity near the coast
</details>

<details>
<summary><b>G. Regulatory &amp; port state</b></summary>

- Port State Control (PSC) inspection/detention
- Missing or expired certificate discovered
- Quarantine/health authority holds the ship
- Immigration problem (crew/passengers)
- Unexpected port congestion / no berth available
- Ship's agent fails to arrange services (bunkers, provisions)
- Environmental violation detected (ballast water, emissions)
</details>

<details>
<summary><b>H. Medical/health</b></summary>

- Outbreak of a contagious disease on board
- Pandemic-related quarantine requirement
- Request for medical assistance from another vessel/fishing boat
- Person overboard from **another** vessel, spotted by own ship
- Need for a medical evacuation (medevac) via detour/helicopter
</details>

<details>
<summary><b>I. Environment &amp; nature</b></summary>

- Marine mammals on the route (mandatory speed reduction/detour)
- Oil spill (own ship, or observed on another vessel — reporting duty)
- Red tide/algal bloom blocking cooling-water intake
- Severe icing requiring de-icing
- Mandatory MARPOL report after an incident
</details>

<details>
<summary><b>J. SAR &amp; humanitarian</b></summary>

- Distress call from another vessel (SOLAS duty to assist)
- Refugee/migrant boat encountered
- Request from the coast guard to assist in a rescue operation
- Collision with debris/floating hazard (UXO/container overboard)
- An unknown wreck/derelict vessel encountered — reporting/investigation duty
</details>

<details>
<summary><b>K. Communication &amp; administrative</b></summary>

- Cargo dispute via satellite communication during the voyage
- Family emergency ashore requiring a crew member's repatriation
- VHF equipment failure
- Satellite communication outage (no contact with shore)
- Urgent instruction from the company that conflicts with safety procedure (the Master must apply ISM Art. 5: safety comes first)
</details>

<details>
<summary><b>L. Rare "old logbook" flavour events (for colour/variety)</b></summary>

- Unexpected salvage opportunity (another vessel in distress offers salvage rights — a Lloyd's Open Form-like situation)
- Unknown/unreported wreck or ship's remains encountered
- Territorial-waters dispute / unclear jurisdiction
- Rumour of sabotage or a stowaway with a hidden agenda
- A historical mine/submarine remnant reported by another vessel
- Unexpected visit from a naval vessel for inspection (e.g. an anti-piracy patrol)
</details>

---

<a id="sec-5"></a>
## 5. Where to find example missions / source material

<a id="sec-5-1"></a>
### 5.1 Already present in this project

*Verified by direct inspection on 2026-09-30, not just filenames.*

> **Correction** to the earlier version of this note: CHIRP was already processed — that had been mis-assessed without reading the pipeline code first. The real, verified situation is more interesting:

**a) `Data/MarineNewsLetters/`** — 67 CHIRP Maritime Feedback (MFB) newsletters, already fully processed by `pipeline/ingest/build_chirp_json.py` (2-column PDF parsing via PyMuPDF, article splitting + "CHIRP Comment" separation) and included in `oow_rag_chunks.json` (3870 chunks total).

| | Count |
|---|---|
| Total parsed CHIRP articles | **608** |
| Used for OOW training (COLREG-relevant) | 181 (30%) |
| **Unused — non-COLREG, ready to mine for Captain** | **427 (70%)** |

The 427 unused articles are exactly the non-COLREG categories the captain must be able to handle: fire, mooring, medical, technical, cargo, etc. This is the strongest, most ready-to-use source for brown-envelope content — no new PDF processing needed, just a new filter/extraction script that takes the non-COLREG articles instead of leaving them unused.

**b) `Data/Captain/Original/{uk,usa}/**/*.pdf`** — on inspection this turns out to be an **exact copy** (806 files, byte-for-byte identical SHA256 hashes for all 343 UK and 463 US files) of `Data/OOW/OOW_Incidents/{uk,usa}`. No new material.

**c) `Data/Captain/Processed Leo/`** — the **real new material**:

| | |
|---|---|
| Files | 108 pdfs (`pdf1.pdf`..`pdf141.pdf`, gaps in numbering) |
| Overlap with the 806-file OOW_Incidents corpus | **0/108** (verified by SHA256 hash) |
| Genuine MAIB collision reports | **105/108 (97%)** |
| Other 3 | 1 Royal Navy "fishing vessel avoidance code of practice" extract + 2 reports not using the word on page 1-2 (one is a man-overboard case) |
| Date range (sampled) | **1990–2014** — i.e. before `OOW_Incidents/uk`'s earliest year-folder (2015) |
| Already processed | Only 2/108 (`pdf2.json`/`pdf3.json`, via an external "ByteIT Worker" parsing service, **not** this project's own pdfplumber/pymupdf pipeline) |

This is a 25-years-older, separately curated, ~97%-collision-relevant MAIB set that has never been used in the pipeline — a much higher COLREG-relevance hit rate (97%) than the broad OOW_Incidents scan ever achieved (~9-11% clears the score threshold in `screen_incidents.py`). Usable both for an enriched OOW/COLREG training round and for Captain-level "what should the captain have decided differently here" material.

**d) `Data/Captain/tools.json`** — a ready-made action schema with 5 tools:

```json
{
  "reroute":    { "urgency": "immediate|planned", "direction": "port|starboard|any",
                  "min_cpa_nm": "float", "speed_kn": "float|null" },
  "slow_down":  { "speed_kn": "float", "reason": "safe_speed|restricted_visibility|traffic_density|precautionary" },
  "stop":       { "reason": "collision_imminent|fog_signal_ahead|await_clearance" },
  "hold":       { "monitor_tcpa_min": "float" },
  "resume":     { "speed_kn": "float|null" }
}
```

This covers almost exactly what was described as the Captain's role (new waypoints, e.g. a different course in a storm, no faster than X after an engine problem) — strong suspicion this is an earlier-designed (possibly by the same "Leo") action space for precisely this Captain layer, not yet wired to any code. Direct candidate for the Captain agent's formal output schema (alongside free-text reasoning/CoT); extended in §8/§9 below.

- Existing generator patterns (`Basic Simulator/generate_imazu_missions.py`, `generate_random_imazu_missions.py`, `generate_impossible_missions.py`) already show how to procedurally generate fictional-but-realistic scenarios + "failure category" tags — the same pattern extends naturally to multi-leg missions with brown-envelope injection.

<a id="sec-5-2"></a>
### 5.2 External, publicly accessible sources

*For research/inspiration — no scraping instructions, just pointers to where to look yourself.*

- **IMO GISIS** (Global Integrated Shipping Information System) — the IMO's own casualty/incident database, including flag-state casualty investigation reports worldwide.
- **EMSA** (European Maritime Safety Agency) — the EU accident database (EMCIP), annual "Annual Overview of Marine Casualties and Incidents".
- **TSB Canada / ATSB Australia** (marine investigation reports) — same format as MAIB/NTSB, additive to what's already in the repo.
- **"Old Weather"** (oldweather.org, a Zooniverse citizen-science project) — transcriptions of real ships' logbooks (Royal Navy, US whaling logs from the 19th/20th century) — authentic daily logbook language and structure.
- **UK National Archives**, ADM series (Admiralty logs) and the **Lloyd's Register Foundation Heritage & Education Centre** — digitised historical ships' logbooks and case material.
- **Lloyd's List / The Maritime Executive / Splash247 / gCaptain** — trade journalism with many "what went wrong here" stories, often centred on Master's decision-making (useful for narrative tone, not as hard data).
- Published memoirs by masters/pilots — good for the "job description" tone and realistic decision-making under pressure, not to be scraped as training data but as reading material for writing credible mission briefings.
- **BIMCO/ICS** operational guidance and the **Nautical Institute's "Mars Reports"** (Mariners' Alerting and Reporting Scheme) — comparable to CHIRP, short anonymous near-miss stories aimed at officers/masters.

---

<a id="sec-6"></a>
## 6. `Data/Captain/` folder structure & Captain's own training pipeline

Mirrors the existing `AgentPaths(domain=..., source_dirname=...)` convention already used for VHF/OOW (`core/paths.py`), and the two-track (rules-&-knowledge / conversational-mission) split described in `.github/copilot-instructions.md`.

<a id="sec-6-1"></a>
### 6.1 Proposed folder layout

| Folder | Purpose | Analogous to (VHF/OOW) |
|---|---|---|
| `Data/Captain/CaptainProtocol/` | Source regulatory texts: ISM Code, SOLAS extracts, STCW extracts, MARPOL extracts, ICS Bridge Procedures Guide summary, Master's Standing Orders / Night Order Book templates, Mission Order (§8) templates | `VHFProtocol/`, `COLREGRules`-equivalent for OOW |
| `Data/Captain/Original/` *(existing)* | Raw incident PDFs — confirmed duplicate of `OOW_Incidents`, kept as-is or later symlinked/removed | `OOW_Incidents/{uk,usa}` |
| `Data/Captain/Processed Leo/` *(existing, rename candidate: `Captain_Incidents_Extra/`)* | The 108 new, unused, 97%-collision-relevant 1990–2014 MAIB reports (§5.1c) | — (new, no OOW equivalent) |
| `Data/Captain/Captain_Eval/` | **Held-out**, never used for training | `VHF_Eval/`, `OOW_Eval/` |
| ↳ `captain_gold_answers.json` | Rules/knowledge Q&A (ISM/SOLAS/STCW/MARPOL), analogous to `vhf_gold_answers.json` | Track 1 eval |
| ↳ `captain_mission_scenarios.json` | Full mission + brown-envelope decision scenarios, analogous to `vhf_colreg_scenarios.json` | Track 2 eval |
| `Data/Captain/Captain_JSON/` | Parsed structured documents (output of a future `build_captain_json.py`) | `OOW_JSON/`, `VHF_JSON/` |
| `Data/Captain/Captain_Agents_Training/` | RAG chunks, KG, PG, reranker pairs, SFT/DPO/Reflection `.jsonl` files, reasoning traces (cache dir) | `OOW_Agents_Training/`, `VHF_Agents_Training/` |
| `Data/Captain/tools.json` *(existing)* | Formal output/tool schema for the Captain agent (§5.1d), extended in §8/§9 | — (new, no OOW equivalent — OOW's action schema lives in `pipeline/oow_agent_spec.py` instead) |
| `_models/Captain/` | Fine-tuned adapters + merged models (own QLoRA SFT/DPO/Reflection chain) | `_models/VHF/`, `_models/OOW/` |

<a id="sec-6-2"></a>
### 6.2 Own training pipeline, mirroring the two-track model

Following the same Track 1 (rules & knowledge) / Track 2 (mission execution / "conversational") split already used for VHF and OOW:

| | Track 1 — rules & knowledge | Track 2 — mission execution ("agentic") |
|---|---|---|
| Eval (held out) | `captain_gold_answers.json` | `captain_mission_scenarios.json` |
| Source material | ISM/SOLAS/STCW/MARPOL text + the 427 unused non-COLREG CHIRP articles (§5.1a) + the 108 new MAIB reports (§5.1c) | Full simulated missions with brown envelopes + captain's sequential decisions (§8/§9) |
| Extraction | `build_captain_json.py` (new) → `extract_captain_reasoning.py` (new, mirrors `extract_chirp_reasoning.py`/`extract_incident_reasoning.py`, minus the COLREG-only concept filter) | `build_captain_scenarios.py` (new, mirrors `pipeline/track2/build_oow_scenarios.py`) + outcome-mining (mirrors `build_outcome_dpo.py`/`build_outcome_reflection.py`) |
| Training files | `captain_sft_direct.jsonl`, `captain_sft_cot.jsonl`, `captain_dpo_pairs.jsonl`, `captain_reflection.jsonl` | `captain_scenario_sft_*.jsonl`, `captain_scenario_dpo_pairs.jsonl`, `captain_scenario_reflection.jsonl`, `captain_outcome_dpo_pairs.jsonl`, `captain_outcome_reflection.jsonl` |

Both tracks feed the **same** Captain QLoRA fine-tune (own model, per the confirmed architecture decision), same as VHF/OOW's own tracks do today.

<a id="sec-6-3"></a>
### 6.3 "Experience" — RAG + reranking + KG + PG

Per explicit user request ("de captain moet ook heel veel ervaring hebben, dat gaan we o.a. via RAG en reranking doen en KG en PG"), the Captain gets the **same 4-part retrieval architecture** already built for OOW, applied to its own corpus:

| Component | OOW (existing) | Captain (proposed) |
|---|---|---|
| **RAG** (dense retrieval over chunked text) | `pipeline/ingest/build_rag.py` over COLREG text + incidents + CHIRP | Same script, own corpus: ISM/SOLAS/STCW/MARPOL text + brown-envelope incident library (§6.2) + historical mission narratives |
| **Reranker** (fine-tuned cross-encoder) | `pipeline/ingest/build_reranker_pairs.py`, trained on COLREG situation-report queries | Same architecture, retrained on Captain's own query distribution (brown-envelope situation reports, not COLREG ones) |
| **KG** (Knowledge Graph, concept-linked chunks) | `pipeline/ingest/build_kg.py` | Same script, concept graph over regulatory/incident vocabulary (e.g. links "engine failure" → relevant MARPOL/ISM sections + past incident cases) |
| **PG** (Procedural Graph, "what to do given condition X") | `pipeline/ingest/build_pg.py` / `pg_guidance.py`, mined from COLREG reasoning traces | Same script, mined from Captain incident reasoning traces + ISM-mandated procedures (e.g. "in case of fire: muster → contact DPA → ...") |

No new retrieval architecture needs to be invented — this is a second domain instance of an already-proven pipeline (`AgentPaths(domain="Captain", source_dirname="CaptainProtocol")` is, by design, all that's needed to point every existing `build_*.py`/`pipeline/ingest/*` script at the new domain).

---

<a id="sec-7"></a>
## 7. Route & waypoints

- **For now**: a mission's route is an ordered list of waypoints (extends the existing `app/missions.py` `Mission.waypoint`/`goal` schema from one optional waypoint to N). **Decided** (2026-09-30): generate a fake route directly — a handful of random waypoints between A and B (e.g. Rotterdam → Den Helder), same procedural pattern as `generate_random_imazu_missions.py`. The route is **provided/faked directly in the Mission Order** (§8), attributed conceptually to "the OOW" (as if the OOW's own planning had produced it), even though for now it is just authored by the scenario generator.
- **Decided** (2026-09-30): fictional obstacles to route around are best modelled as explicit **exclusion zones** (polygons/areas the route must not cross), not just as waypoint detours — e.g. a storm cell, a piracy corridor, a whale-protection area. Genuinely dangerous/real-world areas can be layered onto the waypoint set later; for now exclusion zones are synthetic/fictional, matching this project's existing "fictional but grounded" scenario convention.
- **Decided** (2026-09-30): alongside these mission-specific (dynamic) exclusion zones, the route also needs **static geographic no-go areas** — coastlines and shallow water/shoals — which exist independently of any brown envelope and never change mid-mission. Two distinct categories, both represented with the SAME polygon schema as exclusion zones (no new geometry mechanism needed), just tagged differently:
  - **Coastline** (`type: "coastline"`) — land, always a hard no-go, approximated as a simplified polygon/polyline (a coarse simplification is fine at this abstraction level, exactly the "simple polynomial" level of detail suggested — not full nautical-chart precision).
  - **Shallow water / shoal** (`type: "shallow_water"`) — no-go **conditional on own-ship's draft**: a zone with `min_depth_m` is only a real obstacle once `min_depth_m < draft_m + a safety margin`; this requires the ship profile (§13.A.3) to carry a `draft_m` field, feeding directly into the route planner's feasibility check (§13.B.8).
  - **Recommended sourcing**: rather than generating fictional coastlines per mission, maintain a small, reusable library of named **geographic regions** (e.g. `"nl_north_sea_coast"`), each with its own fixed coastline + shoal polygon set; missions reference a `region_id` (§13.C.13) instead of inventing new coastlines every time — cheaper, more consistent, and reusable across many generated missions for the same corridor (e.g. every Rotterdam↔Den Helder mission shares the same coastline data). Real, free, legitimately-sourceable data exists for this: simplified coastline extracts from OpenStreetMap/OpenSeaMap (the same overlay source this project's own early design notes already considered, see `Basic Simulator/generate_moos_scenarios.py`'s history) and bathymetry from GEBCO/EMODnet — decimated down to a coarse polygon, not used at full chart resolution.
  - Fixed man-made installations (wind farms, platforms) are a third, similarly-static category (`type: "fixed_installation"`) — same polygon schema, no draft-dependence, just a permanent marked no-go area.
- **Decided (2026-09-30): a `ports` library** — the missing piece behind `request_place_of_refuge`/`abort_mission` (§9.1) and the Engine-failure event's own "port of refuge at ~40 nm" premise (§13.A.2): none of these are executable without an actual list of candidate ports to divert to. Same reusable-library pattern as the geographic regions above, not invented per mission: a small, named set of ports per `region_id` (§13.C.13). Schema: `id`, `name`, `position` (lat/lon), `services` (drawn from a fixed vocabulary — e.g. `repair`, `fuel_bunkering`, `medical`, `customs`, `anchorage`), `min_approach_depth_m` (feasibility check against `draft_m`, same mechanism as `shallow_water` zones), `region_id`. The Mission Order's own `scheduled_port_calls` (§8.2) is a list of `id`s **into this same registry**, not a separate freeform list — one canonical port source, not two that could silently diverge. Real, free, legitimately-sourceable data exists here too (a simplified World Port Index extract or OSM port nodes), the same sourcing pattern as the coastline/bathymetry data above. (Land itself is already decided, not an open question: coastline is `type: "coastline"` in the exclusion-zone set above, a hard no-go for the route planner — this is not "open sea with no land", ports sit directly against that same coastline layer.)
- **Decided** (2026-09-30): brown-envelope density scales with route length — roughly **1 brown envelope per 100 nm** (e.g. a 600 nm route carries ~6 brown envelopes). This gives a simple, tunable scenario-generation knob rather than a fixed count per mission.
- **Later**: the OOW can generate the route itself via a real, deterministic route-planning algorithm. **Decided direction** (2026-09-30): most likely via the **PredictWind API** (a real commercial weather-routing service) once available/integrated; a manually downloaded/imported real-world route is also an option in the meantime. This is intentionally **more deterministic, less LLM-dependent** — consistent with this project's general philosophy of using a deterministic algorithm for what is determinable, reserving the LLM for judgement calls. Exact integration not designed yet.
- **Captain's interface to the route**: the Captain does **not** edit waypoints directly. It **instructs** the OOW to avoid a zone / take an alternate route in response to a brown envelope — via the `reroute` tool in `tools.json`, extended with an `avoid_zone` / `preferred_corridor` parameter (see §9). The OOW's own route planner then re-derives the actual new waypoint list. This preserves the real chain-of-command principle: the Captain gives intent/constraint, the OOW executes the actual navigation planning — exactly as a real Master instructs an OOW at a strategic level without dictating exact rudder angles.

---

<a id="sec-8"></a>
## 8. Mission Order format

Adapted from the real NATO/allied-navy **Movement Order (MOVORD)** / **Operation Order (OPORD)** five-paragraph **SMEAC** structure (Situation / Mission / Execution / Admin & Logistics / Command & Signal) — the standard format used to brief a ship's captain on a transport mission, chosen here because merchant "sailing orders"/charter-party instructions follow the same underlying logic (goal, constraints, resources, reporting) even though the paperwork looks different. Military-specific fields (security classification, Rules of Engagement) are dropped or replaced with their civilian equivalents; the structural discipline is kept because it maps 1:1 onto what a Captain agent needs as input.

<a id="sec-8-1"></a>
### 8.1 The six parts, adapted for this project

| # | NATO name | Civilian/merchant adaptation |
|---|---|---|
| 1 | Header & classification | `mission_id`, date-time issued, issuing authority (the company's Fleet Operations Centre / DPA, not a naval command) — no security classification needed for a civilian mission; also carries `t0_utc` (departure wall-clock) and `dt_mission_s` (mission-sim step size, §13.A.1) |
| 2 | Situation | Operational context: weather forecast, known hazard areas (piracy risk zones, ice, restricted waters), other traffic/friendly-vessel info, any brown-envelope risk flags already known before departure |
| 3 | Mission | A clear, concise goal statement — who/what/when/where/why (e.g. "deliver 200 refugees safely to Port X"; "deliver cargo Y intact to Port Z by deadline D") + explicit **success criteria** |
| 4 | Execution | Commander's intent + the **Plan of Intended Movement (PIM)**: waypoint list, Speed of Advance (SOA), restricted zones to avoid, applicable rules of conduct (COLREG + company policy, replacing military ROE) |
| 5 | Admin & Logistics | Cargo manifest, resource budget (fuel/bunkers, crew rest-hour budget), scheduled port calls/replenishment, refuelling plan |
| 6 | Command & Signal | The reporting plan (§3.1 routine/mandatory/emergency triggers), communication channels, **AIS policy** for high-risk zones (the civilian equivalent of military EMCON — real BMP5 anti-piracy guidance already recommends reduced AIS broadcast in high-risk areas), escalation path to the DPA |

<a id="sec-8-2"></a>
### 8.2 Example Mission Order (adapted to this project's schema conventions)

The Mission Order can optionally pre-script known brown envelopes directly into the mission definition, as a scripted `events` array (e.g. an engine failure at a given mission time). This is not a new mechanism to invent: `app/missions.py` already has exactly this pattern for scripted target-vessel manoeuvres (`Mission.target_maneuvers: list[dict]`, schema `{"target","trigger_time_s","new_heading_deg","new_speed_kn"}`, built 2026-09-26 for the IMP01-10 missions). Generalising `target_maneuvers` from "scripted target behaviour" to "scripted brown envelope of any category from §4" is the natural, already-precedented way to make mission-level scenario generation possible, rather than inventing a separate events mechanism.

```json
{
  "mission_id": "MSN-2026-0142",
  "issued_by": "Fleet Operations Centre",
  "issued_at": "2026-10-12T14:00:00Z",
  "vessel": "MV Example Trader",
  "t0_utc": "2026-10-12T18:00:00Z",
  "dt_mission_s": 600,

  "situation": {
    "weather_forecast": "Sea state 4, increasing to 6 near the Azores",
    "known_hazards": ["piracy risk zone: Gulf of Guinea corridor", "reported ice: N Atlantic, lat > 55N"],
    "other_traffic": "convoy of 3 vessels transiting the same TSS 12h ahead"
  },

  "mission": {
    "goal": "Transport 40x TEU containers (general cargo) from Port Norfolk to Port Rota; deliver intact, not later than 2026-10-24T08:00:00Z",
    "success_criteria": [
      "cargo delivered with zero damage",
      "no COLREG violations",
      "arrival within the stated deadline",
      "no avoidable crew injury"
    ]
  },

  "execution": {
    "commanders_intent": "Safe and timely delivery while avoiding the reported piracy corridor",
    "plan_of_intended_movement": {
      "waypoints": [
        {"id": "WPT1", "lat": 36.95, "lon": -76.00, "note": "departure"},
        {"id": "WPT2", "lat": 38.00, "lon": -65.00},
        {"id": "WPT3", "lat": 39.50, "lon": -30.00, "note": "Azores passage"},
        {"id": "WPT4", "lat": 36.62, "lon": -6.32, "note": "arrival"}
      ],
      "speed_of_advance_kn": 12.0,
      "restricted_zones": ["Gulf of Guinea piracy corridor (avoid entirely)"],
      "rules_of_conduct": "COLREG 1972 + company Safety Management System"
    }
  },

  "admin_logistics": {
    "cargo": "40x TEU containers, general cargo, no dangerous goods",
    "resources": {
      "fuel_tonnes_at_departure": 850,
      "fuel_reserve_margin_pct": 15,
      "crew_rest_hour_budget": "STCW A-VIII/1 compliant, no waivers pre-approved"
    },
    "scheduled_port_calls": [],
    "replenishment_plan": "none scheduled — direct passage"
  },

  "command_signal": {
    "reporting_interval_hours": 24,
    "deviation_report_threshold": "position >20nm or ETA >6h off PIM",
    "escalation_path": ["OOW -> Captain", "Captain -> DPA", "DPA -> Flag State (if required)"],
    "ais_policy": "normal broadcast, EXCEPT reduced/off while transiting any listed high-risk zone (BMP5 practice)"
  },

  "events": [
    {"t_s": 32400, "type": "engine_failure", "severity": "moderate",
     "details": "main engine RPM limited to 60%; Chief Engineer caps speed at 8 kn"}
  ]
}
```

<a id="sec-8-3"></a>
### 8.3 The response: Mission Progress Report (the MOVREP-equivalent)

Once the Captain (and, in this project's architecture, the OOW) reviews the PIM for viability (fuel consumption, weather routing, navigational hazards), the ship sends a **Mission Progress Report (MPR)** — the civilian-flavoured equivalent of a naval **MOVREP**, and in substance no different from the **noon report** / position report merchant ships already send their operations department in real life. It says: "we received the order, here is our exact timeline to execute it, and here is when we will cross specific operational boundaries."

- Sent at the `reporting_interval_hours` cadence from §8.2 — this **is** the Routine class from §3.1.
- Sent **immediately** (out of cadence) whenever the ship deviates from the PIM by more than the `deviation_report_threshold` — this **is** the Mandatory class from §3.1, specifically the "any deviation from the planned track/route" trigger.
- A brown envelope that forces a `reroute` (§7/§9) always triggers an out-of-cadence MPR, since it is by definition a PIM deviation.

This closes the loop cleanly: the Mission Order (§8.2) is the Captain's **input contract**, the Mission Progress Report is the Captain's **output contract** back to the company/DPA, and §3 is the **internal** contract between the Captain and its own OOW.

<a id="sec-8-4"></a>
### 8.4 Mission State — the live counterpart of the Mission Order

The Mission Order (§8.2) is issued **once**, at the start of the mission. What the Captain actually manages during the mission is a separate, continuously updated **Mission State** object — not a text memory, an explicit structured object:

- **Goals**, each with a priority and a condition (e.g. "reach Port B before 14:00"; "maintain ≥12 kn"; "avoid Zone X") — initialised from the Mission Order's `mission.success_criteria`, but can be individually relaxed/abandoned as the mission proceeds (see §10.1's refined "Mission outcome" axis).
- **Resources and status** — engines, fuel, steering, sensors, crew — initialised from `admin_logistics.resources`, then depleted/updated by every brown envelope and every Captain decision (a detour spends fuel and time; a slow-down after an engine fault spends time).
- **Hazards and constraints** — whale zones, weather, a hostile vessel — initialised from `situation.known_hazards`, appended to as new brown envelopes are discovered mid-mission.
- **The active plan** — route, speed profile, regime (§9.4), contingencies — this is what actually gets handed down to the OOW (§3.2/§3.3), and is exactly what `reroute`/`slow_down`/`hold` mutate.
- **A log of events and decisions taken** — the running record this project already writes for every OOW run (`_llm_runs/*.json` checkpoints); the Captain-level equivalent extends the same pattern with mission-level entries.

Concretely, the Mission State is `Mission Order (frozen) + accumulated deltas` — the Mission Order never changes once issued, the Mission State does, every time a brown envelope or a Captain decision touches it. This is the object the Captain's prompt is built from at every invocation (§3.3), not a re-read of the original order each time.

---

<a id="sec-9"></a>
## 9. Captain goals, planning & reasoning

Explicit cross-cutting requirement: the Captain must always operate against three inputs, and must **plan**, not just react:

1. **Mission goal + success criteria** — from the Mission Order (§8.2 `mission.goal` / `mission.success_criteria`).
2. **Current resource state** — fuel, crew rest-hour budget, time remaining to the deadline (§8.2 `admin_logistics.resources`), which is **consumed** over the course of the mission and by brown-envelope responses (a detour costs fuel and time; a slow-down after an engine problem costs time; calling at a port of refuge costs both).
3. **An incoming brown-envelope event** (§4).

The Captain's output is a decision **and an updated plan** — not just a one-off patch. After every brown envelope, the Captain should explicitly re-derive whether the **original** mission goal is still achievable given the resources remaining, and decide to (a) continue as planned, (b) continue with a revised plan (reroute/slow down/replenish), (c) accept partial mission fulfilment, or (d) abort — this is genuine re-planning, grounded in the priority ladder from §1.2.8.

<a id="sec-9-1"></a>
### 9.1 Extended tool/output schema (building on `tools.json`, §5.1d)

| Tool | Scope | Status |
|---|---|---|
| `reroute` | OOW-facing | existing in `tools.json`; extend with `avoid_zone` / `preferred_corridor` params per §7 |
| `slow_down` | OOW-facing | existing |
| `stop` | OOW-facing | existing |
| `hold` | OOW-facing | existing |
| `resume` | OOW-facing | existing |
| `contact_dpa` | strategic, not OOW-facing | **new**, needed for §1.2.4/§1.2.9 escalation |
| `declare_emergency` | strategic | **new**, needed for §1.2.5 (SAR/muster/security activation) |
| `request_place_of_refuge` | strategic | **new**, needed for §1.2.9 |
| `abort_mission` / `accept_partial_mission` | strategic | **new**, the explicit re-planning outcomes from the paragraph above |
| `set_regime` | strategic | **new**, switches the vessel's formal COLREG conduct status and/or the operating regime (COLREG vs. security/self-protection, §9.4) |

<a id="sec-9-2"></a>
### 9.2 Grounding in "experience"

Per explicit user request, the Captain's reasoning must be grounded in extensive experience, delivered the same way OOW's is: **RAG + reranking + KG + PG** — see §6.3 for the full mapping of each component onto the Captain's own corpus. This is what turns "goals + resources + an event" into a **grounded** decision rather than an unconstrained LLM guess: the retrieved regulatory text supplies the *legal* constraint, the retrieved incident/CHIRP cases supply *precedent* ("what happened last time a captain faced this"), and the Procedural Graph supplies the *concrete next-step sequence* mandated by ISM/SOLAS/company procedure.

Verifiable reward is more limited here than for OOW (goals are checkable, judgement quality is not) — expect more weight on SFT/DPO/Reflection than on RL. The "critic" tasks this implies (grading a draft mission decision against what an incident report shows really happened) map directly onto this project's existing Reflection-stage draft/critique/refined pattern (`pipeline/track1/build_reflection.py`), and "less RL, more distillation" is consistent with the project's existing distillation pipeline stage (`compress_distill.py`) rather than requiring a new RL loop to be built.

<a id="sec-9-3"></a>
### 9.3 Where the LLM is — and is not — the right tool

Not every brown envelope should be reasoned through from scratch by the LLM:

- **Known emergency procedures** (engine failure, blackout, man overboard, steering failure) are checklist material. These belong in a **deterministic procedure layer** (a state machine or behaviour tree), not free-form reasoning. The Captain LLM's job here is to **select and justify** which procedure applies, not to re-derive the steps themselves.
- **New or ambiguous situations** — a contact behaving abnormally, conflicting goals (on-time arrival vs. avoiding a whale zone vs. sparing the engine), incomplete information — are exactly where the LLM earns its keep: weighing, reprioritising, explaining.
- **Never**: the safety boundaries. Minimum distances, no-go zones, maximum speed in restricted visibility, "no port turn under Rule 19" — these live in a hard-coded **safety shield** beneath the Captain's strategic authority, and apply regardless of what the Captain (or the company) wants. This is not a new invention — it generalises the same hard rule-legality gate `app/oracle_planner.py`'s `required_direction()` already enforces *before* any cost-based ranking for the OOW. The Captain's ISM Art. 5 "overriding authority" (§1.1) overrides *commercial* pressure, never the shield itself — the two are consistent, not in tension.

<a id="sec-9-4"></a>
### 9.4 Regime switching (COLREG vs. security/self-protection)

Under attack (piracy, armed robbery, a hostile naval intercept — §4.F), the applicable rule set stops being COLREG and becomes a **security/self-protection regime** (BMP5-style anti-piracy procedure for merchant shipping, already referenced in §8.1's AIS-policy discussion). This, too, is mostly checklist material — the Captain's role is to **switch regime and justify the switch**, not to improvise a security response from first principles. A regime switch typically also changes the vessel's formal COLREG conduct status (e.g. declaring "restricted in ability to manoeuvre" under Rule 3(g)/Rule 27, which changes which vessel has priority under Rule 18 for every OTHER vessel interacting with it) and the AIS broadcast policy (§8.1) — both need to be reflected as explicit facts in the OOW's own prompt (same "state computed facts directly" convention as everywhere else in this project), not left for the OOW to infer.

---

<a id="sec-10"></a>
## 10. Evaluation

The Captain gets its **own composite evaluation function**, analogous to `Evaluation Functions/evaluate_run.py`, but with different axes reflecting the different job.

<a id="sec-10-1"></a>
### 10.1 Proposed axes

| Axis | What it measures | Hard gate? | Analogous OOW axis |
|---|---|---|---|
| **Safety** | Any avoidable loss of life / catastrophic ship loss | **Yes — composite = 0** | `safety_score` (collision hard gate) |
| **Mission outcome** | Goal fully / partially / not achieved (per Mission Order success criteria) — a goal **legitimately abandoned** after a correct re-plan (§9) under genuine resource constraints is scored favourably, not as a failure; only an unjustified or poorly-reasoned failure is penalised | Total *unjustified* failure capped low (mirrors "did not reach goal" → ≤0.2); a justified, well-reasoned abort/de-scope is **not** hard-gated | (no direct OOW analogue — new) |
| **Regulatory/procedural compliance** | ISM/SOLAS/MARPOL reporting duties correctly followed (e.g. DPA notified when required, MARPOL report filed) | No (weighted) | `compliance_axis()` |
| **Procedure/regime selection accuracy** | Did the Captain invoke the correct deterministic procedure (§9.3) / switch to the correct regime (§9.4) for the situation | No (weighted) | (no direct OOW analogue — new) |
| **Explanation / citation accuracy** | Did the Captain correctly cite the ISM/SOLAS/MARPOL article backing a decision | No (weighted) | `explanation_axis()` |
| **Decision timeliness / "calling discipline"** | Was the Captain engaged exactly when the §3.1 mandatory triggers require — no missed genuine emergencies (false negatives), no "crying wolf" over-escalation (false positives) | No (weighted) | (no direct OOW analogue — new, specific to the Captain's "when to get involved" role) |
| **Resource efficiency** | Fuel/time/crew-rest budget used well relative to what the situation actually demanded | No (weighted) | `temporal_efficiency` / `spatial_efficiency` |
| **Crew welfare** | Rest-hour violations, injuries, how well crew-related brown envelopes (§4.D) were handled | No (weighted) | (no direct OOW analogue — new) |

<a id="sec-10-2"></a>
### 10.2 Composite formula (same two-hard-gate architecture as `evaluate_run.py`)

1. **Hard gate 1**: any avoidable loss of life / catastrophic ship loss → `composite = 0`, regardless of everything else.
2. **Hard gate 2**: total mission failure (goal not achieved at all, no valid abort justification) → `composite` capped at a low value (mirrors the existing `≤0.2` mission-incomplete cap).
3. Otherwise: a weighted blend of the remaining axes. **Decided starting weights** (2026-09-30):

   | Axis | Weight |
   |---|---|
   | Mission outcome | 25% |
   | Safety | 20% |
   | Crew welfare | 20% |
   | Regulatory/procedural compliance | 7% |
   | Procedure/regime selection accuracy | 7% |
   | Explanation/citation accuracy | 7% |
   | Decision timeliness / calling discipline | 7% |
   | Resource efficiency | 7% |

   The remaining 35% (beyond Mission outcome/Safety/Crew welfare) is split evenly across the 5 other axes as a simple starting default. As with OOW's own `DEFAULT_WEIGHTS` (see `pipeline-notes.md`), these are a starting point, expected to be retuned once real Captain-agent runs exist to calibrate against — not fixed forever.

This is explicitly a **first proposal**, mirroring the already-proven evaluator design rather than inventing a new one — the exact "calling discipline" false-positive/negative metric definition is left for a later, data-driven pass once real Captain-agent runs exist to calibrate against.

<a id="sec-10-3"></a>
### 10.3 Starting thresholds for the two metrics deferred in §10.1 (decided 2026-09-30)

Both use the same **80% starting threshold**, symmetrically, as a simple first pass — per the user's own "start simple, calibrate later, much of this is subjective anyway" instruction:

- **Decision timeliness / calling discipline**: recompute, at every decision epoch, whether a §3.1 Mandatory-call condition was actually true (deterministically, from the Mission State's own facts — mirrors the OOW auditor's `_ground_truth_at_checkpoint` pattern). Define:
  - `recall` = (mandatory triggers where the Captain was actually engaged) / (total true mandatory triggers) — the "no missed emergencies" side.
  - `precision` = (Captain engagements that matched a true mandatory trigger) / (total Captain engagements outside scheduled routine reports) — the "no crying wolf" side.
  - **Starting rule**: axis score = 1.0 when `recall ≥ 80%` **and** `precision ≥ 80%`, scaling linearly down to 0 below that. Note (not implemented yet): missing a genuine emergency (low recall) is arguably worse than over-escalating (low precision) — real bridge doctrine treats them very differently — so an asymmetric weighting (e.g. recall counting for more of the axis than precision) is a natural first calibration refinement once real data exists, not built into the starting rule.
- **Legitimately abandoned goal vs. unjustified failure**: when the Captain declares a goal unachievable/aborts it, independently recompute the Mission State's true resource margin at that decision point (same "recompute ground truth independently, don't trust the model's own claim" pattern already used by `build_outcome_dpo.py` for OOW). **Starting rule**: the abandonment is scored as *legitimate* when the recomputed ground truth confirms **≥80%** of the Captain's stated shortfall (i.e. the stated justification is at least 80% consistent with the independently recomputed Mission State); below 80% it is treated as an *unjustified* failure and Hard Gate 2 applies.

Both thresholds are a deliberately simple, symmetric starting point — expected to move once real Captain-agent runs exist to calibrate against, same as the §10.2 composite weights.

---

<a id="sec-11"></a>
## 11. Simulator representation

How this shows up in the actual app — extends `Basic Simulator/app/` (the existing Streamlit simulator) rather than building a separate tool, reusing every already-solved UI pattern (native Plotly scrubbing, the gray-box `st.container(key=...)` Agent panel, the checkpoint/"Inspect moment" replay pattern — see `basic_simulator.md`'s long history of iterating these) instead of re-solving them for a third layer.

<a id="sec-11-1"></a>
### 11.1 Four new UI surfaces for the four things asked about

| Question | New UI surface | Reuses |
|---|---|---|
| The incoming order | **Mission Briefing panel** (§11.2) | Existing "Mission content" sidebar expander pattern |
| The Captain's response | **Captain panel** (§11.4) | Existing Agent panel's gray-box container + checkpoint/"Inspect moment" pattern |
| The route the OOW delivers | **Route overlay on the plot** (§11.3) | Existing `viz_plotly.py` trajectory figure |
| Execution + brown envelopes + everyone's decisions | **Mission Log** (§11.5) | New table, built from data every layer already produces |

<a id="sec-11-2"></a>
### 11.2 Mission Briefing panel

- Renders the Mission Order (§8.2) in human-readable form — the same "expander with a full-text rendering" pattern the existing "Mission content" sidebar expander already uses (`mission.as_text()`), extended to the SMEAC-style sections (§8.1): Situation / Mission (goal + success criteria) / Execution (PIM) / Admin & Logistics / Command & Signal.
- Below it, a live **Mission State** (§8.4) snapshot — the evolving counterpart, refreshed every rerun: remaining fuel/time budget, current regime (§9.4), active Standing/Night Orders, goals still open vs. already (legitimately) abandoned.
- **Design choice to flag, not yet decided**: whether scripted future brown envelopes (§8.2's `events` array) are visible up front (useful for a developer/debug view) or hidden until they actually trigger, "fog of war"-style (closer to a real live mission, avoids trivialising the test). Recommendation: hidden by default in a "Play Mission" viewing mode, with a separate "scenario designer" toggle (mirrors this project's existing pattern of dev-only debug expanders) revealing the full scripted timeline for whoever built the scenario.

<a id="sec-11-3"></a>
### 11.3 Route overlay on the plot

Extends `viz_plotly.py`'s existing `trajectory_figure`/`animated_trajectory_figure` with three new, purely additive layers (none touch the existing collision-marker/decision-annotation code):

- **Planned route (PIM)**: the Mission Order's waypoint list, drawn as a dashed polyline with labelled markers (WPT1, WPT2, …) — visually distinct from the solid line of the ship's actually-sailed trajectory, so a `reroute` deviation is obvious (also exactly the trigger for an out-of-cadence Mission Progress Report, §8.3).
- **Exclusion zones** (§7): semi-transparent filled polygons (Plotly `layout.shapes`), colour-coded by brown-envelope category (§4) — e.g. red for a storm cell, amber for a piracy corridor.
- **Brown-envelope markers**: an icon + hover tooltip at the time/position each brown envelope actually triggered (not its scripted time, per §11.2's fog-of-war default) — reuses the same annotation mechanism already used for the collision marker, just a new icon set keyed by §4's category letter.

<a id="sec-11-4"></a>
### 11.4 Captain panel

Sits **above** the existing OOW "Agent panel" in the layout (visually establishing the chain of command, §1/§3), reusing the exact same solved UI patterns rather than inventing new ones:

- Same gray-box `st.container(key="captain_panel")` styling as the existing Agent panel.
- A "Play Mission" mode reuses the existing native-Plotly-animation + "Inspect moment" selectbox + "Show details" button pattern — but keyed on the Captain's OWN, much sparser checkpoint list (§3.1: routine reports + mandatory triggers + emergencies), not the OOW's dense per-decision-interval checkpoints. The two checkpoint lists coexist independently; scrubbing the Captain's timeline does not have to move in lock-step with the OOW's.
- Each Captain checkpoint shows: the incoming trigger (which §3.1 class fired, or which brown envelope), the Mission State snapshot at that moment (§8.4), the Captain's decision (tool call from §9.1's extended `tools.json` schema + free-text reasoning), and the resulting instruction hand-off to the OOW (rendered exactly as the new constraint line the OOW's own prompt receives, §3.3) — making the Captain→OOW handoff literally visible, not just implied.
- A persistent, always-visible **status strip** just below the header (not inside the collapsible panel): current regime (COLREG / security, §9.4), current speed/course limit in force, count of open vs. abandoned goals — the "cockpit summary" a user should see without opening anything.

<a id="sec-11-5"></a>
### 11.5 Mission Log (unified timeline)

A single, chronologically sorted, filterable-by-actor table merging what today would otherwise live in three separate places:

| Time | Actor | Event | Summary |
|---|---|---|---|
| ... | System | brown envelope | Engine failure, severity moderate |
| ... | Chief Engineer *(fact, §2.5)* | constraint | Speed capped at 8 kn |
| ... | Captain | decision | `slow_down(8kn)` + reasoning |
| ... | OOW | decision | `hold_course` (Rule 17 stand-on) |
| ... | VHF | transmission | PAN PAN broadcast |

This is the direct answer to "execution + brown envelopes + everyone's decisions": one merged, filterable deck-log-style view, built entirely from data each layer already produces (OOW's existing checkpoint decisions, the Captain's new checkpoints, VHF's existing transmission log, and the brown-envelope trigger events) — no new decision-logic needed, only a merge/render step.

<a id="sec-11-6"></a>
### 11.6 Data model implication (for later, not building now)

To support "Play Mission" replay the same way "Play Agent Mission" already works for OOW, a new **mission-level run-log wrapper** is the natural extension of the existing `app/llm_runs.py` schema — not a replacement:

```
mission_run.json
├── mission_order          (§8.2, frozen)
├── mission_state_log      (§8.4 snapshots over time)
├── captain_checkpoints[]  (trigger, mission_state, decision, reasoning, hand-off)
├── brown_envelopes[]      (trigger time/type, resolution)
└── oow_run_refs[]         — references to ordinary, UNCHANGED per-leg
                           `_llm_runs/*.json` files (existing schema),
                           one per OOW-controlled leg between Captain
                           interventions
```

The OOW's own run-log **schema** is deliberately left untouched — the Captain layer only adds a wrapper around it, so every existing OOW replay/dashboard tool keeps reading the nested files unmodified. **Correction (2026-09-30)**: the run-log *schema* staying unchanged does NOT mean the run-log *evaluator* can stay unchanged too — see §13.C.14, `evaluate_run.py` needs a new encounter mode or these nested evaluations are silently wrong.

---

<a id="sec-12"></a>
## 12. Open questions / not yet decided

- **Decided** (2026-09-30): the captain becomes a separate, separately-trained Qwen model (own RAG+CoT), not a shared model with OOW. OOW keeps performing manoeuvres/navigation; the captain can overrule and helps with new waypoints/operational limits arising from brown envelopes.
- **Decided** (2026-09-30, this pass): folder structure (§6), Mission Order format (§8), route/waypoint interface (§7), and evaluation axes (§10) are proposed — not yet finalised/implemented.
- **Decided** (2026-09-30): mission scale — e.g. Rotterdam → Den Helder, with fictional exclusion zones to route around and a handful of waypoints; ~1 brown envelope per 100 nm of route (a 600 nm route ⇒ ~6 brown envelopes). See §7.
- **Decided** (2026-09-30): §10.2's composite weights — Mission outcome 25%, Safety 20%, Crew welfare 20%, remaining 5 axes 7% each. Starting point, expected to be retuned later.
- **Decided** (2026-09-30): route-planning direction — likely the PredictWind API later, or a manually downloaded real route; a fake random-waypoint route for now. See §7.
- Still to be determined: which exact ISM/SOLAS/STCW/MARPOL source text becomes the Captain's RAG corpus (COLREG-Consolidated-2018.pdf played that role for OOW) — no such file exists yet in `Data/Captain/`. **Research done (2026-09-30), VERIFIED (2026-09-30, this pass) via direct fetches, first real documents ACQUIRED via a new scraper (same pass)**: (a) IMO's own "Index of IMO Resolutions" page (imo.org) confirms full resolution text from 2000-present requires **IMODOCS registration** (free account, but not an anonymous bulk download) — pre-2000 resolutions (the ISM Code's own parent, **Resolution A.741(18)**, adopted 1993) are explicitly "available in print" via the Maritime Knowledge Centre, i.e. **NOT a simple free PDF fetch**, correcting the earlier assumption; (b) the free NATIONAL implementing-legislation route is now CONFIRMED with real, working, anonymous public APIs, not just "likely free": **US eCFR** (`ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml`, no auth) covers Coast-Guard/shipping regulation implementing SOLAS/STCW/MARPOL substance — its own real search API even surfaced the DIRECT cross-reference ("a safety management system fully compliant with the ISM Code requirements, implemented in **33 CFR part 96**"), a genuinely useful discovery, not a guess; **UK legislation.gov.uk** (content-negotiated XML at `/{type}/{year}/{number}/data.xml`, Open Government Licence v3.0) covers UK Merchant Shipping statutory instruments — the CURRENT ISM-Code-implementing SI (`uksi/2026/194`) and STCW-implementing SI (`uksi/2022/1342`) were both looked up via the site's own title-search redirect, never guessed; **EUR-Lex** publishes EU shipping directives/regulations freely (exact REST endpoint still to be confirmed, flagged not blocking); **BMP5** remains fully free/industry-published (the official maritimeglobalsecurity.org host 403s scripted requests, the Liberian ship registry's own mirror does not). **New script**: `pipeline/ingest/build_captain_legal_corpus.py` — raw acquisition only (no parsing/chunking yet, that's a deliberately separate later decision), already run end-to-end, **5 real files acquired**: `Data/Captain/Legal_Reference/{bmp5/BMP5.pdf, uk_legislation/{uk_stcw.xml,uk_ism.xml}, ecfr/{ecfr_ism_title46_part138.xml,ecfr_stcw_title46_part13.xml}}`, plus a `manifest.json` recording url/license/sha256/fetched_at per file for provenance. Recommendation UNCHANGED: mirror the existing `simple_colreg.json` pattern (§5.1/§6.3) — official/paid consolidated text where available, plus an own plain-language layer built from these now-ACQUIRED free sources as a copyright-safe supplement. **Next concrete step, not yet done**: also pull 33 CFR part 96 (the directly-referenced ISM Code implementation just discovered) and decide which parts of all these now-acquired documents actually enter RAG/training — a later, separate decision per this section's own recommendation, not made by the scraper itself.
- **Decided** (2026-09-30): Look-out/Helmsman/QMOW/Chief Engineer (§2) stay as structured facts fed into Captain/OOW prompts for now — not separate fine-tuned agents.
- **Decided** (2026-09-30): **model/family for the Captain — starts from the same base Qwen model as the OOW** (not a larger 32B variant or a frontier API model). Own separate QLoRA SFT/DPO(+Reflection) fine-tune on its own corpus (per the earlier-confirmed architecture), same base checkpoint as OOW's own fine-tune starts from. Revisiting a larger/different base model is not ruled out for later, but is not the starting point.
- **Decided** (2026-09-30): the "decision timeliness / calling discipline" and "legitimately abandoned goal" thresholds (§10.1) both start at **80%**, symmetrically — see §10.3 for the exact recall/precision and ground-truth-match formulas. A deliberately simple starting point, to be recalibrated once real Captain-agent runs exist (same as the §10.2 composite weights).

---

<a id="sec-13"></a>
## 13. Computational specification — the missing layer

**Honest assessment (2026-09-30)**: §1–§12 specify roles, doctrine, and formats — they do not yet specify what the code must actually *compute*. This section answers that, point by point, in the same order it was raised: blocking items first, then loose contracts, then data/evaluation. Nothing in this section changes §1–§12 — it makes them computable.

<a id="sec-13-a"></a>
### 13.A Blocking — nothing can run a mission without these

<a id="sec-13-a-1"></a>
#### 13.A.1 Two-timescale simulator

- **Mission-sim** (new): steps in **seconds** (`dt_mission_s`), advances position along the current leg (distance travelled = speed × dt), consumes fuel and time (§13.A.3), advances a simple time-indexed weather schedule, and accumulates rest-hours. This is where a 100–600 nm, multi-hour-to-multi-day mission actually lives.
- **Mission-sim parameters (fixing a real gap, 2026-09-30)**: `dt_mission_s`, referenced elsewhere ("within one mission-sim step", §13.B.7) but never actually fixed. **Decided default: 600 s (10 minutes)**, configurable per mission — mirrors the encounter-sim's own already-adjustable 1–60 s `dt` slider in `app/simulation.py`, one level up. Fine enough for a realistic Mandatory-call response deadline, coarse enough to keep a multi-day mission computationally tractable (a 600 nm/~50 h mission is only ~300 steps at this rate).
- **Wall clock (fixing a real gap, 2026-09-30)**: the Mission Order gains `t0_utc` (an absolute ISO-8601 departure timestamp, §8.2) — mission-sim time becomes `t0_utc + elapsed_s`, a real calendar/clock reference. This is what makes "night" (Night Orders, §3.2; Look-out staffing realism), a genuinely noon-anchored report (§8.3's `reporting_interval_hours` — a real noon report is anchored to actual local noon, not an arbitrary elapsed-time boundary), and human-readable MPR/log timestamps (§8.3/§13.C.12) actually meaningful, rather than a bare elapsed-seconds counter. The STCW rest-hour ledger (§13.A.3) itself stays a pure rolling elapsed-time window and does not need the wall clock to compute correctly — `t0_utc` is for everything that needs a real calendar/clock reference, not the rolling-window math.
- **Backward compatibility (explicitly required, 2026-09-30)**: the whole mission-sim layer is strictly **opt-in** — a mission with no `waypoints`/Mission Order metadata (today's existing single-leg missions: Imazu/UM/RND/IMP, the current "Nomoto"-tagged corpus) has no `dt_mission_s`/`t0_utc` at all and **bypasses the mission-sim entirely**, running directly on the unchanged encounter-sim exactly as it does today. Mirrors this project's own established pattern for exactly this kind of change (`Mission.waypoint: tuple[float, float] | None = None`, optional, defaulting to `None` so every existing mission/run is unaffected — see `basic_simulator.md`'s paused-waypoint-feature note). Zero new fields are added to the legacy `Mission` schema; the new fields only ever exist on mission-level scenario files (§13.C.13).
- **Encounter-sim** (existing, unchanged): `app/simulation.py`'s existing 10 s-step metric simulator, spliced in ONLY when the mission-sim detects a reason to (a scripted contact/hazard within lookahead range, or a scripted event's trigger point). Between encounters the OOW runs on autopilot toward the next waypoint — no LLM calls, no per-step decisions, exactly the "OOW runs autonomously on the Captain's plan between events" behaviour already established in §3.3.
- **Handoff**: at encounter entry, project the mission-sim's lat/lon position to a local flat-earth metric frame (equirectangular projection centred on the entry point — accurate enough over an encounter's bounded few-nm/few-minute window, NOT used for the whole voyage); run the existing OOW loop unchanged for the encounter's duration, with any active Captain instruction (speed cap, `avoid_zone`) passed in as a `VesselConstraints` override, same mechanism as today; project the encounter's final position/heading/speed back to lat/lon to resume mission-sim leg progression.
- **Concretising the encounter-sim splice (fixing a real gap, 2026-09-30)** — the paragraph above was still an intention, not a specification. Five concrete decisions close it:
  1. **Ambient-traffic generator**: `ambient.background_traffic_density` (§13.C.13) becomes a Poisson-process contact-arrival rate along the route (illustrative v1 defaults: light ≈ 0.1/h, moderate ≈ 0.3/h, dense ≈ 1.0/h — tunable, not fixed forever). Each contact's own geometry (bearing/range/course/speed at spawn) is generated with the SAME machinery already used for Imazu-style missions (`app.geometry.compute_target()` / `generate_random_imazu_missions.py`'s pattern) — reused unchanged, not reinvented, just called at the ship's live position/heading/speed at spawn time instead of at a fixed mission start. Overlapping encounters are supported by construction, not a special case: the mission-sim keeps a pool of "currently relevant contacts" and stays spliced into the encounter-sim as long as that pool is non-empty (exactly how today's multi-target missions like Imazu07/IMP05/IMP12 already work) — no separate one-at-a-time restriction.
  2. **Encounter window bounds**: **start** when any contact reaches TCPA ≤ 30 min OR range ≤ 6 nm (whichever first) — a deliberately earlier/more generous threshold than the encounter-sim's own internal quiet/risk classification (`QUIET_TCPA_S`/`QUIET_CPA_M` in `app/narrate.py`), so the OOW gets control well before real risk, and the existing internal COLREG logic still does the actual risk assessment once inside. **End** when BOTH already-computed facts hold for every tracked contact/leg: every contact is already "past and clear" (the existing `narrate.py`/`contact_line()` fact) AND the Goal Course Check (`goal_course_check_line()`) reports back on the planned track within its existing deadband — both facts already exist in the OOW pipeline today, so ending an encounter needs no new detection logic, only a composite condition over facts that already exist.
  3. **Mission clock during an encounter: continues, explicitly** — mission-sim time, fuel, and rest-hours (§13.A.3) never pause and are never double-counted; while spliced in, they simply advance at the encounter-sim's own finer step size (its existing `dt`, e.g. 10 s) instead of the coarser `dt_mission_s` (§13.A.1) — same formulas, finer granularity, then back to the coarse step once the encounter ends. Ship time never stops; only how often the loop "checks in" varies.
  4. **A brown envelope CAN fire mid-encounter — this is the interesting case, and it requires the trigger-checking loop (not just the Captain) to run at whichever granularity is currently active.** Since events (§13.A.2) trigger on mission-sim position/time and those keep advancing per point 3, an event's condition can become true while spliced into an encounter. This means the event-trigger and monitor checks (§13.B.7/§13.B.9) must run at the ENCOUNTER-SIM's own step rate while spliced in, not only at `dt_mission_s` boundaries — the one real architectural consequence of this gap. Once the Captain (or the walking skeleton's deterministic baseline) reacts — e.g. `slow_down` after a mid-crossing engine failure — the resulting `VesselConstraints` change is picked up by the OOW's very next decision inside the SAME, still-running encounter, exactly the same way a Captain instruction is already picked up between encounters (§3.3) — no new plumbing needed there, only the trigger-checking cadence fix. (Flagged as a natural, high-value future test scenario in its own right — engine failure mid-crossing is close to genuine in-extremis COLREG territory — but not built now.)
  5. **`avoid_zone` polygons inside the encounter frame**: at the SAME handoff step that projects own-ship/target positions into the local metric frame (point above), also project the currently-active exclusion zones relevant to that local area — both dynamic (weather/security/wildlife, §7) and the mission's static region layer (coastline/shoal, §7/§13.B.8) — into the identical local coordinates. `app/oracle_planner.py`'s existing hard rule-legality gate is extended to also exclude any candidate heading/action that would cross one of these projected polygons, exactly alongside its existing COLREG-illegality exclusion — the same hard-gate mechanism, a larger exclusion set, not a new one. This is also what makes a Captain's `avoid_zone` instruction (§7/§9.1) automatically effective at the tactical level with no separate instruction to the OOW: the projection step already carries the active zone set along every time it runs.
- **Leg** = the route segment between two consecutive waypoints, or between the current position and a Captain-ordered deviation point (a `reroute` ends the current leg early and starts a new one to the revised waypoint list).
- Coordinate system: WGS84 lat/lon for the route/Mission Order (matches §8.2's schema); local equirectangular metric ONLY inside an encounter window, never for the whole-mission distance/ETA math (that stays in great-circle/rhumb-line nm, standard navigation practice).

<a id="sec-13-a-2"></a>
#### 13.A.2 Event model — state-delta table, reduced to a v1 set of 5

§4's ~70 named events are a taxonomy, not yet a computable model. Per the recommended reduction (adopted, see §14), the following 5 are fully specified now; the rest of §4 remains a backlog until needed.

| Event | Physical/operational delta | Duration & recovery | Observability | Follow-on | World-responder |
|---|---|---|---|---|---|
| **Engine failure** (moderate) | `max_speed_kn` capped (e.g. 8 kn); optionally degraded Nomoto K/T if steering-dynamics fidelity is wanted (deferrable) | Until a scripted repair time, or a Chief Engineer "fixed" report | Immediate — Chief Engineer reports directly (§2.5), unprompted | May combine with fog/traffic to compound risk | None needed, unless a `request_place_of_refuge` is issued |
| **Fog / restricted visibility** | `visibility_m` drops below the Rule 19 safe-speed threshold → regime flag forces safe-speed + sound-signal requirement | Scripted weather window (start/end time) | Immediate (sensor fact) | Raises effective encounter frequency if combined with traffic density | None — self-contained COLREG regime change |
| **Distress call (SOLAS duty to assist)** | A new "target" contact appears requiring an assist/no-assist decision | Resolved when the world-responder fires (see next column) | Via VHF, immediate | May itself trigger a `reroute`/`declare_emergency` | **Scripted, timed reply required** — e.g. after N simulated minutes: "vessel under tow by X, assistance no longer required" or "survivors recovered" — a fixed resolution, not an open-ended negotiation |
| **Whale zone** | Adds an exclusion polygon + speed cap while inside it | While position is inside the polygon | Immediate (charted, known before entry) | None | None needed |
| **Commercial instruction vs. safety (ISM Art. 5)** | A scripted incoming company/DPA message demanding a schedule/route violation | Resolved once the Captain responds | Explicit message event | May escalate to `contact_dpa` | **Scripted, timed reply required** — a fixed company acknowledgement after the Captain's decision (accepts refusal, or escalates once) |

The **world-responder** column is the piece that was completely missing: for the 2 events that imply the outside world must reply (distress call, commercial instruction), it is a small deterministic script/timer table — never an open-ended negotiation — consistent with this project's "deterministic where possible" philosophy (§9.3).

**Correction (2026-09-30), addressing a real critique of the original single-answer procedure model**: the table above is not a fixed (event → one required action) mapping — see §13.A.4's revised 3-layer model. **Engine failure** and **Commercial instruction vs. safety** are only genuinely *table-determined* in their simplest parameterisation; both become real judgement cases once continuous parameters are added — Engine failure gains `distance_to_refuge_nm`/`deadline_slack_h` (continue at capped speed vs. put in to a port of refuge is then a real cost tradeoff, not a lookup — `distance_to_refuge_nm` is itself now a COMPUTED value, not a free parameter: the route planner's shortest path, §13.B.8, to the nearest port in the §7 `ports` library whose `services` cover the need and whose `min_approach_depth_m` clears `draft_m`); Commercial instruction's "breaches a safety margin" becomes a *computed* condition against the live Mission State rather than a fixed always-refuse rule. Fog and Whale zone remain genuinely table-determined (a COLREG-mandated safe speed and a charted/posted speed limit both leave negligible real tradeoff space) — Distress call was already a judgement case. This replaces the original "4 table-determined, 1 judgement case" split with a more accurate one: 2 event types are consistently table-determined, 3 are judgement-capable once properly parameterised — see §13.A.4.

<a id="sec-13-a-3"></a>
#### 13.A.3 Resource model

- **Fuel, exact calibration (fixing a real gap, 2026-09-30)**: `fuel_rate(t) = base_load + k * speed_kn(t)**3` (the real cube-law relationship between ship speed and propulsion power/fuel burn, plus a constant "hotel load" for auxiliary/generator consumption independent of propulsion). A single fuel-budget equation cannot determine two unknowns (`base_load`, `k`) at once, so the calibration is now a fixed two-step procedure, not an independent guess for each:
  1. Fix a **hotel-load fraction** `f_hotel` (default **0.10** — auxiliary/generator load as a fraction of the propulsive fuel rate at the planned cruise speed, a standard simplification): `base_load = f_hotel * k * SOA**3`.
  2. Solve the Mission Order's own fuel-budget equation for `k`, integrated over the **planned route at constant SOA** (as originally framed): `fuel_tonnes_at_departure * (1 - fuel_reserve_margin_pct/100) = (base_load + k * SOA**3) * T`, where `T = total_route_distance_nm / SOA` (in **hours** — nm ÷ kn is unavoidably hours; see §13.A.7). Substituting `base_load` from step 1 and solving: `k = fuel_budget / (SOA**3 * (1 + f_hotel) * T)`, then `base_load = f_hotel * k * SOA**3`.

  Both constants are now fully determined from the Mission Order's own stated numbers (fuel budget, reserve margin, route distance, SOA), recomputed once per mission at generation time (§13.C.9) — never independently guessed, and never re-derived mid-mission. `fuel_rate(t)` is therefore **tonnes/hour**; fuel consumed over one mission-sim step is `fuel_rate(t) * (dt_mission_s / 3600)`, never a raw multiplication by `dt_mission_s` (§13.A.7).
- **ETA**: `remaining_distance_nm / current_speed_kn`, recomputed every mission-sim step (reflects live speed changes from Captain/OOW decisions).
- **Rest-hours, concrete default roster + the Captain's own fatigue (fixing a real gap, 2026-09-30)**: the single aggregated "OOW on watch" ledger now has a concrete default source — a **4-hours-on / 8-hours-off** watch rotation (a standard 3-watch merchant system), advanced mechanically from `t0_utc` (§13.A.1). Under this roster alone, STCW's 10h/24h minimum is comfortably met (8h+8h off > 10h); real violations arise only from **disruptions** to the off-watch period (an emergency, a mandatory drill, doubling up in heavy weather/pilotage), which subtract from the next rest block. **The Captain is not exempt**: a separate `captain_rest_ledger` (same STCW-style rolling-window mechanics) starts fully rested and is decremented by every Mandatory/Emergency engagement (§3.1) that falls outside the Captain's own already-awake periods — being called repeatedly overnight is a real, tracked fatigue cost, not a free action. Both ledgers feed the Crew welfare axis (§10.1) directly, and either one breaching its STCW minimum can itself spawn a fatigue-error brown envelope (§4.D) — exactly the pattern real incident reports (CHIRP/MAIB, §5/§6) already document: a fatigued master, after repeated night calls, making a worse decision.
- **Ship profile speed consistency (fixing a real inconsistency, 2026-09-30)**: `pipeline/nomoto.py`'s existing `SHIP_PROFILES` (e.g. `sawada2021_default`, 12 kt design speed per the cited paper) need two additional fields so a Mission Order's `speed_of_advance_kn` and a ship profile are always mutually consistent: `nominal_speed_kn` (the design/cruise speed already implied by each profile's own Nomoto constants) and `max_speed_kn` (the absolute top speed the engine can produce, ≥ nominal — a real ship typically has a modest margin above design speed, not a large one). The scenario/mission generator (§13.C.9) must **validate `SOA ≤ max_speed_kn`** for whichever ship profile is sampled, rejecting/resampling any combination that violates it — a hard generation-time check, not a runtime judgement call. §8.2's worked example had exactly this inconsistency (`sawada2021_default` at 12 kt design speed, but `speed_of_advance_kn: 16.0`) — corrected to `12.0`, matching the profile it names.
- This is what makes "is the goal still achievable" (§9, §10.3) an actual computation rather than a judgement call.

<a id="sec-13-a-4"></a>
#### 13.A.4 Procedure library — shield, mandatory duties, and a cost-based decision layer (revised 2026-09-30)

**Agreeing with a real critique of the first version of this section**: defining `required_actions` as "the ONLY correct action(s)" and using it simultaneously as shield/label/auditor collapses the Captain into a lookup table — SFT would just learn the lookup, DPO would punish any deviation from it, and no evaluation could ever score a genuinely better decision higher than the table's own entry. That directly contradicts §9.3 ("where the LLM proves its value: weighing, reprioritising") and the "legitimately abandoned goal" logic in §10.1/§10.3. Fixed by splitting the table into three layers with genuinely different roles — not three synonyms for "the correct answer":

1. **Mandatory** — reporting duties and their deadlines (the original `reporting_duties`), independent of which candidate action is chosen. Checklist material, stays deterministic (§9.3's "known procedures" category). Can itself depend on the CHOSEN action (e.g. diverting to a port of refuge triggers its own extra reporting duty) — a function of `(event, chosen_action)`, not just `event`.
2. **Shield** — `forbidden_actions`, unchanged: a hard gate on the action space, validated before any instruction reaches the OOW, mirroring `app/oracle_planner.py`'s `required_direction()` hard rule-legality gate (§9.3). Absolute, never a matter of degree.
3. **Decision layer** (new, replaces the old single `required_actions`) — the only layer that gives the Captain something to actually decide, and the only layer worth training/evaluating judgement on:
   - `candidates(event, mission_state) -> list[action]` — a systematic candidate generator (mirrors `oracle_planner.py`'s own candidate generation: baseline first-step choices + a parameter sweep), **not** an exhaustive enumeration and **not** a hard restriction on what the Captain may propose — it exists to compute a reference, not to fence in the Captain's action space (only the shield does that).
   - `cost(action, mission_state) -> scalar` — fully specified in §13.A.8, not sketched here: five dimensions (life/ship-environment/legal-duty/mission-goal/commercial) combined via a lexicographic-by-construction weighting that mirrors the PRIORITY ORDER of §1.2.8, not merely the shape of `oracle_planner.py`'s `W_GOAL`/`W_CLEARANCE`/`W_EFFORT` pattern — giving `oracle_best = argmin cost over candidates`.
   - **Training/eval signal is regret, not exact match**: `regret = cost(chosen_action, mission_state) − cost(oracle_best, mission_state)`, with `cost()` recomputed for WHATEVER the Captain actually proposed (even an action outside the candidate generator's own list) — so a genuinely better, novel decision scores a lower regret than the oracle's own reference, never penalised just for differing textually. Low/zero regret is SFT-worthy regardless of whether it matches `oracle_best` verbatim; a real, unambiguous shield violation is the only thing that stays hard-gated (layer 2), not the decision layer.

This directly fixes the "there are always exceptions to the rule" problem raised: layers 1–2 stay genuinely rule-like (checklists, hard bans) where that's actually true of the real world, and layer 3 is deliberately open-ended (regret against a computed reference, not membership in a fixed set) precisely because real Captain judgement calls don't have one textually-fixed correct answer.

Proposed home unchanged: **`pipeline/captain_agent_spec.py`**, mirroring `pipeline/oow_agent_spec.py`'s `classify_rules()` role. Source unchanged: ISM/SOLAS/BMP5 + the project's own CHIRP set (§5/§6).

Revised concrete v1 entries:

- *Engine failure*: **mandatory** = fault log entry (+ a place-of-refuge report only if requested). **Shield** = never exceed the Chief-Engineer-declared safe speed. **Decision layer** = continue at the capped speed vs. divert to a nearby port of refuge — a real cost tradeoff once the event carries `distance_to_refuge_nm`/`deadline_slack_h` (§13.A.2, now computed against the §7 `ports` library, not a free parameter); with no time pressure and no refuge nearby (no port in range with matching `services`), `oracle_best` collapses to "continue at capped speed" and there is effectively nothing to weigh — this is what made the original v1 parameterisation look table-determined; it was an under-specified special case, not a property of engine failures in general.
- *Fog, regime change*: **mandatory** = none. **Shield** = never exceed the Rule-19 safe speed for conditions. **Decision layer** = degenerate — COLREG mandates the safe speed itself, no real second candidate worth weighing — genuinely table-determined, unlike engine failure.
- *Distress call*: **mandatory** = notify DPA + flag state (SOLAS reporting), deadline immediate. **Shield** = never ignore the call without logging a reason. **Decision layer** = assist vs. document a valid reason not to — cost trades off time/fuel/mission-goal delay against the legal/moral/reputational cost of not assisting; genuinely a judgement case, as originally identified.
- *Whale zone*: **mandatory** = none. **Shield** = never exceed the zone's posted speed limit inside the polygon. **Decision layer** = degenerate — a charted, fixed limit, no real second candidate — genuinely table-determined.
- *Commercial instruction vs. safety*: **mandatory** = log entry + DPA notification of the outcome. **Shield** = never comply with an instruction that breaches a safety margin (§13.A.6 quantifies exactly what that means). **Decision layer** = whether the instruction actually breaches a margin is now a COMPUTED condition against the live Mission State (fuel/rest-hour/COLREG margins) rather than a fixed always-refuse rule — itself a genuine judgement case once "breaches a margin" isn't hard-coded as always-true.

**Reassessed v1 split** (supersedes the original "4 table-determined, 1 judgement case" claim): Fog and Whale zone are consistently table-determined; Engine failure, Distress call, and Commercial instruction are judgement-capable once properly parameterised. §14's walking skeleton is updated accordingly.

<a id="sec-13-a-5"></a>
#### 13.A.5 Evidence, ambiguity, and reporter reliability

Real captains do exactly what was described: treat a single engine-room sensor alarm with suspicion (a faulty sensor is often more likely than a genuine failure), but take three independent sensors agreeing much more seriously; and an experienced captain also learns which crew members tend to over-report. **Confirming the read already given**: this is genuine judgement material, belongs entirely in the decision layer (§13.A.4), and is explicitly out of scope for the deterministic walking skeleton (§14) — but the architecture needs a few concrete additions so there is actually "room for it" once an LLM captain exists, rather than this staying a hopeful assertion.

1. **Events carry evidence, not just a fact.** §13.A.2's event model gains `evidence: list[{sensor_id, reading, reported_by}]` plus a HIDDEN ground-truth `is_false_alarm: bool` that the scenario file (§13.C.13) knows but never exposes directly to the Captain — only the raw evidence is observable, exactly like a real sensor-fault situation. Whether it's one corroborating reading or three independent ones then becomes a real input to the decision layer's `candidates()`/`cost()` functions (§13.A.4), not a separate mechanism bolted on the side.
2. **The decision layer gains an extra candidate: "treat as low-confidence, investigate/hold before acting."** This is exactly the ambiguous, weighing case §9.3 already earmarked as the LLM's real value — no new evaluation axis is needed: correctly discounting a single-sensor alarm (or correctly escalating on 3-sensor corroboration) is scored by the SAME regret mechanism (§13.A.4), since the cost function knows the true (hidden) state — an LLM that reacts proportionately to weak vs. strong evidence gets low regret; one that either cries wolf on noise or ignores a real multi-sensor failure gets high regret. This generalises cleanly rather than needing a bespoke alarm-discrimination score.
3. **Reports need a `reported_by` field.** The subordinate facts containers already proposed (§15.2 — `EngineStatus`, `LookoutReport`) gain a `reported_by: crew_member_id` field; the Mission State's append-only event/decision log (§13.B.5) already records everything with a timestamp, so a per-mission "track record" (this crew member's last 3 reports: 2 confirmed false, 1 confirmed real) is available to the Captain for free, just by rendering that slice of the log into the facts-only prompt (§13.B.5) — no new mechanism, just a new fact to surface.
4. **Two distinct kinds of "learning", worth keeping separate**:
   - **In-mission**: the Captain reads a specific crew member's own track record so far THIS mission (item 3) — pure in-context reasoning over data already logged, works from day one once the field exists.
   - **Cross-mission learned bias** ("this reporter tends to be trigger-happy" as a general pattern baked into training) — a genuine, bigger design choice with two options: (a) NAMED individuals with a persistent trait across the whole training corpus (higher fidelity, but risks the model learning a specific name → trait association that doesn't generalise to a real, unnamed crew), or (b) a per-mission SAMPLED "reporter reliability profile" (e.g. "tonight's lookout: elevated false-positive rate this mission"), freshly drawn per mission — mirrors the already-proven `ship_profile` sampling pattern from the Nomoto work (`pipeline/nomoto.py`'s `SHIP_PROFILES`/`sample_ship_profile()`) exactly, just for reporter reliability instead of ship dynamics. **Recommended for v1: (b)** — same reasoning as the ship-profile precedent: it teaches the general skill (weigh evidence on its merits, don't blindly trust or distrust a role) rather than memorising a specific name, and comes with a ready-made held-out-profile pattern for genuine generalisation testing.
5. **Scenario generator** (§13.C.9) gains a new sampled dimension: whether a given event instance is a true event or a false alarm, how many/which sensors corroborate it, and (per item 4) an optional sampled reliability profile for whoever reports it.

This gives the walking skeleton (§14) a natural second test scenario once it graduates beyond the single parameterised engine-failure case: the same event, but with a genuinely ambiguous 1-sensor vs. 3-sensor evidence pattern, directly exercises the regret mechanism §13.A.4 was built for.

<a id="sec-13-a-6"></a>
#### 13.A.6 Quantifying "safety margin", and the missing "complies anyway" outcome

**Without this, "forbidden = complying with an instruction that breaches a safety margin" (§13.A.4) is not executable** — a shield rule needs a concrete function to test, not a phrase. Fixed with `check_safety_margins(candidate_action, mission_state) -> list[MarginViolation]` (proposed home: `pipeline/captain_agent_spec.py`, alongside the procedure library, §13.A.4) — deliberately **general-purpose**, usable by the shield for ANY Captain decision, not hardcoded to the commercial-instruction event alone (the same 4 checks apply just as well to, say, a Captain-issued `speed_up` that happens to breach the fuel reserve regardless of what triggered it):

| Margin | Check |
|---|---|
| Engine speed cap | `candidate_action.speed_kn > engine_status.max_speed_kn` (the Chief Engineer's declared cap, §2.5/§13.A.5's `EngineStatus`) |
| Exclusion zone | the candidate's implied route/position enters an active zone (dynamic or static, §7/§13.B.8) it isn't authorised to be in |
| Rest-hours | complying would push the OOW watch ledger OR the `captain_rest_ledger` (§13.A.3) below its STCW minimum |
| Fuel reserve | the recomputed fuel consumption (§13.A.3) at the candidate's implied speed/route breaches the Mission Order's `fuel_reserve_margin_pct` |

The function returns WHICH margins are breached, not just a bare boolean — needed both as an observable fact for the facts-only prompt (§13.B.5, e.g. "this instruction would exceed the declared engine speed cap of 8 kn") and so §13.C.11's regulatory-compliance/citation-accuracy ground truth can check the Captain's own reasoning against the actual breached margin, not just a pass/fail.

**The missing outcome branch, now defined**: §13.B.6's precedence rule (**Shield > active Captain instruction**) already says a shield-violating OOW-facing instruction is rejected/clamped before reaching the OOW — this now explicitly extends to the Captain's own top-level decision too, not only downstream OOW instructions. Two cases:

- **Normal simulation (shield active, the default)**: if the Captain's own proposed decision is "comply" and `check_safety_margins()` finds a real breach, the simulation forcibly substitutes the procedure library's **mandatory** action (refuse/defer, citing ISM Art. 5) before it is ever applied to the Mission State — the ship never actually experiences the bad outcome. The Captain's *original, un-clamped* proposal is still recorded, though: it becomes exactly a category-(a) shield-violation DPO-rejected sample (§13.C.10, already defined), scored against the Procedure/regime selection accuracy axis (§10.1) as a real, negative decision-quality event — a correctly-caught bad decision is not free, it just doesn't sink the ship.
- **Only if run WITHOUT shield enforcement** (a deliberate ablation, or to generate a genuine negative/failure training example on purpose): the compliance is actually applied, and the consequence is a probabilistic **follow-on brown envelope keyed to which margin was breached** — no new mechanism needed, each margin already maps onto an existing category: a speed-cap breach risks a secondary engine failure (§13.A.2's Engine failure, as a follow-on); a zone breach risks that zone's own inherent hazard (grounding on a shoal, a wind-farm allision, a piracy encounter, a whale strike — whichever `type` the zone carries, §7); a rest-hours breach risks a fatigue-error brown envelope (§4.D, already linked in §13.A.3); a fuel-reserve breach risks a new fuel-shortage event (§4.C's backlog category). If the resulting consequence is severe enough, this is exactly what the real Safety hard gate (§10.1) is for — consistent with, not a special case of, the existing gate.

<a id="sec-13-a-7"></a>
#### 13.A.7 Units and time convention (fixing a real inconsistency, 2026-09-30)

**Rule**: any field driving the mission-sim/encounter-sim clock, an event trigger, a deadline, or a monitor threshold is in **seconds**, suffixed `_s` (`dt_mission_s`, `t_s`, `elapsed_time_s`, `timeout_s`, `delay_s`, the encounter-sim's own `dt`) — matching `app/simulation.py`'s existing convention, already in use project-wide. Speeds are always **kn**, distances/ranges always **NM**, fuel always **tonnes**. Text that previously said the mission-sim "steps in minutes" (§13.A.1) or the walking-skeleton's `step_mission(n)` "steps (minutes)" (§15.3) was a documentation inconsistency, not a genuine second unit — both now read seconds; "10 minutes" only ever appears as a human-readable gloss on `dt_mission_s=600`, never as a stored/computed unit.

Two **named, deliberate** exceptions, not further inconsistencies — each matches how the real regulation/industry already states that specific quantity, and is converted to/from seconds only at the point it needs to interact with the second-based clock:

- **STCW rest-hours** (`hours_awake`, `hours_rested_last_24h`, the 10h/24h and 77h/7d minimums, §13.A.3) stay in **hours** — exactly how STCW A-VIII/1 itself is always quoted, and how §13.C.13's `rest_hours_ledger` is already written. Checked against the mission-sim's elapsed seconds only via an explicit `/3600` conversion at the comparison point, never mixed in raw form.
- **Fuel rate and transit time** (§13.A.3): `T = total_route_distance_nm / SOA` is unavoidably in **hours** (nm ÷ kn = hours, standard nautical arithmetic), so `fuel_rate(t)` is **tonnes/hour** (also the real-world convention for reporting bunker consumption). Deducting fuel over one mission-sim step is `fuel_rate(t) * (dt_mission_s / 3600)`, not a raw multiplication by `dt_mission_s`.
- Mission Order fields that describe a **human-facing policy** the way the actual order/regulation would state it — `reporting_interval_hours`, `deviation_report_threshold: "position >20nm or ETA >6h off PIM"` (§8.2) — also stay in hours for the same reason; read by the reporting logic and converted to seconds internally exactly like the rest-hours ledger, never left ambiguous about which representation is authoritative (seconds always is, for anything computed).

`track_deviation_above(20_nm, within_h=6)` (§13.B.7) was the one genuine **code-level** inconsistency (a monitor function signature, not human-facing policy text) — renamed to `track_deviation_above(20_nm, within_s=21600)` to match the primary rule.

<a id="sec-13-a-8"></a>
#### 13.A.8 The `cost()` function — fully specified (fixing a real gap, 2026-09-30)

**Everything in the decision layer hangs off `cost()`** — `oracle_best`, regret, the SFT/DPO split (§13.C.10), the legitimately-abandoned-goal check (§10.3), and the deterministic baseline captain (§14) — yet it was only ever sketched as "resource_cost, risk_cost, goal_cost, weighted like `W_GOAL`/`W_CLEARANCE`". Specified now in three parts.

**a) Dimensions, units, and lexicographic ordering.** Each dimension maps onto exactly one tier of §1.2.8's existing priority ladder — that ladder was always the intended ordering, it was just never turned into arithmetic:

| Tier (§1.2.8) | Dimension | Unit / definition | Weight (illustrative) |
|---|---|---|---|
| 1. Life | `life_risk_cost` | expected life-loss units: Σ over hazards `P(outcome \| action) × severity_weight` (fatality = 1.0, serious injury = 0.1, minor injury = 0.01) — probabilities calibrated against the CHIRP/MAIB incident base rates (§5/§6), the same source already invoked for "a risk proxy grounded in the source incident base rates" | 1e8 |
| 2. Ship/environment | `ship_env_risk_cost` | expected hull/environment-loss units, same `P × severity` construction (total loss = 1.0, major damage/pollution = 0.3, minor damage = 0.05) | 1e6 |
| 3. Legal/COLREG obligation | `duty_cost` | count of §13.A.4 layer-1 mandatory reporting duties the rollout shows as missed/late, plus any distress-assistance obligation still unresolved at horizon end | 1e4 |
| 4. Mission goal | `goal_cost` | `1 − (fraction of Mission Order success_criteria, §8.2, still achievable at rollout horizon)` — **reuses §10.3's own "recompute the true resource margin" ground-truth function directly, not a new formula** | 1e2 |
| 5. Commercial | `commercial_cost` | fuel (t) + schedule delay (h) beyond §13.B.8's unconstrained minimum-resource reference, converted to one illustrative $ figure via a fixed bunker-price/day-rate constant — a pure tie-breaker | 1 |

`cost(action, mission_state)` is the single weighted sum of these five (a scalar is required since `oracle_best = argmin cost` and `regret = cost(chosen) − cost(oracle_best)`, §13.A.4, are both scalar operations) — but the **100x-per-tier gap**, combined with every dimension above being bounded to roughly `[0, 1]` by construction (the one unbounded dimension, `commercial_cost`, is deliberately given the smallest weight so it can never dominate), means no realistic amount of tier-N savings can ever change a ranking already decided at tier-(N-1) — a computable lexicographic ordering, not a soft trade-off. This directly answers "the oracle could ignore a distress call because it's cheaper": ignoring it costs at least one `duty_cost` point (tier 3, weight 1e4, or worse, `life_risk_cost` at tier 1) — no fuel saving (`commercial_cost`, weight 1) can ever close that gap. Weights are illustrative defaults, tunable like `f_hotel`/ambient-traffic rates elsewhere in §13 — the order-of-magnitude gap itself is the actual design decision, not the exact numbers.

**b) The mission-level rollout that produces each dimension.** `rollout(action, mission_state, horizon_h) -> RolloutResult` is the Captain's own mission-scale lookahead — distinct from, and never nested inside, `oracle_planner.py`'s existing tactical `_rollout_cost` (OOW's own 6-step/10s-per-step COLREG rollout, §9.3): the two run at different timescales and answer different questions (tactical heading choice vs. mission-level resource/risk trade-off), so nesting one inside the other would re-run an expensive geometry-level search inside every candidate's mission-level evaluation for no benefit — the Captain needs aggregate effects, not tactical COLREG geometry.

- **Horizon**: `min(time_to_mission_end, N_h)`, illustrative default **`N_h = 48h`** — long enough to span two full STCW rest-hour cycles (so a rest-hours-depleting decision's recovery, or failure to recover, is visible within the horizon) and to reach most missions' next major decision point; capped so the rollout isn't dominated by speculative, far-future, unrelated events.
- **What IS simulated**, all at the mission-sim's own `dt_mission_s` cadence (§13.A.1, §13.A.7) and never finer: route/position progression along the current plan-of-intended-movement, fuel consumption (`fuel_rate`, §13.A.3), the rest-hours ledger(s) (§13.A.3), passage through already-known exclusion zones (§13.B.8), and — for events ALREADY active in the current Mission State — their already-scripted world-responder timer/outcome (§13.A.2), which needs no new random draw since it is deterministic by construction.
- **What is NOT simulated**: (i) the encounter-sim itself — no per-target COLREG tactics are re-run; ordinary ambient traffic's effect is folded in as a single **expected-delay term**, a constant/lookup calibrated once from historical encounter-sim run logs per `ambient.background_traffic_density` category (§13.A.1's light/moderate/dense Poisson rates), an expected value rather than a fresh simulation; (ii) any brown envelope that has not yet triggered in the real Mission State — the rollout only plays out ALREADY-active events plus their scripted resolutions, never speculating about hypothetical future events, since the design has no event-probability model to draw on for that (a stated, deliberate limitation, not an oversight).
- **Determinism**: the rollout is a pure function of `(action, mission_state)` — no fresh random draws. It only ever replays the CURRENTLY-active events' already-fixed scripted outcomes (originally determined by the mission's own `(mission_seed, responder_seed)`, §13.C.9); the ambient-traffic expected-delay term is a calibrated constant, never sampled. This determinism is required, not incidental — `regret = cost(chosen) − cost(oracle_best)` must return the same number every time it is recomputed, or the SFT/DPO split (§13.C.10) becomes non-reproducible. Per §13.A.4's own promise, `rollout()`/`cost()` work identically for a `candidates()` member or a genuinely novel Captain proposal, since both only ever need `(action, mission_state)` as input, never membership in a precomputed set — this is the Captain's counterpart to the Nomoto-lookahead `oracle_planner.py` already has for OOW.

**c) The regret threshold, in `cost()`'s own unit.** The 100x-per-tier gap from (a) makes a single flat threshold, expressed at the **commercial tier's own scale**, automatically tier-aware with no extra machinery needed: `regret_threshold = 0.05 × commercial_cost`'s typical spread across candidates for the same event (illustrative default, same recalibrate-later status as the §10.2/§10.3 starting values) — i.e. "differs from `oracle_best` by no more than a normal amount of commercial-tier variation, and identical on tiers 1–4". Any real tier 1–4 difference, even the smallest nonzero one, produces a regret at least 100x this threshold by construction, so it is automatically classified high-regret/DPO-rejected without a separate per-tier check. Below the threshold → SFT-positive (§13.C.10's near-zero-regret rule); at or above it → the DPO-rejected member of a pair (§13.C.10).

<a id="sec-13-a-9"></a>
#### 13.A.9 One discrete-event scheduler, not two loops with an if-statement (fixing a real gap, 2026-09-30)

**The design has quietly accumulated: two timescales (§13.A.1), distance/time/position-based event triggers (§13.A.2), world-responder timers (§13.A.2), continuous monitors (§13.B.7), deadlines per urgency class (§13.B.7), report cadence (§8.3), encounter start/end (§13.A.1), and events allowed to fire mid-encounter (§13.A.1) — described so far as separate mechanisms, when they are really one priority queue of timestamped events.** This determines whether §13.C.12's determinism guarantee actually holds, and is hard to change once code exists, so it is specified now, not deferred.

**a) Event-type alphabet** (a closed set — every queue entry is one of exactly these 7, each carrying a fixed **tie-break rank**):

| Rank | Event type | What creates it |
|---|---|---|
| 1 (highest) | `EMERGENCY_TRIGGER` | A `MONITOR_TICK` detecting an Emergency-class condition (§3.1) |
| 2 | `MANDATORY_TRIGGER` | A `MONITOR_TICK` detecting a Mandatory-class condition |
| 3 | `WORLD_RESPONDER_TIMER` | Scheduled at event-creation time, `t_now + delay_s` (§13.A.2/§13.C.13) |
| 4 | `ENCOUNTER_BOUNDARY` | A `MONITOR_TICK` detecting the start (TCPA/range threshold) or end (past-and-clear + on-track, §13.A.1) condition |
| 5 | `WAYPOINT_ARRIVAL` | A provisional ETA-based estimate (§13.A.3), revalidated whenever a speed-affecting decision invalidates it |
| 6 | `ROUTINE_REPORT` | Computed in closed form up front for the whole mission from `t0_utc + n × reporting_interval_hours × 3600` (§8.2/§8.3) — clock-based, never revalidated |
| 7 (lowest) | `MONITOR_TICK` | Itself recurring, fires every `dt_mission_s`, or the encounter-sim's own finer `dt` while spliced in (§13.A.1 point 4) |

Every queue entry is the tuple `(timestamp_s, tie_break_rank, sequence_no, event)`, a min-heap ordered first by `timestamp_s`, then `tie_break_rank`, then `sequence_no` (a monotonically increasing counter assigned at ENQUEUE time) — a full total order, so two entries are never ambiguous regardless of how they were produced.

**b) Two categories of scheduling** (the distinction that makes this tractable): **fixed-time** events have a knowable future timestamp the moment they're created and are inserted into the queue ONCE — `WORLD_RESPONDER_TIMER`, `ROUTINE_REPORT`, and any scripted event's own `elapsed_time_s` trigger (§13.C.13). **Condition** events have no closed-form future timestamp (it depends on decisions not yet made) and are instead detected by the recurring `MONITOR_TICK` re-evaluating the live Mission State — continuous monitors (§13.B.7), distance/position-based brown-envelope triggers, and `WAYPOINT_ARRIVAL` (a provisional estimate that must be revalidated, not trusted until it reaches the front of the queue). **`MONITOR_TICK` is not a separate loop layered on top of the mission-sim step — it *is* the mission-sim step** (position/fuel/rest-hours integration, §13.A.1/§13.A.3), and its side effect of possibly enqueueing `EMERGENCY_TRIGGER`/`MANDATORY_TRIGGER`/`ENCOUNTER_BOUNDARY` entries at the SAME timestamp is what replaces the "two loops with an if-statement" shape with one queue.

**c) Tie-break order, exactly**: `EMERGENCY_TRIGGER` > `MANDATORY_TRIGGER` > `WORLD_RESPONDER_TIMER` > `ENCOUNTER_BOUNDARY` > `WAYPOINT_ARRIVAL` > `ROUTINE_REPORT` > `MONITOR_TICK` (lowest, by design): an independently fixed-time event landing at the exact same timestamp as a tick (e.g. a scripted world-responder timer exactly coinciding with a scheduled MPR boundary) is resolved BEFORE the tick "closes out" that timestamp; and anything a tick itself spawns gets the SAME timestamp but a rank strictly above `MONITOR_TICK`, so it is processed before the clock advances to the next tick. Monitors themselves are iterated in the FIXED table order already given in §13.B.7 (visibility → position doubt → track deviation → alarm → machinery fault → distress signal → company instruction), so their own enqueue order — and hence `sequence_no` — is itself fixed, never iteration-order-dependent.

**d) The determinism guarantee this closes**: §13.C.12's `(mission_seed, responder_seed, captain_prompt_hash, oow_prompt_hash)` replay tuple is necessary but not sufficient without a specified PROCESSING order — without one, two "identical-seed" runs could still diverge in which same-timestamp event fires first purely from language/implementation nondeterminism (dict/set iteration order), even though nothing in this project intentionally randomises it. The scheduler's total order is what makes replay actually deterministic, not merely seeded — a standard discrete-event-simulation pattern (priority queue + a fully specified tie-break rule), not a novel mechanism.

<a id="sec-13-b"></a>
### 13.B Loose contracts

<a id="sec-13-b-5"></a>
#### 13.B.5 Mission State — formal schema + the facts-only decision

- Schema split: **immutable** (copied once from the Mission Order at mission start: `mission_id`, `success_criteria`, `restricted_zones`, initial resources) vs. **mutable** (everything §8.4 already lists — goals-with-status, current resources, active hazards, active plan, event/decision log).
- Delta representation: a generic patch entry `{"t": <mission-sim time>, "field_path": "...", "old": ..., "new": ..., "cause": <event_id or decision_id>}` — append-only, so the Mission State at any past time is reconstructable by replaying deltas up to that point (same "log, don't overwrite" principle as this project's existing run-checkpoint files).
- **Decided (2026-09-30, corrected same day)**: the Captain follows the **same facts-only prompt convention already adopted for OOW** (the project's own 2026-09-24 architecture pivot, see `basic_simulator.md`) — Mission State renders to the Captain's prompt as plain facts (fuel remaining, elapsed rest-hours, current regime, distance to next waypoint, active exclusion zones) **and nothing else**. **Correction**: an earlier version of this bullet said the §1.2 priority ladder / "safety beats schedule" lives "once in the system prompt" — that is itself decision logic, not a fact, and is exactly the kind of hand-written imperative the OOW's own facts-only pivot removed (see `basic_simulator.md`'s "Judgment/timing nuance is now deliberately left to (1) RAG/PG-retrieved real COLREG text, (2) future SFT/DPO/Reflection training data — NOT hand-written prompt rules"). Fixed: the priority ladder does **not** appear anywhere in the Captain's prompt, system prompt included. §1.2 remains a description of the real-world role (documentation, for humans designing this system), not literal prompt text. The system is instead expected to acquire and apply that prioritisation via the same three channels already established for OOW — **RAG** (retrieving the actual ISM Art. 5 / SOLAS text when a commercial-vs-safety brown envelope is active, §6.3), **KG/PG** (the procedural graph structurally encoding "safety precedes schedule" as a traversal property of the mined procedures, §6.3), and **the trained weights** (SFT/DPO/Reflection on scenarios that demonstrate correct prioritisation, §13.C.10) — never a restated rule in the prompt.

<a id="sec-13-b-6"></a>
#### 13.B.6 Captain output schema + precedence rules

- Output schema: `{"tool": <name>, "params": {...}, "plan": {"updated_goals": [...], "resource_note": <str>}, "reasoning": <str>}`.
- Invalid response handling: mirrors OOW's `validate_action_json()` pattern — schema-check, and on failure fall back to a safe no-op (`hold`) + flag as a parse error; never silently apply a malformed instruction.
- **Precedence, explicitly decided (2026-09-30)**: **Shield (§13.A.4 `forbidden_actions`) > active Captain instruction > OOW's own plan/goal-check > commercial preference.** A Captain instruction that violates the shield is rejected/clamped before it ever reaches the OOW — mirrors `oracle_planner.py`'s hard-gate-before-cost-ranking pattern (§9.3).
- Simultaneous instructions: **the most recently issued Captain instruction supersedes any earlier unresolved one** — a stack of at most one active instruction per scope (route / speed / regime), not a merge — simplest possible rule, revisit only if it proves inadequate.
- "Resolved" is machine-checkable by construction: every instruction carries an explicit resolution predicate, e.g. `{"type": "exit_polygon", "zone_id": "tss_1"}` or `{"type": "elapsed_minutes", "value": 120}`, evaluated every mission-sim step — never bare free text like "until clear of the TSS" left to interpretation.

<a id="sec-13-b-7"></a>
#### 13.B.7 Trigger monitors

Each §3.1 checklist bullet becomes a named, thresholded function over the Mission State:

| Monitor | Signature (illustrative) |
|---|---|
| Visibility | `visibility_below(threshold_m)` |
| Position doubt | `position_uncertainty_above(threshold_m)` |
| Track deviation | `track_deviation_above(20_nm, within_s=21600)` |
| Unresolved alarm | `alarm_unresolved(timeout_s)` — an "alarm" is defined, self-referentially but concretely, as any OTHER monitor whose own condition has been true for longer than `timeout_s` without an OOW/Captain action addressing it |
| Machinery fault | `machinery_fault_reported()` — **discrete**, true only when the Chief Engineer's fact (§2.5/§15.2 `EngineStatus`) reports a fault; carries the §13.A.5 `evidence` list, but the monitor itself only asks "is there ≥ 1 corroborating reading", not "is it real" (that judgement is the decision layer's job, §13.A.4/§13.A.5, not this monitor's) |
| Distress signal | `distress_signal_received()` — **discrete**, true only when a distress-call event has fired |
| Company instruction | `company_instruction_received()` — **discrete**, true only when a commercial-vs-safety event has fired |

**Continuous vs. discrete monitors (clarifying a real ambiguity, 2026-09-30)**: the first four monitors are **continuous** — they evaluate a live Mission State value against a threshold, and can legitimately become true from *ambient* causes (§13.C.13's `weather_schedule`, ordinary track-keeping) as well as from a scripted brown envelope. That is correct, not a bug: a real ambient fog bank crossing the visibility threshold SHOULD engage the Captain, and doing so is a true positive, never "crying wolf", regardless of whether a scripted event caused it. The last three are **discrete** — they can only become true because a specific event type fired, by construction; "a discrete monitor fires without its event" is therefore impossible by design, not just unlikely.

- **Concurrent triggers**: the highest-severity unresolved trigger is answered first; all are logged regardless; the Captain's response deadline is keyed to the single most urgent unresolved trigger.
- **Deadlines per urgency class** (this is what makes §10.3's recall/precision computable): Emergency = same decision epoch (zero elapsed sim time); Mandatory = within one mission-sim step (`dt_mission_s`, default 600 s / 10 min); Routine = by the next scheduled report cycle.

<a id="sec-13-b-8"></a>
#### 13.B.8 Route planner with exclusion zones

- **v1 algorithm (decided)**: a **visibility graph** over exclusion-zone polygon vertices (nodes = start, goal, every polygon corner; an edge exists where the straight segment crosses no zone; shortest path via Dijkstra/A\*) — geometrically exact for polygon obstacles at this abstraction level, and simpler than a grid-based A\* (which forces a resolution trade-off and produces jagged paths). This is what makes `reroute`/`avoid_zone` (§7/§9.1) actually do something.
- **Obstacle set (revised 2026-09-30)** = the union of (a) the mission's own dynamic exclusion zones (weather/security/wildlife, §7) and (b) the static geographic layer for the mission's `region_id` (§7/§13.C.13) — coastline always included, shallow-water zones included only when `min_depth_m < draft_m` + safety margin for the current ship profile. Same single algorithm over a larger, mixed-source obstacle set — no separate mechanism needed for "real" vs. "fictional" obstacles.
- **Distance-to-port-of-refuge (fixing a real gap, 2026-09-30)** — reuses the SAME visibility-graph shortest path, just with the goal node set to a candidate port position instead of the next waypoint: `distance_to_refuge_nm(mission_state, required_services)` filters the §7 `ports` library to those whose `services` cover the current need (e.g. `repair` for an engine failure) and whose `min_approach_depth_m` clears `draft_m`, then returns the shortest-path distance (around coastline/exclusion zones, never a straight-line cut through land) to the nearest one — this is what makes §13.A.2/§13.A.4's `distance_to_refuge_nm` a computed value rather than scenario flavour text, and feeds directly into §13.A.8's rollout when a candidate action is "divert to port of refuge".
- **Feasibility check**: recompute ETA/fuel (§13.A.3) along any candidate new route; reject/flag it if it breaches the Mission Order's resource margins. Running this planner ONCE, unconstrained, at mission start also produces the "feasibility oracle" baseline §13.C.11 needs for the resource-efficiency axis.

<a id="sec-13-b-9"></a>
#### 13.B.9 Event → monitor mapping (closing the two-ground-truths gap)

**A real gap, now fixed**: scripted events (§13.A.2) and trigger monitors (§13.B.7) are two independent routes to engaging the Captain, but §10.3's recall/precision is computed against monitors only. Without an explicit mapping, an event that happens to trip no monitor is invisible to the metric (an unrecorded miss), and a continuous monitor firing for an ambient reason could be wrongly read as "crying wolf" even when it was correct. Fixed by requiring **every** event type to name exactly which monitor(s) it drives and at what urgency — no event may be added to §4/§13.A.2 without this mapping being extended alongside it, permanently, not just for the v1 set:

| Event type | Monitor(s) it drives | Urgency class |
|---|---|---|
| Engine failure | `machinery_fault_reported()` | Mandatory |
| Fog / restricted visibility | `visibility_below(threshold_m)` | Mandatory |
| Distress call | `distress_signal_received()` | Mandatory |
| Whale zone | *(none — see below)* | **Pre-authorised, not a live monitor at all** |
| Commercial instruction vs. safety | `company_instruction_received()` | Mandatory |

**Whale zone resolved**: neither Routine nor Mandatory — a charted, known-before-departure, fixed-response hazard (§13.A.4's decision layer already called this "degenerate", no real judgement) is exactly what a **Standing Order** (§3.2) is for: pre-authorise the speed reduction once, at mission start, from the Mission Order's own `known_hazards` (§8.2) — mirrors §3.2's own worked example ("reduce to half speed automatically in visibility under 2nm") almost verbatim. The OOW then handles every entry into the zone autonomously under that standing order; the Captain is never re-engaged per instance, and this event is correctly **excluded** from the §10.3 recall/precision metric entirely, rather than forced into an ill-fitting urgency class.

**Two separate questions, two separate metrics** (worth stating explicitly, since §13.A.5's sensor-evidence case makes it easy to conflate them): "was the Captain engaged in time" is §10.3's recall/precision, computed purely from monitor state; "did the Captain correctly judge the engaged situation" (e.g. discount a single uncorroborated sensor reading vs. escalate on three) is §13.A.4/§13.A.5's regret, computed from the decision layer. A monitor firing only ever asks the first question — it never pre-judges whether the underlying alarm turns out to be real.

<a id="sec-13-c"></a>
### 13.C Data & evaluation

<a id="sec-13-c-9"></a>
#### 13.C.9 Scenario-generator specification

- Category/severity distribution: reuse the existing weighted-category pattern from `generate_random_imazu_missions.py` — §4's category letters as the weighted category set, minor/moderate/serious/catastrophic as a secondary weighted draw.
- Dependency rules: a small explicit compatibility table (e.g. fog + engine failure allowed to co-occur; piracy excluded on a North-Sea-tagged route) — a geography/category exclusion list, not a full plausibility model.
- **Held-out split, revised (2026-09-30)** — the original "no event-*combination* shared" rule is not achievable with only 5 v1 event types (§13.A.2): the combinatorial space (≤5 event types × 4 severities × a handful of legal co-occurrence pairs from this same dependency table) is too small to guarantee non-overlap without either an unusably tiny eval set or silently reusing combinations. Fixed with a **composite held-out key** across 4 dimensions instead of 1: `(route_template_id, event_set, trigger_placement_bucket, severity_set)`.
  - **Whole route templates held out**: a fixed subset of route templates (e.g. 2 of N) is reserved for eval ONLY — never appears in any training mission regardless of which events/severities are attached — the strongest single guarantee, and a direct extension of the existing `HELD_OUT_EVAL_PROFILE`/Imazu22 pattern (whole-template holdout, not per-field holdout).
  - **Bucketed trigger placement**: `trigger_placement_bucket` discretises the trigger's distance/time-along-route into a small number of coarse legs (e.g. early/mid/late), rather than the raw continuous value — bucketing is what makes "no combination shared" meaningful at all (two continuous draws are already never bit-identical, so an unbucketed rule would trivially always pass without testing anything real).
  - For missions built on the REMAINING (non-reserved) route templates, the full 4-part key must still differ between any training and any eval mission — a genuine generalisation test across geography, which events, roughly when, and how severe, not just "which events fired".
- Seeds: one seed per mission index (matches `generate_random_imazu_missions.py`'s own convention), plus a **separate** seed for the world-responder's own scripted timing/outcome draws (§13.A.2) — a mission is fully reproducible from `(mission_seed, responder_seed)`.

<a id="sec-13-c-10"></a>
#### 13.C.10 Training-row format

- Input: last-N-events window + the current Mission State snapshot (facts-only, §13.B.5) + retrieved RAG/KG/PG chunks (§6.3) — bounded, not the full mission history, mirroring OOW's own bounded situation-report design.
- Labels, revised (2026-09-30) for the 3-layer model (§13.A.4): for mandatory/shield-only events (fog, whale zone), the deterministic action is a direct SFT label as before; for decision-layer events (engine failure, distress call, commercial instruction), the SFT label is `oracle_best` (§13.A.4) but ONLY when the Captain's actual choice has near-zero regret against it — otherwise the row is better mined as a DPO/reflection pair (chosen vs. `oracle_best`) than force-fit as an SFT positive. The Captain's own free-text `plan`/`reasoning` is CoT training text regardless; the Mission Progress Report (§8.3) is a separate, simpler templated-output row, not mixed with tactical-decision rows.
- **Where a decision-layer `chosen_action` genuinely comes from (fixing a real gap, 2026-09-30)** — the bullet above presupposed a `chosen_action` already exists to compute regret against, without saying where it originates. If it were generated by just running the cost model itself and writing the result up as prose, every decision-layer row would trivially have zero regret by construction (`chosen_action == oracle_best`) — SFT on well-written prose is still SFT on a lookup, the critique's real point. Fixed with an explicit **third, independent source**: a **teacher model** (same mechanism as §11's existing reasoning-trace extraction — a large model called via API, run locally per this project's local/cloud split) is given the Mission State facts and asked for a plan + justification WITHOUT ever seeing `oracle_best` — never contaminated with the cost model's own answer — so its proposals genuinely vary, matching or departing from `oracle_best` on their own merits. A **checker** then (a) rejects any teacher proposal that fails the shield (layer 2, rule-based, cheap) or skips a mandatory duty (layer 1), and (b) computes `regret` against `oracle_best` via the same `cost()` function (§13.A.4) — the SAME two-way split then applies: near-zero regret → SFT positive using the teacher's OWN prose (not a synthetic rewrite — this is what keeps the reasoning genuinely varied, not a lookup wearing a prose costume); high regret → the DPO-rejected member of a pair, with `oracle_best`'s own action (rendered via the same prose formatter, mirroring `build_rlhf.py`'s existing "re-call the same formatter with one field swapped" rule) as the preferred member.
- **Explanation-quality rubric (new, 2026-09-30)** — action-correctness (regret) alone is not sufficient: a correct action justified by the wrong or no reasoning is still a bad training row for a model whose whole value proposition is judgement, not lookup. A short rubric — does the explanation (i) cite the actual applicable ISM/SOLAS/STCW article (feeds §13.C.11's explanation/citation-accuracy axis), (ii) name at least one cost dimension it weighed (fuel/time/risk/goal, §13.A.4), (iii) name at least one alternative it considered and rejected — filters teacher rows before they enter the SFT pool: low-regret-but-low-rubric-score rows are excluded rather than force-included just because the action happened to be right.
- CHIRP (427 unused articles, §5/§6) extraction: **reuses the existing `extract_incident_reasoning.py` schema unchanged** (`situation`/`procedures`/`regulations`/`outcomes`/`key_facts` plus its incident-specific `fault_attribution`/`actual_actions_taken`-vs-`procedures` fields) — this schema already captures exactly "situation, decision, outcome, what should have happened"; no new schema needed.
- **CHIRP → procedure-library key, the missing link (fixing a real gap, 2026-09-30)**: the schema above captures a narrative, but nothing connected it to `(event_type, severity, context_flags)` — the exact key the procedure library (§13.A.4) is looked up on — so none of the 427 articles could be scored against it. Fixed by extending the SAME extraction call (no extra API cost) with 3 more structured fields: `mapped_event_type` (one of the 5 v1 event types, §13.A.2, or `unmapped` if none genuinely fit), `severity` (§13.C.9's minor/moderate/serious/catastrophic scale), `context_flags` (the mission scenario's own free-list vocabulary, §13.C.13). A lightweight rule-based **checker** cross-validates each mapping against a keyword guard (e.g. an `engine_failure` tag requires a propulsion/machinery keyword somewhere in the source text) — a cheap sanity check against a hallucinated tag, not a full independent classifier. **Most articles will map to `unmapped`, and that's expected, not a failure**: only 5 of §4's ~70 named events are fully specified in v1 (§13.A.2), while CHIRP spans the full ~70 — unmapped rows are not discarded, they still train the general reasoning/citation pool exactly as before, they just can't be scored on the Procedure/regime-selection axis (§13.C.11) until their event type graduates out of the §4 backlog.
- DPO-rejected construction, revised (2026-09-30) for the 3-layer model: **two categories, split along layer boundaries, not "wrong vs. late"** — (a) a genuine mandatory/shield violation (a required report missed its §13.B.7 deadline, or a forbidden action was taken) — stays exact-match/rule-based, since layers 1–2 are genuinely rule-like; (b) a high-regret decision-layer choice (`regret` above a threshold against `oracle_best`, §13.A.4) — the action was permissible but clearly cost-dominated, a soft/graded rejection, not a rule violation. This replaces the earlier "wrong procedure vs. correct-but-late" split, which implicitly assumed a single correct procedure existed for every event — no longer true once the decision layer is open-ended.

<a id="sec-13-c-11"></a>
#### 13.C.11 Ground-truth functions per evaluation axis

| §10.1 axis | Ground-truth function |
|---|---|
| Safety | Counterfactual check against §13.A.4: was there a `required_action` that, per the procedure library's own recovery condition, would have prevented the outcome? |
| Mission outcome | Recomputed resource margin (§13.A.3) vs. Mission Order success criteria — already specified in §10.3 |
| Regulatory/procedural compliance | Direct boolean check against §13.A.4's `reporting_duties` table (report X filed within deadline Y?) |
| Procedure/regime selection accuracy | Revised (2026-09-30): for mandatory/shield-only events, a direct rule-compliance check; for decision-layer events, **regret** (§13.A.4) against `oracle_best` — a low-regret action scores well even when it differs from the oracle's own candidate, never a bare exact-match check |
| Explanation/citation accuracy | Needs an **ISM/SOLAS/MARPOL article index** (the Captain's equivalent of `classify_rules()`'s COLREG-rule knowledge) mapping each procedure-library entry to its underlying article/regulation number; the training-side explanation-quality rubric (§13.C.10) checks the same citation at label-generation time, this is its eval-time counterpart |
| Decision timeliness / calling discipline | The §13.B.7 trigger monitors ARE this ground truth already |
| Resource efficiency | The §13.B.8 route planner's own unconstrained minimum-resource run, taken once per mission as the reference |
| Crew welfare | Directly read off the §13.A.3 rest-hour ledger |

Without these, the §10 composite is, as put, "a number without meaning" — this table is what closes that gap.

<a id="sec-13-c-12"></a>
#### 13.C.12 Run format & versioning

- Mission-level run JSON nests ordinary OOW run-log files, same **schema** as today, exactly as already proposed in §11.6's `mission_run.json` wrapper — but see §13.C.14: the *evaluator* run over each nested file is NOT unchanged, only the file format is.
- Add a `captain_prompt_hash` alongside OOW's existing `prompt_hash` mechanism (the exact existing pattern in `run_llm_scenario.py`/`build_outcome_dpo.py`) — any change to the Captain's system prompt or Mission-State-to-text renderer bumps this hash, so DAgger/outcome-mining correctly skips stale-prompt runs, exactly as already done for OOW.
- **`oow_agent: "oracle" | "llm"` (fixing a real gap, 2026-09-30, see §13.C.15)** — which OOW actually sailed the run; `oow_prompt_hash` is only meaningful/present when `oow_agent == "llm"`, `null` otherwise (the oracle is code, not a prompt).
- Deterministic mission replay: `(mission_seed, responder_seed, captain_prompt_hash, oow_prompt_hash, oow_agent)` fully determines a reproducible run (assumes greedy/deterministic decoding, matching this project's existing convention) — the scheduler's own total order (§13.A.9) is what makes "deterministic" actually true, not just seeded.

<a id="sec-13-c-13"></a>
#### 13.C.13 Ground-truth scenario file — putting it all together

Answering directly: **Mission Order + route + brown envelopes + resources is the right core, but 4 more elements are needed** for the file to be genuinely computable and scoreable end-to-end, not just readable:

1. **Exclusion zones as their own element** (§7) — the route is "waypoints + exclusion-zone polygons" together, not waypoints alone. Revised (2026-09-30): this now spans both DYNAMIC zones (weather/security/wildlife, mission-specific) and STATIC geographic zones (coastline, shallow water, fixed installations — reused across missions via a shared `region_id`, §7), tagged with a `type` field so the route planner (§13.B.8) treats them uniformly while the scenario generator (§13.C.9) only ever varies the dynamic ones.
2. **Per-brown-envelope severity + context flags**, not just (type, time) — this is exactly the key the procedure library (§13.A.4) looks up on; without it there is no way to derive the correct required action at all.
3. **A world-responder script** for every event that needs one (§13.A.2) — a distress call or a commercial instruction is otherwise unresolvable (nothing ever answers it).
4. **Starting resource STATE**, not a budget list — the actual values at t=0 (fuel quantity, rest-hour ledger — possibly already partially fatigued as a deliberately harder variant, ship performance/Nomoto profile), since §13.A.3's formulas need a starting point to integrate from.
5. **Ambient/background traffic & weather**, distinct from the scripted brown envelopes — needed so the encounter-sim (§13.A.1) has ordinary COLREG traffic to splice in, not only brown-envelope-triggered contacts.
6. **Two seeds** (`mission_seed`, `responder_seed`) for reproducibility (§13.C.9/§13.C.12).
7. **Held-out/category metadata** (`failure_category` tag, `held_out` flag) so the mission fits the train/eval-split methodology (§13.C.9) already used elsewhere in this project.
8. **Ports library reference** (§7) — the `region_id`'s fixed set of `{id, name, position, services, min_approach_depth_m}` entries, not invented per mission; `request_place_of_refuge`/`abort_mission` and the Engine-failure event's `distance_to_refuge_nm` (§13.A.2/§13.B.8) have no candidate to divert to without it.
8. **A precomputed oracle baseline** (the route planner's own unconstrained minimum-resource run, §13.B.8) stored alongside the scenario, not recomputed at eval time — the fixed reference the resource-efficiency axis (§13.C.11) compares against, and itself inspectable/debuggable.

On timing ("only revealed at the right moment, when it hurts most"): this is already served by each event's `trigger` being a *condition* (distance-along-route, elapsed time, or position), not a fixed reveal-time — the generator can pick/search trigger points adversarially (the same severity-sweep/falsification methodology already used for OOW's Nomoto work) rather than a plain random time.

Combined schema (extends §8.2's Mission Order):

```json
{
  "mission_id": "MSN-2026-0142",
  "seeds": {"mission_seed": 42, "responder_seed": 7},
  "held_out": false,
  "failure_category": "stand_on_timing_pressure",

  "mission_order": { "...": "§8.2, unchanged" },

  "route": {
    "region_id": "nl_north_sea_coast",
    "waypoints": [ "...WPT1..WPT4, as in §8.2..." ],
    "exclusion_zones": [
      {"id": "whale_zone_1", "type": "dynamic_hazard", "polygon": [[0,0],[0,1],[1,1],[1,0]], "speed_limit_kn": 10},
      {"id": "shoal_1", "type": "shallow_water", "polygon": [[2,0],[2,1],[3,1],[3,0]], "min_depth_m": 8}
    ],
    "ports": [
      {"id": "port_denhelder", "name": "Den Helder", "position": [52.96, 4.75],
       "services": ["repair", "fuel_bunkering", "medical"], "min_approach_depth_m": 12}
    ]
  },

  "ambient": {
    "background_traffic_density": "moderate",
    "weather_schedule": [
      {"t_s": 0, "visibility_m": 10000},
      {"t_s": 32000, "visibility_m": 500}
    ]
  },

  "resources_initial": {
    "fuel_tonnes": 850,
    "rest_hours_ledger": {"hours_awake": 4, "hours_rested_last_24h": 8},
    "ship_profile": "sawada2021_default",
    "draft_m": 7.5
  },

  "brown_envelopes": [
    {
      "event_id": "ev1", "type": "engine_failure", "severity": "moderate",
      "context_flags": ["restricted_visibility_nearby"],
      "trigger": {"type": "distance_along_route_nm", "value": 180},
      "world_responder": null
    },
    {
      "event_id": "ev2", "type": "distress_call", "severity": "serious",
      "context_flags": [],
      "trigger": {"type": "elapsed_time_s", "value": 54000},
      "world_responder": {"type": "timed_resolution", "delay_s": 1200,
                           "resolution": "assistance_no_longer_required"}
    }
  ],

  "oracle_baseline": {
    "min_resource_route": { "...": "output of §13.B.8's unconstrained run" },
    "min_fuel_tonnes": 610,
    "min_eta_s": 190000
  }
}
```

**Decided (2026-09-30)**: the "correct answer" for each brown envelope is **derived at eval time from the procedure library (§13.A.4)**, not stored redundantly in the scenario file itself — avoids the scenario's label and the procedure library silently diverging if the latter is later revised. The scenario file only stores what actually happened (type/severity/context/trigger/world-responder), never the expected response.

<a id="sec-13-c-14"></a>
#### 13.C.14 OOW evaluator needs an encounter mode (fixing a real gap, 2026-09-30)

**A real gap, not to be waved away**: §11.6/§13.C.12 said nested OOW run-logs are evaluated "unchanged" by `Evaluation Functions/evaluate_run.py` — but that evaluator assumes a fixed-position mission goal and an `arrived`/`reached_goal` criterion. Inside a mission, an encounter's "goal" is only the projected leg endpoint (a direction to head in, not a destination to arrive at — the encounter simply ENDS once §13.A.1's own end condition is met: every contact past-and-clear and back on the planned track), so directly reusing the standalone arrival/temporal/spatial logic would silently produce wrong numbers (e.g. a spurious mission-incomplete cap firing on every single encounter, since nothing is ever meant to "arrive"). Not accepting that as a known-wrong number — fixed with an explicit **encounter mode**:

- **Kept as-is** (none of these depend on arrival at a fixed goal, only on behaviour during the window): `safety_score` (CPA/collision hard gate), `compliance_axis()` (COLREG rule-following), `explanation_axis()` (citation accuracy), `manoeuvre_and_smoothness_axes()` (turn-count/zigzag detection over the bounded window).
- **Replaced**: the `reached_goal`/mission-incomplete hard gate becomes an **"encounter resolved" gate** — success = the encounter ended via §13.A.1's own end condition without a collision; failure = the encounter's own step budget is exhausted without resolving (own-ship perpetually still avoiding, the same failure signature already seen for VO/DWA in the Nomoto falsification work) OR a literal collision (the existing hard gate, unchanged).
- **Dropped from the per-encounter score entirely, computed ONCE at the whole-mission level instead**: `temporal_efficiency`/`spatial_efficiency` — comparing a bounded encounter segment's path to a straight line toward a temporary leg endpoint isn't meaningful; the mission's own Resource efficiency axis (§10.1), measured against the whole-mission feasibility oracle (§13.B.8/§13.C.11), already covers this at the level it's actually meaningful.
- **No per-encounter `composite_score` at all**: rather than inventing a reduced composite formula that would look superficially like the standalone one but mean something different, each nested OOW run only reports its individual axis scores (safety/compliance/explanation/manoeuvre/smoothness) upward into the Mission Log (§11.5); the **one real composite number** is computed at the mission level (§10), aggregating per-encounter axis scores (e.g. worst-case safety across all encounters) together with the mission-level axes.
- Implementation-wise, this is `evaluate_run.py` gaining an explicit `mode: "standalone" | "encounter" = "standalone"` parameter — an ADDITIVE change (default preserves today's exact behaviour for every existing standalone mission), not a rewrite. This is the second, evidence-based, explicitly-justified exception to this file's normal "reuse UNCHANGED" rule (the first was the 2026-09-23 `PASS_WITH_CPA_VIOLATION` fix, see `pipeline-notes.md`) — consistent with how that exception policy already works, not a violation of it.

<a id="sec-13-c-15"></a>
#### 13.C.15 OOW determinism during Captain training-data generation and evaluation (fixing a real gap, 2026-09-30)

**The gap**: if the LLM-OOW sails a spliced encounter (§13.A.1) while Captain training labels are being mined (§13.C.9/§13.C.10) or the Captain is being evaluated (§13.C.11), an OOW tactical mistake — a genuine collision a competent OOW would have avoided — feeds directly into the mission's ACTUAL recorded outcome: fuel burned, time lost, and critically the `life_risk_cost`/`ship_env_risk_cost` terms §13.A.8 computes from what really happened. That contaminates the Captain's `regret = cost(chosen) − cost(oracle_best)` signal with an OOW competency failure that has nothing to do with what the CAPTAIN decided.

**Rule**: during Captain training-data generation (§13.C.9/§13.C.10) and Captain evaluation (§13.C.11), OOW is ALWAYS `app/oracle_planner.py`'s deterministic `plan()` — the SAME deterministic OOW mechanism §13.A.1's encounter splice, §13.B.8's route planner, and §13.A.8's rollout already treat as the reference/ground truth throughout this design — never the LLM-OOW. Holding the tactical layer constant is what makes a difference in outcome between candidate Captain decisions attributable to the Captain alone, the same "isolate the layer being measured" logic this project's own two-track (rules/knowledge vs. conversational-compliance) held-out eval design already relies on. **Only in the FINAL end-to-end system evaluation** — Captain LLM and OOW LLM sailing together, a deliberate ablation of the full deployed stack — does the LLM-OOW sail underneath the Captain.

**Recorded in run metadata (§13.C.12)**: the new `oow_agent: "oracle" | "llm"` field. Every training-data-generation and Captain-evaluation run MUST have `oow_agent: "oracle"`; `oow_agent: "llm"` is reserved for the explicit final-system ablation, and is checked/excluded automatically if one were ever accidentally fed into §13.C.9/§13.C.10/§13.C.11's pipelines — a hard generation-time/mining-time gate, mirroring the existing `prompt_hash` staleness-gate pattern already used by DAgger/outcome-mining (§13.C.12).

---

<a id="sec-14"></a>
## 14. Approach: walking skeleton first (decided, 2026-09-30)

Adopting the proposed approach as-is:

- **Build first, with no LLM at all**: one mission, 3 waypoints, ONE event (engine failure → speed cap), and a "Captain" that is nothing more than a direct lookup into the procedure library (§13.A.4) — i.e. a deterministic baseline Captain, built exactly the way OOW's own deterministic baselines (`app/baselines/*.py`) were built before any LLM agent existed. It serves the same later purpose: a baseline the eventual LLM-Captain gets measured against, not a throwaway prototype.
- **Revised (2026-09-30)**: this ONE event must be the **parameterised** engine-failure variant (moderate failure + deadline pressure + a port of refuge ~40 nm away — continue at 8 kn, or put in?), not the bare table-determined version — per §13.A.4's reassessment, the bare version exercises zero real judgement, only plumbing. The deterministic baseline Captain for THIS variant is a direct `oracle_best` lookup (§13.A.4) — correct for a baseline, but proves the walking skeleton can compute `cost()`/regret at all, which is the actual thing being tested here, not just that a mission can run end-to-end.
- Mission-sim + encounter-sim coupled (§13.A.1), an MPR emitted (§8.3), evaluation computed (§10) over the whole run. If this runs end-to-end, §13.A, §13.B, and §13.C.11/§13.C.12 are all proven simultaneously — not just designed on paper.
- §4 is reduced to the **v1 set of 5 fully-specified events** (§13.A.2) for this skeleton; the remaining ~65 named events in §4 stay an explicit backlog, not a blocker.
- The RAG-corpus source-text decision (§12) is reaffirmed as a prerequisite **specifically for the RAG pipeline**, not for the walking skeleton — the skeleton needs the procedure library (§13.A.4), not RAG, which is a useful sequencing discovery in its own right: the skeleton can be built before the corpus question is resolved.
- **Facts-only prompt rule — decided**: the Captain follows the same facts-only convention as OOW (§13.B.5).

<a id="sec-14-1"></a>
### 14.1 Data readiness — what must actually be arranged, in priority order (2026-09-30)

Direct answer to "which data do we need to arrange first before we can start": for the walking skeleton itself, almost nothing — the real, user-side sourcing decision (ISM/SOLAS/STCW/MARPOL text) is explicitly NOT on the critical path for step one.

1. **Tier 0 — author directly, zero external sourcing, blocks the skeleton**: one fictional `ports` entry + a trivial/empty `region_id` exclusion-zone set (§7) for the skeleton's own 3-waypoint route (needed so the engine-failure event's "port of refuge ~40 nm away" is a real, reachable point, §13.A.2/§13.B.8) — written alongside the skeleton's own mission JSON, exactly like every existing Imazu/RND mission's fictional geometry (§5.1) is hand-authored, not sourced.
2. **Tier 0.5 — small, quick, blocks the skeleton, needs a literature check not user sourcing**: `nominal_speed_kn`/`max_speed_kn` fields for `pipeline/nomoto.py`'s `SHIP_PROFILES` (§13.A.3) — read each cited paper's (Sawada 2021, Xie 2023, Wang 2023) own stated design/max speed, or document a reasonable margin default if a paper doesn't state one.
3. **Tier 1 — does NOT block the skeleton, but is the one item genuinely on the user's own side and has the longest lead time, worth starting now**: the ISM/SOLAS/STCW/MARPOL RAG-corpus source text (§12). **Done, ahead of schedule (2026-09-30)**: `pipeline/ingest/build_captain_legal_corpus.py` (raw acquisition only, no parsing/chunking yet) has already pulled 5 real documents — BMP5, the current UK STCW/ISM-Code-implementing statutory instruments, and 2 US eCFR parts (discovered via eCFR's own search API, which also surfaced a direct cross-reference to 33 CFR part 96, the ISM Code's actual US implementation — not yet pulled). Which parts of these actually enter RAG/training is still a later, separate decision, per this item's own original recommendation.
4. **Tier 2 — already exist in the repo, zero new sourcing, only need the extraction pipeline BUILT once Track 1 work starts**: the 427 unused non-COLREG CHIRP articles and the 108 new Processed-Leo MAIB reports (§5.1a/§5.1c).
5. **Tier 3 — only needed once the scenario generator (§13.C.9) scales beyond this single skeleton mission across multiple regions**: real port/coastline/bathymetry data (World Port Index, OSM/OpenSeaMap, GEBCO/EMODnet, §7) to grow the `ports`/region-coastline libraries beyond the Tier-0 stub. **Partially done already (2026-09-30)**, ahead of schedule, as a first real sample: the full NGA World Port Index (`Data/Captain/Geo_Reference/world_port_index.csv`, ~29,000 ports, public-domain US government data, includes exactly the fields the §7 `ports` schema needs — `Services - Electrical Repair`, `Supplies - Fuel Oil`/`Diesel Oil`, `Supplies - Potable Water`/`Provisions`, plus lat/lon and multiple depth fields for `min_approach_depth_m`) and a real OSM coastline sample for the Rotterdam–Den Helder corridor (`Data/Captain/Geo_Reference/nl_coastline_rotterdam_denhelder.json`, 11 coastline ways, via the free public Overpass API) are both already downloaded and verified. GEBCO/EMODnet bathymetry grids are confirmed public-domain/free too, but multi-GB — deferred until a real `min_depth_m` shoal check is actually needed beyond the Tier-0 stub, not fetched yet.

---

<a id="sec-15"></a>
<a id="sec-15"></a>
## 15. Concrete data structures & UI controls — what's left before coding

Direct answer to "should we also define interface controls and data structures for everything before we start coding — what do you think?"

<a id="sec-15-1"></a>
### 15.1 My assessment

Two different questions, two different answers:

- **Data structures — mostly already done.** §13 already specifies the Mission State schema (§13.B.5), the Captain output schema (§13.B.6), the event/state-delta model (§13.A.2), the procedure library structure (§13.A.4), the ground-truth scenario file (§13.C.13), and the run format (§13.C.12) — that IS the bulk of "data structures of all components". What remains is a short, concrete list of Python-level dataclass diffs (§15.2): small and mechanical, and in this project's own established practice, the kind of detail usually finalised *while* writing the walking skeleton, not before — it tends to shift once real code hits real friction (see e.g. the `target_heading`/Nomoto saga in `basic_simulator.md`, where several "obvious-looking" fields needed correction only once a real bug surfaced).
- **UI controls — genuinely not done, and I'd recommend NOT fully specifying them yet.** §11 describes what the panels *show*, not the exact buttons/widgets/state machine (per-widget enabled/disabled logic, what triggers a rerun, etc.). But §14's walking skeleton is explicitly backend-only — no Streamlit involved at all, tested the same way OOW's `app/baselines/*.py` were: standalone scripts/pytest, long before any UI existed for them. Fully specifying a polished UI now would mean designing an interface for a backend that doesn't exist yet — and this project's own `basic_simulator.md` iteration history shows the OOW Streamlit UI was built and rebuilt many times **after** the simulation/agent logic already worked, never before. Same order recommended here.
- What I'd specify now, cheaply: a minimal, non-Streamlit **debug control set** (§15.3) — just enough to drive/inspect the walking skeleton by hand while building it. Directly useful for the very next step, unlike a full UI spec.

<a id="sec-15-2"></a>
### 15.2 Remaining concrete data-structure specifics (small, mechanical — do while coding, not before)

- `Mission` dataclass (`app/missions.py`): extend from a single optional `waypoint` to `waypoints: list[tuple[float, float]]`, plus `exclusion_zones: list[dict]` and `brown_envelopes: list[dict]` (schemas already given in §7/§13.C.13 — this is just the literal field diff).
- `VesselConstraints` (`app/simulation.py`): new fields for Captain-imposed limits — e.g. `captain_speed_cap_kn: float | None`, `captain_regime: str` (`"colreg"` / `"security"`) — consistent with the precedence rule already decided in §13.B.6.
- Mission-sim ↔ encounter-sim bridge (§13.A.1): two small pure functions, `to_local_frame(lat, lon, origin) -> (x, y)` / `from_local_frame(x, y, origin) -> (lat, lon)` (equirectangular, as already decided), plus the handoff function that starts/stops an encounter-sim window and passes the active Captain instruction through as a `VesselConstraints` override.
- Subordinate "facts" schema (§2.6): a small dict/dataclass per subordinate role providing structured facts to the Captain/OOW prompts — e.g. `EngineStatus(max_speed_kn, fault, reported_at)`, `LookoutReport(bearing, description, range_est)` — deliberately NOT full agents (§2.6/§12), just typed fact containers.
- World-responder execution mechanism (§13.A.2/§13.C.13): the `world_responder` JSON field already has a schema; the scheduling primitive is now fully specified (§13.A.9) — a `WORLD_RESPONDER_TIMER` entry (`t_now + delay_s`) in the mission-sim's own discrete-event queue, no separate architecture needed.

None of these require a design decision to be made now — they're direct, mechanical translations of what §13 already specified in prose/JSON into Python types.

<a id="sec-15-3"></a>
### 15.3 Minimal debug control set for the walking skeleton (decided scope, not a full UI)

Just enough to drive/inspect the skeleton by hand — a CLI/notebook-level control set, not Streamlit, not polished:

| Control | What it does |
|---|---|
| `step_mission(n=1)` | Advance the mission-sim by `n` steps (each `dt_mission_s` seconds) |
| `run_to_next_event()` | Advance until the next scripted brown envelope fires or the mission ends |
| `force_event(event_id)` | Manually trigger a specific scripted event out of turn (for testing) |
| `show_mission_state()` | Dump the current Mission State (§13.B.5) as text |
| `show_procedure_lookup(event)` | Show what the procedure library (§13.A.4) returns for the current event, without applying it |
| `show_mpr()` | Render the Mission Progress Report (§8.3) at the current point |

This mirrors exactly how OOW's own baselines were first exercised — direct function calls / a small script, before `app/streamlit_app.py` existed at all.

<a id="sec-15-4"></a>
### 15.4 Full UI control spec — explicitly deferred

**Decided (2026-09-30)**: defer the full, polished UI control-by-control specification (exact buttons/widgets per §11's panels, enabled/disabled logic, rerun triggers) until **after** the walking skeleton (§14) is running — the same order this project's own OOW Streamlit UI was actually built in. Revisit §11 at that point to turn its panel *descriptions* into a real control spec.
