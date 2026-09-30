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
  - [13.B Loose contracts](#sec-13-b)
    - [13.B.5 Mission State — formal schema](#sec-13-b-5)
    - [13.B.6 Captain output schema + precedence rules](#sec-13-b-6)
    - [13.B.7 Trigger monitors](#sec-13-b-7)
    - [13.B.8 Route planner with exclusion zones](#sec-13-b-8)
  - [13.C Data & evaluation](#sec-13-c)
    - [13.C.9 Scenario-generator specification](#sec-13-c-9)
    - [13.C.10 Training-row format](#sec-13-c-10)
    - [13.C.11 Ground-truth functions per evaluation axis](#sec-13-c-11)
    - [13.C.12 Run format & versioning](#sec-13-c-12)
    - [13.C.13 Ground-truth scenario file](#sec-13-c-13)
- [14. Approach: walking skeleton first](#sec-14)
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
| 1 | Header & classification | `mission_id`, date-time issued, issuing authority (the company's Fleet Operations Centre / DPA, not a naval command) — no security classification needed for a civilian mission |
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
      "speed_of_advance_kn": 16.0,
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

The OOW's own run-log format is deliberately left untouched — the Captain layer only adds a wrapper around it, so every existing OOW replay/audit/dashboard tool keeps working unmodified on the nested files.

---

<a id="sec-12"></a>
## 12. Open questions / not yet decided

- **Decided** (2026-09-30): the captain becomes a separate, separately-trained Qwen model (own RAG+CoT), not a shared model with OOW. OOW keeps performing manoeuvres/navigation; the captain can overrule and helps with new waypoints/operational limits arising from brown envelopes.
- **Decided** (2026-09-30, this pass): folder structure (§6), Mission Order format (§8), route/waypoint interface (§7), and evaluation axes (§10) are proposed — not yet finalised/implemented.
- **Decided** (2026-09-30): mission scale — e.g. Rotterdam → Den Helder, with fictional exclusion zones to route around and a handful of waypoints; ~1 brown envelope per 100 nm of route (a 600 nm route ⇒ ~6 brown envelopes). See §7.
- **Decided** (2026-09-30): §10.2's composite weights — Mission outcome 25%, Safety 20%, Crew welfare 20%, remaining 5 axes 7% each. Starting point, expected to be retuned later.
- **Decided** (2026-09-30): route-planning direction — likely the PredictWind API later, or a manually downloaded real route; a fake random-waypoint route for now. See §7.
- Still to be determined: which exact ISM/SOLAS/STCW/MARPOL source text becomes the Captain's RAG corpus (COLREG-Consolidated-2018.pdf played that role for OOW) — no such file exists yet in `Data/Captain/`. **Research done (2026-09-30), acquisition still pending on the user's side**: (a) check whether the same channel that supplied `COLREG-Consolidated-2018.pdf` also covers SOLAS/MARPOL/STCW/ISM Code (IMO Publishing sells these as consolidated editions too, likely the same licence bundle); (b) free/legitimate alternatives if not — the ISM Code's full text was adopted via **IMO Resolution A.741(18)**, and IMO Assembly/MSC/MEPC resolutions are typically free PDFs on imo.org (unlike the paid "consolidated editions"), also republished free by several flag-state registries (Marshall Islands, Panama, Liberia, Malta); **BMP5** (Best Management Practices to Deter Piracy) is fully free/industry-published and directly usable for §9.4's security regime; for SOLAS/MARPOL/STCW specifically, each country's own free implementing legislation covers the same substance — US eCFR Title 33/46, UK legislation.gov.uk "Merchant Shipping" statutory instruments, EU EUR-Lex directives/regulations; STCW competence tables are also often freely summarised by maritime training academies. The ICS *Bridge Procedures Guide* itself is commercial (Marisec/Witherby), no known free equivalent. Recommendation: mirror the existing `simple_colreg.json` pattern (§5.1/§6.3) — use official/paid text where available, plus an own plain-language summary layer built from the free sources above as a copyright-safe supplement.
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

- **Mission-sim** (new): steps in **minutes**, advances position along the current leg (distance travelled = speed × dt), consumes fuel and time (§13.A.3), advances a simple time-indexed weather schedule, and accumulates rest-hours. This is where a 100–600 nm, multi-hour-to-multi-day mission actually lives.
- **Encounter-sim** (existing, unchanged): `app/simulation.py`'s existing 10 s-step metric simulator, spliced in ONLY when the mission-sim detects a reason to (a scripted contact/hazard within lookahead range, or a scripted event's trigger point). Between encounters the OOW runs on autopilot toward the next waypoint — no LLM calls, no per-step decisions, exactly the "OOW runs autonomously on the Captain's plan between events" behaviour already established in §3.3.
- **Handoff**: at encounter entry, project the mission-sim's lat/lon position to a local flat-earth metric frame (equirectangular projection centred on the entry point — accurate enough over an encounter's bounded few-nm/few-minute window, NOT used for the whole voyage); run the existing OOW loop unchanged for the encounter's duration, with any active Captain instruction (speed cap, `avoid_zone`) passed in as a `VesselConstraints` override, same mechanism as today; project the encounter's final position/heading/speed back to lat/lon to resume mission-sim leg progression.
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

**Correction (2026-09-30), addressing a real critique of the original single-answer procedure model**: the table above is not a fixed (event → one required action) mapping — see §13.A.4's revised 3-layer model. **Engine failure** and **Commercial instruction vs. safety** are only genuinely *table-determined* in their simplest parameterisation; both become real judgement cases once continuous parameters are added — Engine failure gains `distance_to_refuge_nm`/`deadline_slack_h` (continue at capped speed vs. put in to a port of refuge is then a real cost tradeoff, not a lookup); Commercial instruction's "breaches a safety margin" becomes a *computed* condition against the live Mission State rather than a fixed always-refuse rule. Fog and Whale zone remain genuinely table-determined (a COLREG-mandated safe speed and a charted/posted speed limit both leave negligible real tradeoff space) — Distress call was already a judgement case. This replaces the original "4 table-determined, 1 judgement case" split with a more accurate one: 2 event types are consistently table-determined, 3 are judgement-capable once properly parameterised — see §13.A.4.

<a id="sec-13-a-3"></a>
#### 13.A.3 Resource model

- **Fuel**: `fuel_rate = base_load + k * speed_kn**3` (the real cube-law relationship between ship speed and propulsion power/fuel burn, plus a constant "hotel load" for auxiliary/generator consumption independent of propulsion) — `k` calibrated per mission so the Mission Order's `fuel_tonnes_at_departure` roughly balances the expected voyage duration at `speed_of_advance_kn` (an approximation, not a real ship's exact curve, flagged as such).
- **ETA**: `remaining_distance_nm / current_speed_kn`, recomputed every mission-sim step (reflects live speed changes from Captain/OOW decisions).
- **Rest-hours**: STCW A-VIII/1's real rule — minimum 10 hours' rest in any 24-hour period, minimum 77 hours in any 7-day period, divisible into no more than 2 periods (one ≥6 hours). Tracked cumulatively (v1: one aggregate "OOW currently on watch" ledger, not per named crew member); a violation feeds the Crew welfare axis (§10.1) directly and can itself spawn a fatigue-error brown envelope (§4.D) if pushed far enough.
- This is what makes "is the goal still achievable" (§9, §10.3) an actual computation rather than a judgement call.

<a id="sec-13-a-4"></a>
#### 13.A.4 Procedure library — shield, mandatory duties, and a cost-based decision layer (revised 2026-09-30)

**Agreeing with a real critique of the first version of this section**: defining `required_actions` as "the ONLY correct action(s)" and using it simultaneously as shield/label/auditor collapses the Captain into a lookup table — SFT would just learn the lookup, DPO would punish any deviation from it, and no evaluation could ever score a genuinely better decision higher than the table's own entry. That directly contradicts §9.3 ("where the LLM proves its value: weighing, reprioritising") and the "legitimately abandoned goal" logic in §10.1/§10.3. Fixed by splitting the table into three layers with genuinely different roles — not three synonyms for "the correct answer":

1. **Mandatory** — reporting duties and their deadlines (the original `reporting_duties`), independent of which candidate action is chosen. Checklist material, stays deterministic (§9.3's "known procedures" category). Can itself depend on the CHOSEN action (e.g. diverting to a port of refuge triggers its own extra reporting duty) — a function of `(event, chosen_action)`, not just `event`.
2. **Shield** — `forbidden_actions`, unchanged: a hard gate on the action space, validated before any instruction reaches the OOW, mirroring `app/oracle_planner.py`'s `required_direction()` hard rule-legality gate (§9.3). Absolute, never a matter of degree.
3. **Decision layer** (new, replaces the old single `required_actions`) — the only layer that gives the Captain something to actually decide, and the only layer worth training/evaluating judgement on:
   - `candidates(event, mission_state) -> list[action]` — a systematic candidate generator (mirrors `oracle_planner.py`'s own candidate generation: baseline first-step choices + a parameter sweep), **not** an exhaustive enumeration and **not** a hard restriction on what the Captain may propose — it exists to compute a reference, not to fence in the Captain's action space (only the shield does that).
   - `cost(action, mission_state) -> {resource_cost, risk_cost, goal_cost}` (§13.A.3's fuel/time formulas + a risk proxy grounded in the source incident base rates + deviation from the Mission State's own goals), combined via weights exactly like `oracle_planner.py`'s existing `W_GOAL`/`W_CLEARANCE`/`W_EFFORT` pattern — giving `oracle_best = argmin cost over candidates`.
   - **Training/eval signal is regret, not exact match**: `regret = cost(chosen_action, mission_state) − cost(oracle_best, mission_state)`, with `cost()` recomputed for WHATEVER the Captain actually proposed (even an action outside the candidate generator's own list) — so a genuinely better, novel decision scores a lower regret than the oracle's own reference, never penalised just for differing textually. Low/zero regret is SFT-worthy regardless of whether it matches `oracle_best` verbatim; a real, unambiguous shield violation is the only thing that stays hard-gated (layer 2), not the decision layer.

This directly fixes the "there are always exceptions to the rule" problem raised: layers 1–2 stay genuinely rule-like (checklists, hard bans) where that's actually true of the real world, and layer 3 is deliberately open-ended (regret against a computed reference, not membership in a fixed set) precisely because real Captain judgement calls don't have one textually-fixed correct answer.

Proposed home unchanged: **`pipeline/captain_agent_spec.py`**, mirroring `pipeline/oow_agent_spec.py`'s `classify_rules()` role. Source unchanged: ISM/SOLAS/BMP5 + the project's own CHIRP set (§5/§6).

Revised concrete v1 entries:

- *Engine failure*: **mandatory** = fault log entry (+ a place-of-refuge report only if requested). **Shield** = never exceed the Chief-Engineer-declared safe speed. **Decision layer** = continue at the capped speed vs. divert to a nearby port of refuge — a real cost tradeoff once the event carries `distance_to_refuge_nm`/`deadline_slack_h` (§13.A.2); with no time pressure and no refuge nearby, `oracle_best` collapses to "continue at capped speed" and there is effectively nothing to weigh — this is what made the original v1 parameterisation look table-determined; it was an under-specified special case, not a property of engine failures in general.
- *Fog, regime change*: **mandatory** = none. **Shield** = never exceed the Rule-19 safe speed for conditions. **Decision layer** = degenerate — COLREG mandates the safe speed itself, no real second candidate worth weighing — genuinely table-determined, unlike engine failure.
- *Distress call*: **mandatory** = notify DPA + flag state (SOLAS reporting), deadline immediate. **Shield** = never ignore the call without logging a reason. **Decision layer** = assist vs. document a valid reason not to — cost trades off time/fuel/mission-goal delay against the legal/moral/reputational cost of not assisting; genuinely a judgement case, as originally identified.
- *Whale zone*: **mandatory** = none. **Shield** = never exceed the zone's posted speed limit inside the polygon. **Decision layer** = degenerate — a charted, fixed limit, no real second candidate — genuinely table-determined.
- *Commercial instruction vs. safety*: **mandatory** = log entry + DPA notification of the outcome. **Shield** = never comply with an instruction that breaches a safety margin. **Decision layer** = whether the instruction actually breaches a margin is now a COMPUTED condition against the live Mission State (fuel/rest-hour/COLREG margins) rather than a fixed always-refuse rule — itself a genuine judgement case once "breaches a margin" isn't hard-coded as always-true.

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

<a id="sec-13-b"></a>
### 13.B Loose contracts

<a id="sec-13-b-5"></a>
#### 13.B.5 Mission State — formal schema + the facts-only decision

- Schema split: **immutable** (copied once from the Mission Order at mission start: `mission_id`, `success_criteria`, `restricted_zones`, initial resources) vs. **mutable** (everything §8.4 already lists — goals-with-status, current resources, active hazards, active plan, event/decision log).
- Delta representation: a generic patch entry `{"t": <mission-sim time>, "field_path": "...", "old": ..., "new": ..., "cause": <event_id or decision_id>}` — append-only, so the Mission State at any past time is reconstructable by replaying deltas up to that point (same "log, don't overwrite" principle as this project's existing run-checkpoint files).
- **Decided (2026-09-30)**: the Captain follows the **same facts-only prompt convention already adopted for OOW** (the project's own 2026-09-24 architecture pivot, see `basic_simulator.md`). Mission State renders to the Captain's prompt as plain facts (fuel remaining, elapsed rest-hours, current regime, distance to next waypoint, active exclusion zones) — the §1.2 priority ladder and "safety beats schedule" live ONCE in the Captain's system prompt, not repeated as an imperative annotation on every situational fact. Exactly mirrors OOW's system-prompt-vs-situation-report split.

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
| Track deviation | `track_deviation_above(20_nm, within_h=6)` |
| Unresolved alarm | `alarm_unresolved(timeout_s)` — an "alarm" is defined, self-referentially but concretely, as any OTHER monitor whose own condition has been true for longer than `timeout_s` without an OOW/Captain action addressing it |

- **Concurrent triggers**: the highest-severity unresolved trigger is answered first; all are logged regardless; the Captain's response deadline is keyed to the single most urgent unresolved trigger.
- **Deadlines per urgency class** (this is what makes §10.3's recall/precision computable): Emergency = same decision epoch (zero elapsed sim time); Mandatory = within one mission-sim step (a few simulated minutes); Routine = by the next scheduled report cycle.

<a id="sec-13-b-8"></a>
#### 13.B.8 Route planner with exclusion zones

- **v1 algorithm (decided)**: a **visibility graph** over exclusion-zone polygon vertices (nodes = start, goal, every polygon corner; an edge exists where the straight segment crosses no zone; shortest path via Dijkstra/A\*) — geometrically exact for polygon obstacles at this abstraction level, and simpler than a grid-based A\* (which forces a resolution trade-off and produces jagged paths). This is what makes `reroute`/`avoid_zone` (§7/§9.1) actually do something.
- **Feasibility check**: recompute ETA/fuel (§13.A.3) along any candidate new route; reject/flag it if it breaches the Mission Order's resource margins. Running this planner ONCE, unconstrained, at mission start also produces the "feasibility oracle" baseline §13.C.11 needs for the resource-efficiency axis.

<a id="sec-13-c"></a>
### 13.C Data & evaluation

<a id="sec-13-c-9"></a>
#### 13.C.9 Scenario-generator specification

- Category/severity distribution: reuse the existing weighted-category pattern from `generate_random_imazu_missions.py` — §4's category letters as the weighted category set, minor/moderate/serious/catastrophic as a secondary weighted draw.
- Dependency rules: a small explicit compatibility table (e.g. fog + engine failure allowed to co-occur; piracy excluded on a North-Sea-tagged route) — a geography/category exclusion list, not a full plausibility model.
- Held-out split at **mission** level: no event-*combination* shared between train and eval sets — mirrors the existing `HELD_OUT_EVAL_PROFILE`/Imazu22 held-out pattern (a fixed, deterministic held-out fraction of mission templates).
- Seeds: one seed per mission index (matches `generate_random_imazu_missions.py`'s own convention), plus a **separate** seed for the world-responder's own scripted timing/outcome draws (§13.A.2) — a mission is fully reproducible from `(mission_seed, responder_seed)`.

<a id="sec-13-c-10"></a>
#### 13.C.10 Training-row format

- Input: last-N-events window + the current Mission State snapshot (facts-only, §13.B.5) + retrieved RAG/KG/PG chunks (§6.3) — bounded, not the full mission history, mirroring OOW's own bounded situation-report design.
- Labels, revised (2026-09-30) for the 3-layer model (§13.A.4): for mandatory/shield-only events (fog, whale zone), the deterministic action is a direct SFT label as before; for decision-layer events (engine failure, distress call, commercial instruction), the SFT label is `oracle_best` (§13.A.4) but ONLY when the Captain's actual choice has near-zero regret against it — otherwise the row is better mined as a DPO/reflection pair (chosen vs. `oracle_best`) than force-fit as an SFT positive. The Captain's own free-text `plan`/`reasoning` is CoT training text regardless; the Mission Progress Report (§8.3) is a separate, simpler templated-output row, not mixed with tactical-decision rows.
- CHIRP (427 unused articles, §5/§6) extraction: **reuses the existing `extract_incident_reasoning.py` schema unchanged** (`situation`/`procedures`/`regulations`/`outcomes`/`key_facts` plus its incident-specific `fault_attribution`/`actual_actions_taken`-vs-`procedures` fields) — this schema already captures exactly "situation, decision, outcome, what should have happened"; no new schema needed.
- DPO-rejected construction, revised (2026-09-30) for the 3-layer model: **two categories, split along layer boundaries, not "wrong vs. late"** — (a) a genuine mandatory/shield violation (a required report missed its §13.B.7 deadline, or a forbidden action was taken) — stays exact-match/rule-based, since layers 1–2 are genuinely rule-like; (b) a high-regret decision-layer choice (`regret` above a threshold against `oracle_best`, §13.A.4) — the action was permissible but clearly cost-dominated, a soft/graded rejection, not a rule violation. This replaces the earlier "wrong procedure vs. correct-but-late" split, which implicitly assumed a single correct procedure existed for every event — no longer true once the decision layer is open-ended.

<a id="sec-13-c-11"></a>
#### 13.C.11 Ground-truth functions per evaluation axis

| §10.1 axis | Ground-truth function |
|---|---|
| Safety | Counterfactual check against §13.A.4: was there a `required_action` that, per the procedure library's own recovery condition, would have prevented the outcome? |
| Mission outcome | Recomputed resource margin (§13.A.3) vs. Mission Order success criteria — already specified in §10.3 |
| Regulatory/procedural compliance | Direct boolean check against §13.A.4's `reporting_duties` table (report X filed within deadline Y?) |
| Procedure/regime selection accuracy | Revised (2026-09-30): for mandatory/shield-only events, a direct rule-compliance check; for decision-layer events, **regret** (§13.A.4) against `oracle_best` — a low-regret action scores well even when it differs from the oracle's own candidate, never a bare exact-match check |
| Explanation/citation accuracy | Needs an **ISM/SOLAS/MARPOL article index** (the Captain's equivalent of `classify_rules()`'s COLREG-rule knowledge) mapping each procedure-library entry to its underlying article/regulation number |
| Decision timeliness / calling discipline | The §13.B.7 trigger monitors ARE this ground truth already |
| Resource efficiency | The §13.B.8 route planner's own unconstrained minimum-resource run, taken once per mission as the reference |
| Crew welfare | Directly read off the §13.A.3 rest-hour ledger |

Without these, the §10 composite is, as put, "a number without meaning" — this table is what closes that gap.

<a id="sec-13-c-12"></a>
#### 13.C.12 Run format & versioning

- Mission-level run JSON nests ordinary, unchanged OOW run-log files exactly as already proposed in §11.6's `mission_run.json` wrapper.
- Add a `captain_prompt_hash` alongside OOW's existing `prompt_hash` mechanism (the exact existing pattern in `run_llm_scenario.py`/`build_outcome_dpo.py`) — any change to the Captain's system prompt or Mission-State-to-text renderer bumps this hash, so DAgger/outcome-mining correctly skips stale-prompt runs, exactly as already done for OOW.
- Deterministic mission replay: `(mission_seed, responder_seed, captain_prompt_hash, oow_prompt_hash)` fully determines a reproducible run (assumes greedy/deterministic decoding, matching this project's existing convention).

<a id="sec-13-c-13"></a>
#### 13.C.13 Ground-truth scenario file — putting it all together

Answering directly: **Mission Order + route + brown envelopes + resources is the right core, but 4 more elements are needed** for the file to be genuinely computable and scoreable end-to-end, not just readable:

1. **Exclusion zones as their own element** (§7) — the route is "waypoints + exclusion-zone polygons" together, not waypoints alone.
2. **Per-brown-envelope severity + context flags**, not just (type, time) — this is exactly the key the procedure library (§13.A.4) looks up on; without it there is no way to derive the correct required action at all.
3. **A world-responder script** for every event that needs one (§13.A.2) — a distress call or a commercial instruction is otherwise unresolvable (nothing ever answers it).
4. **Starting resource STATE**, not a budget list — the actual values at t=0 (fuel quantity, rest-hour ledger — possibly already partially fatigued as a deliberately harder variant, ship performance/Nomoto profile), since §13.A.3's formulas need a starting point to integrate from.
5. **Ambient/background traffic & weather**, distinct from the scripted brown envelopes — needed so the encounter-sim (§13.A.1) has ordinary COLREG traffic to splice in, not only brown-envelope-triggered contacts.
6. **Two seeds** (`mission_seed`, `responder_seed`) for reproducibility (§13.C.9/§13.C.12).
7. **Held-out/category metadata** (`failure_category` tag, `held_out` flag) so the mission fits the train/eval-split methodology (§13.C.9) already used elsewhere in this project.
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
    "waypoints": [ "...WPT1..WPT4, as in §8.2..." ],
    "exclusion_zones": [
      {"id": "whale_zone_1", "polygon": [[0,0],[0,1],[1,1],[1,0]], "speed_limit_kn": 10}
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
    "ship_profile": "sawada2021_default"
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
- World-responder execution mechanism (§13.A.2/§13.C.13): the `world_responder` JSON field already has a schema; what's missing is just the scheduling primitive — a `(trigger_time, resolution_fn)` entry in the mission-sim's own event queue, evaluated every step, no new architecture needed.

None of these require a design decision to be made now — they're direct, mechanical translations of what §13 already specified in prose/JSON into Python types.

<a id="sec-15-3"></a>
### 15.3 Minimal debug control set for the walking skeleton (decided scope, not a full UI)

Just enough to drive/inspect the skeleton by hand — a CLI/notebook-level control set, not Streamlit, not polished:

| Control | What it does |
|---|---|
| `step_mission(n=1)` | Advance the mission-sim by `n` steps (minutes) |
| `run_to_next_event()` | Advance until the next scripted brown envelope fires or the mission ends |
| `force_event(event_id)` | Manually trigger a specific scripted event out of turn (for testing) |
| `show_mission_state()` | Dump the current Mission State (§13.B.5) as text |
| `show_procedure_lookup(event)` | Show what the procedure library (§13.A.4) returns for the current event, without applying it |
| `show_mpr()` | Render the Mission Progress Report (§8.3) at the current point |

This mirrors exactly how OOW's own baselines were first exercised — direct function calls / a small script, before `app/streamlit_app.py` existed at all.

<a id="sec-15-4"></a>
### 15.4 Full UI control spec — explicitly deferred

**Decided (2026-09-30)**: defer the full, polished UI control-by-control specification (exact buttons/widgets per §11's panels, enabled/disabled logic, rerun triggers) until **after** the walking skeleton (§14) is running — the same order this project's own OOW Streamlit UI was actually built in. Revisit §11 at that point to turn its panel *descriptions* into a real control spec.
