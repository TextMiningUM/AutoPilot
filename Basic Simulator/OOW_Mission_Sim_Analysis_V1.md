# OOW-simulatie-analyse — top problemen per model (bare_qwen, v0–v6) en voorstel vervolgtraject

**Bron:** `Basic Simulator/Data/missions/_llm_runs/*__units_v1.json` — 147 run-logs (van in totaal
152 verwacht; de sweep loopt op dit moment nog door voor UM10/v3–v6, zie `_sweep_status.json`).
Elke run = één volledige missie-uitvoering (own-ship stap-voor-stap bestuurd door `ask_oow()`),
niet de eerdere Track 1/2 Q&A-ablatie uit `full report evaluations V1.md` — dit is de
**gedrags-/uitvoeringslaag**: reed het schip echt veilig van start tot doel.

**Missieset (19 missies × 8 configs):**
- **Imazu01–09** — de 9 klassieke Imazu head-on/crossing/overtaking-scenario's (Sawada et al. 2021).
- **UM01–02** — echte "quiet"-missies (geen reëel aanvaringsgevaar; correcte respons = niets doen).
- **UM03–10** — een systematische bearing-sweep van give-way/stand-on/overtaking-situaties
  (Rule 13/14/15/16/17), UM10 is een 2-doel-scenario. **Dit zijn dus GEEN quiet-missies** ondanks de
  "UM"-naam — alleen UM01/UM02 testen degeneratie (overreageren op niets).
- **Configs:** `bare_qwen` (geen COLREG-framing), `v0_base` (framing, geen extra's), `v1_rag` (+RAG),
  `v2_cot` (+chain-of-thought), `v3_rag_cot` (RAG+CoT), `v4_pg`/`v5_pg_incident`/`v6_pg_scenario`
  (+procedural-graph guidance, resp. samengevoegd/alleen-incidenten/alleen-Track2-scenario's).

Analysescript: `Basic Simulator/_tmp_analyze_mission_runs_v2.py` (bewaard voor reproduceerbaarheid).

---

## 1 — Kernresultaten per config

| Config | n | Composite | Imazu | UM | Onveilig (CPA-schending) | Compliance | Aankomst% |
|---|---|---|---|---|---|---|---|
| `bare_qwen` | 19 | 0.281 | 0.141 | 0.406 | **9/19 (47%)** | 0.039 | 37% |
| `v0_base` | 19 | 0.635 | 0.566 | 0.696 | 0/19 | 0.263 | **95%** |
| `v1_rag` | 19 | **0.333** | 0.364 | 0.305 | 5/19 (26%) | 0.368 | **26%** |
| `v2_cot` | 19 | 0.569 | 0.586 | 0.553 | 1/19 | 0.500 | 74% |
| `v3_rag_cot` | 18 | 0.627 | 0.684 | 0.570 | 0/18 | 0.458 | 83% |
| `v4_pg` | 18 | 0.632 | 0.652 | 0.613 | 0/18 | 0.306 | 94% |
| `v5_pg_incident` | 18 | 0.652 | 0.677 | 0.627 | 0/18 | 0.458 | 89% |
| `v6_pg_scenario` | 17 | **0.662** | 0.692 | 0.635 | 0/17 | **0.544** | 88% |

**Twee opvallende, verrassende uitkomsten** (in tegenspraak met de eerdere Track-1/2 Q&A-ablatie,
waar `v1_rag`/`v3_rag_cot` juist de winnaars waren):
- `v1_rag` (kaal RAG, geen CoT) is in **live missie-uitvoering het slechtste geframede model** —
  slechtste aankomstpercentage (26%) én nog steeds 5 onveilige uitkomsten. RAG-fragmenten alleen
  lijken het model in "regel-citeren"-modus te duwen zonder dat het tot een besluit komt.
- `v0_base` (puur de systeemprompt, geen enkele extra) staat qua Composite gewoon **in de kopgroep**
  (0.635, 2e plek) — bijna gelijk aan v3_rag_cot/v4_pg/v5_pg_incident/v6_pg_scenario. Maar zijn
  compliance-score (0.263) en foutieve-draairichting-telling (zie §2) laten zien dat dit hoge cijfer
  deels op geluk berust, niet op correcte regeltoepassing.

---

## 2 — Top 10 problemen (over alle modellen heen, op prevalentie/ernst)

### 1. Gefabriceerde regelcitaties ("fabricated risk") — het meest voorkomende probleem, overal
Het model citeert een COLREG-regel (bijna altijd Rule 15) terwijl de eigen situatie-rapportage al
zegt dat er geen reëel gevaar is (CPA/TCPA te groot, of CPA al gepasseerd). Aantal per config:
`bare_qwen` 197, `v0_base` 102, `v4_pg` 73, `v1_rag` 79, `v2_cot` 72, `v3_rag_cot`/`v5_pg_incident` 64,
`v6_pg_scenario` 45 (laagste, maar nog steeds fors). Voorbeeld (`Imazu01/v0_base`, t=0s): CPA=0.000NM
maar TCPA=1800s (30 minuten) — het model ziet "CPA 0" en trekt meteen Rule 15, zonder TCPA mee te
wegen. Dit is verreweg het grootste, meest systemische probleem.

### 2. Verkeerde draairichting bij give-way-regels (Rule 14/15/16 → toch naar bakboord)
Rule 14/15/16 vereisen (bijna) altijd een stuurboordwending; het model kiest geregeld toch
`turn_left`. Telling: `v0_base` **52**, `v2_cot` 34, `v3_rag_cot` 29, `v4_pg` 25, `v5_pg_incident` 21,
`v6_pg_scenario` 14, `v1_rag` 6, `bare_qwen` 0 (citeert vaak helemaal geen regel). Letterlijk
voorbeeld (`Imazu01/v2_cot`, `v5_pg_incident`, `v6_pg_scenario`, stap 0): *"Rule 15 applies for
head-on situations requiring port-to-port passage"* — dit is dubbel fout: (a) Rule 15 is de
kruisende-regel, niet head-on (dat is Rule 14), en (b) zelfs Rule 14 vereist stuurboord, nooit
bakboord. Dit is precies de "verkeerde kant passeren" die je noemde, en het is het scherpst
aanwezig in `v0_base` — het model met de op-één-na-hoogste Composite-score.

### 3. Zigzagkoersen
`turn_left`/`turn_right` wisselt meerdere keren snel af zonder dat de situatie dat rechtvaardigt.
`v3_rag_cot` **15/18 (83%)**, `v0_base` 14/19 (74%), `v5_pg_incident` 12/18, `v4_pg` 10/18,
`v6_pg_scenario` 9/17, `v2_cot` 8/19, `v1_rag` 2/19, `bare_qwen` 0/19. Voorbeeld (`Imazu01/v2_cot`):
`turn_left, turn_right, turn_left, turn_right, turn_left, turn_right, turn_right, turn_right,
turn_left, ...` — 6 richtingwisselingen in de eerste 12 besluiten. Zigzaggen is dus juist bij de
"betere" geframede configs het ergst, niet bij de kale baseline.

### 4. Fysiek onmogelijke stuurcommando's (>30° gevraagd in één stap)
De systeemprompt vermeldt expliciet een limiet van 30°/stap; toch vraagt het model geregeld meer.
`v2_cot` **65** keer, `v3_rag_cot` 44, `v6_pg_scenario` 42, `v4_pg` 40, `v5_pg_incident` 37,
`v0_base` 29, `v1_rag` 3, `bare_qwen` 0. Voorbeeld: `Imazu01/v0_base` stap 0 vraagt 90°, stap 20
vraagt 32° — in beide gevallen wordt het commando stilzwijgend afgekapt door de simulator, wat
verklaart waarom sommige modellen tóch aankomen ondanks dit "gevaarlijke" verzoek — maar het is nog
steeds een teken dat het model geen correct beeld heeft van het eigen scheepsvermogen.

### 5. RAG-alleen leidt tot handelingsverlamming ("nooit aankomen")
`v1_rag` heeft veruit het laagste aankomstpercentage (26%, 9/19 "did not reach the goal") en blijft
ook het vaakst "hangen" in een snelheidscyclus (zie #6). Regelcitaties zijn hier het minst fout van
alle geframede configs (6 wrong-direction, laagste), maar dat gaat ten koste van voortgang: het
model lijkt voorzichtiger te worden zodra ruwe COLREG-tekstfragmenten in de prompt staan, zonder de
CoT-structuur om dat om te zetten in een besluit.

### 6. Snelheid verlaagd maar nooit blijvend hersteld (eindeloze slow/speed-up-cyclus)
`v1_rag` **11/19** missies eindigen in een oneindige `slow_down ↔ speed_up`-lus zonder de
kruissnelheid echt te herpakken (bv. `Imazu07/v1_rag`: 9 wisselingen op rij). `bare_qwen` 3/19,
`v0_base` 1/19, overige configs 0. Dit is exact het "snelheid wel verlagen maar niet weer
versnellen"-patroon dat je noemde.

### 7. Onveilige uitkomsten / CPA-schending (incl. aanvaringen)
`bare_qwen`: **9/19 (47%) eindigt in een aanvaring** — bijna een muntworp. `v1_rag`: 5/19 (26%),
waarvan sommige ondanks RAG-context. `v2_cot`: 1/19. Alle PG-configs en `v3_rag_cot`: **0/18** —
sinds de eerdere PG-fix (zie repo-memory, 24e pass) is de veiligheidslaag voor CoT/PG-configs
volledig hersteld. `v0_base` ook 0/19, dus alleen framing (geen extra's) is al genoeg om aanvaringen
te vermijden — het echte probleem zit dus niet meer in "botsen", maar in correcte regeltoepassing
(#1/#2) en efficiëntie (#8/#9).

### 8. Te lang redeneren — operationeel onhoudbare latency voor CoT/PG-configs
Gemiddelde latency per besluit: `bare_qwen` 5.7s, `v0_base` 7.7s, `v1_rag` 9.6s — tegenover
`v2_cot` **65s**, `v6_pg_scenario` 58s, `v5_pg_incident` 67s, `v4_pg` 75s, en `v3_rag_cot`
**116s gemiddeld, met een piek van 1971s (ruim 32 minuten) voor ÉÉN enkel besluit**. Elke CoT/PG-
config heeft nagenoeg 100% van zijn besluiten boven de 20s-drempel. Dit is 10–20× trager dan de
baseline voor een compositescore-winst van hooguit +0.08–0.10 — in de huidige vorm onbruikbaar voor
iets dat op interactieve snelheid moet reageren.

### 9. Niet aankomen op tijd / helemaal niet aankomen, en waarom
Buiten aanvaringen (#7) is "FAIL — did not reach the goal" de andere hoofdverdict: `v1_rag` 9/19,
`v2_cot` 4/19, `v3_rag_cot` 3/19, `bare_qwen` 3/19, `v4_pg`/`v6_pg_scenario` 1–2/19. Hoofdoorzaken,
uit de checkpoints: (a) de eindeloze snelheidscyclus (#6), (b) zigzaggen dat de koers per saldo niet
laat vorderen (#3), en (c) te lang redeneren (#8) waardoor te weinig besluit-checkpoints binnen
`max_steps` passen.

### 10. Quiet-missies (UM01/UM02): licht risico op vals-positieve manoeuvres
Slechts 1 van de 147 runs deed ten onrechte iets in een écht risicoloze missie
(`UM02/bare_qwen`: `slow_down` bij CPA 1.62NM/TCPA 89s — buiten de veilige-afstandsdrempel, dus
technisch geen vals alarm, maar wel overdreven voorzichtig). Alle geframede configs (v0–v6) hielden
hier correct koers/snelheid aan. Dit specifieke DTU-paper-risico (kleine modellen die altijd
hetzelfde antwoorden) is dus **niet** het hoofdprobleem hier — de framing zelf werkt goed voor
UM01/UM02; het echte probleem zit in de bearing-sweep-missies (UM03–10) waar de RICHTING/regel
vaak fout is (zie #2).

---

## 3 — Per-model korte diagnose

| Model | Sterktes | Zwaktes | Advies |
|---|---|---|---|
| `bare_qwen` | Snelst (5.7s/besluit) | 47% aanvaringen, 197 gefabriceerde citaties, slechtste aankomst | **Laten vallen** als kandidaat — enkel nog nuttig als absolute vloer-baseline |
| `v0_base` | Beste aankomst (95%), 0 aanvaringen, snel | Meeste wrong-direction-fouten van alle configs (52), lage compliance (0.263), 74% zigzag | **Verder onderzoeken** — sterke basis maar de regelcitatie-laag moet apart aangepakt worden |
| `v1_rag` | Minste wrong-direction (6) | Slechtste aankomst (26%), 26% onveilig, eindeloze snelheidscycli | **Niet los verder testen** — alleen samen met CoT/rerank (zie §4) |
| `v2_cot` | Redelijke Composite | Meeste >30°-verzoeken (65), 65s gemiddelde latency, nog 1 aanvaring | **Verder onderzoeken**, maar latency moet eerst opgelost |
| `v3_rag_cot` | Beste Imazu-score (0.684), 0 aanvaringen | Ergste zigzag (83% van missies!), extreme latency-piek (1971s) | **Verder onderzoeken** — sterk op papier, maar zigzag+latency zijn blokkerend |
| `v4_pg` (merged) | Hoge aankomst (94%) | Zwakste compliance van de PG's (0.306), nog steeds veel fabricated risk (73) | Twijfelgeval — overwegen te laten vallen ten gunste van v5/v6 |
| `v5_pg_incident` | Goede balans (compliance 0.458, composite 0.652) | Nog 21 wrong-direction | **Verder onderzoeken** |
| `v6_pg_scenario` | **Beste Composite (0.662) én beste compliance (0.544)**, minste fabricated-risk (45), minste wrong-direction van de PG's (14) | Nog steeds 42 >30°-verzoeken, 9/17 zigzag | **Sterkste kandidaat — topprioriteit voor vervolgonderzoek** |

---

## 4 — Welke modellen verder onderzoeken, en welke laten vallen

**Verder onderzoeken (in aflopende prioriteit):** `v6_pg_scenario` → `v5_pg_incident` → `v0_base` →
`v3_rag_cot` → `v2_cot`. Dit zijn de 5 configs die zowel 0(–1) aanvaringen als een Composite ≥ 0.57
combineren.

**Laten vallen als losse, verder te onderzoeken kandidaat:**
- `bare_qwen` — dient alleen nog als vloer-referentie, geen doel op zich.
- `v1_rag` (kaal, zonder CoT) — slechtste live-uitvoeringscijfers van alle geframede configs; het
  eerdere Q&A-voordeel van RAG vertaalt zich niet naar veilige/tijdige missie-uitvoering. RAG blijft
  wel waardevol, maar **alleen in combinatie met CoT/reranking**, nooit los.
- `v4_pg` (merged/alle bronnen) — wordt op elk vlak verslagen door `v5_pg_incident`/`v6_pg_scenario`
  afzonderlijk; het samenvoegen van bronnen lijkt de guidance te verdunnen in plaats van te
  versterken.

---

## 5 — 10 concrete aanpassingen die zin hebben

1. **Fix de CPA/TCPA-afweging bij regelcitatie (probleem #1, hoogste prioriteit).** Voeg een
   expliciete tussenstap toe in de prompt/CoT die *eerst* controleert of CPA/TCPA een reëel risico
   vormen vóórdat een regelnummer wordt genoemd — bv. een harde regel: "noem GEEN Rule-nummer als
   CPA boven de veilige afstand ligt, ongeacht wat er verder in de tekst staat."
2. **Voeg een expliciete stuurboord-default toe voor Rule 14/15/16** (probleem #2): een
   harde constraint/validatiestap ("als rule_applied ∈ {14,15,16} en action=turn_left → herzie") —
   dit is het soort fout waar `v10_dpo_contrast`'s in-context-mismatch-voorbeeld al voor bedoeld was,
   maar het treedt nog steeds op in v2/v3/v5/v6 — dat contrastvoorbeeld werkt dus nog niet genoeg.
3. **Zigzag-onderdrukking versterken** (probleem #3): de bestaande "nooit zigzaggen"-instructie in
   `SYSTEM_OOW_AGENT` wordt genegeerd in tot 83% van de `v3_rag_cot`-missies — voeg een expliciete
   "vorige 2 besluiten"-samenvatting toe aan de prompt (het model ziet nu geen geschiedenis).
4. **Cap `degrees` clientside vóór het besluit wordt gelogd** (probleem #4): laat de agent het
   fysieke maximum (30°) al in de prompt herhalen vlak vóór de JSON-output, of valideer/clip
   server-side en voed dat als feedback terug in de volgende stap.
5. **Los de eindeloze snelheidscyclus op** (probleem #6): voeg een expliciete "je vaart nu N%
   onder kruissnelheid, herstel als er geen reëel risico meer is"-regel toe, specifiek getriggerd
   als er 2+ keer achter elkaar `slow_down`/`speed_up` afwisselt zonder netto voortgang.
6. **Beperk de CoT-lengte/`max_new_tokens` hard voor busy missies** (probleem #8): de huidige
   4-stappen-CoT-instructie loopt soms uit tot 1971s — voeg een striktere tokenlimiet per stap toe
   of dwing een kortere `<think>`-batterij af (bv. max 150 tokens redenering) i.p.v. de huidige losse
   grens.
7. **Voeg PG-guidance NIET meer toe als aparte configs, maar als context binnen RAG of CoT**
   (rechtstreeks antwoord op je vraag "waar past PG het beste"): PG's toegevoegde waarde
   (beste compliance-score, minste fabricated-risk) lijkt vooral te komen van *extra grondtekst*, wat
   qua rol identiek is aan RAG-fragmenten — voeg PG-guidance toe als een extra input naast de
   RAG-chunks in dezelfde group, in plaats van als 3 aparte ablatie-armen.
8. **Onderzoek waarom `v1_rag` slechter presteert dan `v0_base` in live uitvoering** — dit is
   tegengesteld aan de eerdere Q&A-ablatie en verdient een gerichte kleine studie (bv. lengte van de
   RAG-chunks, of ze concrete actiegerichte taal bevatten of alleen regeltekst) vóórdat RAG blind
   wordt meegenomen in de nieuwe hoofdgroep.
9. **Voeg de bestaande SFT/DPO/Reflectie-trainingsdata toe als in-context voorbeelden** (zoals
   `v9_fewshot`/`v10_dpo_contrast`/`v11_reflect` al prototypen, maar dan gevoed met échte
   `Data/OOW/OOW_Agents_Training/oow_sft_*.jsonl` / `oow_dpo_pairs.jsonl` / `oow_reflection.jsonl`
   voorbeelden in plaats van de 3 handgeschreven voorbeelden nu) — dit test rechtstreeks of de data
   die je voor een toekomstige fine-tune hebt verzameld al via prompting waarde toevoegt, vóórdat er
   GPU-uren in een echte SFT/DPO/Reflectie-training gaan.
10. **Bouw een lichte, deterministische "sanity-laag" rond de agent** (client-side, geen extra
    modelcall) die vóór het uitvoeren van een besluit checkt: (a) rule_applied ∈{14,15,16} → action
    moet turn_right/hold_course zijn, (b) CPA boven veilige afstand → rule_applied moet "none" zijn,
    (c) degrees ≤ fysieke limiet. Dit lost niet het onderliggende redeneerprobleem op, maar voorkomt
    dat de 3 grootste, meest voorkomende fouten (#1/#2/#4) ooit het schip echt aansturen — een
    pragmatische vangnet-oplossing die los staat van welk promptontwerp/model je uiteindelijk kiest.

---

## 6 — Voorstel: vervolg-testmatrix (minder combinaties, gerichter)

In plaats van 8 losse configs, stel ik 4 hoofdgroepen voor zodat elke vervolgronde (na een
aanpassing uit §5) met een beheersbaar aantal runs opnieuw getest kan worden:

| Hoofdgroep | Samenstelling | Doel |
|---|---|---|
| **A — Baseline** | `v0_base` (systeemprompt, geen extra's) | Referentiepunt voor elke vervolgronde |
| **B — RAG+KG+Rerank** | `v1_rag` + KG-retrieval (`pipeline/ingest/build_kg.py`, al beschikbaar) + de fine-tuned reranker (`v7_rerank`-prototype) samengevoegd tot ÉÉN config | Test of de volledige retrieval-stack (niet RAG alleen) het probleem uit §2/§5.8 oplost |
| **C — CoT (incl. gerichte prompts)** | `v2_cot`, mét de aanpassingen uit §5 punt 2/3/4/6 (stuurboord-default, geschiedenis, degree-cap, tokenlimiet) verwerkt in de CoT-instructie zelf | Test of CoT nuttig wordt zodra de bekende faalpatronen expliciet in de instructie worden geadresseerd |
| **D — PG als context binnen B of C** | PG-guidance (`v6_pg_scenario`'s bron, de sterkste van de 3) toegevoegd als extra contextblok in ZOWEL groep B als groep C (2 varianten: B+PG, C+PG) i.p.v. 3 aparte v4/v5/v6-armen | Beantwoordt direct je vraag "waar past PG het beste" — door het in beide te testen i.p.v. te gokken |

Dit brengt de matrix terug naar **A, B, C, B+PG, C+PG = 5 configs** in plaats van 8, met PG's
plaats empirisch bepaald in plaats van vooraf aangenomen. `bare_qwen` en `v4_pg`/`v1_rag`-los
vervallen als aparte armen (zie §4).

### Waar de SFT/DPO/Reflectie-trainingsdata in past
Zoals gevraagd: eerst **in-context** (geen training), pas daarna een echte fine-tune:
1. **Stap 1 (nu voorgesteld, nog geen training):** voeg voorbeelden uit
   `Data/OOW/OOW_Agents_Training/oow_sft_direct.jsonl`/`oow_sft_cot.jsonl` (en de incident-/scenario-
   varianten) toe als few-shot-voorbeelden binnen groep **C (CoT)** — dat is waar `v9_fewshot` al een
   prototype voor bouwde, nu gevoed met echte gemineerde voorbeelden i.p.v. 3 handgeschreven.
   `oow_dpo_pairs.jsonl`/`oow_incident_dpo_pairs.jsonl` (fout-vs-correct contrastparen) passen om
   dezelfde reden ook het beste bij groep **C**, niet bij B — ze zijn per definitie
   redenerings-contrast, geen retrieval-content. `oow_reflection.jsonl` past bij een vijfde,
   optionele variant (C+reflect) als losse zelfcontrole-stap bovenop C.
2. **Stap 2 (pas na Stap 1's resultaten):** als een van deze in-context-varianten een duidelijke,
   reproduceerbare verbetering laat zien op de problemen uit §2 (met name #1/#2/#3), is dát het
   signaal om de bijbehorende trainingsdata daadwerkelijk te gebruiken voor een echte
   SFT/DPO/Reflectie-fine-tune (op de cloud, zoals gebruikelijk in dit project) — en pas die
   gefinetunede varianten dan te evalueren tegen dezelfde A/B/C/B+PG/C+PG-hoofdgroepen op deze
   simulatie-missies, niet alleen op de bestaande Track 1/2 Q&A-evaluatie.

---

## 7 — Kanttekeningen bij deze analyse

- De sweep was **147/152 (~97%) compleet** op het moment van analyse (UM10 voor v3–v6 liep nog) —
  de cijfers voor `v3_rag_cot`/`v4_pg`/`v5_pg_incident`/`v6_pg_scenario` zijn dus op n=17–18 i.p.v.
  n=19; de conclusies zijn stabiel genoeg om op te handelen, maar een korte herbevestiging na
  afronding van de sweep kost niets.
- "Compliance"-schendingen (§2 #1/#2) komen uit een deterministische ground-truth-check
  (CPA/TCPA-gebaseerd), niet uit een LLM-judge — dit is dus een betrouwbare, reproduceerbare
  telling, geen subjectieve beoordeling.
- Er is bewust **geen code aangepast** in deze sessie — dit document is uitsluitend een analyse en
  voorstel, zoals gevraagd.
