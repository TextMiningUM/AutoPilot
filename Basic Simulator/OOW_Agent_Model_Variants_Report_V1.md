# OOW-agent: modelvarianten in de missie-simulator — technisch referentierapport

Doel van dit document: **precies** vastleggen wat er vandaag (2026-09-22) in `Basic Simulator/app/agents.py` en de bijbehorende pipeline-modules gebeurt voor elke config die in de missie-sweeps wordt gebruikt (`bare_qwen` t/m `v6_pg_scenario`), plus de 3 "super"-configs die de daadwerkelijke experimentmatrix-promptkolommen vormen (`v7_super_rag`/`v8_super_cot_pg`/`v9_super_all`, die later op 2026-09-22 de nooit-gedraaide `v7_rerank`/`v8_rerank_cot`/`v9_fewshot`/`v10_dpo_contrast`/`v11_reflect`-prototypeslots vervingen), en de echte SFT/DPO/Reflectie-fine-tune. Alles hieronder is direct uit de broncode en de data-bestanden zelf gehaald (geen aannames) — bestandspaden en regelnummers staan erbij zodat dit reproduceerbaar is.

**Belangrijkste feit vooraf, dat de rest van dit rapport kadert:** er is maar **één** taalmodel-gewichtenset in de hele missie-simulator: de kale, niet-gefinetunede `Qwen/Qwen3-8B` (4-bit NF4), geladen door `_load_qwen()` in [Basic Simulator/app/agents.py](Basic%20Simulator/app/agents.py#L292). Er wordt **nergens** in `agents.py` een LoRA-adapter of gefinetuned checkpoint geladen (geen `PeftModel`-import, geen `.from_pretrained(..., adapter)` aanroep). Elke config — ook de "super"-configs — is dus uitsluitend een **prompt-variant** op exact dezelfde onderliggende gewichten ("gewichten" wordt een aparte as zodra fase F4 het checkpoint-laadpad bouwt; zie sectie 3). Dat is expliciet zo ontworpen (zie de docstring bovenaan `agents.py`): eerst goedkoop testen via prompting of retrieval/CoT/PG iets oplevert, vóórdat er cloud-GPU-uren in een echte fine-tune gaan.

***

## 1 — Gedeelde technische basis (geldt voor ALLE 11 configs)

| Onderdeel                | Waarde                                                                                                                                                                   | Bron                                                                              |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Basismodel               | `Qwen/Qwen3-8B`, 4-bit NF4 (bitsandbytes, dubbele quantisatie, bf16-rekenprecisie), `attn_implementation="sdpa"`                                                         | `agents.py` `_load_qwen()`, [L292-306](Basic%20Simulator/app/agents.py#L292-L306) |
| Device                   | 1 GPU, hard pinned (`device_map={"": 0}`), géén CPU-offload                                                                                                              | `agents.py` `_load_qwen()`                                                        |
| Context-venster (prompt) | tokenizer-truncatie op **max_length=4096 tokens** (`tok(text, ..., truncation=True, max_length=4096)`)                                                                   | `agents.py` `_generate()`, [L\~730](Basic%20Simulator/app/agents.py)              |
| Sampling                 | **greedy, deterministisch** — `do_sample=False`, `temperature=1.0`, `top_p=1.0` (die twee zijn irrelevant zolang `do_sample=False`)                                      | `agents.py` `_generate()`                                                         |
| Herhaling-onderdrukking  | `repetition_penalty=1.15` (géén `no_repeat_ngram_size` — bewust teruggedraaid, zie code-comment: blokkeerde ook correcte herhaalde getallen en gaf soms CJK-tekens)      | `agents.py` `_generate()`                                                         |
| Stopcriterium            | `stop_strings="\"}"` — stopt zodra de JSON-`reasoning`-waarde sluit, i.p.v. altijd het volledige tokenbudget te verbruiken                                               | `agents.py` `_generate()`                                                         |
| Retrieval-embedder       | `BAAI/bge-base-en-v1.5` (768d), CPU-only, asymmetrische queryprefix `"Represent this sentence for searching relevant passages: "`                                        | `core/embedding.py`                                                               |
| JSON-schema (antwoord)   | Elke config (behalve fewshot-voorbeelden zelf) moet exact teruggeven: \`{"action": "...", "degrees": \<float                                                             | null\>, "rule_applied": "...", "reasoning": "1-2 zinnen"}\`                       |
| Parsing                  | Balanced-brace scanner pakt het **laatste** geldige `{...}`-object met een `"action"`-sleutel (Qwen3 herhaalt soms de JSON 2x rond een `<think>`-blok)                   | `agents.py` `_extract_json_objects()`/`_parse_json_action()`                      |
| Evaluatie                | `Evaluation Functions/evaluate_run.py` (ongewijzigd hergebruikt) — hard gate op veiligheid (CPA/aanvaring), dan gewogen compliance/temporal/spatial/manoeuvre/smoothness | `app/evaluation.py`                                                               |

***

## 2 — De 8 configs die daadwerkelijk in de huidige sweep draaien (`bare_qwen` … `v6_pg_scenario`)

Alle 8 worden gedefinieerd in `_CONFIG_SPECS` in [Basic Simulator/app/agents.py](Basic%20Simulator/app/agents.py) als een simpele dict van booleans/strings (`bare`, `rag`, `rerank`, `cot`, `pg`) — `build_oow_prompt()` bouwt de uiteindelijke prompt puur door deze vlaggen af te lopen, dus onderstaande tabel is letterlijk wat er per config aan- of uitstaat. **Sinds 2026-09-22 zijn dit een gearchiveerde ablatie-arm** (147 runs hangen eraan; blijven reproduceerbaar, maar staan niet meer in de standaard-sweep se `--configs`-default — zie sectie 3):

| Config           | `rag` | `cot` | `pg`         | `enable_thinking` (effectief) | `max_new_tokens` (effectief) |
|------------------|-------|-------|--------------|-------------------------------|------------------------------|
| `bare_qwen`      | –     | –     | –            | false                         | 256 (caller-default)         |
| `v0_base`        | –     | –     | –            | false                         | 256                          |
| `v1_rag`         | ✓     | –     | –            | false                         | 256                          |
| `v2_cot`         | –     | ✓     | –            | **true** (geforceerd)         | **3072** (geforceerd)        |
| `v3_rag_cot`     | ✓     | ✓     | –            | **true**                      | **3072**                     |
| `v4_pg`          | –     | –     | `"merged"`   | **true**                      | **3072**                     |
| `v5_pg_incident` | –     | –     | `"incident"` | **true**                      | **3072**                     |
| `v6_pg_scenario` | –     | –     | `"scenario"` | **true**                      | **3072**                     |

De "geforceerd"-kolom komt uit `effective_generation_params()` ([agents.py L\~635](Basic%20Simulator/app/agents.py)): **elke** `cot`- of `pg`-config negeert de meegegeven `enable_thinking`/`max_new_tokens` en dwingt altijd `enable_thinking=True` + minimaal 3072 tokens af — ook PG-configs, want zonder Qwen3's `<think>`-kanaal negeerde het model de PG-tekst gewoon (zie code-comment: "PG decisions were near-identical to v0_base's, same failures" vóór deze fix).

### 2.1 — `bare_qwen`: absolute vloer-baseline

-   **Systeemprompt (volledig, woordelijk):**

```
You are an AI assistant helping a ship's navigation system decide on a heading/speed change. Reply with ONLY a JSON object, no other text:
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": <float, only for turn_left/turn_right>,
 "rule_applied": "<e.g. Rule 15, or 'none' if no rule applies>",
 "reasoning": "<one or two sentences>"}
```

(`BARE_SYSTEM`, [agents.py L119-124](Basic%20Simulator/app/agents.py#L119-L124))

-   **User-bericht:** alleen de kale `narrate()`-situatietekst (numeriek: posities/koersen/ CPA/TCPA) + "Recommend exactly ONE manoeuvre...". Geen COLREG-framing, geen fysieke limieten, geen mission-context.
-   Dit is de **enige** config met een compleet andere system prompt — bedoeld om te meten hoeveel de COLREG-framing zelf waard is.

### 2.2 — `v0_base`: framing zonder extra's

-   **Systeemprompt:** `SYSTEM_OOW_AGENT`, de gedeelde prompt voor v0–v6 (en v7–v11), volledig:

```
You are the navigator on a large commercial vessel. Decide the next helm order.

MISSION REQUIREMENTS -- this mission is only completed successfully if ALL of these hold, not just
the first one you happen to satisfy:
1. Every contact's CPA stays at or above this mission's safe passing distance (given further below)
   at all times -- exactly as mandatory as actually reaching the goal. There is no automatic safety
   net correcting your choice if you get this wrong: your own action each step is what the ship
   actually does.
2. Every manoeuvre you take while a real collision risk exists complies with COLREG.
3. You reach the mission goal.

PRIORITY ORDER when these pull in different directions -- always in this order, never reversed:
1. Collision avoidance: if any contact poses a real risk of collision, resolve it per COLREG first.
2. Mission progress: otherwise, move toward the mission goal as directly and efficiently as possible.

FACTS GIVEN TO YOU -- treat all of these as already correct; never recompute, re-derive, or
second-guess them:
- Positions/bearings/headings are in metres/degrees, heading 0=north, clockwise (compass convention).
- rel.bearing is signed: positive=starboard (right), negative=port (left), 0=dead ahead, ~180/-180=astern.
- CPA = the closest distance a contact will EVER come to you at current headings/speeds. TCPA = seconds
  until that closest point.
- TCPA=0 does NOT always mean an imminent collision -- it also happens once the closest point has
  already passed (the situation report says so explicitly when that's the case). Judge real risk from
  CPA alone, never from TCPA alone.
- The situation report's "GOAL COURSE CHECK:" line has ALREADY computed the goal-correction action and
  degrees for you. Never substitute a contact's rel.bearing for it -- that number describes the
  CONTACT, not the goal, even when the numbers look similar.

DECISION PROCEDURE -- follow in order:
1. Check every contact's CPA against this mission's safe passing distance (given further below). If
   none are below it, there is no real collision risk right now -- go to step 3.
2. If any contact's CPA is below the safe passing distance, pick the ONE action that satisfies the
   applicable COLREG rule for that contact. This step overrides everything below it.
3. Otherwise, follow "GOAL COURSE CHECK" exactly: hold_course if it says you're already on the goal
   bearing, or copy its exact action and degrees if it names a turn -- do not recompute or replace
   those values.
4. Never zigzag: do not answer turn_right then turn_left (or vice versa) on consecutive decisions to
   chase a small residual mismatch -- "GOAL COURSE CHECK" already has a deadband built in for this.
5. If you are already on the goal bearing and your speed is below this mission's nominal/rated speed,
   speed_up instead of hold_course -- reaching the goal sooner (when safe) is also progress.

Ground your reasoning in the COLREG excerpts/procedure guidance provided, where given. Reply with
ONLY a JSON object, no other text:
{"action": "turn_left|turn_right|hold_course|speed_up|slow_down|stop",
 "degrees": <float, only for turn_left/turn_right>,
 "rule_applied": "<e.g. Rule 15, or 'none' if no rule applies>",
 "reasoning": "<one or two sentences>"}
```

(`SYSTEM_OOW_AGENT`, [agents.py L143-186](Basic%20Simulator/app/agents.py#L143-L186) — runtime aanpasbaar via de sidebar-popover, geldt voor elke config behalve `bare_qwen`.)

-   **User-bericht:** fysieke-limieten-zin (turn-rate/max rudder/versnelling/veilige afstand, alleen als `constraints` is meegegeven) + `narrate()`-situatietekst + de vaste slotinstructie. Géén RAG, géén CoT-instructie, géén PG.
-   **Voorbeeld situatietekst** (echt, uit `Imazu01__v0_base__units_v1.json`, stap 0):

```
Own-ship's physical limits: heading changes at most 3.0 deg/s (~30 deg per 10s step). A single
turn_left/turn_right command can request AT MOST 30 degrees -- a larger request will be silently
capped, so a course change bigger than that needs several separate turn commands across multiple
steps, not one big one. Speed is capped at 10.0 m/s, changing gradually (0.20 m/s² up / 0.20 m/s²
down) -- speed_up/slow_down are not instant. This mission's safe passing distance is 500m: CPA
below that is a real collision risk, CPA well above it is safe regardless of how small it looks.

Situation:
Own-ship at (0.000, -6.000) NM, heading 0.0, speed 12.00 kt (nominal/rated speed for this mission:
12.00 kt).
Mission goal at (0.000, 6.000) NM, 12.000 NM away, bearing 0.0 deg.
GOAL COURSE CHECK: heading is ALREADY on the goal bearing (within 10 deg) -- no turn needed for
the goal.
At current speed, ETA to goal ≈ 3600s if heading straight there.
1 other ship:
  - Ship named "ts1": range 12.000 NM, rel.bearing 0.0 deg, heading 180.0, speed 12.00 kt, CPA
    0.000 NM, TCPA 1800s

Recommend exactly ONE manoeuvre as the specified JSON object.
```

### 2.3 — `v1_rag`: + retrieved COLREG-excerpts

-   **Retrieval-techniek:** `kg_retrieve()` in [pipeline/ingest/build_kg.py L233-275](pipeline/ingest/build_kg.py#L233-L275) — **hybride dense + concept-graaf**:
    1.  Dense pool: cosine-similarity tussen de bge-embedding van de situatietekst en alle chunk-embeddings, top-`dense_n` kandidaten.
    2.  Concept-boost: trefwoorden/alias-match (`_OOW_ALIASES`, o.a. "cpa"/"tcpa"→`cpa_tcpa`, "give-way", "head-on", ...) geeft +0.15 aan chunks die het concept letterlijk bevatten, +0.08 aan chunks die een **co-occurrerend** concept bevatten (top-5 sterkste buren).
    3.  Top-`k` van die gecombineerde pool wordt behouden.
-   **Parameters zoals daadwerkelijk gebruikt in de sweep:** `k=2` (CLI-default in `run_llm_scenario.py`/`sweep_llm_params.py`), `dense_n=40` (`build_oow_prompt()`'s eigen default — er is **geen** CLI-vlag om dit te overschrijven, dus dit staat in de huidige sweep altijd vast op 40).
-   **Formattering in de prompt:** `format_context()` in [pipeline/eval/prep_ablation.py L126-133](pipeline/eval/prep_ablation.py#L126-L133):

```
[Excerpt 1 — {source_file} / {chapter_title}]
{chunk text}

[Excerpt 2 — ...]
...
```

-   **Voorbeeld van een echte chunk** (`chunk_94bba771`, uit `Data/OOW/OOW_Agents_Training/oow_rag_chunks.json`):

```json
{
  "chunk_id": "chunk_94bba771",
  "source_file": "COLREG-Consolidated-2018.pdf",
  "chapter_title": "Convention Articles",
  "text": "Convention on the International Regulations for Preventing Collisions at Sea, 1972 ... The Parties to the present Convention undertake to give effect to the Rules and other Annexes ..."
}
```

(Corpus: uitsluitend `COLREG-Consolidated-2018.pdf`, geparsed door `pipeline/ingest/build_oow_json.py` — géén incident-rapporten of scenario-data in déze index; dat gaat via PG, zie §2.5.)

-   **Belangrijk gemeten resultaat (zie het aparte missie-analyserapport):** `v1_rag` is in live missie-uitvoering juist de **slechtst presterende geframede config** (26% aankomstpercentage) — RAG-fragmenten alleen lijken het model in "regels citeren"-modus te duwen zonder CoT om dat om te zetten in een besluit.

### 2.4 — `v2_cot`: + chain-of-thought instructie

-   **Extra user-content,** `COT_INSTR` ([agents.py L188-197](Basic%20Simulator/app/agents.py#L188-L197)), woordelijk:

```
Before answering, write your reasoning as EXACTLY these 4 steps, one short sentence each --
no more steps, no re-deriving bearings/CPA/TCPA (they are already given -- just quote them):
1. Contacts: name each contact and say whether its CPA is below or above the safe passing
distance.
2. Rule: for any contact below it, name the applicable COLREG rule and the action it requires.
3. Goal: if no contact is below the safe passing distance, state what GOAL COURSE CHECK says.
4. Decision: state the one action you will take and why.
Then give the final JSON answer.
```

-   Geen RAG, geen PG. `enable_thinking=True` + `max_new_tokens=3072` geforceerd (zie §2 boven) zodat de 4-stappenredenering ergens naartoe kan (Qwen3's verborgen `<think>`-kanaal).
-   **Gemeten prijs:** gemiddeld 65s/besluit (tegen 6-10s voor bare/v0/v1), pieken tot 248s.

### 2.5 — `v3_rag_cot`: v1 + v2 gecombineerd

-   Simpelweg beide bovenstaande blokken na elkaar in het user-bericht: eerst `COT_INSTR`, dan de RAG-excerpten, dan de situatietekst. Geen nieuwe techniek t.o.v. §2.3/2.4.
-   **Gemeten prijs:** duurste config van alle 8 — gemiddeld **116s/besluit**, piek **1971s** (ruim 32 minuten voor één besluit) in de huidige sweep-data.

### 2.6/2.7/2.8 — `v4_pg` / `v5_pg_incident` / `v6_pg_scenario`: + procedural-graph guidance

-   **Wat een Procedural Graph (PG) is:** een graaf van genormaliseerde procedurestappen (`pg_NNNN`-nodes met een `label`, `n_occurrences`, `families`) + gerichte edges met `support`/`condition`/`pitfalls`, gebouwd door `pipeline/ingest/build_pg.py` uit reasoning-traces (COLREG-regeltekst, echte ongevalsrapporten, of Track-2-scenario's). Voorbeeld-node (`Data/OOW/OOW_Agents_Training/oow_pg_scenario.json`):

```json
"pg_0000": {
  "label": "Assess the fused contact picture (bearing/range/CPA/TCPA) against own-ship's course and speed.",
  "n_occurrences": 390, "families": {"colreg_encounter": 390}, "n_starts": 390
},
"pg_0001": {"label": "Determine own-ship's role: give way.", "n_occurrences": 360},
"pg_0002": {"label": "Execute the action: alter_course (+30 degrees to starboard, lookahead distance 200 m).", "n_occurrences": 250}
```

-   **De 3 varianten verschillen uitsluitend in welk graafbestand geladen wordt:** `oow_pg.json` (merged, alle bronnen) voor `v4_pg`, `oow_pg_incident.json` (alleen echte MAIB/NTSB-ongevalsrapporten) voor `v5_pg_incident`, `oow_pg_scenario.json` (alleen synthetische Track-2-scenario's) voor `v6_pg_scenario` — zie `_CONFIG_SPECS`'s `pg="merged"/"incident"/"scenario"` sleutel.
-   **Retrieval van de guidance:** `_pg_match_query()` ([agents.py L385-408](Basic%20Simulator/app/agents.py#L385-L408)) bouwt een KORTE zoekfrase uit `narrate.contact_line()`'s AL BEREKENDE `encounter`/`rules`-velden (bv. *"crossing target on port encounter, applicable Rule 15 and Rule 16, give-way/stand-on obligations"*) — deze frase wordt NOOIT aan het model getoond, alleen gebruikt om `render_guidance()` de juiste graafpad te laten vinden (embedding-match tegen node-labels, drempel `ANCHOR_THRESH=0.65`). `render_guidance()` (`pipeline/ingest/pg_guidance.py` L143-177) loopt vanaf het best-matchende ankerpunt het sterkst-ondersteunde pad af (max 6 stappen) en rendert:

```
Procedure guidance (typical colreg encounter sequence from the procedure graph):
  1. Assess the fused contact picture (bearing/range/CPA/TCPA) against own-ship's course and speed.
  2. Determine own-ship's role: give way.
  3. Execute the action: alter_course (+30 degrees to starboard, lookahead distance 200 m).
Applies when: ...
Avoid: ...
Follow this order where applicable, but adapt to the specific situation.
```

-   Ook deze 3 configs krijgen `COT_INSTR` (de "4-stappen"-instructie, zie §2.4) áls extra user-content — PG-configs hebben dus ook een expliciete redeneerstructuur, niet alleen de procedure-tekst.
-   **Status/kwaliteitshistorie:** deze 3 configs waren tot recent (24e pass, zie repo-memory) stuk — scoorden identiek aan `v0_base` (\~0.14, vrijwel altijd aanvaringen) omdat (a) `enable_thinking` niet geforceerd werd voor `pg`-configs, en (b) de retrieval- query de rauwe numerieke `narrate()`-tekst gebruikte i.p.v. de COLREG-vocabulaire, dus altijd op één generieke "algemeen"-node terechtkwam. Beide zijn gefixt; sindsdien presteren alle 3 configs het best van de 8 (zie het aparte missie-analyserapport).

***

## 3 — De experimentmatrix-promptkolommen: `v7_super_rag` / `v8_super_cot_pg` / `v9_super_all`

**Belangrijke correctie t.o.v. de eerdere versie van dit rapport (2026-09-22, zelfde dag):**
de oorspronkelijke `v7_rerank`/`v8_rerank_cot`/`v9_fewshot`/`v10_dpo_contrast`/`v11_reflect`
prototypeslots die hier eerder beschreven stonden **zijn nooit gedraaid** — een grep over
`Basic Simulator/Data/missions/**llm_runs/**` (inclusief alle archieven/tags) leverde **0**
resultaatbestanden met een van die 5 namen op, dus er ging geen data verloren door ze
opnieuw te definiëren. Ze zijn vervangen door de 3 configs hieronder, die de daadwerkelijke
promptkolommen (P1/P2/P3) van de experimentmatrix zijn. De handgeschreven regelkennis-
voorbeelden die de oude `v9_fewshot`/`v10_dpo_contrast` gebruikten (bv. *"Rule 14 requires
BOTH vessels to alter to STARBOARD, never port"*) zijn volledig uit de code verwijderd —
dat soort in-prompt regelkennis is precies wat het projectprincipe verbiedt (regelkennis
komt uit data, niet uit prompts).

| Config             | `rag` | `rerank` | `cot` | `pg`                  | `enable_thinking` (effectief) | `max_new_tokens` (effectief) |
|--------------------|-------|----------|-------|-----------------------|-------------------------------|-------------------------------|
| `v7_super_rag`     | ✓     | ✓        | –     | –                     | false (blijft snel)           | 256 (caller-default)          |
| `v8_super_cot_pg`  | –     | –        | ✓     | `"scenario+incident"` | **true** (geforceerd)          | **3072** (geforceerd)          |
| `v9_super_all`     | ✓     | ✓        | ✓     | `"scenario+incident"` | **true**                       | **3072**                       |

### 3.1 — `v7_super_rag`: de volledige retrieval-stack op corpus v2, geen CoT/PG

-   Dense RAG (`kg_retrieve()`) op de herbouwde corpus v2 (bge-large-en-v1.5, 1024d, CHIRP +
    Leo MOOS-cases + marginale incidenten toegevoegd) + herscoring door de fine-tuned
    cross-encoder (`rerank_hits()`, [build_kg.py](pipeline/ingest/build_kg.py)) + KG-concept-boost.
    Géén CoT-instructie, géén PG-blok — bewust gehouden op v0_base-achtige latency
    (`enable_thinking` blijft `False` tenzij de aanroeper het expliciet aanzet).
-   De reranker (`_models/OOW/oow_reranker/`, `cross-encoder/ms-marco-MiniLM-L6-v2` basis,
    ~22M parameters, CPU-only) is opnieuw getraind op corpus v2 op 2026-09-22:
    `dev_accuracy_at_1` 0.768 → 1.000 (n=138 evaluatie-queries) — zelfde eindscore als de
    eerdere training op de oude corpus, dus de herscoring blijft even sterk na de
    corpus-uitbreiding.

### 3.2 — `v8_super_cot_pg`: CoT + procedure-guidance uit BEIDE grafen, geen retrieval

-   `COT_INSTR` (4-stappen redeneertemplate) + procedure-guidance uit de herbouwde
    `oow_pg_scenario.json` **én** `oow_pg_incident.json` **samen** — géén dense/reranked
    RAG-excerpten. `oow_pg_rule.json` zit hier expliciet NIET in (blijft een losse,
    optionele test-arm).
-   Nieuw laadpad: `pipeline/ingest/pg_guidance.py`'s `load_merged_pg()` laadt beide
    PG-bestanden en merget ze tot één `ProceduralGraph` — beide bronbestanden gebruiken
    onafhankelijk hetzelfde `"pg_0000", "pg_0001", ...`-idschema, dus elke node-id wordt
    genamespaced per bronindex (`"0:pg_0000"` voor scenario, `"1:pg_0000"` voor incident)
    vóór het mergen, anders zouden nodes uit de twee bronnen elkaar overschrijven.
    Geverifieerd (`tests/test_pg_guidance_merge.py`): het gemergede object heeft precies
    `len(scenario.nodes) + len(incident.nodes)` = 7 + 126 = 133 nodes, geen enkele
    node-id-botsing, en een query op elke bron z'n eigen labeltekst matcht terug naar een
    node in de juiste namespace (score 0.89-0.95).

### 3.3 — `v9_super_all`: v7_super_rag + v8_super_cot_pg gecombineerd

-   Precies de optelsom van beide hierboven: volledige retrieval-stack (RAG+rerank+KG) ÉN
    CoT + scenario+incident PG-guidance in dezelfde prompt. Zowel `cot` als `pg` staan aan,
    dus `effective_generation_params()` forceert hier ook `enable_thinking=True` +
    `max_new_tokens>=3072`.

***

## 4 — De ECHTE SFT/DPO/Reflectie fine-tune: apart traject, ook nog niet in de simulator

Dit is een **volledig ander stuk code** dan v7-v9 hierboven (die zijn allemaal in-context/prompting-only) — dit traject traint daadwerkelijk LoRA-gewichten. **Belangrijk: gewichten zijn geen configs** — welk checkpoint antwoordt (W0 base / W1 SFT / W2 SFT+DPO / W3 SFT+DPO+Reflectie) is een aparte as van welke prompt gebruikt wordt (`v0_base`...`v9_super_all`), geselecteerd via het (nog te bouwen, fase F4) checkpoint-laadpad. De run-bestandsnaam draagt beide assen apart: `{mission}__{config}__{weights}__{tag}.json` (bv. `Imazu04__v8_super_cot_pg__W0_base__units_v2.json`); voor nu is `weights` altijd `"W0_base"`, aangezien dit laadpad nog niet bestaat.



| Stage     | Script                               | Basismodel + quantisatie              | LoRA                                                                    | Belangrijkste hyperparameters                                                                                                        | Trainingsdata (voorbeeld)                                                                                                                                                                                                |
|-----------|--------------------------------------|---------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SFT       | `pipeline/train/train_sft.py`        | `Qwen/Qwen3-8B`, 4-bit NF4            | r=16, alpha=32, dropout=0.05, target=`q/k/v/o_proj`+`gate/up/down_proj` | epochs=1, batch=1×grad_accum16=16 effectief, lr=2e-4, cosine+30 warmup-steps, `max_seq_length=1024`, loss alleen op assistant-tokens | `oow_sft_direct.jsonl`/`_cot`/`_rag`/`_multihop`, `oow_incident_sft_*`, `oow_scenario_sft_*`, `oow_scenario_Leo_sft_*`, `oow_pg_sft.jsonl`, `oow_pg_incident_sft.jsonl` (13 bestanden, zie §12/12.5/12.6 in de notebook) |
| DPO       | `pipeline/train/train_dpo.py`        | idem, laadt de SFT-LoRA als startpunt | r=8, alpha=16                                                           | epochs=0.5, lr=5e-5, **beta=0.05** (bewust laag — beta=0.1 gaf eerder hallucinatie, Faithfulness 0.36→0.02)                          | `oow_dpo_pairs.jsonl`, `oow_incident_dpo_pairs.jsonl`, `oow_scenario_dpo_pairs.jsonl`, `oow_scenario_Leo_dpo_pairs.jsonl` (4 bestanden)                                                                                  |
| Reflectie | `pipeline/train/train_reflection.py` | idem, laadt SFT+DPO-LoRA              | r=8, alpha=16                                                           | (zelfde schema als DPO-config)                                                                                                       | `oow_reflection.jsonl`, `oow_incident_reflection.jsonl`, `oow_scenario_reflection.jsonl`, `oow_scenario_Leo_reflection.jsonl` (4 bestanden)                                                                              |

**Voorbeeld van een echte SFT-rij** (`oow_sft_direct.jsonl`, 1e regel):

```json
{"messages": [
  {"role": "system", "content": "You are the Officer of the Watch, an AI navigation agent responsible for COLREG-compliant collision avoidance. Answer accurately, cite the correct COLREG rule number(s), and follow the give-way/stand-on obligations exactly. Be concise."},
  {"role": "user", "content": "are the consequences of neglecting COLREG compliance?"},
  {"role": "assistant", "content": "Vessels and their personnel are responsible for compliance with COLREG rules, and special circumstances may require deviation from standard rules to avoid danger."}
]}
```

**Voorbeeld van een echt DPO-paar** (`oow_dpo_pairs.jsonl`, perturbatie `drop_regulation` — het "rejected"-antwoord laat bewust één regelverwijzing weg):

```json
{"prompt": [...], 
 "chosen":  [{"role":"assistant","content":"... This follows COLREG 1972. Be careful: neglecting to comply with the Rules may lead to consequences. Done correctly, ensures safety by adhering to navigation rules."}],
 "rejected":[{"role":"assistant","content":"... Be careful: neglecting to comply with the Rules may lead to consequences. Done correctly, ensures safety by adhering to navigation rules."}]}
```

**Voorbeeld van een echte reflectie-rij** (`oow_reflection.jsonl`, `reflection_mode: skip_proword` — de "Draft" mist bewust een verplicht proword, de "Critique"/"Refined" herstellen het):

```
Draft: Vessels must maintain a safe speed to avoid collisions. Proceed at a safe speed.
Critique: Reviewing the Draft: it omits the required prowords (fishing vessel, safe speed). That is safety-critical and must be included.
Refined: Vessels must maintain a safe speed to avoid collisions. Proceed at a safe speed. Use the prowords fishing vessel, safe speed as appropriate. ...
```

### 4.1 — Status: WEL al getraind en geëvalueerd (Track 1/2 Q&A) — maar NIET in de missie-simulator

Volgens `full report evaluations V1.md` (§3) bestaat de samengevoegde SFT+DPO+Reflectie- checkpoint (`oow_qwen_full`) al, en is die al geëvalueerd op de Track 1/2 Q&A-eval: Track 2 (toegepaste besluiten) verbeterde sterk (Composite 0.436→0.574, DirectionCorrect 0.0→0.714), maar Track 1 (kennisvragen) ging juist AC HTERUIT (0.551→0.375). Dit is dus een op de **cloud** al doorlopen traject (train_sft→train_dpo→train_reflection→ merge_adapter, zie `cloud/run_all_oow.sh`), niet iets dat nog gestart moet worden.

**Lokaal in deze repo-kopie** (`_models/OOW/`) staat echter:

-   `oow_qwen_sft_lora/` — **lege map** (geen gewichten lokaal aanwezig; per project-conventie draait echte training alleen op de cloud, `_models/` is gitignored).
-   Geen `oow_qwen_dpo_lora/`, `oow_qwen_reflect_lora/`, gemergde of gequantiseerde varianten lokaal aanwezig — die bestaan alleen op de cloud-pod (`models_v1/` volgens het eval-rapport).

**Cruciaal voor dit rapport:** `agents.py`**'s** `_load_qwen()` **laadt ALTIJD de kale** `Qwen/Qwen3-8B`**, ongeacht welke config je kiest — er is geen enkele code-pad dat** `oow_qwen_sft_lora`**/de gemergde checkpoint in de Basic Simulator-app laadt.** Dat betekent: zelfs al zijn SFT/DPO/Reflectie op de cloud getraind en op de Q&A-eval getest, voor de **missie-sweeps in dit rapport (bare_qwen t/m v6) is dat gefinetunede model nooit gebruikt** — exact wat je vroeg te bevestigen.

## 5 — Reproductie: hoe elke stap opnieuw te draaien

```powershell
# 1) Bestaande 8-config sweep (bare_qwen..v6_pg_scenario) opnieuw/verder draaien (archief, reproductie):
cd "Basic Simulator"
..\.venv\Scripts\python.exe -m app.sweep_llm_params --tag units_v1 --configs bare_qwen v0_base v1_rag v2_cot v3_rag_cot v4_pg v5_pg_incident v6_pg_scenario

# 2) Standaard-sweep (de 5 --configs-default): baselines + de 3 super-configs:
..\.venv\Scripts\python.exe -m app.sweep_llm_params --tag units_v2

# 3) Eén super-config los uittesten:
..\.venv\Scripts\python.exe -m app.run_llm_scenario --missions Imazu01 --configs v8_super_cot_pg --tag units_v2

# 4) Reranker opnieuw trainen (CPU, lokaal, ~minuten):
..\.venv\Scripts\python.exe -m pipeline.ingest.build_reranker_pairs
..\.venv\Scripts\python.exe -m pipeline.train.train_reranker --epochs 3

# 5) Echte SFT/DPO/Reflectie-fine-tune (CLOUD ONLY, NOOIT lokaal starten):
#    zie cloud/run_all_oow.sh stages voor de exacte volgorde train_sft -> train_dpo ->
#    train_reflection -> merge_adapter (main-plan fase F4 sluit dit checkpoint dan aan
#    op de "weights"-as van de run-bestandsnaam).
```

***

## 6 — Samenvatting in één oogopslag

|                                                                               | Config(s)                                                                                             | Gewichten                                                                                      | Prompt-techniek                                                                                         | Status                                                                                                              |
|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **Gearchiveerde ablatie-arm (niet meer in de standaard-sweep; herdefinieer nooit)** | `bare_qwen`, `v0_base`, `v1_rag`, `v2_cot`, `v3_rag_cot`, `v4_pg`, `v5_pg_incident`, `v6_pg_scenario` | Kale `Qwen/Qwen3-8B` (W0_base, identiek voor alle 8) | Systeemprompt-varianten + RAG (dense+KG) + CoT + PG (procedure-graaf) — losse en gecombineerde ablaties | ✅ Volledig gedraaid (147/152 runs, tag=units_v1) |
| **Standaard-sweep sinds 2026-09-22: de 3 experimentmatrix-promptkolommen** | `v7_super_rag` (P1), `v8_super_cot_pg` (P2), `v9_super_all` (P3) | Kale `Qwen/Qwen3-8B` (W0_base) + `oow_reranker` cross-encoder (herget., dev-acc@1 0.768→1.000 op corpus v2) | Volledige RAG+rerank+KG-stack / CoT+scenario+incident-PG / beide gecombineerd | ⏳ Code + tests klaar (`test_agents_configs.py`, `test_pg_guidance_merge.py`), nog geen echte generatie/sweep gedraaid |
| **Losstaand traject: echte fine-tune**                                        | (geen aparte config-naam in `agents.py`; wordt de "weights"-as W1/W2/W3 in de run-bestandsnaam, fase F4) | LoRA SFT(r16)+DPO(r8,β0.05)+Reflectie(r8) bovenop `Qwen/Qwen3-8B`                              | Traint op alle 21 SFT/DPO/Reflectie-bestanden hierboven                                                 | ✅ Getraind + geëvalueerd op Track 1/2 Q&A (cloud) — ❌ nog NOOIT geladen/getest in de Basic Simulator missie-sweep |
