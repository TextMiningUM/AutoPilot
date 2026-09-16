# COLREG Exam Question Bank (500 items)

500 question/answer pairs covering the full International Regulations for
Preventing Collisions at Sea, 1972 (COLREGs, as amended): Rules 1-38 and
Annexes I-IV.

## Files
- `colreg_qa_500.json` — full dataset, list of objects (recommended for training pipelines)
- `colreg_qa_500.csv` — same data, flat CSV (recommended for spreadsheet review / labeling tools)

## Schema (per item)
| Field | Description |
|---|---|
| `id` | 1–500, stable index |
| `category` | Part A/B/C/D/E section or Annex the item belongs to |
| `rule_ref` | Specific Rule number(s) or Annex section the item tests |
| `question_type` | `recall`, `definition`, `application`, `scenario`, or `identification` |
| `question` | The question text |
| `answer` | The correct/expected answer |
| `explanation` | Why the answer is correct, with the specific rule citation |
| `difficulty` | Currently all `"medium"` — see Next Steps below to refine |

## Coverage breakdown (n=500)
- Part A – General (Rules 1–3): 21
- Section I – Any condition of visibility (Rules 4–10): 63
- Section II – In sight of one another, knowledge (Rules 11–18): 38
- Section II – Scenario/applied (crossing, head-on, overtaking, Rule 18 hierarchy, narrow-channel/TSS interplay): 86
- Section III – Restricted visibility (Rule 19): 32
- Part C – Lights and shapes (Rules 20–31), incl. identification-style items: 100
- Part D – Sound and light signals (Rules 32–37): 43
- Part E – Exemptions (Rule 38): 14
- Annex I – Technical details of lights/shapes: 41
- Annex II – Additional fishing-fleet signals: 19
- Annex III – Technical details of sound appliances: 20
- Annex IV – Distress signals: 23

## Question types
- **recall** (241) — direct rule text / factual lookup
- **application** (120) — "why," "does this satisfy," edge-case reasoning
- **scenario** (83) — geometric/operational situations (bearings, vessel types, visibility) with a derived correct action, closest to what a collision-avoidance agent needs to reason about
- **identification** (34) — given a light/shape/sound description, identify vessel status
- **definition** (22) — Rule 3/21/32 term definitions

## Notes on accuracy
Content was authored directly from the structure and substance of the 1972
COLREGs as amended (Rules 1–38, Annexes I–IV), which are public international
treaty text (IMO), not copyrighted prose from a third-party exam bank. A small
number of items intentionally flag genuine ambiguity or non-standard
light/shape combinations (see the Part C "Identification" items near the end
of that section) — these are included because a real collision-avoidance
agent needs to recognize uncertainty, not just pattern-match textbook cases.

**Before using this at scale for training a safety-relevant agent**, have a
licensed deck officer / marine COLREG instructor spot-check a sample —
especially the Annex I technical figures (exact heights/ranges/candela
formulas), Annex III sound-pressure-level figures, and Rule 38 exemption
details, which are the areas most sensitive to edition-specific numeric
detail and are most likely to need verification against the current IMO
consolidated text.

## Suggested next steps for the collision-avoidance agent
1. **Split the data**: use `recall`/`definition`/`identification` items as a
   knowledge-comprehension eval set; use `scenario` items as a closer proxy
   for operational decision-making.
2. **Scale the scenario set programmatically**: the `scenario` items here are
   templated (bearing angle, vessel type, visibility condition →
   give-way/stand-on + required action). This logic can be extended into a
   full procedural generator producing thousands of labeled
   (state → action) pairs for RL/imitation-style training, rather than
   relying only on hand-written exam questions.
3. **Add difficulty labels**: currently a placeholder; consider scoring by
   how many rules must be combined (e.g., Rule 13 overriding Rule 18 is
   harder than a single-rule lookup).
4. **Cross-check against a second source** (e.g., Cockcroft & Lameijer,
   IMO Model Course 7.03) for the most safety-critical items before using
   this as ground truth for a deployed system.
