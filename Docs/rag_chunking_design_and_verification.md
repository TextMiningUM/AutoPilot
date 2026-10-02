# RAG Chunking & Text Segmentation — Design & Verification

**Scope:** documents the text-to-JSON ingestion standard (de-hyphenation) and the
chunking standard (embedding-based topic-boundary detection) applied to EVERY source
document — PDF, TXT, or MD — across every domain (VHF/OOW/CHIRP/incident reports today;
Captain once `build_captain_json.py` exists). This is the answer to two concrete
questions: "how do we detect that a text fragment is actually on 2 separated topics?"
and "how do we keep part of the next page's text with the current text when it belongs
together?" — the second question turned out to already be solved by this pipeline's
existing architecture (see §4); the first did not have an answer before this work.

**Source of truth:** `core/text_segmentation.py` (the pure algorithmic logic — no model
import, fast to unit-test), `pipeline/ingest/build_rag.py` (the one shared choke point
every domain's parsed JSON flows through before becoming a RAG chunk), and the
de-hyphenation call sites in `build_vhf_json.py` / `build_oow_json.py` /
`build_chirp_json.py` / `screen_incidents.py`.

---

## 1. Motivation

Two related but distinct data-quality risks in the ingest pipeline, both raised and
researched directly with the user (2026-10-03):

1. **A single chunk silently containing two unrelated topics.** The existing chunker
   (`build_rag.py`'s `chunk_chapter()`) only ever used a coarse hand-built
   keyword-Jaccard check (`TOPIC_JACCARD_MIN`) to decide whether to MERGE two already
   -separated sections — nothing checked whether a single parsed section's OWN text had
   silently drifted topic partway through (e.g. a long un-headed PDF paragraph covering
   two different subjects). No code path in the ingest pipeline detected this before
   this work.
2. **A word or sentence broken across a page/line boundary losing its other half.**
   Investigated and found to be a non-issue for SECTION-level text (see §4) but a real,
   separate issue for individual WORDS broken by a PDF's own line-wrapping
   (`contin-\nued`), which no ingest script de-hyphenated before this work.

## 2. Literature survey (full detail given to the user interactively; summarized here)

Methods considered for topic-boundary detection, cheapest/fastest to most expensive:

| Method | Approach | Verdict for this project |
|---|---|---|
| TextTiling (Hearst 1997) | Lexical (bag-of-words) sliding-window cosine similarity + "depth score" valley detection | Lexical-only, weak on subtle shifts — but the **depth-score mechanism itself** is reused below, just fed embeddings instead of word-count vectors |
| C99 (Choi 2000) | Rank-normalized similarity matrix + divisive clustering | Similar lexical-only limitation |
| GraphSeg (Glavaš et al. 2016, *SEM) | TextTiling-shaped segmentation but using **sentence embeddings** | Reported to outperform TextTiling/C99 while staying unsupervised — the closest published match to what's implemented here |
| Embedding sliding-window + adaptive threshold (LangChain `SemanticChunker` / LlamaIndex `SemanticSplitterNodeParser`) | Consecutive-sentence embedding cosine distance, percentile/std-dev adaptive breakpoint | **Adopted** — current practitioner standard for RAG chunking, no training needed, reuses an already-loaded embedder |
| NSP/coherence classifier | BERT next-sentence-prediction score as a boundary signal | Needs a model with an NSP head (most modern embedders dropped this); not clearly better for passage-level (not sentence-level) boundaries |
| Supervised segmentation (Koshorek et al. 2018; Lukasik et al. 2020 "Cross-Segment Attention") | Sequence-labeling model trained on Wikipedia section boundaries | Best published numbers, but no maritime/legal checkpoint exists; would need fine-tuning/threshold transfer |
| RST discourse parsing | Full discourse tree, nucleus/satellite relations | High fidelity, heavyweight dependency, overkill for bulk ingestion |
| LLM-prompted segmentation | Ask an LLM where to split | Highest semantic accuracy, but costly at corpus scale and conflicts with this repo's "determinism by construction" convention for pipeline-build steps |

**Chosen approach:** Hearst's depth-score valley-detection mechanism, fed **sentence
embeddings** (from the SAME `SentenceTransformer` `build_rag.py` already loads for final
chunk embedding — no new dependency, no extra model download) instead of bag-of-words
vectors, with a **percentile-adaptive** threshold (never a fixed cosine cutoff) — the
same idea practitioner RAG-chunking tooling (LangChain/LlamaIndex) converges on, and the
closest practical match to the published GraphSeg method.

Two more recent, directly-relevant techniques were noted but NOT adopted (documented
here so a future reviewer doesn't have to re-discover them):
- **Contextual Retrieval** (Anthropic, 2024) — prepend a generated context blurb to each
  chunk. This project already does a cheap non-LLM version of this via
  `_finalise_chunk()`'s `text_with_context` field (`"Source: ... Chapter: ..."` prefix).
- **Late Chunking** (Jina AI, 2024) — embed the whole document first, pool into chunks
  afterward. Would require restructuring the embedding pipeline shape; out of scope here.

## 3. Algorithm

### 3.1 De-hyphenation (`core/text_segmentation.py::join_hyphenated_linebreaks`)

Regex `(\w)-\n(?=[a-z])` → join. Deliberately narrow: only fires when a hyphen is
immediately followed by a single newline and then a LOWERCASE letter — this excludes a
genuine compound hyphen followed by more text on the same line (`well-known`), and
excludes a hyphen/dash followed by a new, separately-capitalized line (far more likely a
real sentence/list break than a broken word, e.g. `"See Annex I-\nAnnex II covers..."`).

Applied at every raw-text-assembly point across every parser (see §5) — BEFORE any
heading/sentence-boundary detection runs, since a hyphen-broken word can otherwise defeat
both (`classify_section`'s pattern matching, `split_sentences` below, heading regexes).

### 3.2 Topic-boundary detection (`core/text_segmentation.py`)

1. `split_sentences(text)` — the sentence splitter, consolidated from 2 previously
   independent copies (`build_rag.py`'s old inline regex, `pg_guidance.py`'s
   `split_step_sentences`) into this one shared function.
2. `cosine_similarities(embeddings)` — consecutive-pair cosine similarity (plain dot
   product, since every embedding call in this project already L2-normalizes — see
   `core/embedding.py`).
3. `depth_scores(similarities)` — Hearst's depth score at each internal gap: how far
   similarity dips below the highest point on EACH side, summed. Uses a simplified
   "running peak over the whole prefix/suffix" rather than Hearst's original "nearest
   local peak" — monotonic, robust to a spurious nearby ripple being picked as the peak.
4. `semantic_split_sentence_indices(embeddings, percentile_cutoff=85.0, min_sentences=6,
   min_depth_score=0.05)` — combines TWO independent conditions:
   - **Percentile rank** (adaptive): only the top `(100-percentile_cutoff)`% deepest
     valleys in THIS text's own distribution count as candidates.
   - **Absolute floor** (`min_depth_score`): a percentile rank ALONE is not sufficient —
     on a short or genuinely single-topic text, the single largest ripple in an
     otherwise-flat similarity curve still ranks in the "top 15%" by construction, even
     though it's pure embedding noise. This was caught directly during test development
     (`tests/test_text_segmentation.py::test_no_split_when_all_sentences_are_on_the_same_topic`
     initially failed without this floor). `min_depth_score=0.05` is far below a genuine
     topic shift's depth score (~1.5-2.0 in testing) and far above the measured noise
     floor (<0.01).
   - Requires at least `min_sentences` (default 6) sentences total — below this, the
     depth score's "peak on both sides" logic doesn't have enough signal to mean
     anything.

### 3.3 Wiring into the chunker (`pipeline/ingest/build_rag.py`)

Two integration points, both reusing the SAME `semantic_split_sentence_indices`/
`cosine_similarities` machinery:

1. **Intra-section split** (`_split_section_by_topic`) — runs on EVERY parsed section
   before the existing token-budget split (`_split_oversized_section`), independent of
   section size. Splits sentences at the detected boundary indices, producing sibling
   sections (`section_id` suffixed `_t1`/`_t2`/...) each tagged `semantic_split: True`
   for audit purposes. No confident boundary → section passes through unchanged (the
   common case, so this stays cheap for most sections).
2. **Inter-section merge-blocking** (`_can_extend`'s new `boundary_sim`/`merge_floor`
   parameters) — `chunk_chapter()` embeds every section's FULL text once per chapter,
   computes the chapter's own adjacent-section similarity distribution, and sets
   `merge_floor = percentile(these similarities, 25)` — i.e. "don't merge across a
   boundary that's weaker than the bottom quartile of THIS document's own adjacent-
   section similarities." This is an ADAPTIVE, per-chapter floor, never a fixed global
   cosine constant (deliberately — a fixed constant would need the same per-model
   calibration process `core/embedding.py`'s other thresholds required, which wasn't run
   for this; the adaptive/relative approach sidesteps that entirely and matches the
   literature's own "percentile/std-dev" recommendation). This AUGMENTS (never replaces)
   the existing keyword-topic-Jaccard check — both must pass for a merge to happen.

Every chunk record gets a new `"contains_semantic_split"` boolean field (audit trail —
did any of this chunk's constituent sections get produced by the topic splitter).
`build_rag.py`'s `main()` prints a count of these at the end of each run.

## 4. Cross-page continuation — already solved, not a new mechanism

Checked directly (both `build_vhf_json.py::parse_pdf_file` and
`build_oow_json.py::parse_document`): neither ever force-flushes a section at a page
boundary. Lines from consecutive pages are appended into one continuous buffer, and a
new section only starts when a heading/structural marker (font-size jump, bold,
`Rule N`, `Article N`, ...) is detected — page number is carried along purely as
per-section metadata (`"pages": [...]`). A paragraph that happens to straddle a page
break is therefore already kept as one section by construction, with no special-case
code needed. The ONLY real gap found in this area was the hyphenation issue (§3.1/§1.2),
which is a word-level, not section-level, concern.

## 5. Where de-hyphenation was applied (every raw-text-assembly point, by file)

| File | Function | Input format |
|---|---|---|
| `build_vhf_json.py` | `parse_txt_file` (on `raw` right after reading) | TXT/MD |
| `build_vhf_json.py` | `parse_pdf_file`'s `flush_section` | PDF |
| `build_vhf_json.py` | `parse_pdf_file`'s flat-document fallback | PDF |
| `build_oow_json.py` | `parse_document`'s `flush_section` (COLREG text) | PDF |
| `build_oow_json.py` | `parse_markdown_doc`'s `flush_section` | MD |
| `build_oow_json.py` | `parse_radar_workbook`'s `flush` | PDF |
| `build_chirp_json.py` | `extract_pages_cached` (applied once, then cached) | PDF |
| `screen_incidents.py` | `extract_pages_cached` (applied once, then cached) | PDF |

Two of these (`build_chirp_json.py`, `screen_incidents.py`) cache extracted page text to
disk — a PRE-EXISTING cache file from before this change will NOT retroactively benefit
until its cache is deliberately cleared and the source PDF re-extracted. This is an
accepted, expected limitation of any content-addressed cache, not a bug; re-processing a
corpus is a deliberate pipeline-rebuild decision, not a side effect of a code change.

## 6. Verification

- `tests/test_text_segmentation.py` (16 tests) — pure algorithmic unit tests against
  small synthetic embedding arrays, no model/GPU/network needed: de-hyphenation edge
  cases (compound hyphen, hyphen before a capitalized line, hyphen at a paragraph
  break), sentence splitting, cosine similarity, depth scores (including the "deep
  valley vs. shallow dip" discrimination), percentile, and `semantic_split_sentence_indices`
  (genuine two-topic split detected; no split on a single-topic or too-short text).
- `tests/test_build_rag_chunking.py` (5 tests) — `build_rag.py`'s chunker with a FAKE
  2D embedding model (keyword → orthogonal direction, deterministic, no real model
  load): confirms `_split_section_by_topic` splits a genuinely mixed-topic section,
  leaves a single-topic or too-short section alone, and confirms `chunk_chapter()`
  blocks a merge across a real topic shift EVEN WHEN the coarse keyword-Jaccard tags
  coincidentally match (isolating the embedding-based check as the deciding factor).
- Full suites re-run after the change: `pytest tests` and `pytest "Basic Simulator/tests"`
  — see the session's own record for the exact pass counts; the 2 known pre-existing
  unrelated failures (`test_build_incident_reflection_real.py`,
  `test_rag_exclusions.py::test_no_chunk_leaks_eval_scenario_text`) are unaffected by
  this work.

## 7. Known gaps / deliberately deferred

- **No real corpus rebuild was run as part of this change.** `build_rag.py`'s cache
  outputs (`Data/<domain>/<domain>_Agents_Training/*_rag_chunks.json` etc.) are TRACKED
  production files that feed KG/PG/training-data generation — regenerating them is a
  deliberate decision with downstream consequences (re-running `build_kg.py`, sanity-
  checking retrieval quality, likely re-running affected training stages), not something
  to trigger as a side effect of a code change's own verification. Run
  `python -m pipeline.ingest.build_rag --force` for VHF and OOW when ready to actually
  regenerate.
- **Pre-existing CHIRP/incident-screening text caches are not retroactively
  de-hyphenated** until their cache is cleared (see §5).
- **The per-chapter merge-floor percentile (25) and split percentile (85) are not
  empirically calibrated against this project's real corpus** (e.g. via a recall@k-style
  diagnostic the way `core/embedding.py`'s other thresholds were) — they were chosen
  from the adaptive-percentile literature convention and verified against synthetic
  test cases, not tuned against a real retrieval-quality metric. If a real corpus
  rebuild shows over- or under-splitting in practice, these two constants are the first
  place to adjust, OR revisit whether per-domain calibration (same `calibrate_embedder_
  thresholds.py` methodology) is worth the effort here too.
- **Captain has no parser yet** (`build_captain_json.py` doesn't exist) — but since
  `build_rag.py` is the single shared choke point, Captain's future parser gets this
  topic-segmentation standard automatically, with no extra wiring required, as long as
  it emits the same chapters/sections JSON schema every other domain already does.
