---
description: "Use when editing .ipynb notebooks or any deep learning/NLP/IR training and evaluation code (build_*.py, train_*.py, eval_*.py). Covers safe notebook editing, notebook hygiene, reproducibility, and ML-specific pitfalls."
applyTo: "**/*.ipynb,**/*.py"
---

# Critical: How to edit notebooks

`.ipynb` files are **JSON documents**, not plain text files. Treat them as structured data, not a text blob.

- **Never perform raw text search-and-replace, sed-style edits, or "global replace" directly on the `.ipynb` file's raw contents.** This risks corrupting JSON structure, mangling cell metadata/outputs, breaking execution counts, or silently duplicating/misplacing cells.
- **Always use the notebook-aware editing tool/API available in this environment** (the editor's native notebook cell tools, nbformat, jupytext, or the IDE's "edit cell" action) rather than opening the `.ipynb` as a raw text file and patching strings in it.
- **Edit at the cell level.** Identify the specific cell(s) that need to change, and replace/update the full content of that cell — don't try to patch a substring across cell boundaries or across multiple cells in one blind replace.
- If a change needs to apply to **many cells** (e.g., renaming a variable used throughout the notebook), do it as a **sequence of individual, explicit cell edits** — show each affected cell and the change — rather than one sweeping find/replace across the whole file. Confirm the list of affected cells first if there are more than a handful.
- **Never manually hand-edit cell `outputs`, `execution_count`, or `metadata` fields** unless explicitly asked to. These are managed by the kernel/notebook runtime — inconsistent manual edits (e.g., stale outputs left in place after changing code) are worse than no edit.
- If unsure whether a proposed change is safe to apply directly to the notebook, **say so and ask**, rather than applying a risky bulk edit and hoping it round-trips correctly.
- After editing, verify the notebook still parses as valid JSON / valid nbformat before considering the task done.
- Prefer working in `.py` (e.g., via jupytext-paired notebooks) for large refactors, and only touch the `.ipynb` directly for small, targeted changes — if the repo uses paired `.py`/`.ipynb` files, edit the `.py` and let jupytext sync, rather than editing both independently.

## Notebook hygiene

- **Preserve execution order intent.** Don't reorder cells or introduce reliance on out-of-order execution. If a fix requires moving code between cells, keep the logical top-to-bottom re-run order intact (running all cells top to bottom should still work).
- **Don't leave dead/commented-out experimental code** in cells you touch — either remove it or move it to a scratch/experiments notebook.
- **Clear large or sensitive outputs before suggesting a commit** — don't leave megabytes of embedded images, dataframes, or logs in cell outputs when proposing changes meant to be committed. Flag this rather than silently stripping outputs that might be wanted.
- **Never commit outputs containing secrets, API keys, tokens, or credentials** (e.g., printed env vars, auth headers in request debug output). If you see one, flag it and suggest redacting/clearing that specific output rather than the whole notebook.
- **Keep notebooks focused.** If a notebook is accumulating unrelated exploration, model training, and reporting all in one file, suggest splitting into separate notebooks (e.g., `01_data_exploration.ipynb`, `02_train.ipynb`, `03_evaluate.ipynb`) rather than growing one monolithic file indefinitely.
- **Move reusable code out of notebooks.** Functions/classes used across multiple notebooks or across multiple cells repeatedly should live in a proper `.py` module (this repo: `core/`) and be imported — don't duplicate the same helper function across notebooks or across cells within one.

## Deep learning / training code

- **Reproducibility**: set and surface random seeds (Python, NumPy, framework-specific — e.g., `torch.manual_seed`, `tf.random.set_seed`) explicitly; note any remaining source of non-determinism (e.g., cuDNN algorithms, data loader shuffling, multi-GPU non-determinism, Python's hash-randomized `set` ordering) rather than claiming full reproducibility if it isn't guaranteed.
- **Config management**: prefer a config object/file (YAML/JSON/dataclass) over scattered hardcoded hyperparameters across cells. If hyperparameters are hardcoded inline, flag it and suggest consolidating.
- **Device handling**: use device-agnostic code (`device = "cuda" if available else "cpu"`), and don't hardcode GPU indices or assume a specific number of GPUs.
- **Memory management**: watch for patterns that accumulate GPU/CPU memory across cells (e.g., re-running training cells without clearing old model/optimizer references, unbounded lists of tensors/activations kept around). Flag and suggest explicit cleanup (`del`, `torch.cuda.empty_cache()`, moving tensors off-GPU) where relevant.
- **Checkpointing**: ensure model/optimizer/scheduler state (and ideally epoch/step count and RNG state) are saved together for resumability, not just raw weights, unless inference-only saving is explicitly intended.
- **Experiment tracking**: if the project uses an experiment tracker (MLflow, Weights & Biases, TensorBoard, etc.), log new metrics/params through it rather than only `print()`-ing results into cell output.
- **Data leakage**: watch for train/val/test contamination — shared preprocessing statistics (e.g., normalization computed on the full dataset instead of train-only), leakage through data augmentation applied before splitting, or shuffling that mixes splits. (This repo's own convention: every training-data builder must cosine-filter against both held-out eval files — see the main copilot-instructions.md.)

## NLP-specific

- **Tokenizer/model version pinning**: note the exact model checkpoint/tokenizer version used (e.g., a pinned model id or revision hash) so results are reproducible as upstream models change.
- **Tokenization consistency**: ensure the same tokenizer/preprocessing is used consistently at train, eval, and inference time — flag any place where preprocessing logic is duplicated (and could drift) between training and inference code paths instead of shared from one source.
- **Sequence length / truncation**: be explicit about max sequence length, truncation, and padding strategy, and flag silent truncation that could quietly drop meaningful content.
- **Text cleaning**: don't apply irreversible or lossy text normalization (aggressive lowercasing, stripping punctuation/unicode) without flagging the tradeoff, especially for tasks where that information may be meaningful.
- **Multilingual/encoding issues**: watch for encoding assumptions (e.g., ASCII-only cleaning regexes) that break on non-English or non-Latin-script text if the project's data isn't guaranteed to be English-only.

## Information retrieval-specific

- **Index/embedding versioning**: flag when the embedding model or preprocessing changes would invalidate an existing index — an index built with one embedding model shouldn't be silently queried with vectors from a different one.
- **Train/eval separation for retrieval**: ensure query sets used for evaluation aren't leaking into the corpus used to build/tune the index in a way that inflates metrics.
- **Metric correctness**: for IR evaluation (precision@k, recall@k, MRR, NDCG, etc.), double check the exact definition being implemented matches the standard one — these are easy to subtly get wrong (off-by-one in k, wrong handling of ties, ignoring relevance grades vs. binary relevance).
- **Scale-appropriate approach**: flag if a brute-force similarity search is being used somewhere that will clearly need an approximate nearest-neighbor index (FAISS, ScaNN, HNSW, etc.) at production scale, or vice versa — don't over-engineer indexing for a small dev-time dataset.

## General ground rules

- Prefer **many small, explicit, reviewable edits** over one large notebook-wide transformation, especially for anything touching more than a couple of cells.
- If a requested change is ambiguous about **which cells** it should apply to, ask rather than guessing and editing the whole notebook.
- When adding new cells, keep **markdown documentation cells** alongside new code cells explaining what the code does and why, matching the notebook's existing documentation style/density.
- Don't silently change library versions, imports, or add new dependencies without flagging it — notebook environments are often fragile to dependency drift.
