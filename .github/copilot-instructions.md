# Auto Pilot — Project Guidelines

Auto Pilot is a multi-domain maritime agent pipeline, not a single VHF project. Three fine-tuned domains share one architecture — **VHF** (radio procedure, most mature), **OOW** (officer-of-the-watch COLREG navigation, actively being built), **Captain** (mission-level command, design-only as of 2026-09-30, no code yet) — plus **Basic Simulator**, a separate interactive Streamlit app/eval harness (see its own section below). Rules below apply to all domains unless scoped to one; when you build something for one domain, check whether the others need the matching piece too.

## Domains: one pipeline shape, `AgentPaths(domain=...)`

VHF/OOW/Captain reuse the exact same pipeline scripts (`pipeline/ingest/`, `pipeline/track1/`, `pipeline/track2/`, `pipeline/train/`, `pipeline/eval/`), parameterized by `core.paths.AgentPaths` and the `AUTOPILOT_DOMAIN` env var (`VHF`/`OOW`/`Captain`) — most pipeline modules resolve `AgentPaths.from_env()` at **import time**, so set `AUTOPILOT_DOMAIN` before importing them, not after. Adding or extending a domain means:
- add/extend an `AgentPaths` classmethod for its directory layout (mirrors the existing VHF/OOW ones);
- reuse the existing `build_rag.py`/`build_kg.py`/`build_pg.py`/`build_sft.py`/`build_rlhf.py`/`build_reflection.py`/`build_multihop.py`/`train_*.py`/`eval_*.py` scripts unchanged wherever possible;
- **never fork a second copy of a pipeline script for a new domain** — if the new domain genuinely needs different logic, parameterize the existing script (this is how OOW's incident pipeline and Track 2 scenario builders were added without duplicating VHF's scripts).

Captain is the newest domain — as of now it only exists as a design document (`design_captain_missions.md`, repo root) plus some raw acquired source data (`Data/Captain/`); there is no Captain training/eval code yet. Don't assume Captain-specific pipeline files exist — check the design doc and `Data/Captain/` first.

## Architecture: two tracks per domain, always extended together

Every domain's pipeline (training-data generation AND evaluation) has **two parallel tracks**. When you change or add something for one track, add the matching piece for the other — they are not optional extras, both are first-class. VHF's layout (the original, most mature example — OOW mirrors this with rule-text+incident-report Track 1 and mission-scenario Track 2; Captain is expected to mirror it again once built):

| | Track 1 — Rules & knowledge | Track 2 — Conversational compliance |
|---|---|---|
| Eval data (held out, NEVER used for training) | `Data/VHF/VHF_Eval/vhf_gold_answers.json` (540 Q&A) | `Data/VHF/VHF_Eval/vhf_colreg_scenarios.json` (498 collision-avoidance scenarios) |
| Training data | `vhf_sft_direct/cot/rag.jsonl`, `vhf_multihop.jsonl`, `vhf_dpo_pairs.jsonl`, `vhf_reflection.jsonl` | `vhf_conversations.jsonl` (raw dialogues) + `vhf_colreg_sft_*.jsonl`, `vhf_colreg_multihop.jsonl`, `vhf_colreg_dpo_pairs.jsonl`, `vhf_colreg_reflection.jsonl` (mined from those dialogues via `extract_conversation_reasoning.py`) |
| Eval script | `eval_finetuned.py` | `eval_colreg_scenarios.py` |

Both tracks feed the **same** QLoRA fine-tune (`train_sft.py`/`train_dpo.py`/`train_reflection.py` each load a list of files spanning both tracks), but stay in **separate files** so each competency's contribution is traceable. Every model tag (`qwen_base`, `vhf_qwen`, `vhf_qwen_awq`, `distill_vhf`) must be evaluated on **both** tracks — never just one.

## Local vs cloud: two environments, different jobs

There are exactly two places this project runs, and they have **different responsibilities** — don't blur them:

| | Local (laptop) | Cloud (LeafCloud GPU pod) |
|---|---|---|
| OS | Windows | Ubuntu |
| GPU | NVIDIA RTX 4070 Laptop, 8 GB VRAM | NVIDIA RTX 6000, 96 GB VRAM |
| Workspace | `C:\Users\jcsch\Documents\Python\Auto Pilot` | `/home/ubuntu/AutoPilot` |
| Python env | `.venv` (project-local) | `.venv` (project-local, same layout) |
| SSH | n/a | key file + host/user are kept in local-only notes (never committed) |
| What runs here | Data pipeline: JSON parsing (§8), chunking/RAG/KG (§9-10), reasoning-trace extraction (§11, OpenAI API calls), all `build_*.py` dataset builders (§12/§12.5/§12.6). Also used to sanity-check that `train_sft.py`/`train_dpo.py`/`train_reflection.py`'s data-loading functions (`load_all_sft()`, `load_dpo()`, `load_reflection()`) run cleanly against the current datasets — **without** loading the actual model. | **All actual QLoRA fine-tuning, merging, AWQ quantization, pruning, distillation, and evaluation** (`train_sft.py`, `train_dpo.py`, `train_reflection.py`, `merge_adapter.py`, `compress_*.py`, `eval_finetuned.py`, `eval_colreg_scenarios.py`, ablation). |
| Why the split | 8 GB VRAM is enough to verify the data pipeline and run tiny smoke tests, but a full SFT→DPO→Reflection→AWQ→prune→distill→ablation chain takes 8-12+ hours — that needs the cloud's 96 GB RTX 6000 and needs to survive disconnects. | See `tmux` note below. |

**Rule of thumb: never launch actual model training/compression/distillation locally.** Verify data changes locally (fast, free, no GPU risk), then push to git and run the real training chain only on the cloud.

**Always use the cloud's own existing `.venv`** at `/home/ubuntu/AutoPilot/.venv` (activate it, or call `../.venv/bin/python` directly) for every command run on the cloud pod — never fall back to system Python, `pip install` into a fresh/ad-hoc env, or another user's clone/venv on the same pod (the pod hosts multiple per-user clones under `/home/<user>/AutoPilot`, e.g. for JupyterHub logins; only the `ubuntu` account's clone is the one this project's automation/SSH commands and the overnight sweeps actually run against). This `.venv` must stay equivalent (same installed packages) to the local machine's `.venv` — if a package is added/upgraded on one side, mirror it on the other. Likewise `.env` lives at the repo root (`~/AutoPilot/.env`, NOT inside `.venv`) on both sides, same as `core.paths.AgentPaths.env_file` expects — if a required key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) is missing there, that's a real gap to flag/fix, not a sign to look for a key somewhere else.

`AUTOPILOT_MODELS_DIR` env var (used by `train_sft.py`, `merge_adapter.py`, `compress_*.py`, `eval_finetuned.py`) points at a shared cloud model-storage location when set, so multiple users/clones on the pod don't each download/merge their own multi-GB copy of Qwen3-8B; falls back to the repo-local `_models/` when unset (laptop use).

The cloud pod also runs a multi-user **JupyterHub** (`/etc/jupyterhub/jupyterhub_config.py`) so the notebook can be opened there interactively too — but see the next section for why the heavy chain doesn't run through it.

## The notebook(s) and `cloud/run_all*.sh` are independent front-ends to the SAME scripts

Each domain has its own notebook + orchestration-script pair — `VHF_Agent_Training_Pipeline.ipynb`+`cloud/run_all.sh` for VHF, `OOW_Agent_Training_Pipeline.ipynb`+`cloud/run_all_oow.sh` for OOW; Captain will need its own pair once it has code. Both members of a pair call the exact same standalone `.py` scripts via subprocess (`build_sft.py`, `train_sft.py`, `eval_finetuned.py`, `eval_colreg_scenarios.py`, ...). No pipeline logic lives in either front-end directly.

- The **notebook** is the interactive, documented front-end (used locally or via the cloud's JupyterHub) for step-by-step exploration.
- **`cloud/run_all*.sh`** is the headless orchestration script run inside a `tmux` session on the cloud GPU pod for long (8-12h+) unattended runs — `tmux` survives SSH/browser disconnects, which is why the heavy training chain is never run as live notebook cells.
- **They do not stay in sync automatically.** If you add/change a pipeline stage in one (e.g. a new eval call), you must manually mirror it in the other, for that domain's pair.

## Training-data style: fluent prose, never telegraphic label:value dumps

Every domain's `build_sft.py`, `build_rlhf.py`, `build_reflection.py`, `build_multihop.py` must produce natural sentences (e.g. "First, hail the vessel on Channel 16..."), never `"Steps: ... Channels: ... Warning: ..."` label concatenation. This telegraphic style previously contaminated every VHF training stage and caused the fine-tuned model to score *worse* than the untuned base model — treat this as a hard rule for every new domain/builder, not just a historical VHF fix. Perturbation-based DPO pairs (`build_rlhf.py`) must be generated by re-calling the same prose formatter with one field swapped — never by string search/replace on formatted text.

## Contamination filtering is mandatory for all new training-data builders

Any script generating training Q&A must embed questions and drop any with cosine similarity ≥ 0.85 against **every** held-out eval file for that domain it could plausibly overlap with (VHF: `vhf_gold_answers.json` + `vhf_colreg_scenarios.json`; OOW: its own gold-answers + scenario eval files; Captain: once it has eval files, the same rule applies) — never just one.

## Basic Simulator: a separate interactive project, own conventions

`Basic Simulator/` (top-level folder, NOT under `Data/`) is a distinct sub-project from the training pipeline above — an interactive Streamlit app (`app/streamlit_app.py`) plus standalone CLI tools (`run_llm_scenario.py`, `run_baseline_scenario.py`, `sweep_llm_params.py`) that exercise the OOW agent and deterministic baselines in closed-loop mission simulations. It has its **own test suite** (`Basic Simulator/tests/`, separate from repo-root `tests/`) and its own design docs (`Docs/nomoto_dynamics_design_and_verification.md`, `Docs/evaluation_function_design_and_verification.md`). Established conventions here that any new code must follow:

- **Facts-only prompts.** The live agent prompt (`pipeline/oow_agent_spec.py`'s `SYSTEM_OOW_AGENT`/`constraint_line()`) states computed facts only (CPA/TCPA, "already past closest point", a real measured manoeuvre time) — never hand-written imperative decision logic ("you must...", "do not answer X"). Judgement is learned via RAG/PG retrieval and SFT/DPO/Reflection training data, not prompt engineering. This was a deliberate architecture pivot after the opposite approach taught the model to pattern-match trigger phrases instead of reasoning — don't reintroduce it, including for the future Captain agent.
- **State computed facts directly — don't trust the model to re-derive them** (e.g. `stand_on_deadline_passed`, `closing: bool`). If a value has two possible interpretations depending on context (a classic example: `TCPA == 0` can mean "collision now" or "already past and diverging"), compute the boolean and state it in plain text; don't rely on the model to correctly infer it from a raw number.
- **Hard safety gates are code, never prompt text.** `app/oracle_planner.py`'s `required_direction()` rejects illegal candidates in code before cost-ranking — never rely on the LLM to self-censor via a prompt instruction. Any future Captain "shield" layer must follow the same hard-gate-in-code pattern.
- Six deterministic, non-LLM baselines (`app/baselines/*.py`) share one `decide(mission, own, targets, constraints) -> (decision, debug)` interface — reuse it for any new deterministic decision function instead of inventing a new signature.
- See `/memories/repo/basic_simulator.md` (this agent's own accumulated session notes) for the detailed iteration history before re-deriving already-settled UI/flicker/Streamlit-caching/physics lessons from scratch.

## Python coding best practices

This project already spans 3 domains + a simulator + a shared pipeline — treat these as load-bearing conventions, not style preference, since duplicated/drifted logic has caused real, hard-to-find bugs here before:

- **Reuse before you rebuild.** Check `core/` (paths/io/embedding/prose helpers) and the relevant `pipeline/` subpackage for an existing function before writing a new one. Shared renderer functions (e.g. `constraint_line()`, `goal_course_check_line()`) must stay the SAME function called from both training-data generation and the live prompt — a hand-copied second version WILL drift out of sync.
- **Type hints on every new or changed function signature** (parameters + return type). Don't retrofit unrelated existing code in the same change, but never add a new untyped function.
- **One-line, purpose-stating docstrings — not restated code.** Say *why*/*what it's for*, not what the next line already shows.
- **No bare `except:`.** Catch the specific exception you expect and let unexpected errors surface with a real traceback — several real bugs in this repo (quantization mismatches, stale checkpoints, schema drift) were only caught because an exception was allowed to propagate.
- **Determinism by construction.** Any function that samples/generates data (`sample_row_limits*`, mission/scenario generators) must take an explicit seed/row-id and be reproducible from it — never a bare unseeded random call. Document any remaining real non-determinism (LLM API calls, torch/cuDNN) explicitly rather than claiming full reproducibility.
- **Config via `AgentPaths`/env vars/CLI args — never a hardcoded path or a constant duplicated across files.** A value shared between a training-data generator and the live prompt (safe distance, turn-rate limits, ship profiles, etc.) must live in exactly one module that both import.
- **`pathlib.Path`, not string path concatenation** (matches `core/paths.py`).
- **Small, focused functions.** If a function parses AND computes AND renders, split it — this repo's worst bugs were caught faster once the offending logic was isolated into one small, independently testable function.
- **Grep every call site before changing a shared function's signature.** A hard-won lesson here: adding a new `constraint_line()` parameter once silently broke fallback limits-dicts in generators that weren't the obvious call sites.

## Documentation obligations

- **Every new module gets a short top-of-file docstring/comment**: what it does, and — for anything touching training data or models — whether it's safe to run **locally** or is **cloud-only** (see the Local vs cloud table above; state this explicitly, don't make the reader infer it).
- **Every function used by more than one caller gets a one-line docstring.** Private helpers used exactly once don't need one if the code is self-explanatory.
- **Every CLI script (`build_*.py`, `run_*.py`, `sweep_*.py`) needs real `argparse` `help=` text**, not a bare flag name — these scripts get invoked from memory/notebooks/shell scripts weeks later; `--help` is the actual documentation for how to run them.
- **Non-trivial design decisions get written into a `Docs/*_design_and_verification.md` file** (established pattern: `Docs/nomoto_dynamics_design_and_verification.md`, `Docs/evaluation_function_design_and_verification.md`) — not left only in chat/session history. `Docs/` is gitignored (local-only), but it's the durable source of truth for *why* a design choice was made.
- **Flag deliberate exceptions to a stated convention with a one-line comment at the exception site** (e.g. `evaluate_run.py`'s "reuse UNCHANGED" rule has documented, deliberate exceptions) — an undocumented exception reads as a bug to the next person/agent.

## Unit testing obligations

Two separate suites exist and must both stay green — repo-root `tests/` (VHF/OOW/Captain pipeline + `core/`) and `Basic Simulator/tests/` (simulator/baselines/oracle planner), run via `pytest tests` and `pytest "Basic Simulator/tests"` respectively (they are not merged into one suite — run both whenever you touch `pipeline/`, `core/`, or `Basic Simulator/`).

- **Every new deterministic function needs a test**: classifiers (`classify_encounter`, `classify_rules`), scorers (`evaluate_run.py`'s axis functions), samplers (`sample_row_limits*`), geometry/physics (`pipeline/nomoto.py`, `app/geometry.py`), and any data-shape/schema assumption another script depends on (e.g. "every trace record has `chunk_id`/`source_file`") — schema-compatibility assumptions silently broken have been this repo's most common real bug class, and a one-line schema test catches them immediately.
- **Tests must run without a GPU or a live API key.** Mock/stub any model or Anthropic/OpenAI call — follow the existing tests' pattern, never add a test that silently skips or spends real API credits.
- **Any new `build_*.py` generator needs a cheap dry-run/smoke-test path** (the established `--skip-llm`/small-`--n`-style flag) so it's runnable without spending API credits — and that smoke test must write to a path that **cannot** collide with the script's real tracked output (a repeated real bug here: a "quick smoke test" silently overwrote tracked production `.jsonl` files because the default output path was reused).
- **Before calling a change done, actually run the new/affected tests** — not just an import/compile check. Multiple real bugs in this repo's history were reported "fixed" after only a syntax check and turned out broken the moment a test or real run exercised them.
- **A regression you fix gets a named regression test** (matches the existing style: `test_oow_diverging_turn_fix.py`, `test_leo_goal_course_consistency.py`), not just a silent code change.

## Critical: how to edit notebooks

`.ipynb` files are **JSON documents**, not plain text files. Treat them as structured data, not a text blob.

- **Never perform raw text search-and-replace, sed-style edits, or "global replace" directly on the `.ipynb` file's raw contents.** This risks corrupting JSON structure, mangling cell metadata/outputs, breaking execution counts, or silently duplicating/misplacing cells.
- **Always use the notebook-aware editing tool/API available in this environment** (e.g., the editor's native notebook cell tools, nbformat, jupytext, or the IDE's "edit cell" action) rather than opening the `.ipynb` as a raw text file and patching strings in it.
- **Edit at the cell level.** Identify the specific cell(s) that need to change, and replace/update the full content of that cell — don't try to patch a substring across cell boundaries or across multiple cells in one blind replace.
- If a change needs to apply to **many cells** (e.g., renaming a variable used throughout the notebook), do it as a **sequence of individual, explicit cell edits** — show each affected cell and the change — rather than one sweeping find/replace across the whole file. Confirm the list of affected cells with the user first if there are more than a handful.
- **Never manually hand-edit cell `outputs`, `execution_count`, or `metadata` fields** unless explicitly asked to. These are managed by the kernel/notebook runtime — inconsistent manual edits (e.g., stale outputs left in place after changing code) are worse than no edit.
- If unsure whether a proposed change is safe to apply directly to the notebook, **say so and ask**, rather than applying a risky bulk edit and hoping it round-trips correctly.
- After editing, verify the notebook still parses as valid JSON / valid nbformat before considering the task done.
- Prefer working in `.py` (e.g., via jupytext-paired notebooks) for large refactors, and only touch the `.ipynb` directly for small, targeted changes — if the repo uses paired `.py`/`.ipynb` files, edit the `.py` and let jupytext sync, rather than editing both independently.

### Notebook hygiene

- **Preserve execution order intent.** Don't reorder cells or introduce reliance on out-of-order execution. If a fix requires moving code between cells, keep the logical top-to-bottom re-run order intact (running all cells top to bottom should still work).
- **Don't leave dead/commented-out experimental code** in cells you touch — either remove it or move it to a scratch/experiments notebook.
- **Clear large or sensitive outputs before suggesting a commit** — don't leave megabytes of embedded images, dataframes, or logs in cell outputs when proposing changes meant to be committed. Flag this to the user rather than silently stripping outputs they might want to keep.
- **Never commit outputs containing secrets, API keys, tokens, or credentials** (e.g., printed env vars, auth headers in request debug output). If you see one, flag it and suggest redacting/clearing that specific output rather than the whole notebook.
- **Keep notebooks focused.** If a notebook is accumulating unrelated exploration, model training, and reporting all in one file, suggest splitting into separate notebooks (e.g., `01_data_exploration.ipynb`, `02_train.ipynb`, `03_evaluate.ipynb`) rather than growing one monolithic file indefinitely.
- **Move reusable code out of notebooks.** Functions/classes used across multiple notebooks or across multiple cells repeatedly should live in a proper `.py` module (this repo: `core/`) and be imported — don't duplicate the same helper function across notebooks or across cells within one.

### General ground rules for notebooks

- Prefer **many small, explicit, reviewable edits** over one large notebook-wide transformation, especially for anything touching more than a couple of cells.
- If a requested change is ambiguous about **which cells** it should apply to, ask rather than guessing and editing the whole notebook.
- When adding new cells, keep **markdown documentation cells** alongside new code cells explaining what the code does and why, matching the notebook's existing documentation style/density.
- Don't silently change library versions, imports, or add new dependencies without flagging it — notebook environments are often fragile to dependency drift.

## Repo conventions

- Only `Docs/`, `.env`, and `_models/` are gitignored by default. Everything else — including all generated `Data/VHF/VHF_Agents_Training/*.jsonl`, `Data/VHF/VHF_JSON/*.json`, and the design docs at repo root (`design_captain_missions.md`, `design_ideas.txt`, `design_sim_agents.txt`) — is tracked in git and must be committed (models are the only thing too large for git).
- Scoped, deliberate gitignore exceptions are allowed but must be narrow and documented, never a blanket folder-wide ignore added silently: e.g. `Data/Captain/Legal_Reference/uk_legislation/*.xml` is gitignored specifically because legislation.gov.uk's own internal ID scheme (`key-<32 hex chars>`) is a byte-identical false-positive match for GitHub's API-key secret scanner — the raw XML is re-downloadable any time via `pipeline/ingest/build_captain_legal_corpus.py`, so excluding it is safe. Follow this pattern (narrow path, one-line reason in `.gitignore`) rather than widening an ignore rule to "fix" a push-protection or size problem.
- The cloud repo (path: `~/AutoPilot` on the GPU pod) frequently has stale local modifications to tracked data files from in-progress runs. Never do a plain `git pull` there — use `git fetch` + scoped `git checkout origin/main -- <files>`, or `git stash --include-untracked && git reset --hard origin/main` when a full resync is intended.
