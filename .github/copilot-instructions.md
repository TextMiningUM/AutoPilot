# Auto Pilot / VHF Agent Pipeline — Project Guidelines

## Architecture: two tracks, always extended together

Every stage of the VHF pipeline (training-data generation AND evaluation) has **two parallel tracks**. When you change or add something for one track, add the matching piece for the other — they are not optional extras, both are first-class:

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
| GPU | NVIDIA RTX 4070 Laptop, 8 GB VRAM | NVIDIA A30, 24 GB VRAM |
| Workspace | `C:\Users\jcsch\Documents\Python\Auto Pilot` | `/home/ubuntu/AutoPilot` |
| Python env | `.venv` (project-local) | `.venv` (project-local, same layout) |
| SSH | n/a | key file + host/user are kept in local-only notes (never committed) |
| What runs here | Data pipeline: JSON parsing (§8), chunking/RAG/KG (§9-10), reasoning-trace extraction (§11, OpenAI API calls), all `build_*.py` dataset builders (§12/§12.5/§12.6). Also used to sanity-check that `train_sft.py`/`train_dpo.py`/`train_reflection.py`'s data-loading functions (`load_all_sft()`, `load_dpo()`, `load_reflection()`) run cleanly against the current datasets — **without** loading the actual model. | **All actual QLoRA fine-tuning, merging, AWQ quantization, pruning, distillation, and evaluation** (`train_sft.py`, `train_dpo.py`, `train_reflection.py`, `merge_adapter.py`, `compress_*.py`, `eval_finetuned.py`, `eval_colreg_scenarios.py`, ablation). |
| Why the split | 8 GB VRAM is enough to verify the data pipeline and run tiny smoke tests, but a full SFT→DPO→Reflection→AWQ→prune→distill→ablation chain takes 8-12+ hours — that needs the cloud's 24 GB A30 and needs to survive disconnects. | See `tmux` note below. |

**Rule of thumb: never launch actual model training/compression/distillation locally.** Verify data changes locally (fast, free, no GPU risk), then push to git and run the real training chain only on the cloud.

**Always use the cloud's own existing `.venv`** at `/home/ubuntu/AutoPilot/.venv` (activate it, or call `../.venv/bin/python` directly) for every command run on the cloud pod — never fall back to system Python, `pip install` into a fresh/ad-hoc env, or another user's clone/venv on the same pod (the pod hosts multiple per-user clones under `/home/<user>/AutoPilot`, e.g. for JupyterHub logins; only the `ubuntu` account's clone is the one this project's automation/SSH commands and the overnight sweeps actually run against). This `.venv` must stay equivalent (same installed packages) to the local machine's `.venv` — if a package is added/upgraded on one side, mirror it on the other. Likewise `.env` lives at the repo root (`~/AutoPilot/.env`, NOT inside `.venv`) on both sides, same as `core.paths.AgentPaths.env_file` expects — if a required key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) is missing there, that's a real gap to flag/fix, not a sign to look for a key somewhere else.

`AUTOPILOT_MODELS_DIR` env var (used by `train_sft.py`, `merge_adapter.py`, `compress_*.py`, `eval_finetuned.py`) points at a shared cloud model-storage location when set, so multiple users/clones on the pod don't each download/merge their own multi-GB copy of Qwen3-8B; falls back to the repo-local `_models/` when unset (laptop use).

The cloud pod also runs a multi-user **JupyterHub** (`/etc/jupyterhub/jupyterhub_config.py`) so the notebook can be opened there interactively too — but see the next section for why the heavy chain doesn't run through it.

## The notebook and `cloud/run_all.sh` are two independent front-ends to the SAME scripts

`VHF_Agent_Training_Pipeline.ipynb` and `cloud/run_all.sh` both call the exact same standalone `.py` scripts via subprocess (`build_sft.py`, `train_sft.py`, `eval_finetuned.py`, `eval_colreg_scenarios.py`, ...). No pipeline logic lives in either of them directly.

- The **notebook** is the interactive, documented front-end (used locally or via the cloud's JupyterHub) for step-by-step exploration.
- **`cloud/run_all.sh`** is the headless orchestration script run inside a `tmux` session on the cloud GPU pod for long (8-12h+) unattended runs — `tmux` survives SSH/browser disconnects, which is why the heavy training chain is never run as live notebook cells.
- **They do not stay in sync automatically.** If you add/change a pipeline stage in one (e.g. a new eval call), you must manually mirror it in the other.

## Training-data style: fluent prose, never telegraphic label:value dumps

`build_sft.py`, `build_rlhf.py`, `build_reflection.py`, `build_multihop.py` must produce natural sentences (e.g. "First, hail the vessel on Channel 16..."), never `"Steps: ... Channels: ... Warning: ..."` label concatenation. This telegraphic style previously contaminated every training stage and caused the fine-tuned model to score *worse* than the untuned base model. Perturbation-based DPO pairs (`build_rlhf.py`) must be generated by re-calling the same prose formatter with one field swapped — never by string search/replace on formatted text.

## Contamination filtering is mandatory for all new training-data builders

Any script generating training Q&A must embed questions and drop any with cosine similarity ≥ 0.85 against **both** held-out eval files (`vhf_gold_answers.json` and `vhf_colreg_scenarios.json`), not just one.

## Repo conventions

- Only `Docs/`, `.env`, and `_models/` are gitignored. Everything else — including all generated `Data/VHF/VHF_Agents_Training/*.jsonl` and `Data/VHF/VHF_JSON/*.json` — is tracked in git and must be committed (models are the only thing too large for git).
- The cloud repo (path: `~/AutoPilot` on the GPU pod) frequently has stale local modifications to tracked data files from in-progress runs. Never do a plain `git pull` there — use `git fetch` + scoped `git checkout origin/main -- <files>`, or `git stash --include-untracked && git reset --hard origin/main` when a full resync is intended.
