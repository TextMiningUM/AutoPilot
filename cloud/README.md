# Cloud VHF-QWEN Training

Same fine-tune pipeline, but on a rented GPU (24-48 GB VRAM) so we can finish in **~4-6 hours** instead of days on the 8 GB laptop.

## Why go to cloud

- Laptop RTX 4070 (8 GB) is thermally constrained → 35 W TGP → crashes after 15-45 min
- Cloud A10G (24 GB) at ~$0.30/hr = **~$1.50 for the whole job**
- No thermal issues, sustained 100 % load is designed for
- Same code, same data, same recipe — only compute moves

## Recommended provider: Runpod

- Cheapest reliable GPU rental
- Web terminal + SSH + SCP
- Pay per second, stop when done
- https://runpod.io

Alternatives:
- **Vast.ai** — even cheaper (community hosts) but less predictable
- **Modal Labs** — Python-first, serverless, ~$0.60/hr on A10G
- **Google Colab Pro+** ($50/month) — A100 access with limits
- **HuggingFace AutoTrain** — even simpler but locked to their recipe

## Step-by-step (Runpod)

### 1. Rent a pod
1. Sign up + add ~$10 credit
2. **Deploy → GPU Pods → 1× A10G 24 GB** (or L40 48 GB for extra headroom)
3. Template: **PyTorch 2.4 (CUDA 12.4)** — pre-installed
4. Volume size: 40 GB (enough for Qwen weights + checkpoints)
5. Click **Deploy On-Demand**
6. Pod boots in ~30s → grab the SSH command shown

### 2. Bootstrap the pod

SSH into the pod (Runpod shows the command), then:

```bash
curl -sSL https://raw.githubusercontent.com/TextMiningUM/AutoPilot/main/cloud/bootstrap.sh | bash
```

This does:
- clones the AutoPilot repo
- creates a venv
- installs torch (cu124) + transformers + peft + trl + bitsandbytes
- downloads Qwen3-8B to `_models/hf_cache/`

Takes ~5 min.

### 3. Upload your data

From your **laptop**, in the `Auto Pilot` directory:

```powershell
pwsh -File cloud/upload_bundle.ps1 -Host <RUNPOD_HOST> -Port <RUNPOD_PORT>
```

This uploads only the required minimum via `scp`:
- `Data/VHF/VHF_Agents_Training/*.jsonl` (SFT/DPO/reflection datasets)
- `Data/VHF/VHF_Eval/vhf_gold_answers.json` (for eval)

Total: ~30 MB.

### 4. Kick off training

Back on the pod:

```bash
cd ~/AutoPilot
source .venv/bin/activate

# Full 3-stage fine-tune (~4-6 h total on A10G)
tmux new -s train
python train_sft.py --epochs 1 --max_seq_length 2048
# ...disconnect: Ctrl-b then d
# ...reconnect later: tmux attach -t train
```

`tmux` keeps the training running if your SSH drops. After SFT completes, run `train_dpo.py` and `train_reflection.py` sequentially.

Or use the same overnight script:
```bash
bash cloud/run_all.sh  # equivalent of run_overnight.ps1 for Linux
```

### 5. Download the results

When done, from your **laptop**:

```powershell
pwsh -File cloud/download_results.ps1 -Host <RUNPOD_HOST> -Port <RUNPOD_PORT>
```

Pulls back the merged model + eval summaries (~14 GB — takes ~15 min).

### 6. Stop the pod

Very important! Runpod bills per second while the pod exists.

```
Runpod dashboard → Pod → Stop
```

## Cost estimates

| Provider | GPU | $/hr | Full fine-tune (SFT+DPO+Reflect+Merge+Eval+Prune+Distill+Eval) | Total cost |
|---|---|---:|---:|---:|
| Runpod | A10G 24 GB | 0.30 | ~5 h | **$1.50** |
| Runpod | L40 48 GB | 0.60 | ~4 h | **$2.40** |
| Runpod | A100 40 GB | 1.20 | ~3 h | **$3.60** |
| Vast.ai | A10G | 0.20 | ~5 h | **$1.00** |
| Modal | A10G | 0.60 | ~5 h | **$3.00** |

A10G is the sweet spot: 24 GB VRAM = enough for Qwen 7B in bf16, cheap, plentiful.

## What DOESN'T get uploaded

- `_models/` — cloud recreates it fresh (Qwen from HF hub, LoRA from training)
- `Data/VHF/VHFProtocol/` — raw sources not needed once JSONL is built
- `Data/VHF/VHF_JSON/` — same
- `Docs/` — private, stays local (see `.gitignore`)
- `.env` — sensitive; you'll set `OPENAI_API_KEY` on the pod directly if needed for eval judge

## What to bring BACK from cloud

- `_models/VHF/VHF-QWEN/` — merged fine-tuned model (~14 GB)
- `_models/VHF/DistillVHF-QWEN/` — student model (~3 GB) — optional
- `Data/VHF/VHF_Agents_Training/eval_*.json` — per-question metrics + summaries

If you don't need the merged model locally (e.g. planning to deploy to more cloud inference anyway), skip the download and just pull the summaries.
