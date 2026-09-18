#!/bin/bash
# Bootstrap a fresh Runpod / Vast.ai / GCP GPU pod for VHF-QWEN training.
# Assumes: Ubuntu 22.04, Python 3.11+, CUDA 12.4, nvidia-smi works.
#
# Usage on the pod:
#   curl -sSL https://raw.githubusercontent.com/TextMiningUM/AutoPilot/main/cloud/bootstrap.sh | bash

set -euo pipefail

echo "=== VHF-QWEN Cloud Bootstrap ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

REPO_URL="https://github.com/TextMiningUM/AutoPilot.git"
CLONE_DIR="$HOME/AutoPilot"

echo ""
echo "=== 1/5 Clone repo ==="
if [ ! -d "$CLONE_DIR" ]; then
    git clone "$REPO_URL" "$CLONE_DIR"
else
    echo "Repo already present, pulling latest..."
    (cd "$CLONE_DIR" && git pull)
fi
cd "$CLONE_DIR"

echo ""
echo "=== 2/5 Create venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

echo ""
echo "=== 3/5 Install torch (cu124) ==="
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=== 4/5 Install rest ==="
pip install -q \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    accelerate>=0.29.0 \
    bitsandbytes>=0.43.0 \
    sentence-transformers>=2.7.0 \
    scikit-learn>=1.4.0 \
    pdfplumber>=0.11.0 \
    openai>=1.40.0 \
    anthropic>=0.34.0 \
    tiktoken>=0.7.0 \
    tqdm>=4.66.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    networkx>=3.2

echo ""
echo "=== 5/5 Prepare directories ==="
mkdir -p Data/VHF/VHF_Agents_Training
mkdir -p Data/VHF/VHF_Eval
mkdir -p _models/hf_cache
mkdir -p _models/VHF

# Pre-download Qwen3-8B so training starts immediately once data arrives
echo ""
echo "=== Pre-downloading Qwen3-8B (~5 GB in 4-bit, ~16 GB in bf16) ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
mid = 'Qwen/Qwen3-8B'
print('Tokenizer...'); AutoTokenizer.from_pretrained(mid, cache_dir='_models/hf_cache')
print('Model in bf16 (memory-map, no VRAM alloc)...')
AutoModelForCausalLM.from_pretrained(mid, cache_dir='_models/hf_cache', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
print('OK - Qwen cached.')
"

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload from your laptop:"
echo "     pwsh -File cloud/upload_bundle.ps1 -Host <THIS_HOST> -Port <THIS_PORT>"
echo ""
echo "  2. On this pod, kick off training:"
echo "     cd $CLONE_DIR"
echo "     source .venv/bin/activate"
echo "     tmux new -s train"
echo "     bash cloud/run_all.sh"
echo ""
echo "  3. When done, download from your laptop:"
echo "     pwsh -File cloud/download_results.ps1 -Host <THIS_HOST> -Port <THIS_PORT>"
