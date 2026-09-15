#!/bin/bash
# Full training + compression chain for a cloud pod (Linux equivalent of run_overnight.ps1).
# Runs the same 8-stage sequence; each stage logs to overnight_logs/<NN_stage>.log.
#
# Usage:
#   cd ~/AutoPilot && source .venv/bin/activate && bash cloud/run_all.sh

set -uo pipefail

WORKSPACE="$HOME/AutoPilot"
LOG_DIR="$WORKSPACE/Data/VHF/VHF_Agents_Training/overnight_logs"
MASTER="$LOG_DIR/_master.log"

cd "$WORKSPACE"
mkdir -p "$LOG_DIR"

log() {
    local ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$MASTER"
}

run_stage() {
    local idx="$1"; shift
    local name="$1"; shift
    local required="$1"; shift
    local -a cmd=("$@")

    local tag="$(printf '%02d_%s' "$idx" "$name")"
    local log="$LOG_DIR/$tag.log"

    log "================================================================================"
    log "STAGE $tag  required=$required"
    log "cmd: python ${cmd[*]}"
    log "log: $log"
    log "--------------------------------------------------------------------------------"

    local t0=$(date +%s)
    python -X utf8 "${cmd[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    local dur_min=$(( ($(date +%s) - t0) / 60 ))

    log "STAGE $tag  exit=$rc  duration=${dur_min}min"

    if [ "$rc" -ne 0 ]; then
        if [ "$required" = "true" ]; then
            log "!!! REQUIRED STAGE FAILED -- aborting chain."
            exit 1
        else
            log "... optional stage failed, continuing."
        fi
    fi
}

log "===== CLOUD CHAIN START ====="
log "workspace: $WORKSPACE"
log "python:    $(which python)"
log "gpu:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Cloud GPUs have 24-48 GB VRAM so we can use larger settings than on the laptop.
run_stage 1 sft            true  train_sft.py       --epochs 1 --max_seq_length 2048 --save_steps 50
run_stage 2 dpo            true  train_dpo.py       --epochs 1 --max_length 2048     --save_steps 50
run_stage 3 reflection     true  train_reflection.py --epochs 2 --max_length 2048    --save_steps 50
run_stage 4 merge          true  merge_adapter.py
run_stage 5 eval_qwen_base false eval_finetuned.py --model Qwen/Qwen2.5-7B-Instruct  --tag qwen_base --n 540
run_stage 6 eval_vhfqwen   false eval_finetuned.py --model _models/VHF/VHF-QWEN         --tag vhf_qwen --n 540

log "--- optional: AWQ int4 quantization (pip install autoawq may fail on some setups) ---"
pip install -q autoawq 2>&1 | tee "$LOG_DIR/07_awq_install.log" || log "autoawq install failed -- skipping AWQ stage."
run_stage 7 awq_quantize false compress_quantize_awq.py --n-calibration 128 --skip-eval
run_stage 8 eval_awq     false eval_finetuned.py --model _models/VHF/VHF-QWEN-awq-int4 --tag vhf_qwen_awq --n 540

run_stage 9  prune        false compress_prune.py --n-prune 4
run_stage 10 distill      false compress_distill.py --merge-final
run_stage 11 eval_distill false eval_finetuned.py --model _models/VHF/DistillVHF-QWEN --tag distill_vhf --n 540

log "--- prompt-injection ablation (base Qwen: no-context / RAG / CoT / RAG+CoT), full 540 ---"
run_stage 12 ablation_prep  false prep_ablation.py --n 540
run_stage 13 ablation_run   false run_ablation.py
run_stage 14 ablation_score false score_ablation.py

log "===== CLOUD CHAIN DONE ====="
