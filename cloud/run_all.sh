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
run_stage 1 sft          true  train_sft.py       --epochs 1 --max_seq_length 2048 --save_steps 50
run_stage 2 dpo          true  train_dpo.py       --epochs 1 --max_length 2048     --save_steps 50
run_stage 3 reflection   true  train_reflection.py --epochs 2 --max_length 2048    --save_steps 50
run_stage 4 merge        true  merge_adapter.py
run_stage 5 eval_vhfqwen false eval_finetuned.py --model _models/VHF/VHF-QWEN         --tag vhf_qwen
run_stage 6 prune        false compress_prune.py --n-prune 4
run_stage 7 distill      false compress_distill.py --merge-final
run_stage 8 eval_distill false eval_finetuned.py --model _models/VHF/DistillVHF-QWEN --tag distill_vhf

log "===== CLOUD CHAIN DONE ====="
