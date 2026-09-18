#!/bin/bash
# Full training + compression chain for a cloud pod (Linux equivalent of run_overnight.ps1).
# Every stage that evaluates a model runs BOTH tracks (Track 1 rules/knowledge via
# eval_finetuned.py, Track 2 conversational compliance via eval_colreg_scenarios.py) --
# Track 2 is a permanent, first-class part of the pipeline, not an optional extra.
# Each stage logs to overnight_logs/<NN_stage>.log.
#
# Usage:
#   cd ~/AutoPilot && source .venv/bin/activate && bash cloud/run_all.sh
#   RESUME_FROM=10 bash cloud/run_all.sh   # skip stages < 10 (e.g. after a reboot mid-chain)

set -uo pipefail

WORKSPACE="$HOME/AutoPilot"
LOG_DIR="$WORKSPACE/Data/VHF/VHF_Agents_Training/overnight_logs"
MASTER="$LOG_DIR/_master.log"
COLREG_EVAL="Data/VHF/VHF_Eval/vhf_colreg_scenarios.json"
RESUME_FROM="${RESUME_FROM:-1}"

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

    if [ "$idx" -lt "$RESUME_FROM" ]; then
        log "STAGE $tag  skipped (RESUME_FROM=$RESUME_FROM)"
        return 0
    fi

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

log "--- Track 2: mine agentic training data from the 360 training conversations (notebook § 12.6) ---"
run_stage  1 colreg_extract    false -m pipeline.track2.extract_conversation_reasoning
run_stage  2 colreg_sft        false -m pipeline.track1.build_sft --traces-file Data/VHF/VHF_Agents_Training/vhf_conversation_traces.jsonl --out-prefix vhf_colreg_sft --extra-gold-file "$COLREG_EVAL"
run_stage  3 colreg_dpo        false -m pipeline.track1.build_rlhf --traces-file Data/VHF/VHF_Agents_Training/vhf_conversation_traces.jsonl --out-file Data/VHF/VHF_Agents_Training/vhf_colreg_dpo_pairs.jsonl --extra-gold-file "$COLREG_EVAL"
run_stage  4 colreg_reflection false -m pipeline.track1.build_reflection --traces-file Data/VHF/VHF_Agents_Training/vhf_conversation_traces.jsonl --out-file Data/VHF/VHF_Agents_Training/vhf_colreg_reflection.jsonl --extra-gold-file "$COLREG_EVAL"
run_stage  5 colreg_multihop   false -m pipeline.track1.build_multihop --traces-file Data/VHF/VHF_Agents_Training/vhf_reasoning_traces.jsonl Data/VHF/VHF_Agents_Training/vhf_conversation_traces.jsonl --out-file Data/VHF/VHF_Agents_Training/vhf_colreg_multihop.jsonl --extra-gold-file "$COLREG_EVAL"

# Cloud GPUs have 24-48 GB VRAM so we can use larger settings than on the laptop.
run_stage  6 sft            true  -m pipeline.train.train_sft       --epochs 1 --max_seq_length 2048 --save_steps 50
run_stage  7 dpo            true  -m pipeline.train.train_dpo       --epochs 0.5 --beta 0.05 --max_length 2048 --save_steps 25
run_stage  8 reflection     true  -m pipeline.train.train_reflection --epochs 2 --max_length 2048    --save_steps 50
run_stage  9 merge          true  -m pipeline.train.merge_adapter

log "--- Track 1 (rules/knowledge) + Track 2 (conversational compliance), every model tag ---"
log "    (defaults to the claim-level suite v2 -- requires the *_claims.json gold files uploaded by upload_bundle.ps1)"
run_stage 10 eval_qwen_base        false -m pipeline.eval.eval_finetuned --model Qwen/Qwen3-8B --tag qwen_base --n 540
run_stage 11 eval_qwen_base_colreg false -m pipeline.eval.eval_colreg_scenarios --model Qwen/Qwen3-8B --tag qwen_base

log "--- prompt-injection ablation (base Qwen only: V0-V4 incl. PG guidance if vhf_pg.json is present), full 540, ---"
log "    Track 1 only -- runs right after the base-Qwen eval above since it needs nothing else (no fine-tuning ---"
log "    dependency), instead of waiting until after fine-tuning/probes/compression like earlier runs did ---"
run_stage 12 ablation_prep  false -m pipeline.eval.prep_ablation --n 540
run_stage 13 ablation_run   false -m pipeline.eval.run_ablation
run_stage 14 ablation_score false -m pipeline.eval.score_ablation

run_stage 15 eval_vhfqwen          false -m pipeline.eval.eval_finetuned --model _models/VHF/VHF-QWEN --tag vhf_qwen --n 540
run_stage 16 eval_vhfqwen_colreg   false -m pipeline.eval.eval_colreg_scenarios --model _models/VHF/VHF-QWEN --tag vhf_qwen

log "--- Stage-attribution probes (DPO axis win-rates incl. swap_step_order + reflection Delta) ---"
run_stage 17 probe_qwen_base false -m pipeline.eval.probe_dpo --model Qwen/Qwen3-8B --tag qwen_base
run_stage 18 probe_vhfqwen   false -m pipeline.eval.probe_dpo --model _models/VHF/VHF-QWEN --tag vhf_qwen

log "--- optional: AWQ int4 quantization (pip install autoawq may fail on some setups) ---"
pip install -q autoawq 2>&1 | tee "$LOG_DIR/19_awq_install.log" || log "autoawq install failed -- skipping AWQ stage."
run_stage 19 awq_quantize      false -m pipeline.compress.compress_quantize_awq --n-calibration 128 --skip-eval
run_stage 20 eval_awq          false -m pipeline.eval.eval_finetuned --model _models/VHF/VHF-QWEN-awq-int4 --tag vhf_qwen_awq --n 540
run_stage 21 eval_awq_colreg   false -m pipeline.eval.eval_colreg_scenarios --model _models/VHF/VHF-QWEN-awq-int4 --tag vhf_qwen_awq

run_stage 22 prune             false -m pipeline.compress.compress_prune --n-prune 4
run_stage 23 distill           false -m pipeline.compress.compress_distill --merge-final
run_stage 24 eval_distill      false -m pipeline.eval.eval_finetuned --model _models/VHF/DistillVHF-QWEN --tag distill_vhf --n 540
run_stage 25 eval_distill_colreg false -m pipeline.eval.eval_colreg_scenarios --model _models/VHF/DistillVHF-QWEN --tag distill_vhf

log "===== CLOUD CHAIN DONE ====="
