#!/bin/bash
# Full OOW pipeline chain for a cloud pod -- exactly mirrors every executable cell
# of OOW_Agent_Training_Pipeline.ipynb (both Track 1 -- COLREG rules & knowledge --
# and Track 2 -- applied helm/engine-order decisions), in the same order. Sample
# size, file tag and how far to run are all configurable via env vars so the same
# script covers a tiny smoke test AND a bigger sanity-check pass (e.g. n=50 through
# the error analysis, stopping before fine-tuning). See cloud/run_all.sh for the
# VHF equivalent / full-run conventions this mirrors.
#
# Usage:
#   cd ~/AutoPilot && source .venv/bin/activate && bash cloud/run_all_oow.sh
#   RESUME_FROM=24 bash cloud/run_all_oow.sh          # skip stages < 24 (e.g. after a reboot mid-chain)
#   N=50 TAG=n50 MAX_STAGE=33 bash cloud/run_all_oow.sh  # n=50 through §12 error analysis, no training

set -uo pipefail

export AUTOPILOT_DOMAIN=OOW

WORKSPACE="$HOME/AutoPilot"
CACHE_DIR="Data/OOW/OOW_Agents_Training"
MODELS_DIR="_models/OOW"
LOG_DIR="$WORKSPACE/$CACHE_DIR/overnight_logs"
N="${N:-2}"                 # gold-sample size for baseline/ablation stages (§10-§11)
TAG="${TAG:-smoke}"         # suffix for every eval/ablation/model file this run produces
MAX_STAGE="${MAX_STAGE:-999}"  # stages > this are skipped -- e.g. 33 to stop before §13 training
MASTER="$LOG_DIR/_master_oow_${TAG}.log"
NORM_FILE="Data/OOW/OOW_Eval/colreg_qa_500_normalised.json"
NORM_CLAIMS="Data/OOW/OOW_Eval/colreg_qa_500_normalised_claims.json"
RESUME_FROM="${RESUME_FROM:-1}"

SFT_DIR="$MODELS_DIR/oow_qwen_sft_lora_${TAG}"
DPO_DIR="$MODELS_DIR/oow_qwen_dpo_lora_${TAG}"
REFLECT_DIR="$MODELS_DIR/oow_qwen_reflect_lora_${TAG}"
MERGED_DIR="$MODELS_DIR/OOW-QWEN_${TAG}"
AWQ_DIR="$MODELS_DIR/OOW-QWEN-awq-int4_${TAG}"
PRUNED_DIR="$MODELS_DIR/OOW-QWEN-pruned_${TAG}"
DISTILL_DIR="$MODELS_DIR/DistillOOW-QWEN_${TAG}"

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
    if [ "$idx" -gt "$MAX_STAGE" ]; then
        log "STAGE $tag  skipped (MAX_STAGE=$MAX_STAGE)"
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

log "===== OOW CHAIN START (N=$N TAG=$TAG MAX_STAGE=$MAX_STAGE) ====="
log "workspace: $WORKSPACE"
log "python:    $(which python)"
log "gpu:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

log "--- § 1: normalise the 500-Q&A gold file -> {question, gold_answer, expected_points} ---"
run_stage  1 gold_norm        true  -m pipeline.ingest.build_oow_gold_norm

log "--- § 2-4: COLREG source -> JSON -> RAG chunks -> Knowledge Graph ---"
run_stage  2 oow_json         false -m pipeline.ingest.build_oow_json
run_stage  3 rag              false -m pipeline.ingest.build_rag
run_stage  4 kg               false -m pipeline.ingest.build_kg

log "--- § 5: rule-text reasoning-trace extraction (needs OPENAI_API_KEY, resume-safe) ---"
run_stage  5 reasoning        false -m pipeline.track1.extract_reasoning

log "--- § 5.1: real-world incident reports (screen -> excerpt -> extract) ---"
run_stage  6 screen_incidents  false -m pipeline.ingest.screen_incidents
run_stage  7 incident_excerpts false -m pipeline.ingest.build_incident_excerpts --min-score 15
run_stage  8 incident_reasoning false -m pipeline.track1.extract_incident_reasoning

log "--- § 6: Track 1 training data from rule text ---"
run_stage  9 sft_rule    false -m pipeline.track1.build_sft --out-prefix oow_sft --gold-file "$NORM_FILE"
run_stage 10 multihop    false -m pipeline.track1.build_multihop \
    --traces-file "$CACHE_DIR/oow_reasoning_traces.jsonl" \
                  "$CACHE_DIR/oow_incident_reasoning_traces.jsonl" \
                  "$CACHE_DIR/oow_scenario_reasoning_traces.jsonl" \
    --out-file "$CACHE_DIR/oow_multihop.jsonl" --gold-file "$NORM_FILE"
run_stage 11 dpo_rule        false -m pipeline.track1.build_rlhf \
    --out-file "$CACHE_DIR/oow_dpo_pairs.jsonl" --gold-file "$NORM_FILE"
run_stage 12 reflection_rule false -m pipeline.track1.build_reflection \
    --out-file "$CACHE_DIR/oow_reflection.jsonl" --gold-file "$NORM_FILE"

log "--- § 6.1: Track 1 training data from real incident excerpts ---"
run_stage 13 sft_incident false -m pipeline.track1.build_sft \
    --traces-file "$CACHE_DIR/oow_incident_reasoning_traces.jsonl" \
    --out-prefix oow_incident_sft --gold-file "$NORM_FILE"
run_stage 14 dpo_incident false -m pipeline.track1.build_rlhf \
    --traces-file "$CACHE_DIR/oow_incident_reasoning_traces.jsonl" \
    --out-file "$CACHE_DIR/oow_incident_dpo_pairs.jsonl" --gold-file "$NORM_FILE"
run_stage 15 reflection_incident false -m pipeline.track1.build_reflection \
    --traces-file "$CACHE_DIR/oow_incident_reasoning_traces.jsonl" \
    --out-file "$CACHE_DIR/oow_incident_reflection.jsonl" --gold-file "$NORM_FILE"

log "--- § 7 / § 12.9: gold-claims enrichment (needs ANTHROPIC_API_KEY, resume-safe) ---"
run_stage 16 gold_claims_enrich false -m pipeline.eval.enrich_gold_claims --files "$NORM_FILE"
run_stage 17 gold_claims_check  false -m pipeline.eval.gold_claims --check "$NORM_CLAIMS"

log "--- § 8: Procedural Graphs -- merged + 3 source-scoped, then step-order SFT data ---"
run_stage 18 pg_merged false -m pipeline.ingest.build_pg
if [ "$RESUME_FROM" -le 19 ]; then
    [ -f "$CACHE_DIR/oow_reasoning_traces.jsonl" ] && run_stage 19 pg_rule false \
        -m pipeline.ingest.build_pg --traces-file "$CACHE_DIR/oow_reasoning_traces.jsonl" \
        --out-file "$CACHE_DIR/oow_pg_rule.json" || log "STAGE 19_pg_rule  skipped (traces file not found)"
fi
if [ "$RESUME_FROM" -le 20 ]; then
    [ -f "$CACHE_DIR/oow_incident_reasoning_traces.jsonl" ] && run_stage 20 pg_incident false \
        -m pipeline.ingest.build_pg --traces-file "$CACHE_DIR/oow_incident_reasoning_traces.jsonl" \
        --out-file "$CACHE_DIR/oow_pg_incident.json" || log "STAGE 20_pg_incident  skipped (traces file not found)"
fi
if [ "$RESUME_FROM" -le 21 ]; then
    [ -f "$CACHE_DIR/oow_scenario_reasoning_traces.jsonl" ] && run_stage 21 pg_scenario false \
        -m pipeline.ingest.build_pg --traces-file "$CACHE_DIR/oow_scenario_reasoning_traces.jsonl" \
        --out-file "$CACHE_DIR/oow_pg_scenario.json" || log "STAGE 21_pg_scenario  skipped (traces file not found)"
fi
run_stage 22 pg_sft false -m pipeline.track1.build_pg_sft --gold-file "$NORM_FILE"
if [ "$RESUME_FROM" -le 23 ]; then
    [ -f "$CACHE_DIR/oow_pg_incident.json" ] && run_stage 23 pg_sft_incident false \
        -m pipeline.track1.build_pg_sft --gold-file "$NORM_FILE" \
        --pg-file "$CACHE_DIR/oow_pg_incident.json" --out-file "$CACHE_DIR/oow_pg_incident_sft.jsonl" \
        || log "STAGE 23_pg_sft_incident  skipped (oow_pg_incident.json not found)"
fi

log "--- § 10 / § 10.1: base-Qwen baseline, Track 1 + Track 2 (n=$N) ---"
run_stage 24 eval_qwen_base        false -m pipeline.eval.eval_finetuned \
    --model Qwen/Qwen3-8B --tag "oow_qwen_base_${TAG}" --gold-file "$NORM_FILE" --force-4bit --n "$N"
run_stage 25 eval_qwen_base_colreg false -m pipeline.eval.eval_oow_scenarios \
    --model Qwen/Qwen3-8B --tag "oow_qwen_base_${TAG}" --force-4bit --n "$N"

log "--- § 11 / § 11.1: prompt ablation (V0-V6), Track 1 + Track 2 (n=$N) ---"
run_stage 26 ablation_prep    false -m pipeline.eval.prep_ablation --gold-file "$NORM_FILE" --n "$N" --tag "$TAG"
run_stage 27 ablation_run     false -m pipeline.eval.run_ablation --tag "$TAG"
run_stage 28 ablation_score   false -m pipeline.eval.score_ablation --gold-file "$NORM_FILE" --tag "$TAG"
run_stage 29 ablation_prep_t2 false -m pipeline.eval.prep_ablation --track2 --n "$N" --tag "$TAG"
run_stage 30 ablation_run_t2  false -m pipeline.eval.run_ablation --track2 --tag "$TAG"
run_stage 31 ablation_score_t2 false -m pipeline.eval.score_ablation --track2 --tag "$TAG"

log "--- § 12 / § 12.1: data-coverage gap analysis + cross-source consistency check ---"
run_stage 32 gap_analysis      false -m pipeline.eval.analyze_gaps \
    --eval-file "$CACHE_DIR/eval_oow_qwen_base_${TAG}.jsonl" --group-by section_id
run_stage 33 consistency_check false -m pipeline.eval.check_consistency

log "--- § 13 / § 13.1: fine-tune OOW-QWEN (SFT -> DPO -> Reflection -> merge), TAG=$TAG ---"
run_stage 34 sft         true -m pipeline.train.train_sft       --out-dir "$SFT_DIR" --force
run_stage 35 dpo         true -m pipeline.train.train_dpo       --out-dir "$DPO_DIR" --sft-dir "$SFT_DIR" --force
run_stage 36 reflection  true -m pipeline.train.train_reflection --out-dir "$REFLECT_DIR" \
    --sft-dir "$SFT_DIR" --dpo-dir "$DPO_DIR" --force
run_stage 37 merge       true -m pipeline.train.merge_adapter --output "$MERGED_DIR" \
    --sft-dir "$SFT_DIR" --dpo-dir "$DPO_DIR" --reflect-dir "$REFLECT_DIR" --force

log "--- § 13.2 / § 13.3: evaluate OOW-QWEN, Track 1 + Track 2 (n=$N) ---"
run_stage 38 eval_oowqwen        false -m pipeline.eval.eval_finetuned \
    --model "$MERGED_DIR" --tag "oow_qwen_${TAG}" --gold-file "$NORM_FILE" --force-4bit --n "$N"
run_stage 39 eval_oowqwen_colreg false -m pipeline.eval.eval_oow_scenarios \
    --model "$MERGED_DIR" --tag "oow_qwen_${TAG}" --force-4bit --n "$N"

log "--- § 13.4: stage-attribution probes (DPO axis win-rates + reflection delta), TAG=$TAG ---"
run_stage 40 probe_qwen_base false -m pipeline.eval.probe_dpo \
    --model Qwen/Qwen3-8B --tag "oow_qwen_base_${TAG}" --gold-file "$NORM_FILE" --force-4bit \
    --per-axis 5 --n-reflect 5
run_stage 41 probe_oowqwen   false -m pipeline.eval.probe_dpo \
    --model "$MERGED_DIR" --tag "oow_qwen_${TAG}" --gold-file "$NORM_FILE" --force-4bit \
    --per-axis 5 --n-reflect 5

log "--- § 14: AWQ int4 quantization (pip install autoawq may fail on some setups), TAG=$TAG ---"
pip install -q autoawq 2>&1 | tee "$LOG_DIR/42_awq_install.log" || log "autoawq install failed -- skipping AWQ stage."
run_stage 42 awq_quantize    false -m pipeline.compress.compress_quantize_awq \
    --input "$MERGED_DIR" --output "$AWQ_DIR"
run_stage 43 eval_awq        false -m pipeline.eval.eval_finetuned \
    --model "$AWQ_DIR" --tag "oow_qwen_awq_${TAG}" --gold-file "$NORM_FILE" --force-4bit --n "$N"
run_stage 44 eval_awq_colreg false -m pipeline.eval.eval_oow_scenarios \
    --model "$AWQ_DIR" --tag "oow_qwen_awq_${TAG}" --force-4bit --n "$N"

log "--- § 14.1: layer pruning (ShortGPT-style), TAG=$TAG ---"
run_stage 45 prune             false -m pipeline.compress.compress_prune \
    --n-prune 4 --input "$MERGED_DIR" --output "$PRUNED_DIR"
run_stage 46 eval_pruned        false -m pipeline.eval.eval_finetuned \
    --model "$PRUNED_DIR" --tag "oow_qwen_pruned_${TAG}" --gold-file "$NORM_FILE" --force-4bit --n "$N"
run_stage 47 eval_pruned_colreg false -m pipeline.eval.eval_oow_scenarios \
    --model "$PRUNED_DIR" --tag "oow_qwen_pruned_${TAG}" --force-4bit --n "$N"

log "--- § 14.2: knowledge distillation -> DistillOOW-QWEN, TAG=$TAG ---"
run_stage 48 distill            true  -m pipeline.compress.compress_distill \
    --teacher-dir "$MERGED_DIR" --output "$DISTILL_DIR"
run_stage 49 eval_distill        false -m pipeline.eval.eval_finetuned \
    --model "$DISTILL_DIR" --tag "distill_oow_qwen_${TAG}" --gold-file "$NORM_FILE" --force-4bit --n "$N"
run_stage 50 eval_distill_colreg false -m pipeline.eval.eval_oow_scenarios \
    --model "$DISTILL_DIR" --tag "distill_oow_qwen_${TAG}" --force-4bit --n "$N"

log "===== OOW CHAIN DONE (TAG=$TAG) ====="
