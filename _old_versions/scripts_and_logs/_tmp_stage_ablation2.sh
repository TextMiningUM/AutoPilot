#!/bin/bash
# Redo DPO with fixed hyperparameters (beta 0.1->0.05, epochs 1->0.5) then re-run the
# 3-way stage ablation: SFT-only, SFT+DPO, SFT+DPO+Reflection -- both eval tracks each.
set -uo pipefail
cd /home/ubuntu/AutoPilot
source .venv/bin/activate
export AUTOPILOT_DOMAIN=VHF
LOG_DIR=Data/VHF/VHF_Agents_Training/overnight_logs

run() {
  echo "[ablation2] ===== $* ====="
  "$@" 2>&1 | tee -a "$LOG_DIR/ablation2_stages.log"
}

# Retrain DPO (new beta/epochs) and Reflection (on top of the new DPO adapter) in place.
run python -m pipeline.train.train_dpo --epochs 0.5 --beta 0.05 --max_length 2048 --save_steps 25 --force
run python -m pipeline.train.train_reflection --epochs 2 --max_length 2048 --save_steps 50 --force

# 1) SFT only (SFT adapter unaffected by the DPO fix -- reuse as-is)
run python -m pipeline.train.merge_adapter --sft-only --output _models/VHF/VHF-QWEN-sft-only --force
run python -m pipeline.eval.eval_finetuned --model _models/VHF/VHF-QWEN-sft-only --tag vhf_qwen_sft_only
run python -m pipeline.eval.eval_colreg_scenarios --model _models/VHF/VHF-QWEN-sft-only --tag vhf_qwen_sft_only

# 2) SFT + DPO (new params, no reflection)
run python -m pipeline.train.merge_adapter --sft --dpo --output _models/VHF/VHF-QWEN-sft-dpo --force
run python -m pipeline.eval.eval_finetuned --model _models/VHF/VHF-QWEN-sft-dpo --tag vhf_qwen_sft_dpo
run python -m pipeline.eval.eval_colreg_scenarios --model _models/VHF/VHF-QWEN-sft-dpo --tag vhf_qwen_sft_dpo

# 3) SFT + DPO(new) + Reflection(new) -- full stack, overwrite VHF-QWEN
run python -m pipeline.train.merge_adapter --output _models/VHF/VHF-QWEN --force
run python -m pipeline.eval.eval_finetuned --model _models/VHF/VHF-QWEN --tag vhf_qwen
run python -m pipeline.eval.eval_colreg_scenarios --model _models/VHF/VHF-QWEN --tag vhf_qwen

echo "ABLATION2_DONE_EXIT_CODE=$?" > /home/ubuntu/AutoPilot/ABLATION2_FINISHED
