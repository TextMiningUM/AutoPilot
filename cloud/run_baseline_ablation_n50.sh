#!/bin/bash
# One-off: base-Qwen baseline + prompt ablation (Track 1 + Track 2) at n=50,
# NO training. Uses tag 'n50' so files never collide with the earlier
# n=2 'smoke'-tagged run. Mirrors OOW_Agent_Training_Pipeline.ipynb §10/§10.1/§11/§11.1.
set -uo pipefail
export AUTOPILOT_DOMAIN=OOW
cd "$HOME/AutoPilot"
source .venv/bin/activate
NORM_FILE="Data/OOW/OOW_Eval/colreg_qa_500_normalised.json"
LOG_DIR="Data/OOW/OOW_Agents_Training/overnight_logs"
mkdir -p "$LOG_DIR"

run() {
    local name="$1"; shift
    echo "=== $name ==="
    python -X utf8 "$@" 2>&1 | tee "$LOG_DIR/n50_$name.log"
    echo "--- $name exit=${PIPESTATUS[0]} ---"
}

run eval_base_t1      -m pipeline.eval.eval_finetuned --model Qwen/Qwen3-8B --tag oow_qwen_base_n50 --gold-file "$NORM_FILE" --force-4bit --n 50
run eval_base_t2      -m pipeline.eval.eval_oow_scenarios --model Qwen/Qwen3-8B --tag oow_qwen_base_n50 --force-4bit --n 50
run ablation_prep_t1  -m pipeline.eval.prep_ablation --gold-file "$NORM_FILE" --n 50 --tag n50
run ablation_run_t1   -m pipeline.eval.run_ablation --tag n50
run ablation_score_t1 -m pipeline.eval.score_ablation --gold-file "$NORM_FILE" --tag n50
run ablation_prep_t2   -m pipeline.eval.prep_ablation --track2 --n 50 --tag n50
run ablation_run_t2    -m pipeline.eval.run_ablation --track2 --tag n50
run ablation_score_t2  -m pipeline.eval.score_ablation --track2 --tag n50
echo "===== DONE ====="
