#!/bin/bash
# Cloud batch chain: wait for the running Imazu01-03 qwen_base+nomoto job to finish,
# then Fase 1A (new qwen_sftdpo_nomoto model, all remaining Imazu/UM missions),
# then Fase 2A (qwen_base, same remaining Imazu/UM missions) -- sequential, single GPU.
set -uo pipefail
cd "/home/ubuntu/AutoPilot/Basic Simulator"
export AUTOPILOT_DOMAIN=OOW

echo "=== waiting for qwen_base_nomoto_test to finish ==="
while tmux has-session -t qwen_base_nomoto_test 2>/dev/null; do
  sleep 15
done

MISSIONS="Imazu04 Imazu05 Imazu06 Imazu07 Imazu08 Imazu09 Imazu10 Imazu11 Imazu12 Imazu13 Imazu14 Imazu15 Imazu16 Imazu17 Imazu18 Imazu19 Imazu20 Imazu21 Imazu22 UM01_rescaled UM02_rescaled UM03_rescaled UM04_rescaled UM05_rescaled UM06_rescaled UM07_rescaled UM08_rescaled UM09_rescaled UM10_rescaled UM11_rescaled UM12_rescaled UM13_rescaled"
CONFIGS="v0_base v10_super_colreg_rag v11_super_colreg_rag_cot"

echo "=== FASE 1A: new model (qwen_sftdpo_nomoto), all remaining Imazu/UM missions ==="
../.venv/bin/python -m app.run_llm_scenario --missions $MISSIONS --configs $CONFIGS \
    --model qwen_sftdpo_nomoto --kinematics-model nomoto --tag nomoto_full_sweep_v1 2>&1 | \
    tee /tmp/nomoto_full_sweep_v1_newmodel.log

echo "=== FASE 2A: qwen_base, all remaining Imazu/UM missions ==="
../.venv/bin/python -m app.run_llm_scenario --missions $MISSIONS --configs $CONFIGS \
    --model qwen_base --kinematics-model nomoto --tag nomoto_full_sweep_v1 2>&1 | \
    tee /tmp/nomoto_full_sweep_v1_qwenbase.log

echo "=== CLOUD CHAIN DONE ==="
