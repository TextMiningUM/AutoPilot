#!/bin/bash
# Corrected cloud chain (2026-09-26, revised to avoid duplicating local's qwen_base+Nomoto
# work on Imazu/UM): new model (qwen_sftdpo_nomoto) covers Imazu/UM AND all IMP missions;
# qwen_base only covers the still-missing IMP11-13 (IMP01-10 already done earlier).
set -uo pipefail
cd "/home/ubuntu/AutoPilot/Basic Simulator"
export AUTOPILOT_DOMAIN=OOW

IMAZU_UM="Imazu04 Imazu05 Imazu06 Imazu07 Imazu08 Imazu09 Imazu10 Imazu11 Imazu12 Imazu13 Imazu14 Imazu15 Imazu16 Imazu17 Imazu18 Imazu19 Imazu20 Imazu21 Imazu22 UM01_rescaled UM02_rescaled UM03_rescaled UM04_rescaled UM05_rescaled UM06_rescaled UM07_rescaled UM08_rescaled UM09_rescaled UM10_rescaled UM11_rescaled UM12_rescaled UM13_rescaled"
IMP_ALL="IMP01 IMP02 IMP03 IMP04 IMP05 IMP06 IMP07 IMP08 IMP09 IMP10 IMP11 IMP12 IMP13"
IMP_1113="IMP11 IMP12 IMP13"
CONFIGS="v0_base v10_super_colreg_rag v11_super_colreg_rag_cot"

echo "=== STAGE 1: new model (qwen_sftdpo_nomoto), Imazu04-22 + UM01-13 ==="
../.venv/bin/python -m app.run_llm_scenario --missions $IMAZU_UM --configs $CONFIGS \
    --model qwen_sftdpo_nomoto --kinematics-model nomoto --tag nomoto_full_sweep_v1 2>&1 | \
    tee /tmp/nomoto_full_sweep_v1_newmodel_imazu_um.log

echo "=== STAGE 2: new model (qwen_sftdpo_nomoto), ALL IMP01-13 ==="
../.venv/bin/python -m app.run_llm_scenario --missions $IMP_ALL --configs $CONFIGS \
    --model qwen_sftdpo_nomoto --kinematics-model nomoto --tag nomoto_full_sweep_v1 2>&1 | \
    tee /tmp/nomoto_full_sweep_v1_newmodel_imp.log

echo "=== STAGE 3: qwen_base, IMP11-13 (IMP01-10 already done earlier) ==="
../.venv/bin/python -m app.run_llm_scenario --missions $IMP_1113 --configs $CONFIGS \
    --model qwen_base --kinematics-model nomoto --tag imp_qwen_base_v1 2>&1 | \
    tee /tmp/imp_qwen_base_v1_1113_cloud.log

echo "=== CLOUD CHAIN DONE ==="
