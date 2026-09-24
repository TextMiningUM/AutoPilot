# Audit report

Generated: 2026-09-24T09:44:42.351145+00:00

Schema versions seen: {'two_field': 2}
Runs audited: 2

## ✅ No blockers

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_base::W0_base | 1 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 |
| v7_super_rag::W0_base | 1 | 100.00% | 0.00% | 0.00% | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 24 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| quiet | 2 | 50.00% | 0.00% | 0.00% | 0.00% |

## Canary (UM01/UM02) -- A-rate should be ~0: **50.00%**

## Top-10 worst checkpoints
- weight=6 `UM01__v7_super_rag__W0_base__additional_um_facts_only_prompt.json` step=0: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM01__v7_super_rag__W0_base__additional_um_facts_only_prompt.json` step=20: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM01__v7_super_rag__W0_base__additional_um_facts_only_prompt.json` step=40: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM01__v7_super_rag__W0_base__additional_um_facts_only_prompt.json` step=60: ['E_encounter_mismatch', 'E_role_fabrication', 'A_fabricated_risk']
- weight=6 `UM01__v7_super_rag__W0_base__additional_um_facts_only_prompt.json` step=80: ['E_encounter_mismatch', 'E_role_fabrication', 'A_fabricated_risk']
- weight=6 `UM01__v7_super_rag__W0_base__additional_um_facts_only_prompt.json` step=100: ['E_encounter_mismatch', 'E_role_fabrication', 'A_fabricated_risk']
- weight=6 `UM01__v7_super_rag__W0_base__additional_um_facts_only_prompt.json` step=120: ['E_encounter_mismatch', 'E_role_fabrication', 'A_fabricated_risk']
- weight=6 `UM01__v7_super_rag__W0_base__additional_um_facts_only_prompt.json` step=140: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']