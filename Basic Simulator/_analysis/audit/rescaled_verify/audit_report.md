# Audit report

Generated: 2026-09-24T11:47:31.564501+00:00

Schema versions seen: {'two_field': 2}
Runs audited: 2

## 🛑 1 BLOCKER(s)
- `BLOCKER_1_2_goal_course_check_mismatch` step=180: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_base::W0_base | 1 | 0.00% | 0.00% | 5.26% | 26.32% | 21.05% | 5.26% | 0.00% | 0.00% | 10.53% | 0.00% | 15 |
| v7_super_rag::W0_base | 1 | 47.37% | 0.00% | 0.00% | 89.47% | 42.11% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 36 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| um_bearing_sweep | 2 | 23.68% | 0.00% | 2.63% | 2.63% |


## Top-10 worst checkpoints
- weight=8 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=180: ['BLOCKER_1_2_goal_course_check_mismatch', 'A_fabricated_risk', 'B_17c']
- weight=6 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=200: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=220: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=240: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=260: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=280: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=300: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=320: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `UM07_rescaled__v7_super_rag__W0_base__rescaled_verify.json` step=340: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=5 `UM07_rescaled__v0_base__W0_base__rescaled_verify.json` step=40: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_d_direction_mismatch', 'INFO_early_action']