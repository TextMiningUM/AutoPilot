# Audit report

Generated: 2026-09-23T18:43:13.717743+00:00

Schema versions seen: {'two_field': 24}
Runs audited: 24

## 🛑 1 BLOCKER(s)
- `BLOCKER_1_5_verdict_inconsistent` step=None: evaluation.safety.min_cpa_m does not match the recomputed minimum separation over the trajectory.

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_base::W0_base | 6 | 0.00% | 4.41% | 2.94% | 17.65% | 4.41% | 17.65% | 14.71% | 13.24% | 0.00% | 78 |
| v7_super_rag::W0_base | 6 | 0.00% | 0.00% | 2.33% | 13.95% | 9.30% | 48.84% | 13.95% | 16.28% | 69.77% | 108 |
| v8_super_cot_pg::W0_base | 6 | 0.00% | 2.50% | 0.00% | 3.75% | 2.50% | 20.00% | 60.00% | 15.00% | 1.25% | 133 |
| v9_super_all::W0_base | 6 | 1.25% | 5.00% | 1.25% | 3.75% | 3.75% | 22.50% | 71.25% | 7.50% | 5.00% | 166 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| imazu_scenario | 12 | 0.53% | 2.66% | 0.53% | 30.32% |
| quiet | 4 | 0.00% | 0.00% | 0.00% | 0.00% |
| um_bearing_sweep | 8 | 0.00% | 8.51% | 6.38% | 21.28% |

## Canary (UM01/UM02) -- A-rate should be ~0: **0.00%**

## Top-10 worst checkpoints
- weight=15 `UM04__v9_super_all__W0_base__screening_standard_cloud.json` step=30: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=12 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=140: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'G_gate_b_number_fabricated', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=40: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=10 `Imazu01__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=140: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'G_gate_b_number_fabricated', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=8 `Imazu01__v9_super_all__W0_base__screening_standard_cloud.json` step=180: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'G_gate_b_number_fabricated', 'W_deliberation_loop']
- weight=8 `Imazu04__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=220: ['E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_b_number_fabricated', 'G_gate_c_risk_unknown', 'C_degrees_over_limit', 'W_deliberation_loop']
- weight=8 `Imazu04__v9_super_all__W0_base__screening_standard_cloud.json` step=200: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'W_deliberation_loop', 'W_repetition']
- weight=8 `UM09__v9_super_all__W0_base__screening_standard_cloud.json` step=19: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'W_deliberation_loop', 'W_repetition']
- weight=7 `Imazu01__v0_base__W0_base__screening_standard_cloud.json` step=160: ['D_no_action_despite_risk', 'E_rule_matrix_give_way_no_action', 'E_encounter_mismatch', 'G_gate_c_risk_mismatch', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch']
- weight=7 `Imazu04__v7_super_rag__W0_base__screening_standard_cloud.json` step=20: ['ERROR_2_0_parse_error', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'G_gate_b_number_fabricated', 'INFO_early_action']