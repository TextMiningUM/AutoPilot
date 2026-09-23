# Audit report

Generated: 2026-09-23T19:17:58.866493+00:00

Schema versions seen: {'two_field': 24}
Runs audited: 24

## ✅ No blockers

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_base::W0_base | 6 | 0.00% | 4.41% | 2.94% | 17.65% | 4.41% | 17.65% | 0.00% | 0.00% | 13.24% | 0.00% | 67 |
| v7_super_rag::W0_base | 6 | 27.91% | 0.00% | 2.33% | 13.95% | 9.30% | 48.84% | 69.77% | 0.00% | 16.28% | 0.00% | 114 |
| v8_super_cot_pg::W0_base | 6 | 0.00% | 2.50% | 0.00% | 3.75% | 2.50% | 20.00% | 0.00% | 0.00% | 15.00% | 1.25% | 85 |
| v9_super_all::W0_base | 6 | 1.25% | 5.00% | 1.25% | 3.75% | 3.75% | 22.50% | 0.00% | 1.25% | 8.75% | 3.75% | 109 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| imazu_scenario | 12 | 0.53% | 2.66% | 0.53% | 30.32% |
| quiet | 4 | 25.00% | 0.00% | 0.00% | 0.00% |
| um_bearing_sweep | 8 | 6.38% | 8.51% | 6.38% | 21.28% |

## Canary (UM01/UM02) -- A-rate should be ~0: **25.00%**

## Top-10 worst checkpoints
- weight=14 `UM04__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=30: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=40: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=140: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=9 `Imazu01__v8_super_cot_pg__W0_base__screening_standard_cloud_v2.json` step=140: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=7 `Imazu01__v0_base__W0_base__screening_standard_cloud_v2.json` step=160: ['D_no_action_despite_risk', 'E_rule_matrix_give_way_no_action', 'E_encounter_mismatch', 'G_gate_c_risk_mismatch', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch']
- weight=7 `Imazu04__v8_super_cot_pg__W0_base__screening_standard_cloud_v2.json` step=220: ['E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_unknown', 'C_degrees_over_limit', 'W_deliberation_loop']
- weight=7 `Imazu04__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=200: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_c_risk_mismatch', 'W_deliberation_loop', 'W_repetition']
- weight=7 `UM02__v7_super_rag__W0_base__screening_standard_cloud_v2.json` step=20: ['ERROR_2_0_invalid_json', 'E_rule_matrix_none_with_conduct', 'G_gate_c_risk_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=7 `UM02__v7_super_rag__W0_base__screening_standard_cloud_v2.json` step=30: ['ERROR_2_0_invalid_json', 'E_rule_matrix_none_with_conduct', 'G_gate_c_risk_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=7 `UM02__v7_super_rag__W0_base__screening_standard_cloud_v2.json` step=50: ['ERROR_2_0_invalid_json', 'E_rule_matrix_none_with_conduct', 'G_gate_c_risk_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']