# Audit report

Generated: 2026-09-23T23:02:32.839553+00:00

Schema versions seen: {'two_field': 3}
Runs audited: 3

## ✅ No blockers

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v9_super_all::W0_base | 3 | 1.75% | 3.51% | 0.00% | 3.51% | 3.51% | 28.07% | 0.00% | 0.00% | 7.02% | 3.51% | 76 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| imazu_scenario | 3 | 1.75% | 3.51% | 0.00% | 28.07% |


## Top-10 worst checkpoints
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=40: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=140: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=7 `Imazu04__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=200: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_c_risk_mismatch', 'W_deliberation_loop', 'W_repetition']
- weight=5 `Imazu04__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=0: ['E_encounter_mismatch', 'E_role_fabrication', 'INFO_early_action', 'W_deliberation_loop']
- weight=5 `Imazu04__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=220: ['E_unclassified_encounter', 'G_gate_c_risk_mismatch', 'C_degrees_over_limit']
- weight=5 `Imazu07__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=200: ['E_unclassified_encounter', 'G_gate_c_risk_unknown', 'INFO_early_action', 'C_degrees_over_limit', 'W_premature_resume']
- weight=4 `Imazu01__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=180: ['ERROR_2_0_invalid_json', 'G_gate_c_risk_mismatch', 'W_deliberation_loop']
- weight=4 `Imazu07__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=160: ['G_gate_d_direction_mismatch', 'G_gate_e_rule_mismatch', 'INFO_early_action', 'B_wrong_direction', 'W_deliberation_loop']
- weight=4 `Imazu07__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=180: ['B_wrong_direction', 'W_deliberation_loop', 'W_repetition']
- weight=4 `Imazu07__v9_super_all__W0_base__screening_standard_cloud_v2.json` step=220: ['E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_unknown']