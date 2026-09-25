# Audit report

Generated: 2026-09-25T12:03:05.092082+00:00

Schema versions seen: {'two_field': 7}
Runs audited: 7

## 🛑 9 BLOCKER(s)
- `BLOCKER_1_2_goal_course_check_mismatch` step=153: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=20: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=40: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=60: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=80: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=100: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=120: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=135: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=150: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_base::oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2+oow_qwen_reflect_lora_v2 | 1 | 0.00% | 0.00% | 40.91% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 90 |
| v0_base::oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2 | 1 | 2.63% | 5.26% | 5.26% | 31.58% | 0.00% | 10.53% | 0.00% | 0.00% | 15.79% | 0.00% | 31 |
| v0_base::oow_qwen_sft_lora_v2 | 1 | 0.00% | 0.00% | 40.91% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 83 |
| v10_super_colreg_rag::oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2+oow_qwen_reflect_lora_v2 | 1 | 0.00% | 0.00% | 40.91% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 9.09% | 100.00% | 110 |
| v10_super_colreg_rag::oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2 | 1 | 25.58% | 0.00% | 0.00% | 27.91% | 25.58% | 2.33% | 0.00% | 0.00% | 0.00% | 0.00% | 48 |
| v10_super_colreg_rag::oow_qwen_sft_lora_v2 | 1 | 0.00% | 0.00% | 40.91% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 83 |
| v11_super_colreg_rag_cot::oow_qwen_sft_lora_v2 | 1 | 0.00% | 0.00% | 40.91% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% | 72.73% | 143 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| imazu_scenario | 7 | 6.28% | 1.05% | 24.61% | 60.21% |


## Top-10 worst checkpoints
- weight=21 `Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2__compare_v2.json` step=171: ['ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_unknown', 'W_premature_resume']
- weight=19 `Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2__compare_v2.json` step=141: ['ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'G_gate_c_risk_unknown', 'INFO_early_action', 'W_repetition']
- weight=18 `Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2__compare_v2.json` step=60: ['ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'G_gate_c_risk_unknown', 'INFO_early_action']
- weight=18 `Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2__compare_v2.json` step=80: ['ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'G_gate_c_risk_unknown', 'INFO_early_action']
- weight=18 `Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2__compare_v2.json` step=120: ['ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'G_gate_c_risk_unknown', 'INFO_early_action']
- weight=16 `Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2__compare_v2.json` step=40: ['ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'G_gate_c_risk_unknown', 'INFO_early_action']
- weight=13 `Imazu01__v10_super_colreg_rag__oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2+oow_qwen_reflect_lora_v2__compare_v2.json` step=156: ['ERROR_2_0_parse_error', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_unknown', 'W_truncated', 'W_repetition', 'W_premature_resume']
- weight=13 `Imazu01__v10_super_colreg_rag__oow_qwen_sft_lora_v2+oow_qwen_dpo_lora_v2+oow_qwen_reflect_lora_v2__compare_v2.json` step=171: ['ERROR_2_0_parse_error', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_unknown', 'W_truncated', 'W_repetition', 'W_premature_resume']
- weight=13 `Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2__compare_v2.json` step=156: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_unknown', 'W_premature_resume']
- weight=13 `Imazu01__v11_super_colreg_rag_cot__oow_qwen_sft_lora_v2__compare_v2.json` step=159: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_unknown', 'W_premature_resume']