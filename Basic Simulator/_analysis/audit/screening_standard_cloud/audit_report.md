# Audit report

Generated: 2026-09-23T14:31:55.656162+00:00

Schema versions seen: {'two_field': 24}
Runs audited: 24

## 🛑 8 BLOCKER(s)
- `BLOCKER_1_5_verdict_inconsistent` step=None: A CPA-safety-distance violation occurred (min separation below this mission's safe distance) but the run is labelled a plain PASS/reached_goal without any safety flag.
- `BLOCKER_1_5_verdict_inconsistent` step=None: A CPA-safety-distance violation occurred (min separation below this mission's safe distance) but the run is labelled a plain PASS/reached_goal without any safety flag.
- `BLOCKER_1_5_verdict_inconsistent` step=None: A CPA-safety-distance violation occurred (min separation below this mission's safe distance) but the run is labelled a plain PASS/reached_goal without any safety flag.
- `BLOCKER_1_5_verdict_inconsistent` step=None: A CPA-safety-distance violation occurred (min separation below this mission's safe distance) but the run is labelled a plain PASS/reached_goal without any safety flag.
- `BLOCKER_1_5_verdict_inconsistent` step=None: A CPA-safety-distance violation occurred (min separation below this mission's safe distance) but the run is labelled a plain PASS/reached_goal without any safety flag.
- `BLOCKER_1_5_verdict_inconsistent` step=None: A CPA-safety-distance violation occurred (min separation below this mission's safe distance) but the run is labelled a plain PASS/reached_goal without any safety flag.
- `BLOCKER_1_5_verdict_inconsistent` step=None: A CPA-safety-distance violation occurred (min separation below this mission's safe distance) but the run is labelled a plain PASS/reached_goal without any safety flag.
- `BLOCKER_1_5_verdict_inconsistent` step=None: A CPA-safety-distance violation occurred (min separation below this mission's safe distance) but the run is labelled a plain PASS/reached_goal without any safety flag.

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | parse-fail |
|---|---|---|---|---|---|---|---|
| v0_base::W0_base | 6 | 35.29% | 4.41% | 4.41% | 2.94% | 0.00% | 0.00% |
| v7_super_rag::W0_base | 6 | 54.93% | 0.00% | 4.23% | 0.00% | 0.00% | 14.08% |
| v8_super_cot_pg::W0_base | 6 | 18.67% | 0.00% | 4.00% | 4.00% | 2.67% | 9.33% |
| v9_super_all::W0_base | 6 | 20.93% | 2.33% | 3.49% | 4.65% | 1.16% | 6.98% |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate |
|---|---|---|---|---|
| imazu_scenario | 12 | 36.21% | 0.43% | 1.72% |
| quiet | 4 | 3.12% | 0.00% | 0.00% |
| um_bearing_sweep | 8 | 27.78% | 11.11% | 22.22% |

## Canary (UM01/UM02) -- A-rate should be ~0: **3.12%**

## Top-10 worst checkpoints
- weight=14 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=180: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'G_gate_a_contact_missing', 'G_gate_b_number_fabricated', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=12 `UM09__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=19: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'G_gate_b_number_fabricated', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu04__v9_super_all__W0_base__screening_standard_cloud.json` step=160: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'G_gate_d_direction_mismatch', 'A_fabricated_risk', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu04__v9_super_all__W0_base__screening_standard_cloud.json` step=200: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=20: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=80: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `UM04__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=30: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `UM09__v9_super_all__W0_base__screening_standard_cloud.json` step=0: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=10 `Imazu04__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=0: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_b_number_fabricated', 'G_gate_c_risk_mismatch', 'A_fabricated_risk', 'W_deliberation_loop', 'W_repetition']
- weight=10 `Imazu07__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=20: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'G_gate_b_number_fabricated', 'W_truncated', 'W_deliberation_loop', 'W_repetition']