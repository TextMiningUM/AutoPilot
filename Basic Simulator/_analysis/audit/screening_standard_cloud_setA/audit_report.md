# Audit report

Generated: 2026-09-23T19:17:57.203934+00:00

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
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_base::W0_base | 6 | 4.41% | 8.82% | 4.41% | 2.94% | 0.00% | 33.82% | 0.00% | 0.00% | 7.35% | 0.00% | 66 |
| v7_super_rag::W0_base | 6 | 32.39% | 0.00% | 4.23% | 0.00% | 0.00% | 42.25% | 14.08% | 0.00% | 7.04% | 0.00% | 105 |
| v8_super_cot_pg::W0_base | 6 | 4.00% | 2.67% | 2.67% | 4.00% | 2.67% | 12.00% | 0.00% | 0.00% | 20.00% | 6.67% | 114 |
| v9_super_all::W0_base | 6 | 2.33% | 6.98% | 3.49% | 4.65% | 1.16% | 16.28% | 0.00% | 0.00% | 36.05% | 5.81% | 174 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| imazu_scenario | 12 | 12.07% | 3.45% | 1.72% | 25.00% |
| quiet | 4 | 3.12% | 0.00% | 0.00% | 0.00% |
| um_bearing_sweep | 8 | 5.56% | 16.67% | 19.44% | 50.00% |

## Canary (UM01/UM02) -- A-rate should be ~0: **3.12%**

## Top-10 worst checkpoints
- weight=16 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=180: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu04__v9_super_all__W0_base__screening_standard_cloud.json` step=200: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=20: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=80: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=20: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=80: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `UM04__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=30: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `UM09__v9_super_all__W0_base__screening_standard_cloud.json` step=0: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=10 `Imazu07__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=0: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop']
- weight=10 `Imazu07__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=100: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop']