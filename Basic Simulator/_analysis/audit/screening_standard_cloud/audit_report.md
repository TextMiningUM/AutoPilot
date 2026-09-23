# Audit report

Generated: 2026-09-23T16:26:05.511859+00:00

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
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_base::W0_base | 6 | 4.41% | 8.82% | 4.41% | 2.94% | 0.00% | 33.82% | 0.00% | 7.35% | 0.00% | 66 |
| v7_super_rag::W0_base | 6 | 21.13% | 0.00% | 4.23% | 0.00% | 0.00% | 42.25% | 1.41% | 7.04% | 14.08% | 98 |
| v8_super_cot_pg::W0_base | 6 | 4.00% | 2.67% | 4.00% | 4.00% | 2.67% | 13.33% | 2.67% | 20.00% | 9.33% | 120 |
| v9_super_all::W0_base | 6 | 2.33% | 6.98% | 3.49% | 4.65% | 1.16% | 17.44% | 1.16% | 36.05% | 6.98% | 177 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| imazu_scenario | 12 | 9.05% | 3.45% | 1.72% | 25.43% |
| quiet | 4 | 3.12% | 0.00% | 0.00% | 0.00% |
| um_bearing_sweep | 8 | 2.78% | 16.67% | 22.22% | 52.78% |

## Canary (UM01/UM02) -- A-rate should be ~0: **3.12%**

## Top-10 worst checkpoints
- weight=16 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=180: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_a_contact_missing', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=14 `UM09__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=19: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_c_risk_mismatch', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu04__v9_super_all__W0_base__screening_standard_cloud.json` step=200: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=20: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=80: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=20: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `Imazu07__v9_super_all__W0_base__screening_standard_cloud.json` step=80: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `UM04__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=30: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=11 `UM09__v9_super_all__W0_base__screening_standard_cloud.json` step=0: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop', 'W_repetition']
- weight=10 `Imazu07__v8_super_cot_pg__W0_base__screening_standard_cloud.json` step=0: ['ERROR_2_0_parse_error', 'ERROR_2_0_thinking_leak', 'ERROR_2_0_invalid_json', 'E_unclassified_encounter', 'INFO_early_action', 'W_truncated', 'W_deliberation_loop']