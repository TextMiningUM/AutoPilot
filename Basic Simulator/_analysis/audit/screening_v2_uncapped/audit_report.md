# Audit report

Generated: 2026-09-24T12:38:11.776653+00:00

Schema versions seen: {'two_field': 3}
Runs audited: 3

## 🛑 2 BLOCKER(s)
- `BLOCKER_1_2_goal_course_check_mismatch` step=20: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.
- `BLOCKER_1_2_goal_course_check_mismatch` step=130: GOAL COURSE CHECK line does not byte-match the recomputation from own-ship's trajectory position/heading.

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_base::W0_base | 2 | 0.00% | 0.00% | 2.67% | 26.67% | 1.33% | 2.67% | 0.00% | 0.00% | 10.67% | 0.00% | 45 |
| v7_super_rag::W0_base | 1 | 6.25% | 0.00% | 4.69% | 43.75% | 25.00% | 3.12% | 0.00% | 0.00% | 20.31% | 0.00% | 72 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| imazu_scenario | 2 | 3.67% | 0.00% | 4.59% | 3.67% |
| um_bearing_sweep | 1 | 0.00% | 0.00% | 0.00% | 0.00% |


## Top-10 worst checkpoints
- weight=7 `Imazu01__v7_super_rag__W0_base__screening_v2_uncapped.json` step=177: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_c_risk_mismatch', 'G_gate_e_rule_mismatch', 'B_17c']
- weight=6 `Imazu01__v0_base__W0_base__screening_v2_uncapped.json` step=20: ['BLOCKER_1_2_goal_course_check_mismatch', 'C_degrees_over_limit']
- weight=6 `Imazu01__v7_super_rag__W0_base__screening_v2_uncapped.json` step=162: ['D_no_action_despite_risk', 'E_encounter_mismatch', 'G_gate_c_risk_mismatch', 'W_speed_oscillation']
- weight=6 `Imazu01__v7_super_rag__W0_base__screening_v2_uncapped.json` step=468: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=6 `Imazu01__v7_super_rag__W0_base__screening_v2_uncapped.json` step=478: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_e_rule_mismatch', 'G_gate_e_rule_mismatch', 'A_fabricated_risk']
- weight=5 `Imazu01__v0_base__W0_base__screening_v2_uncapped.json` step=174: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_c_risk_mismatch']
- weight=5 `Imazu01__v0_base__W0_base__screening_v2_uncapped.json` step=202: ['D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_c_risk_mismatch']
- weight=5 `Imazu01__v0_base__W0_base__screening_v2_uncapped.json` step=205: ['D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_c_risk_mismatch']
- weight=5 `Imazu01__v7_super_rag__W0_base__screening_v2_uncapped.json` step=168: ['E_encounter_mismatch', 'E_role_fabrication', 'W_speed_oscillation']
- weight=5 `Imazu01__v7_super_rag__W0_base__screening_v2_uncapped.json` step=174: ['E_encounter_mismatch', 'E_role_fabrication', 'G_gate_c_risk_mismatch']