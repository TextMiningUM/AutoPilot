# Audit report

Generated: 2026-09-23T23:51:07.158602+00:00

Schema versions seen: {'two_field': 1}
Runs audited: 1

## ✅ No blockers

## Primary metrics per config x weights
| config::weights | runs | A-rate | B-rate | D-rate | E_enc | E_role | E_unclass | E_rule_matrix_none_with_conduct | gate_b | gate_c | parse-fail | real findings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v7_super_rag::W0_base | 1 | 0.00% | 0.00% | 3.57% | 10.71% | 0.00% | 3.57% | 0.00% | 0.00% | 0.00% | 0.00% | 7 |

## Per mission-type
| type | runs | A-rate | B-rate | D-rate | E_unclass |
|---|---|---|---|---|---|
| imazu_scenario | 1 | 0.00% | 0.00% | 3.57% | 3.57% |


## Top-10 worst checkpoints
- weight=6 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=200: ['D_no_action_despite_risk', 'E_unclassified_encounter', 'G_gate_a_contact_missing']
- weight=2 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=60: ['E_encounter_mismatch', 'INFO_early_action']
- weight=2 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=80: ['E_encounter_mismatch', 'INFO_early_action']
- weight=2 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=140: ['E_encounter_mismatch', 'INFO_early_action']
- weight=2 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=180: ['G_gate_e_rule_mismatch', 'INFO_early_action', 'C_degrees_over_limit']
- weight=0 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=0: ['G_gate_e_rule_mismatch', 'INFO_early_action']
- weight=0 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=20: ['INFO_early_action']
- weight=0 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=40: ['INFO_early_action']
- weight=0 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=100: ['G_gate_e_rule_mismatch', 'INFO_early_action']
- weight=0 `Imazu07__v7_super_rag__W0_base__pastclear_fix_verify.json` step=120: ['INFO_early_action']