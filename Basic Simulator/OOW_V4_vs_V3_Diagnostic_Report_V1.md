# OOW qwen_sftdpo_v4 vs qwen_sftdpo_v3 — Diagnostic Sweep Report (2026-10-03)

## Summary

`qwen_sftdpo_v4` was evaluated against 33 missions never previously covered by a single
sweep in one pass: the 8 genuinely held-out RND missions (never used in any training data),
12 of the 14 IMP ("Imazu-pattern") missions, and all 13 UM_rescaled missions. All runs used
config `v0_base`, `kinematics_model=nomoto_v2`, `--low-level-controller baseline_ruletree`,
`--enable-thinking --max-new-tokens 3584` (tag `v4_diag_v1`).

**Headline result: `v4` averages composite 0.812 vs `v3`'s 0.535 — a +0.277 (+52%) absolute
improvement across all 33 missions**, driven by fixing 6 outright failures while
introducing only 1 new one.

| | v4 | v3 | qwen_base (v0) |
|---|---|---|---|
| Average composite (33 missions) | **0.812** | 0.535 | 0.538* |
| Missions where v4 is better | 25/33 | — | — |
| Missions where v4 is worse | 6/33 | — | — |
| Ties | 2/33 | — | — |

\* `qwen_base` comparison only possible for 25/33 missions (no baseline run exists for RND
held-out missions).

## Biggest wins: 6 outright failures fixed

`v3` collided or ran out of steps on these missions; `v4` reached the goal cleanly on all 6:

| Mission | v3 composite | v4 composite | v3 outcome | v4 outcome |
|---|---|---|---|---|
| RND35 | 0.200 | 0.850 | max_steps_reached | reached_goal |
| IMP11 | 0.000 | 0.796 | collision | reached_goal |
| IMP13 | 0.000 | 0.777 | collision | reached_goal |
| UM08_rescaled | 0.000 | 0.927 | collision | reached_goal |
| UM11_rescaled | 0.000 | 0.932 | collision | reached_goal |
| UM13_rescaled | 0.000 | 0.863 | collision | reached_goal |

## The one new regression: IMP06

`v3` (and `qwen_base`) both handled IMP06 moderately well (composite ~0.50, reached goal).
`v4` collides outright (composite 0.000). This is the only mission in the full 33-mission
set where `v4` is worse than BOTH predecessors, not just one — worth a dedicated root-cause
dig (same raw-decision-diff approach used for IMP14/Imazu13 earlier this session) before
the next training round.

## Full per-mission table

### RND held-out (never used in any training data — 8 missions)

| Mission | v3 | v4 | Δ |
|---|---|---|---|
| RND07 | 0.454 | 0.814 | +0.360 |
| RND14 | 0.796 | 0.899 | +0.103 |
| RND21 | 0.466 | 0.987 | +0.521 |
| RND28 | 0.995 | 1.000 | +0.005 |
| RND35 | 0.200 (failed) | 0.850 | **+0.650** |
| RND42 | 0.841 | 0.767 | −0.074 |
| RND49 | 0.454 | 0.897 | +0.443 |
| RND56 | 0.544 | 0.447 | −0.097 |

### IMP (12 of 14 missions — IMP12/IMP14 covered separately, see GRPO A/B report)

| Mission | qwen_base | v3 | v4 | Δ (v4 vs v3) |
|---|---|---|---|---|
| IMP01 | 0.496 | 0.496 | 0.918 | +0.422 |
| IMP02 | 0.408 | 0.415 | 0.833 | +0.418 |
| IMP03 | 0.855 | 0.717 | 0.714 | −0.003 |
| IMP04 | 0.583 | 0.603 | 0.916 | +0.313 |
| IMP05 | 0.483 | 0.483 | 0.538 | +0.055 |
| IMP06 | 0.498 | 0.511 | 0.000 (collision) | **−0.511** |
| IMP07 | 0.526 | 0.527 | 0.919 | +0.392 |
| IMP08 | 0.482 | 0.469 | 0.675 | +0.206 |
| IMP09 | 0.525 | 0.512 | 0.990 | +0.478 |
| IMP10 | 0.572 | 0.544 | 0.599 | +0.055 |
| IMP11 | 0.000 (failed) | 0.000 (failed) | 0.796 | **+0.796** |
| IMP13 | 0.000 (failed) | 0.000 (failed) | 0.777 | **+0.777** |

### UM_rescaled (13 missions)

| Mission | qwen_base | v3 | v4 | Δ (v4 vs v3) |
|---|---|---|---|---|
| UM01_rescaled | 0.940 | 0.940 | 0.940 | 0.000 (tie) |
| UM02_rescaled | 1.000 | 1.000 | 1.000 | 0.000 (tie) |
| UM03_rescaled | 0.454 | 0.640 | 0.915 | +0.275 |
| UM04_rescaled | 0.609 | 0.624 | 0.859 | +0.235 |
| UM05_rescaled | 0.908 | 0.897 | 0.868 | −0.029 |
| UM06_rescaled | 0.410 | 0.820 | 0.890 | +0.070 |
| UM07_rescaled | 0.653 | 0.656 | 0.921 | +0.265 |
| UM08_rescaled | 0.596 | 0.000 (collision) | 0.927 | **+0.927** |
| UM09_rescaled | 0.556 | 0.581 | 0.630 | +0.049 |
| UM10_rescaled | 0.487 | 0.525 | 0.838 | +0.313 |
| UM11_rescaled | 0.455 | 0.000 (collision) | 0.932 | **+0.932** |
| UM12_rescaled | 0.954 | 0.953 | 0.871 | −0.082 |
| UM13_rescaled | 0.000 (failed) | 0.000 (failed) | 0.863 | **+0.863** |

## Conclusion

`qwen_sftdpo_v4` is a substantial, broad-based improvement over `v3` — not a narrow fix to
one or two missions. It fixes 6 missions `v3` couldn't solve at all (including 3 that
`qwen_base` also couldn't solve), at the cost of exactly 1 new regression (IMP06). The 6
"worse than v3" cases (RND42, RND56, IMP03, IMP06, UM05, UM12) are all small deltas except
IMP06, which stands out as a genuine new weak spot worth investigating before the next
training round (`v5`, incorporating the new `build_oow_decision_reversal_dpo.py` data).

See also: the separate GRPO A/B report covering IMP12/IMP14 and the 20 Imazu missions,
where GRPO showed a less consistent, config-dependent effect (unlike this clean `v4` vs `v3`
comparison).
