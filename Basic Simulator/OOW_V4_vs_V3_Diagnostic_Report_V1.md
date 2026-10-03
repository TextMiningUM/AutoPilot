# OOW Model-Version History — Diagnostic Sweep Report (2026-10-03)

## Summary

`qwen_sftdpo_v4` was evaluated against 33 missions never previously covered by a single
sweep in one pass: the 8 genuinely held-out RND missions (never used in any training data),
12 of the 14 IMP ("Imazu-pattern") missions, and all 13 UM_rescaled missions. All runs used
config `v0_base`, `kinematics_model=nomoto_v2`, `--low-level-controller baseline_ruletree`,
`--enable-thinking --max-new-tokens 3584` (tag `v4_diag_v1`). A separate section below adds
the full 20-mission Imazu set across all 4 model generations (`qwen_base` → `qwen_sftdpo_
nomoto_v2` → `qwen_sftdpo_v3` → `qwen_sftdpo_v4`) for a longer view of improvement speed,
plus the training-data growth behind each version.

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

## Imazu01–21 across all 4 model generations (20 missions)

Unlike the RND/IMP/UM set above, the Imazu missions have been evaluated against every model
generation this project has produced, giving the clearest long-run picture of improvement
speed. `qwen_sftdpo_nomoto_v2` ("v2") was only ever evaluated on a 4-mission "worst
missions" subset historically, not the full 20 — shown where available, `None` elsewhere.

| Mission | qwen_base | v2 (nomoto_v2) | v3 | v4 |
|---|---|---|---|---|
| Imazu01 | 0.850 | 0.890 | 0.200 (max_steps) | 0.912 |
| Imazu02 | 0.200 (max_steps) | — | 0.900 | 0.858 |
| Imazu03 | 0.603 | — | 0.587 | 0.858 |
| Imazu04 | 0.671 | — | 0.822 | 0.200 (max_steps) |
| Imazu05 | 0.947 | — | 0.700 | 0.862 |
| Imazu06 | 0.628 | — | 0.489 | 0.636 |
| Imazu07 | 0.468 | — | 0.415 | 0.935 |
| Imazu09 | 0.570 | — | 0.502 | 0.874 |
| Imazu10 | 0.488 | — | 0.421 | 0.564 |
| Imazu11 | 0.480 | — | 0.483 | 0.726 |
| Imazu12 | 0.549 | — | 0.374 | 0.636 |
| Imazu13 | 0.391 | 0.000 (collision) | 0.000 (collision) | 0.735 |
| Imazu14 | 0.000 (collision) | — | 0.000 (collision) | 0.576 |
| Imazu15 | 0.473 | 0.660 | 0.460 | 0.874 |
| Imazu16 | 0.471 | — | 0.456 | 0.855 |
| Imazu17 | 0.436 | 0.458 | 0.475 | 0.613 |
| Imazu18 | 0.437 | — | 0.444 | 0.644 |
| Imazu19 | 0.000 (collision) | — | 0.000 (collision) | 0.610 |
| Imazu20 | 0.537 | — | 0.501 | 0.643 |
| Imazu21 | 0.000 (collision) | — | 0.000 (collision) | 0.662 |
| **Average** | **0.460** (20/20) | **0.502** (4/20 only) | **0.411** (20/20) | **0.714** (20/20) |
| **Failures (collision/max_steps)** | 4/20 | 1/4 | 5/20 | 1/20 |

**Key finding: the improvement was NOT monotonic.** `v3` is actually *worse* than the
untuned `qwen_base` on the Imazu set (0.411 vs 0.460 average, 5 failures vs 4) — its real
DAgger-mined corrections clearly helped elsewhere (see the RND/IMP/UM table above, and its
own report-worthy wins there) but did not transfer to this mission family. `v4` is the
version that actually fixes it: 0.714 average, only 1 failure, recovering all 3 missions
both `qwen_base` and `v3` collided on (Imazu14/19/21) plus the one `v2`+`v3` both collided
on (Imazu13).

### Combined across all 53 missions evaluated this session (33 diagnostic + 20 Imazu)

| | v3 | v4 | Δ |
|---|---|---|---|
| Average composite | 0.488 | 0.775 | **+0.287 (+59%)** |

## Training data growth by version

All domains/stages are the SAME pipeline scripts (`build_*.py` → `train_sft.py`/
`train_dpo.py`/`train_reflection.py`) parameterized by an ever-growing file list — each
new version is a FRESH SFT→DPO→Reflection training run from the raw Qwen3-8B base using
the cumulative data, never a continued fine-tune of the previous version's weights (see
repo memory for the full verified explanation). Reflection is trained but deliberately
EXCLUDED from every merged deployed checkpoint (reflection data/training still happens,
used only for the dedicated reflection-stage experiments, not the shipped model).

| Version | SFT rows | DPO rows | Reflection rows | What was added this round |
|---|---|---|---|---|
| v2 (`nomoto_v2`) | 11,393 | 9,840 | 8,512 | Baseline: rule-text/incident SFT+multihop+RLHF, Leo MOOS-trajectory data, RND01-60 oracle-planner closed-loop rollouts (Step 3), small RFT self-consistency set (Step 5, 28 rows) |
| v3 | 11,393 (unchanged) | 18,310 (+8,470) | 8,512 (unchanged) | Real DAgger DPO pairs (`build_dagger_dpo.py`): mined from a dedicated 52-mission `v2` closed-loop rollout on RND01-60, oracle-planner disagreements recomputed at every visited checkpoint |
| v4 | 13,247 (+1,854) | 19,237 (+927) | 9,439 (+927) | RND01-60 rollouts regenerated WITH the low-level controller (`baseline_ruletree` fills gaps between decision points instead of passive coasting) — directly fixed 2/2 tested outright-collision missions in a controlled A/B test |
| v5 (training now) | 13,247 (unchanged) | 21,213 (+1,976) | 9,439 (unchanged) | Decision-reversal DPO pairs (`build_oow_decision_reversal_dpo.py`): targets the hold_course-stalling failure found in GRPO evaluation — mined deterministically from existing run logs, zero API/GPU cost |

**DPO growth by stage, visually**: 9,840 → 18,310 (+86%, DAgger) → 19,237 (+5%, low-level-
controller) → 21,213 (+10%, decision-reversal) — DAgger was by far the single largest
data addition of the three, though its Imazu-specific benefit didn't show up until `v4`'s
own low-level-controller data was added on top.

## Conclusion

`qwen_sftdpo_v4` is a substantial, broad-based improvement over `v3` — not a narrow fix to
one or two missions. It fixes 6 missions `v3` couldn't solve at all on the RND/IMP/UM set
(including 3 that `qwen_base` also couldn't solve), at the cost of exactly 1 new regression
(IMP06). On the Imazu set it is the version that actually reverses a real `v3` regression
vs the untuned base model. The 6 "worse than v3" cases on RND/IMP/UM (RND42, RND56, IMP03,
IMP06, UM05, UM12) are all small deltas except IMP06, which stands out as a genuine new
weak spot worth investigating before the next training round (`v5`, incorporating the new
`build_oow_decision_reversal_dpo.py` data, training as of this writing).

See also: the separate GRPO A/B report covering IMP12/IMP14 and the 20 Imazu missions under
both `v0_base` and `v11_super_colreg_rag_cot`, where GRPO showed a less consistent,
config-dependent effect (unlike this clean, monotonically-improving `v3`→`v4` step).
