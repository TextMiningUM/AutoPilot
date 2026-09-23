# Full Report — OOW/VHF Evaluations V1

Snapshot of **every** evaluation run so far, in the same order as `OOW_Agent_Training_Pipeline.ipynb` (§10 → §15). All numbers pulled directly from the `_summary.json`/`ablation_summary_*.json`/`probe_*.json` files now archived at `Eval_v1/` on the cloud pod (and `models_v1/` for the corresponding model checkpoints). Metric suite: RAGAS v2.0, judge model `gpt-4o-mini`.

- **Track 1 (T1)** = held-out COLREG Q&A (knowledge/reasoning), 500 gold questions.
- **Track 2 (T2)** = applied helm/engine-order decisions on the 300 held-out COLREG scenarios.
- **Composite** is the headline score (weighted blend of the axes below it) — use it to compare configs/stages at a glance; the per-axis columns explain *why*.

---

## 1 — § 10/10.1: QWEN base, plain (no prompt engineering)

| Run (tag) | Track | n | AnswerCorrectness | ClaimF1 | CorpusGrounded | NumericF1 | SemSim | Composite |
|---|---|---|---|---|---|---|---|---|
| `oow_qwen_base_full` | T1 | 500 | 0.441 | 0.355 | 0.418 | 0.506 | 0.700 | **0.551** |
| `oow_qwen_base_n50` | T1 | 50 | 0.437 | 0.359 | 0.537 | 0.571 | 0.673 | 0.594 |
| `oow_qwen_base_smoke`⚠️ | T1 | 50 | 0.472 | 0.440 | 0.630 | 0.711 | 0.569 | 0.587 |
| `oow_qwen_base_full` | T2 | 300 | 0.390 | 0.269 | 0.217 | 0.246 | 0.753 | **0.436** |
| `oow_qwen_base_n50` | T2 | 50 | 0.360 | 0.231 | 0.210 | 0.216 | 0.748 | 0.416 |
| `oow_qwen_base_smoke` | T2 | 2 | 0.258 | 0.095 | 0.000 | 0.129 | 0.745 | 0.297 |

⚠️ `oow_qwen_base_smoke`'s T1 run recorded `model: "Qwen/Qwen2.5-7B-Instruct"` — an early smoke test run **before** the switch to Qwen3-8B. Not directly comparable to the other rows (all Qwen3-8B); kept for completeness only.

Extra T2-only axes for base: `ActionCorrect` 0.54 (full) / 0.68 (n50), `RuleCite` 0.661 (full) / 0.76 (n50), `DirectionCorrect` **0.0** across the board (the base model never got a turn *direction* right without prompt help — this is exactly the gap the RAG/CoT ablation below targets).

---

## 2 — § 11/11.1: Prompt ablation (RAG / CoT / Procedural-Graph), full scale (n=500 T1, n≈275-300 pairs T2)

### Track 1 (Q&A)

| Config | AnswerCorrectness | ClaimF1 | CorpusGrounded/Faithfulness | NumericF1 | SemSim | **Composite** | Δ vs v0_base | Significant? |
|---|---|---|---|---|---|---|---|---|
| v0_base (no extras) | 0.430 | 0.340 | 0.424 | 0.505 | 0.701 | **0.547** | — | — |
| v1_rag | 0.478 | 0.403 | 0.666 (Faithfulness) | 0.541 | 0.703 | **0.609** | +0.061 | ✅ yes |
| v2_cot | 0.435 | 0.347 | 0.434 | 0.494 | 0.700 | **0.543** | −0.004 | ❌ no |
| **v3_rag_cot** | **0.495** | **0.425** | 0.635 (Faithfulness) | **0.550** | 0.707 | **0.614** ⭐ | **+0.067** | ✅ yes |
| v4_pg | 0.431 | 0.342 | 0.454 | 0.512 | 0.701 | 0.556 | +0.009 | ❌ no |
| v5_pg_incident | 0.426 | 0.334 | 0.445 | 0.490 | 0.703 | 0.548 | +0.001 | ❌ no |
| v6_pg_scenario | 0.433 | 0.343 | 0.420 | 0.502 | 0.702 | 0.546 | −0.001 | ❌ no |

### Track 2 (applied decisions)

| Config | ActionCorrect | DirectionCorrect | RuleCite | ClaimF1 | **Composite** | Δ vs v0_base | Significant? |
|---|---|---|---|---|---|---|---|
| v0_base | 0.673 | 0.000 | 0.690 | 0.280 | **0.443** | — | — |
| v1_rag | 0.677 | 0.194 | 0.843 | 0.382 | **0.488** | +0.046 | ✅ yes |
| v2_cot | 0.693 | 0.005 | 0.675 | 0.260 | 0.439 | −0.004 | ❌ no |
| **v3_rag_cot** | **0.710** | **0.199** | **0.898** | **0.375** | **0.506** ⭐ | **+0.060** | ✅ yes |
| v4_pg | 0.667 | 0.044 | 0.763 | 0.283 | 0.466 | +0.024 | ✅ yes |
| v5_pg_incident | 0.630 | 0.063 | 0.730 | 0.289 | 0.469 | +0.025 | ✅ yes |
| v6_pg_scenario | 0.627 | 0.063 | 0.587 | 0.332 | 0.462 | +0.019 | ✅ yes |

**Takeaway:** `v3_rag_cot` (RAG excerpts + chain-of-thought) is the clear winner on both tracks — the only config combining a large, statistically significant Composite gain with the biggest jump in `DirectionCorrect` (turn-direction accuracy, T2). CoT alone (v2_cot) does nothing measurable; PG configs (v4-v6) help T2's `RuleCite`/Composite significantly but not T1, and never beat v3_rag_cot outright. This matches why `agents.py`'s deployed OOW agent combines RAG+CoT rather than picking a single narrower ablation winner.

---

## 3 — § 13.2/13.3: Fine-tuned OOW-QWEN (SFT → DPO → Reflection, merged)

| Run (tag) | Track | n | AnswerCorrectness | ClaimF1 | NumericF1 | ActionCorrect | DirectionCorrect | **Composite** |
|---|---|---|---|---|---|---|---|---|
| `oow_qwen_smoke` (fine-tuned, smoke) | T2 | 2 | 0.270 | 0.116 | 0.113 | 0.500 | 0.000 | 0.301 |
| `oow_qwen_full` | T1 | 500 | 0.259 | 0.134 | 0.281 | — | — | **0.375** |
| `oow_qwen_full` | T2 | 300 | 0.560 | 0.453 | 0.716 | **0.763** | **0.714** | **0.574** |

**Takeaway — a genuine split result:**
- **Track 2 improved substantially** over base (0.574 vs 0.436, +0.138) and over the best prompt-ablation config v3_rag_cot (0.574 vs 0.506) — `ActionCorrect` jumped to 0.763 and, notably, `DirectionCorrect` reached 0.714 (base was 0.0, v3_rag_cot was only 0.199). Fine-tuning clearly taught the model to pick the *correct turn direction*, not just recognize that *a* turn is needed.
- **Track 1 got WORSE** than base (0.375 vs 0.551) and worse than every single ablation config including plain v0_base (0.547). `ClaimF1` dropped from 0.355 (base) to 0.134 — the fine-tuned model answers COLREG *knowledge* questions with noticeably less grounded/complete claims than it did before fine-tuning, even though its applied-decision skill (T2) improved a lot. This is the single most important finding in this report: **the SFT/DPO/Reflection data mix currently trades Track 1 knowledge-recall for Track 2 decision-making** — worth investigating specifically once the new training round runs (e.g. check whether Track 1 training data is under-represented relative to Track 2/Leo-scenario data in the combined SFT mix).

---

## 4 — § 13.4: Stage-attribution probes (did DPO/Reflection actually do their jobs?)

Small, targeted probe set (perturbation win-rate: does the model correctly prefer the non-perturbed "gold" answer over a deliberately-broken one — wrong proword, missing step, dropped regulation/warning, swapped step order — per `probe_set.json`), **not** a full 500/300-example re-eval.

| Model | wrong_proword | missing_step | dropped_regulation | dropped_warning | swapped_step_order | **Overall win-rate** |
|---|---|---|---|---|---|---|
| `oow_qwen_base_full` (pre-fine-tune) | 1.00 (n=5) | 1.00 (n=1) | 1.00 (n=5) | 1.00 (n=3) | 0.00 (n=1) | **0.933** (n=15) |
| `oow_qwen_full` (post-fine-tune) | 1.00 (n=5) | 1.00 (n=1) | 1.00 (n=5) | 1.00 (n=3) | **1.00** (n=1) | **1.000** (n=15) |

- `wrong_channel` perturbation had **n=0** both times (no eligible probe items) — not measured.
- **Reflection probe: SKIPPED both times** — `"skipped": "no gold_claims"`. The reflection-specific stage-attribution check never actually ran (missing prerequisite `gold_claims` data), so **there is currently no direct evidence Reflection training did anything** — only DPO's contribution is verified here (small improvement, 0.933→1.000, driven entirely by fixing the one `swapped_step_order` case).

---

## 5 — § 14/14.1/14.2: Compression (AWQ, Pruning, Distillation)

| Run (tag) | Track | n | AnswerCorrectness | ClaimF1 | ActionCorrect | DirectionCorrect | **Composite** | vs `oow_qwen_full` |
|---|---|---|---|---|---|---|---|---|
| **AWQ int4** | — | — | — | — | — | — | **not evaluated** | eval permanently broken on this cloud pod (transformers 5.17.0 needs `gptqmodel`, which forces an incompatible torch upgrade — see repo memory `cloud_sync.md`) |
| `oow_qwen_pruned_full` | T1 | 500 | 0.195 | 0.069 | — | — | 0.370 | −0.005 (≈flat) |
| `oow_qwen_pruned_full` | T2 | 300 | 0.294 | 0.173 | 0.103 | 0.010 | 0.360 | **−0.214** (large regression) |
| `distill_oow_qwen_full` | T1 | 500 | 0.383 | 0.361 | — | — | 0.342 | −0.033 |
| `distill_oow_qwen_full` | T2 | 300 | 0.120 | 0.000 | **0.000** | **0.000** | **0.102** | **−0.472** (catastrophic) |

**Takeaways:**
- **AWQ**: quantization itself succeeded (the checkpoint exists in `models_v1/OOW/OOW-QWEN-awq-int4_full/`), but it has **never been evaluated** — the eval step is blocked by a dependency conflict, not a model-quality issue. Still an open gap.
- **Pruning**: roughly holds T1 steady but **badly hurts T2** (`ActionCorrect` collapses from 0.763→0.103, `DirectionCorrect` from 0.714→0.010) — the layer-pruned model essentially loses its applied-decision competence while its Q&A competence survives.
- **Distillation: confirms your suspicion — it did not work.** T2 numbers are degenerate (`ActionCorrect`, `DirectionCorrect`, `ClaimPrec/Rec/F1`, `CorpusGrounded`, `AnswerRelevancy`, `NumericF1` **all exactly 0.0**), and even T1 is the weakest "real" (non-zero) result of any full-scale run. This is consistent with a distilled student that never learned the task at all, not just a mildly worse model.

---

## 6 — VHF track (for reference — smoke-scale only, no full eval run yet)

| Run | Track | n | Composite |
|---|---|---|---|
| `qwen_base_smoke` | T1 | 2 | 0.683 |
| `qwen_base_smoke` | T2 (conversational compliance) | 2 | 0.278 |

No fine-tuned, pruned, AWQ, or distilled VHF evaluation exists yet — VHF has not progressed past the smoke-test stage on this pod.

---

## 7 — Summary ranking (full-scale runs only, Composite)

| Rank | Tag | Track 1 | Track 2 |
|---|---|---|---|
| 1 | `oow_qwen_full` (fine-tuned) | 0.375 | **0.574** ⭐ best T2 |
| 2 | `v3_rag_cot` (ablation, no fine-tune) | **0.614** ⭐ best T1 | 0.506 |
| 3 | `oow_qwen_base_full` (plain base) | 0.551 | 0.436 |
| 4 | `oow_qwen_pruned_full` | 0.370 | 0.360 |
| 5 | `distill_oow_qwen_full` | 0.342 | 0.102 ⚠️ broken |
| — | AWQ int4 | not evaluated | not evaluated |

**Net picture going into the new training round:** fine-tuning is a clear win for *applying* COLREG decisions (T2) but currently regresses *knowledge recall* (T1) below even the untuned base model; pruning trades away most of the T2 gain; distillation has not produced a usable model on either track; AWQ quality is simply unknown. All of this is exactly what the new training data (kinematics-aware missions + Leo scenarios) and a fresh SFT/DPO/Reflection/compression round should be judged against.

---

*Data sources archived alongside this report: `~/AutoPilot/Eval_v1/{OOW,VHF}/*_summary.json`, `ablation_summary_full.json`, `ablation_summary_track2_full.json`, `probe_oow_qwen_base_full.json`, `probe_oow_qwen_full.json` on the cloud pod. Corresponding model checkpoints archived at `~/AutoPilot/models_v1/{OOW,VHF}/`.*
