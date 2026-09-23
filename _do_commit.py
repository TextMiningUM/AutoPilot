import subprocess

msg = """Add prompt-ablation (v0-v6) + stage-attribution probe results, full-scale base/full/pruned/distill evals

Full-scale (n=500 T1 / n=300 T2) evaluation results for every model tag run so far:
oow_qwen_base (full, n50, smoke), oow_qwen_full (merged SFT+DPO+Reflection), 
oow_qwen_pruned_full, distill_oow_qwen_full -- plus the n=40 COLREG-only ablation
variant. Includes the prompt-ablation study (v0_base through v6_pg_scenario, both
tracks) and the DPO/Reflection stage-attribution probes.

Also includes live pipeline fixes made while running these evals on the cloud:
- compress_distill.py: sanitize teacher/student logits with nan_to_num right after
  the forward pass (a single NaN/Inf logit at a masked-out position was poisoning
  the whole KD loss reduction even after upcasting to float32 -- "NaN * 0" is still
  NaN); hard-label CE now computed manually with the same mask-then-normalize
  pattern as the KD loss instead of relying on F.cross_entropy's ignore_index
  (same NaN-through-masked-position issue); keep logits in bf16 instead of
  upcasting to float32 to roughly halve peak VRAM (large vocab dominates memory).
- eval_finetuned.py: left-padding for batched generation + a new generate_batch()
  helper (throughput scaling for eval speed, mirroring run_ablation.py's approach).
"""

subprocess.run(["git", "add", "-A"], check=True)
result = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
