#!/bin/bash
cd ~/AutoPilot/Data/OOW/OOW_Agents_Training/overnight_logs || exit 1
echo "=== progress ==="
grep -c 'done in' n50_ablation_run_t1.log
echo "=== avg/total seconds so far ==="
grep 'done in' n50_ablation_run_t1.log | sed -E 's/.*done in ([0-9.]+)s/\1/' | awk '{sum+=$1; n++} END {printf "avg=%.1f  n=%d  total=%.0f\n", sum/n, n, sum}'
echo "=== per-config avg seconds ==="
grep 'starting cfg' n50_ablation_run_t1.log | sed -E 's/.*cfg=([a-z0-9_]+).*done in ([0-9.]+)s/\1 \2/' | awk '{sum[$1]+=$2; n[$1]++} END {for (c in n) printf "%-15s n=%d avg=%.1fs\n", c, n[c], sum[c]/n[c]}'
echo "=== now (UTC) ==="
date -u
