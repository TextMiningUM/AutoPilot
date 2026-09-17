# Download the trained VHF-QWEN artefacts from the cloud GPU pod back to the laptop.
# Pulls:
#   - _models/VHF/VHF-QWEN/         (merged model, ~14 GB)
#   - _models/VHF/DistillVHF-QWEN/  (if distillation ran, ~3 GB)
#   - Data/VHF/VHF_Agents_Training/eval_*.jsonl + eval_*_summary.json (both tracks)
#   - Data/VHF/VHF_Agents_Training/ablation_*.json(l) + probe_*.json  (suite v2 results)
#   - Data/VHF/VHF_Agents_Training/ragas_judge_cache.jsonl            (merge back locally)
#   - Data/VHF/VHF_Agents_Training/overnight_logs/                    (training logs)
#
# Usage:
#   pwsh -File cloud/download_results.ps1 -Server <IP> -Port 22 [-User ubuntu] [-SkipMerged]

param(
    [Parameter(Mandatory=$true)] [string]$Server,
    [int]$Port = 22,
    [string]$User = "ubuntu",
    [string]$RemotePath = "~/AutoPilot",
    [string]$KeyPath = "C:\Users\jcsch\Documents\Python\LeafCloud\AutoPilot\KeyPairAutoPilot.txt",
    [switch]$SkipMerged    # skip the big 14 GB merged model, only pull eval + adapters
)

$W = "C:\Users\jcsch\Documents\Python\Auto Pilot"
Set-Location $W

$sshTarget = "${User}@${Server}"
$sshOpts   = "-P", $Port, "-i", $KeyPath, "-o", "StrictHostKeyChecking=no", "-r"

New-Item -ItemType Directory -Force -Path "$W\_models\VHF"                       | Out-Null
New-Item -ItemType Directory -Force -Path "$W\Data\VHF\VHF_Agents_Training"     | Out-Null

Write-Host "Downloading results from ${sshTarget}:${RemotePath}"
Write-Host ""

# 1. Eval summaries + answers (both tracks; suite v2 + legacy stamps), ablation, probes
Write-Host "== 1/5 Eval + ablation + probe results (suite v2) =="
& scp @sshOpts "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/eval_*.jsonl" "$W\Data\VHF\VHF_Agents_Training\"
& scp @sshOpts "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/eval_*_summary.json" "$W\Data\VHF\VHF_Agents_Training\"
& scp @sshOpts "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/ablation_*.json" "$W\Data\VHF\VHF_Agents_Training\"
& scp @sshOpts "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/ablation_scored.jsonl" "$W\Data\VHF\VHF_Agents_Training\"
& scp @sshOpts "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/probe_*.json" "$W\Data\VHF\VHF_Agents_Training\"
# Judge cache: pull to a separate file, don't overwrite the local one -- merge manually
# (both sides may have scored different rows since the bundle was uploaded).
& scp @sshOpts "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/ragas_judge_cache.jsonl" "$W\Data\VHF\VHF_Agents_Training\ragas_judge_cache_from_cloud.jsonl"

# 2. Training logs
Write-Host ""
Write-Host "== 2/5 Training logs =="
& scp @sshOpts "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/overnight_logs" "$W\Data\VHF\VHF_Agents_Training\"

# 3. LoRA adapters (small, ~100 MB each)
Write-Host ""
Write-Host "== 3/5 LoRA adapters =="
& scp @sshOpts "${sshTarget}:${RemotePath}/_models/VHF/vhf_qwen_sft_lora"     "$W\_models\VHF\"
& scp @sshOpts "${sshTarget}:${RemotePath}/_models/VHF/vhf_qwen_dpo_lora"     "$W\_models\VHF\"
& scp @sshOpts "${sshTarget}:${RemotePath}/_models/VHF/vhf_qwen_reflect_lora" "$W\_models\VHF\"

# 4. Merged model (BIG - 14 GB, skip if only need eval)
if (-not $SkipMerged) {
    Write-Host ""
    Write-Host "== 4/5 Merged VHF-QWEN (~14 GB, ~15 min on 100 Mbit) =="
    & scp @sshOpts "${sshTarget}:${RemotePath}/_models/VHF/VHF-QWEN" "$W\_models\VHF\"
} else {
    Write-Host ""
    Write-Host "== 4/5 Merged VHF-QWEN =="
    Write-Host "   Skipped (-SkipMerged flag)."
}

# 5. Distilled model (if it exists on the pod)
Write-Host ""
Write-Host "== 5/5 DistillVHF-QWEN (if present, ~3 GB) =="
$exists = & ssh -p $Port -i $KeyPath -o StrictHostKeyChecking=no $sshTarget "test -d $RemotePath/_models/VHF/DistillVHF-QWEN && echo yes || echo no"
if ($exists.Trim() -eq "yes") {
    & scp @sshOpts "${sshTarget}:${RemotePath}/_models/VHF/DistillVHF-QWEN" "$W\_models\VHF\"
} else {
    Write-Host "   Not on pod (distillation stage did not complete). Skipping."
}

Write-Host ""
Write-Host "Download done."
if (Test-Path "$W\Data\VHF\VHF_Agents_Training\ragas_judge_cache_from_cloud.jsonl") {
    Write-Host ""
    Write-Host "NOTE: cloud judge cache saved as ragas_judge_cache_from_cloud.jsonl -- append its lines"
    Write-Host "      to ragas_judge_cache.jsonl (dedup by the 'k' field) to reuse those verdicts locally."
}
Write-Host ""
Write-Host "Sanity check:"
Get-ChildItem "$W\_models\VHF" -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host ("  {0,-30}  {1,6:N2} GB" -f $_.Name, $size)
}
Get-ChildItem "$W\Data\VHF\VHF_Agents_Training\eval_*_summary.json" | ForEach-Object {
    Write-Host "  $($_.Name)"
}
