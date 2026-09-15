# Upload the minimal training-data bundle to a cloud GPU pod via scp.
# Only sends what's needed:
#   - Data/VHF/VHF_Agents_Training/*.jsonl  (SFT/DPO/reflection datasets)
#   - Data/VHF/VHF_Eval/vhf_gold_answers.json
#   - .env  (optional; needed only if you want to run eval judge on the pod)
#
# Usage:
#   pwsh -File cloud/upload_bundle.ps1 -Server <IP_OR_HOSTNAME> -Port 22 [-User ubuntu] [-Env]

param(
    [Parameter(Mandatory=$true)] [string]$Server,
    [int]$Port = 22,
    [string]$User = "ubuntu",
    [string]$RemotePath = "~/AutoPilot",
    [switch]$IncludeEnv
)

$W = "C:\Users\jcsch\Documents\Python\Auto Pilot"
Set-Location $W

$sshTarget = "${User}@${Server}"
$sshOpts   = "-p", $Port, "-o", "StrictHostKeyChecking=no"

Write-Host "Uploading training bundle to ${sshTarget}:${RemotePath}"
Write-Host ""

# Make sure the target dirs exist on the pod.
& ssh @sshOpts $sshTarget "mkdir -p $RemotePath/Data/VHF/VHF_Agents_Training $RemotePath/Data/VHF/VHF_Eval"

# 1. Training-data JSONL (SFT + DPO + reflection + multihop + traces)
Write-Host "== 1/3 Training datasets =="
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_sft_direct.jsonl")    "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_sft_cot.jsonl")       "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_sft_rag.jsonl")       "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_multihop.jsonl")      "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_dpo_pairs.jsonl")     "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_reflection.jsonl")    "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_reasoning_traces.jsonl") "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"

# 2. RAG/KG cache (needed if you want to run eval with RAG on the pod)
Write-Host ""
Write-Host "== 2/3 RAG/KG cache =="
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_rag_chunks.json")     "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_rag_embeddings.npy")  "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_rag_chunk_ids.json")  "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Agents_Training\vhf_kg.json")             "${sshTarget}:${RemotePath}/Data/VHF/VHF_Agents_Training/"

# 3. Held-out eval assets
Write-Host ""
Write-Host "== 3/3 Eval assets =="
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Eval\vhf_gold_answers.json")    "${sshTarget}:${RemotePath}/Data/VHF/VHF_Eval/"
& scp @sshOpts (Join-Path $W "Data\VHF\VHF_Eval\VHF Exam Questions.txt")   "${sshTarget}:${RemotePath}/Data/VHF/VHF_Eval/"

# 4. Optional: .env with OPENAI_API_KEY for eval judge
if ($IncludeEnv) {
    Write-Host ""
    Write-Host "== .env (contains OPENAI_API_KEY) =="
    & scp @sshOpts (Join-Path $W ".env") "${sshTarget}:${RemotePath}/.env"
}

Write-Host ""
Write-Host "Upload done."
Write-Host "On the pod, verify:"
Write-Host "  ssh -p $Port ${sshTarget} 'ls -lh $RemotePath/Data/VHF/VHF_Agents_Training/*.jsonl'"
