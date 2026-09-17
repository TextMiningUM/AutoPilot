# Upload the minimal training-data + eval bundle to a cloud GPU pod via scp.
# Everything train_sft.py / train_dpo.py / train_reflection.py / the ablation /
# the suite-v2 eval scripts need to run unattended on the pod, nothing more
# (no raw source PDFs/TXT, no _json/ intermediate, no local model weights).
#
# Usage:
#   pwsh -File cloud/upload_bundle.ps1 -Server <IP_OR_HOSTNAME> -Port 22 [-User ubuntu] [-IncludeEnv]

param(
    [Parameter(Mandatory=$true)] [string]$Server,
    [int]$Port = 22,
    [string]$User = "ubuntu",
    [string]$RemotePath = "~/AutoPilot",
    [string]$KeyPath = "C:\Users\jcsch\Documents\Python\LeafCloud\AutoPilot\KeyPairAutoPilot.txt",
    [switch]$IncludeEnv
)

$W = "C:\Users\jcsch\Documents\Python\Auto Pilot"
Set-Location $W

$sshTarget = "${User}@${Server}"
# ssh uses lowercase -p for port; scp uses uppercase -P for port.
$sshOpts   = "-p", $Port, "-i", $KeyPath, "-o", "StrictHostKeyChecking=no"
$scpOpts   = "-P", $Port, "-i", $KeyPath, "-o", "StrictHostKeyChecking=no"

$TRAIN_DIR = "Data\VHF\VHF_Agents_Training"
$EVAL_DIR  = "Data\VHF\VHF_Eval"

function Send-File([string]$relPath) {
    $local = Join-Path $W $relPath
    if (-not (Test-Path $local)) {
        Write-Host "  [skip] $relPath (not found locally)" -ForegroundColor Yellow
        return
    }
    $remoteDir = "$RemotePath/" + (Split-Path $relPath -Parent).Replace('\', '/')
    & scp @scpOpts $local "${sshTarget}:${remoteDir}/"
}

Write-Host "Uploading VHF training + eval bundle to ${sshTarget}:${RemotePath}"
Write-Host ""

& ssh @sshOpts $sshTarget "mkdir -p $RemotePath/$($TRAIN_DIR.Replace('\','/')) $RemotePath/$($EVAL_DIR.Replace('\','/'))"

Write-Host "== 1/6 Track 1 + Track 2 + PG training data (SFT/DPO/reflection) =="
foreach ($f in @(
    "vhf_sft_direct.jsonl", "vhf_sft_cot.jsonl", "vhf_sft_rag.jsonl", "vhf_multihop.jsonl",
    "vhf_conversations.jsonl",
    "vhf_colreg_sft_direct.jsonl", "vhf_colreg_sft_cot.jsonl", "vhf_colreg_sft_rag.jsonl",
    "vhf_colreg_multihop.jsonl",
    "vhf_pg_sft.jsonl",
    "vhf_dpo_pairs.jsonl", "vhf_colreg_dpo_pairs.jsonl",
    "vhf_reflection.jsonl", "vhf_colreg_reflection.jsonl"
)) { Send-File "$TRAIN_DIR\$f" }

Write-Host ""
Write-Host "== 2/6 Reasoning traces (needed if the pod re-runs any build_* stage, e.g. run_all.sh's Track 2 mining) =="
foreach ($f in @("vhf_reasoning_traces.jsonl", "vhf_conversation_traces.jsonl")) {
    Send-File "$TRAIN_DIR\$f"
}

Write-Host ""
Write-Host "== 3/6 RAG / KG / Procedural Graph cache (ablation V1/V3/V4 configs, CorpusGrounded retrieval) =="
foreach ($f in @(
    "vhf_rag_chunks.json", "vhf_rag_embeddings.npy", "vhf_rag_chunk_ids.json",
    "vhf_kg.json", "vhf_pg.json", "probe_set.json"
)) { Send-File "$TRAIN_DIR\$f" }

Write-Host ""
Write-Host "== 4/6 Held-out eval assets + gold_claims (suite v2 requires the *_claims.json siblings) =="
foreach ($f in @(
    "vhf_gold_answers.json", "vhf_gold_answers_claims.json",
    "vhf_colreg_scenarios.json", "vhf_colreg_scenarios_claims.json",
    "VHF Exam Questions.txt"
)) { Send-File "$EVAL_DIR\$f" }

Write-Host ""
Write-Host "== 5/6 Judge cache (re-scoring reuses cached verdicts instead of re-paying for judge calls) =="
Send-File "$TRAIN_DIR\ragas_judge_cache.jsonl"

Write-Host ""
Write-Host "== 6/6 .env (OPENAI_API_KEY for judge calls; ANTHROPIC_API_KEY only needed if re-enriching gold claims) =="
if ($IncludeEnv) {
    Send-File ".env"
} else {
    Write-Host "  Skipped (pass -IncludeEnv to send it)."
}

Write-Host ""
Write-Host "Upload done."
Write-Host "On the pod, verify:"
Write-Host "  ssh -p $Port ${sshTarget} 'ls -lh $RemotePath/Data/VHF/VHF_Agents_Training/*.jsonl $RemotePath/Data/VHF/VHF_Agents_Training/*.json'"
Write-Host "  ssh -p $Port ${sshTarget} 'ls -lh $RemotePath/Data/VHF/VHF_Eval/'"
