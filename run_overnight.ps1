# Overnight training + compression chain for the VHF agent.
#
# Usage:
#   pwsh -File run_overnight.ps1
#
# The script runs the following stages in sequence and logs each stage
# to Data/VHF/VHF_Agents_Training/overnight_logs/<NN_stage>.log:
#
#   01 SFT           train_sft.py           REQUIRED (~4-6 h)
#   02 DPO           train_dpo.py           REQUIRED (~1-2 h)   needs SFT adapter
#   03 Reflection    train_reflection.py    REQUIRED (~30 min)  needs SFT+DPO adapters
#   04 Merge         merge_adapter.py       REQUIRED (~15 min)  produces _models/VHF/VHF-QWEN
#   05 Eval VHF-QWEN eval_finetuned.py      OPTIONAL (~40 min)  full 540-Q evaluation
#   06 Prune         compress_prune.py      OPTIONAL (~15 min)
#   07 Distill       compress_distill.py    OPTIONAL (~2-4 h)   student LoRA
#   08 Eval Distill  eval_finetuned.py      OPTIONAL (~40 min)  full 540-Q evaluation
#
# AWQ (§ 17) is intentionally skipped -- it needs `pip install autoawq` first
# and would otherwise abort the chain.
#
# If a REQUIRED stage fails, the chain aborts. If an OPTIONAL stage fails, we
# log the failure and continue with the next stage.

$ErrorActionPreference = "Continue"
$W       = "C:\Users\jcsch\Documents\Python\Auto Pilot"
$PY      = "$W\.venv\Scripts\python.exe"
$LOG_DIR = "$W\Data\VHF\VHF_Agents_Training\overnight_logs"
$MASTER  = "$LOG_DIR\_master.log"

Set-Location $W
New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null

function Write-Master($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $MASTER -Value $line
}

function Run-Stage($idx, $name, [string[]]$scriptArgs, [bool]$required) {
    $tag = "{0:D2}_{1}" -f $idx, $name
    $log = Join-Path $LOG_DIR "$tag.log"
    Write-Master ("=" * 80)
    Write-Master ("STAGE $tag  required=$required")
    Write-Master ("cmd: $PY -X utf8 " + ($scriptArgs -join ' '))
    Write-Master ("log: $log")
    Write-Master ("-" * 80)

    $t0 = Get-Date
    # Combine stdout + stderr via 2>&1 and tee into the log so we can watch live.
    & $PY -X utf8 @scriptArgs *>&1 | Tee-Object -FilePath $log
    $rc = $LASTEXITCODE
    $dur = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    Write-Master ("STAGE $tag  exit=$rc  duration=${dur}min")

    if ($rc -ne 0) {
        if ($required) {
            Write-Master ("!!! REQUIRED STAGE FAILED -- aborting chain.")
            throw "Stage $tag failed with exit code $rc"
        } else {
            Write-Master ("... optional stage failed, continuing.")
        }
    }
}

Write-Master "===== OVERNIGHT CHAIN START ====="
Write-Master ("workspace: $W")
Write-Master ("python:    $PY")
Write-Master ("log dir:   $LOG_DIR")

try {
    Run-Stage 1 "sft"           @("train_sft.py")                                                                     $true
    Run-Stage 2 "dpo"           @("train_dpo.py")                                                                     $true
    Run-Stage 3 "reflection"    @("train_reflection.py")                                                              $true
    Run-Stage 4 "merge"         @("merge_adapter.py")                                                                 $true
    Run-Stage 5 "eval_vhfqwen"  @("eval_finetuned.py", "--model", "_models/VHF/VHF-QWEN",         "--tag", "vhf_qwen")     $false
    Run-Stage 6 "prune"         @("compress_prune.py", "--n-prune", "4")                                              $false
    Run-Stage 7 "distill"       @("compress_distill.py", "--merge-final")                                             $false
    Run-Stage 8 "eval_distill"  @("eval_finetuned.py", "--model", "_models/VHF/DistillVHF-QWEN",  "--tag", "distill_vhf")  $false
    Write-Master "===== OVERNIGHT CHAIN DONE ====="
    exit 0
}
catch {
    Write-Master ("FATAL: {0}" -f $_.Exception.Message)
    Write-Master "===== OVERNIGHT CHAIN ABORTED ====="
    exit 1
}
