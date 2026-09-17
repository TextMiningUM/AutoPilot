# Polls nvidia-smi every N seconds and writes a CSV so we can see the thermal
# curve during long training runs (spot BSOD / driver-crash correlations).
#
# Usage:
#   pwsh -File monitor_gpu.ps1 [-Interval 5] [-OutFile gpu_log.csv]
#
# Output columns:
#   timestamp,temperature_C,power_W,gpu_util_pct,mem_used_MiB,mem_total_MiB,
#   pstate,clocks_gr_MHz,clocks_mem_MHz,throttled_reason

param(
    [int]$Interval = 5,
    [string]$OutFile = "Data\VHF\VHF_Agents_Training\overnight_logs\gpu_monitor.csv"
)

$dir = Split-Path -Parent $OutFile
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

$header = "timestamp,temp_C,power_W,gpu_util_pct,mem_used_MiB,mem_total_MiB,pstate,clocks_gr_MHz,clocks_mem_MHz,throttled_reason"
if (-not (Test-Path $OutFile)) {
    $header | Out-File -FilePath $OutFile -Encoding utf8
}

Write-Host "GPU monitor started - interval ${Interval}s, log $OutFile"
Write-Host "Press Ctrl-C to stop (or kill the process)."
Write-Host ""
Write-Host "time     temp  power  util  mem_used/total   pstate  clk_gr  clk_mem  throttled"
Write-Host "-------- ----  -----  ----  ---------------  ------  ------  -------  ---------"

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    try {
        $csv = & nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total,pstate,clocks.gr,clocks.mem,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $csv) {
            $line = "$ts,ERR,,,,,,,"
        } else {
            # nvidia-smi returns space+comma separators
            $parts = ($csv -split ',\s*') | ForEach-Object { $_.Trim() }
            $temp   = $parts[0]
            $power  = $parts[1]
            $util   = $parts[2]
            $mem_u  = $parts[3]
            $mem_t  = $parts[4]
            $pst    = $parts[5]
            $clk_g  = $parts[6]
            $clk_m  = $parts[7]
            $thr    = $parts[8]
            $line   = "$ts,$temp,$power,$util,$mem_u,$mem_t,$pst,$clk_g,$clk_m,$thr"
            $hhmmss = (Get-Date -Format "HH:mm:ss")
            $memf   = "{0,5}/{1,5}" -f $mem_u, $mem_t
            Write-Host ("{0}  {1,3}°  {2,5}W  {3,3}%  {4}  {5,-6}  {6,-6}  {7,-7}  {8}" -f `
                $hhmmss, $temp, $power, $util, $memf, $pst, $clk_g, $clk_m, $thr)
        }
        Add-Content -Path $OutFile -Value $line -Encoding utf8
    } catch {
        Write-Host "$ts,ERR polling nvidia-smi: $_"
    }
    Start-Sleep -Seconds $Interval
}
