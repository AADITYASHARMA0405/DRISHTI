<#
Stop a job started with launch_detached.ps1: the runner script, this repository's train.py / eval.py
processes, and their data-loader workers. The next launch continues from the last completed epoch.

    powershell -File scripts/stop_training.ps1
#>
$repo = (Resolve-Path "$PSScriptRoot\..").Path
$all = Get-CimInstance Win32_Process
$runners = $all | Where-Object { $_.Name -in @("bash.exe", "cmd.exe") -and $_.CommandLine -match "scripts/run_phase" }
$mains = $all | Where-Object {
    $_.Name -eq "python.exe" -and $_.CommandLine -match [regex]::Escape($repo) -and $_.CommandLine -match "train\.py|eval\.py" }
$mainIds = @($mains | ForEach-Object { $_.ProcessId })
$workers = $all | Where-Object { $_.Name -eq "python.exe" -and $mainIds -contains $_.ParentProcessId }
$targets = @($runners) + @($mains) + @($workers) | Where-Object { $_ }
if (-not $targets) { "No training job is running."; return }
foreach ($p in $targets) {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    "stopped $($p.Name) $($p.ProcessId)"
}
