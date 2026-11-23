<#
Start a long bash job (e.g. scripts/run_phase1.sh) so it keeps running after the terminal or editor that
started it closes. The job is created through WMI, so it is not a child of this shell.

    powershell -File scripts/launch_detached.ps1 -Script scripts/run_phase1.sh -ScriptArgs "1 2 3" -HfHome D:\ANN\.hf-cache

Progress:  outputs\logs\<script-name>.out  and the per-run logs in outputs\logs\
Stop it:   Stop-Process -Id <pid printed below>  (and its python children); training resumes on the next launch.
#>
param(
    [Parameter(Mandatory = $true)][string]$Script,
    [string]$ScriptArgs = "",
    [string]$HfHome = "",
    [string]$Bash = "C:\Program Files\Git\bin\bash.exe"
)
$repo = (Resolve-Path "$PSScriptRoot\..").Path
$name = [IO.Path]::GetFileNameWithoutExtension($Script)
New-Item -ItemType Directory -Force "$repo\outputs\logs" | Out-Null
$log = "$repo\outputs\logs\$name.out"
$envPart = if ($HfHome) { "set HF_HOME=$HfHome&& " } else { "" }
$cmd = "cmd.exe /c `"$envPart`"$Bash`" $Script $ScriptArgs >> `"$log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $repo }
if ($r.ReturnValue -ne 0) { throw "Could not start the job (Win32_Process.Create returned $($r.ReturnValue))" }
"Started $Script $ScriptArgs  (pid $($r.ProcessId)); log: $log"
