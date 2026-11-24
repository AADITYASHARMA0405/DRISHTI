<#
Start a long bash job (e.g. scripts/run_phase1.sh) so it keeps running after the terminal, editor or
assistant session that started it closes.

    powershell -File scripts/launch_detached.ps1 -Script scripts/run_phase1.sh -ScriptArgs "1 2 3" -HfHome D:\ANN\.hf-cache

The job is created with CREATE_BREAKAWAY_FROM_JOB in its own minimised console titled "DRISHTI training -
do not close", so closing the parent (or its Windows job object) does not end it.

Progress:  outputs\logs\<script-name>.out  and the per-run logs in outputs\logs\
Stop it:   powershell -File scripts/stop_training.ps1   (training resumes from its last epoch on the next launch)
Keep the laptop plugged in: training pauses itself on battery and continues when the charger is back.
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
# FOR_DISABLE_CONSOLE_CTRL_HANDLER stops Intel's Fortran runtime (inside PyTorch's MKL) from aborting the
# whole job ("forrtl: error (200)") when a console close / Ctrl-C event arrives.
$envPart = "set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1&& "
if ($HfHome) { $envPart += "set HF_HOME=$HfHome&& " }
$cmd = "$env:SystemRoot\System32\cmd.exe /c `"$envPart`"$Bash`" $Script $ScriptArgs >> `"$log`" 2>&1`""
$procId = & "$PSScriptRoot\_breakaway.ps1" -CommandLine $cmd -WorkDir $repo
"Started $Script $ScriptArgs  (pid $procId); log: $log"
