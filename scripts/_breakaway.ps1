# Helper for launch_detached.ps1: start a command line outside the caller's Windows job object
# (CREATE_BREAKAWAY_FROM_JOB) in a new, minimised console. Returns the new process id, or throws.
param([Parameter(Mandatory = $true)][string]$CommandLine, [Parameter(Mandatory = $true)][string]$WorkDir,
      [string]$Title = "DRISHTI training - do not close")
if (-not ("Drishti.Launch" -as [type])) {
Add-Type -Namespace Drishti -Name Launch -MemberDefinition @'
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
public struct STARTUPINFO { public int cb; public string lpReserved; public string lpDesktop; public string lpTitle;
  public int dwX; public int dwY; public int dwXSize; public int dwYSize; public int dwXCountChars;
  public int dwYCountChars; public int dwFillAttribute; public int dwFlags; public short wShowWindow;
  public short cbReserved2; public System.IntPtr lpReserved2; public System.IntPtr hStdInput;
  public System.IntPtr hStdOutput; public System.IntPtr hStdError; }
[StructLayout(LayoutKind.Sequential)]
public struct PROCESS_INFORMATION { public System.IntPtr hProcess; public System.IntPtr hThread; public int dwProcessId;
  public int dwThreadId; }
[DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
public static extern bool CreateProcess(string app, System.Text.StringBuilder cmd, System.IntPtr pa, System.IntPtr ta,
  bool inherit, uint flags, System.IntPtr env, string dir, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
[DllImport("kernel32.dll")] public static extern bool CloseHandle(System.IntPtr h);
'@
}
$si = New-Object Drishti.Launch+STARTUPINFO
$si.cb = [Runtime.InteropServices.Marshal]::SizeOf($si)
$si.lpTitle = $Title
$si.dwFlags = 0x1          # STARTF_USESHOWWINDOW
$si.wShowWindow = 7        # SW_SHOWMINNOACTIVE
$pi = New-Object Drishti.Launch+PROCESS_INFORMATION
$flags = [uint32](0x01000000 -bor 0x00000010 -bor 0x00000200)  # BREAKAWAY_FROM_JOB | NEW_CONSOLE | NEW_PROCESS_GROUP
# [NullString]::Value, not $null: PowerShell would pass $null as "" and CreateProcess then fails with error 3
$ok = [Drishti.Launch]::CreateProcess([NullString]::Value, (New-Object Text.StringBuilder $CommandLine), [IntPtr]::Zero,
    [IntPtr]::Zero, $false, $flags, [IntPtr]::Zero, $WorkDir, [ref]$si, [ref]$pi)
if (-not $ok) { throw "CreateProcess failed (Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error()))" }
[void][Drishti.Launch]::CloseHandle($pi.hThread); [void][Drishti.Launch]::CloseHandle($pi.hProcess)
$pi.dwProcessId
