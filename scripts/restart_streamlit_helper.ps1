Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    Write-Host "Checking for processes with 'streamlit' in command line..."
    $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'streamlit' }
    if ($procs) {
        foreach ($p in $procs) {
            try {
                Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
                Write-Host "Stopped PID: $($p.ProcessId)"
            } catch {
                Write-Host "Failed to stop PID: $($p.ProcessId): $($_.Exception.Message)"
            }
        }
    } else {
        Write-Host 'No streamlit process found'
    }

    Write-Host 'Starting Streamlit (headless) via scripts/run_streamlit.ps1'
    Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/run_streamlit.ps1','-Headless' -WorkingDirectory $repoRoot
    Write-Host 'Start command issued.'
} finally {
    Pop-Location
}
