[CmdletBinding()]
param(
    # Which Streamlit entrypoint to run.
    # - frontend: recommended modular UI
    # - dashboard: compatibility entrypoint (defaults to frontend)
    # - legacy-dashboard: force the legacy monolithic dashboard
    [ValidateSet('frontend', 'dashboard', 'legacy-dashboard')]
    [string]$Target = 'frontend',

    # Streamlit server port
    [int]$Port = 8501,

    # Run without opening a browser
    [switch]$Headless,

    # Additional args passed to Streamlit after `run <app.py>`
    [string[]]$ExtraArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-PythonCommand {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) { return 'py' }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) { return 'python' }

    throw 'Neither `py` nor `python` was found on PATH.'
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $appPath = switch ($Target) {
        'frontend' { 'frontend/app.py' }
        'dashboard' { 'dashboard/app.py' }
        'legacy-dashboard' { 'dashboard/app.py' }
    }

    if ($Target -eq 'legacy-dashboard') {
        $env:TFIDF_USE_LEGACY_DASHBOARD = '1'
    } else {
        # Unset to avoid accidentally pinning legacy mode.
        if (Test-Path Env:TFIDF_USE_LEGACY_DASHBOARD) {
            Remove-Item Env:TFIDF_USE_LEGACY_DASHBOARD
        }
    }

    $pythonCmd = Get-PythonCommand

    $argsList = @('-m', 'streamlit', 'run', $appPath, '--server.port', "$Port")
    if ($Headless) {
        $argsList += @('--server.headless', 'true')
    }
    if ($ExtraArgs) {
        $argsList += $ExtraArgs
    }

    Write-Host "Running: $pythonCmd $($argsList -join ' ')" -ForegroundColor Cyan
    & $pythonCmd @argsList
}
finally {
    Pop-Location
}
