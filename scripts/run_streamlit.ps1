[CmdletBinding()]
param(
    # Which Streamlit entrypoint to run.
    # - frontend: recommended modular UI
    [ValidateSet('frontend')]
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
    $appPath = 'frontend/app.py'

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
