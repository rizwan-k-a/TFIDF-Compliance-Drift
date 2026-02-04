[CmdletBinding()]
param(
    [string[]]$PytestArgs = @('-q')
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
    $pythonCmd = Get-PythonCommand
    $argsList = @('-m', 'pytest') + $PytestArgs

    Write-Host "Running: $pythonCmd $($argsList -join ' ')" -ForegroundColor Cyan
    & $pythonCmd @argsList
}
finally {
    Pop-Location
}
