param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

Write-Output '[1/4] Preparing Python environment...'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    } else {
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed. Python 3.13 is recommended.' }
}
$arenaPython = (Resolve-Path '.venv\Scripts\python.exe').Path
& $arenaPython -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ is required; 3.13 is recommended'"
if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python version.' }

Write-Output '[2/4] Installing pinned platform dependencies...'
& $arenaPython -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $arenaPython -m pip install -e '.[demo,web]' --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Platform installation failed.' }

Write-Output '[3/4] Preparing assets, runtime, and real simulation self-check...'
& $arenaPython -m embodied_arena.platform.deploy
if ($LASTEXITCODE -ne 0) { throw 'Runtime deployment failed. Review the error above and retry.' }

Write-Output '[4/4] Starting the portal...'
& (Join-Path $PSScriptRoot 'start_platform.ps1') -NoBrowser:$NoBrowser
if ($LASTEXITCODE -ne 0) { throw 'Portal startup failed.' }
