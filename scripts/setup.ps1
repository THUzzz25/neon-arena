$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    } else {
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment.' }
}
$arenaPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
& $arenaPython -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ is required'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
& $arenaPython -m pip install -e '.[demo,web,test]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $arenaPython scripts/fetch_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Asset download or checksum verification failed.' }
Write-Output 'Ready. Run the demo launcher or: .\.venv\Scripts\python.exe -m embodied_arena run --viewer --realtime'
