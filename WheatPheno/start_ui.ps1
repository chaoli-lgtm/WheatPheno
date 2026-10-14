$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$wheatPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $wheatPython)) {
    Write-Host 'Create .venv with Python 3.12 and install: python -m pip install -e ".[ui,test]"'
    exit 1
}
& $wheatPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.address 127.0.0.1
