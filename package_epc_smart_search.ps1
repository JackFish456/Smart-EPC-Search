Set-Location $PSScriptRoot

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Error "PyInstaller is not installed in this Python environment."
    exit 1
}

pyinstaller `
    --noconfirm `
    --windowed `
    --name "EPC Smart Search" `
    --add-data "assets;assets" `
    epc_smart_search_app.py
