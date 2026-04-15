param(
    [string]$PrebuiltDbPath = $env:EPC_PREBUILT_DB_PATH
)

Set-Location $PSScriptRoot

$probeCode = "import PyInstaller, PySide6"
$candidates = @()

$localVenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $localVenvPython) {
    $candidates += @{
        Label = "project .venv"
        FilePath = $localVenvPython
        PrefixArgs = @()
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    $candidates += @{
        Label = "py -3.12"
        FilePath = "py"
        PrefixArgs = @("-3.12")
    }
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    $candidates += @{
        Label = "python"
        FilePath = "python"
        PrefixArgs = @()
    }
}

$pythonCommand = $null
foreach ($candidate in $candidates) {
    try {
        & $candidate.FilePath @($candidate.PrefixArgs + @("-c", $probeCode)) | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $pythonCommand = $candidate
            break
        }
    } catch {
        continue
    }
}

if ($null -eq $pythonCommand) {
    Write-Error "Could not find a Python interpreter with PyInstaller and PySide6 installed. Install requirements into a project .venv or run 'py -3.12 -m pip install -r requirements-dev.txt'."
    exit 1
}

& $pythonCommand.FilePath @($pythonCommand.PrefixArgs + @("-m", "epc_smart_search.preflight", "--mode", "package", "--prebuilt-db", $PrebuiltDbPath))
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$stageRoot = Join-Path $env:TEMP ("epc_smart_search_pkg_" + [guid]::NewGuid().ToString("N"))
$stageAssetsDir = Join-Path $stageRoot "assets"
$stagedDb = Join-Path $stageAssetsDir "contract_store.prebuilt.db"
New-Item -ItemType Directory -Force -Path $stageAssetsDir | Out-Null
Copy-Item -LiteralPath $PrebuiltDbPath -Destination $stagedDb -Force

try {
    & $pythonCommand.FilePath @(
        $pythonCommand.PrefixArgs + @(
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--windowed",
            "--name",
            "EPC Smart Search",
            "--add-data",
            "assets\kiewey.png;assets",
            "--add-data",
            "assets\semantic_model.json;assets",
            "--add-data",
            "$stagedDb;assets",
            "epc_smart_search_app.py"
        )
    )
    exit $LASTEXITCODE
} finally {
    if (Test-Path $stageRoot) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}
