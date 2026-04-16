param(
    [ValidateSet("Lite", "AI")]
    [string]$Profile = "Lite",
    [string]$PrebuiltDbPath = $env:EPC_PREBUILT_DB_PATH,
    [string]$ModelDir = $env:EPC_SMART_SEARCH_MODEL_DIR
)

Set-Location $PSScriptRoot

function Resolve-PythonCommand {
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

    foreach ($candidate in $candidates) {
        try {
            & $candidate.FilePath @($candidate.PrefixArgs + @("-c", $probeCode)) | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
            continue
        }
    }
    return $null
}

function Invoke-PyInstaller {
    param(
        [hashtable]$PythonCommand,
        [string[]]$Arguments
    )

    & $PythonCommand.FilePath @($PythonCommand.PrefixArgs + $Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
}

function Compress-DirectoryToZip {
    param(
        [hashtable]$PythonCommand,
        [string]$Source,
        [string]$Destination
    )

    $zipCode = @'
from __future__ import annotations

import pathlib
import sys
import zipfile

source = pathlib.Path(sys.argv[1]).resolve()
destination = pathlib.Path(sys.argv[2]).resolve()

with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            bundle.write(path, path.relative_to(source))
'@

    & $PythonCommand.FilePath @($PythonCommand.PrefixArgs + @("-c", $zipCode, $Source, $Destination))
    if ($LASTEXITCODE -ne 0) {
        throw "Zip packaging failed with exit code $LASTEXITCODE."
    }
}

function Copy-DirectoryContents {
    param(
        [string]$Source,
        [string]$Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
    }
}

$pythonCommand = Resolve-PythonCommand
if ($null -eq $pythonCommand) {
    Write-Error "Could not find a Python interpreter with PyInstaller and PySide6 installed. Install requirements into a project .venv or run 'py -3.12 -m pip install -r requirements-dev.txt'."
    exit 1
}

$preflightArgs = @(
    "-m",
    "epc_smart_search.preflight",
    "--mode",
    "package",
    "--profile",
    $Profile,
    "--prebuilt-db",
    $PrebuiltDbPath
)
if ($Profile -eq "AI") {
    $preflightArgs += @("--model-dir", $ModelDir)
}

& $pythonCommand.FilePath @($pythonCommand.PrefixArgs + $preflightArgs)
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$appName = if ($Profile -eq "AI") { "EPC Smart Search AI" } else { "EPC Smart Search Lite" }
$zipName = if ($Profile -eq "AI") { "EPC-Smart-Search-AI-win64.zip" } else { "EPC-Smart-Search-Lite-win64.zip" }
$releaseRoot = Join-Path $PSScriptRoot "release"
$distRoot = Join-Path $PSScriptRoot "dist"
$appEntryPoint = Join-Path $PSScriptRoot "epc_smart_search_app.py"
$serviceEntryPoint = Join-Path $PSScriptRoot "gemma_service.py"
$kieweyAsset = Join-Path $PSScriptRoot "assets\kiewey.png"
$stageRoot = Join-Path $env:TEMP ("epc_smart_search_pkg_" + [guid]::NewGuid().ToString("N"))
$buildRoot = Join-Path $stageRoot "pyinstaller"
$stageAssetsDir = Join-Path $stageRoot "assets"
$stagedDb = Join-Path $stageAssetsDir "contract_store.prebuilt.db"
$zipPath = Join-Path $releaseRoot $zipName

New-Item -ItemType Directory -Force -Path $stageAssetsDir | Out-Null
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
Copy-Item -LiteralPath $PrebuiltDbPath -Destination $stagedDb -Force

try {
    if (Test-Path (Join-Path $distRoot $appName)) {
        Remove-Item -LiteralPath (Join-Path $distRoot $appName) -Recurse -Force
    }
    if (Test-Path $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }

    $appArgs = @(
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        $appName,
        "--distpath",
        $distRoot,
        "--workpath",
        $buildRoot,
        "--specpath",
        $buildRoot,
        "--add-data",
        "$kieweyAsset;assets",
        "--add-data",
        "$stagedDb;assets"
    )
    if ($Profile -eq "Lite") {
        $appArgs += @(
            "--exclude-module", "torch",
            "--exclude-module", "transformers",
            "--exclude-module", "accelerate",
            "--exclude-module", "safetensors",
            "--exclude-module", "tokenizers",
            "--exclude-module", "bitsandbytes"
        )
    }
    $appArgs += $appEntryPoint
    Invoke-PyInstaller -PythonCommand $pythonCommand -Arguments $appArgs

    $appDistPath = Join-Path $distRoot $appName

    if ($Profile -eq "AI") {
        $stagedModelRoot = Join-Path $stageRoot "models"
        $stagedModelDir = Join-Path $stagedModelRoot "gemma"
        Copy-DirectoryContents -Source $ModelDir -Destination $stagedModelDir

        if (Test-Path (Join-Path $distRoot "gemma_service")) {
            Remove-Item -LiteralPath (Join-Path $distRoot "gemma_service") -Recurse -Force
        }

        $serviceArgs = @(
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--console",
            "--name",
            "gemma_service",
            "--distpath",
            $distRoot,
            "--workpath",
            $buildRoot,
            "--specpath",
            $buildRoot,
            "--hidden-import",
            "flask",
            "--hidden-import",
            "torch",
            "--hidden-import",
            "transformers",
            "--hidden-import",
            "accelerate",
            "--hidden-import",
            "safetensors",
            "--hidden-import",
            "tokenizers",
            "--add-data",
            "$stagedModelDir;models/gemma",
            $serviceEntryPoint
        )
        Invoke-PyInstaller -PythonCommand $pythonCommand -Arguments $serviceArgs

        $serviceDistPath = Join-Path $distRoot "gemma_service"
        $aiRuntimeDir = Join-Path $appDistPath "ai_runtime"
        Copy-DirectoryContents -Source $serviceDistPath -Destination $aiRuntimeDir
    }

    Compress-DirectoryToZip -PythonCommand $pythonCommand -Source $appDistPath -Destination $zipPath
    Write-Output "PACKAGED_PROFILE=$Profile"
    Write-Output "APP_DIST=$appDistPath"
    Write-Output "ZIP_PATH=$zipPath"
    exit 0
} finally {
    if (Test-Path $stageRoot) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}
