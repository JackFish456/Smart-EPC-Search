Set-Location $PSScriptRoot

$appScript = Join-Path $PSScriptRoot "epc_smart_search_app.py"
$probeCode = "import PySide6"

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
    Write-Error "Could not find a Python interpreter with PySide6 installed. Install requirements into a project .venv or run 'py -3.12 -m pip install -r requirements.txt'."
    exit 1
}

& $pythonCommand.FilePath @($pythonCommand.PrefixArgs + @($appScript))
