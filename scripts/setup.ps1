param([switch]$Dev)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $environmentPython)) {
    $pythonCommand = $null
    $pythonArguments = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $probePreference = $ErrorActionPreference
        $ErrorActionPreference = 'SilentlyContinue'
        & py -3.12 -c 'import sys; print(sys.version)' 2>$null
        $probeSucceeded = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $probePreference
        if ($probeSucceeded) {
            $pythonCommand = 'py'
            $pythonArguments = @('-3.12')
        }
    }
    if (-not $pythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $probePreference = $ErrorActionPreference
        $ErrorActionPreference = 'SilentlyContinue'
        & python -c 'import sys; assert sys.version_info >= (3,12)' 2>$null
        $probeSucceeded = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $probePreference
        if ($probeSucceeded) { $pythonCommand = 'python' }
    }
    # Codex's bundled interpreter is a fallback for this machine; normal Python
    # installations are preferred so the project also works outside Codex.
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not $pythonCommand -and (Test-Path -LiteralPath $bundledPython)) {
        $pythonCommand = $bundledPython
    }
    if (-not $pythonCommand) {
        throw 'Python 3.12+ is required by the pinned packages (3.12 recommended). Install Python from python.org, reopen the terminal, then run setup.bat again.'
    }
    Write-Host 'Creating the project Python environment...'
    & $pythonCommand @pythonArguments -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
}

$requirementsFile = if ($Dev) { 'requirements-dev.txt' } else { 'requirements.txt' }
Write-Host "Installing packages from $requirementsFile..."
& $environmentPython -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. Check your internet connection and retry setup.bat.' }
& $environmentPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Package consistency check failed.' }
Write-Host ''
Write-Host 'Setup complete. Start the website with .\run.bat'
if ($Dev) { Write-Host 'Run tests with .\.venv\Scripts\python.exe -m pytest -q' }
