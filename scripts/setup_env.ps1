<#
.SYNOPSIS
Creates (if missing) a virtual environment at .venv and installs packages from requirements.txt

.EXAMPLE
From project root (PowerShell):
    .\scripts\setup_env.ps1

#>
param(
    [string]$VenvPath = ".venv",
    [string]$RequirementsFile = "requirements.txt"
)

Write-Output "Using venv path: $VenvPath"

$pythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Output "Virtual environment not found. Creating $VenvPath..."
    python -m venv $VenvPath
} else {
    Write-Output "Virtual environment already exists at $VenvPath"
}

Write-Output "Upgrading pip, setuptools, wheel in venv..."
& $pythonExe -m pip install --upgrade pip setuptools wheel

if (-not (Test-Path $RequirementsFile)) {
    Write-Error "Requirements file '$RequirementsFile' not found in current directory. Please run this from the project root or pass -RequirementsFile path."
    exit 1
}

Write-Output "Installing packages from $RequirementsFile into venv..."
& $pythonExe -m pip install -r $RequirementsFile

Write-Output "Setup complete. Activate the venv with:`n    & ${VenvPath}\Scripts\Activate.ps1"
