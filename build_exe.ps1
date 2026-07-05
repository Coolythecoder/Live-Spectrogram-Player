param(
    [switch]$OneDir
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$name = "LiveSpectrogramPlayer"
$iconPath = Join-Path $scriptDir "assets\app_icon.ico"
if (!(Test-Path -LiteralPath $iconPath -PathType Leaf)) {
    throw "App icon was not found: $iconPath"
}
$soundcardHookDir = python -c "import pathlib, soundcard; print(pathlib.Path(soundcard.__file__).parent / '__pyinstaller')"
$pyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--noupx",
    "--windowed",
    "--name", $name,
    "--icon", $iconPath,
    "--version-file", "$scriptDir\version_info.txt",
    "--add-data", "$iconPath;assets",
    "--additional-hooks-dir", $soundcardHookDir,
    "--collect-data", "customtkinter",
    "--collect-binaries", "soundfile",
    "--hidden-import", "soxr",
    "--hidden-import", "customtkinter",
    "--exclude-module", "PyQt5",
    "--exclude-module", "PyQt6",
    "--exclude-module", "PySide2",
    "--exclude-module", "PySide6",
    "$scriptDir\live_spectrogram_player.py"
)

if (!$OneDir) {
    $pyInstallerArgs = @("--onefile") + $pyInstallerArgs
}

python -m PyInstaller @pyInstallerArgs

$exePath = if ($OneDir) {
    Join-Path $scriptDir "dist\$name\$name.exe"
} else {
    Join-Path $scriptDir "dist\$name.exe"
}

if (!(Test-Path -LiteralPath $exePath)) {
    throw "Build finished but executable was not found: $exePath"
}

Write-Host "Built $exePath"
