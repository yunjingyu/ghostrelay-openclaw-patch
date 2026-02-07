@echo off
setlocal

REM Wrapper to run PowerShell (Unicode-safe) script
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%") do set SCRIPT_DIR=%%~fI

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start.ps1"

endlocal

