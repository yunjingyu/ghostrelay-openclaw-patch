@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist "..\venv\Scripts\activate.bat" (
  call ..\venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
  call venv\Scripts\activate.bat
)

set "PY_EXE=python"
set "PYW_EXE=pythonw"
if exist ".\venv\Scripts\python.exe" set "PY_EXE=.\venv\Scripts\python.exe"
if exist ".\venv\Scripts\pythonw.exe" set "PYW_EXE=.\venv\Scripts\pythonw.exe"
if exist "..\venv\Scripts\python.exe" set "PY_EXE=..\venv\Scripts\python.exe"
if exist "..\venv\Scripts\pythonw.exe" set "PYW_EXE=..\venv\Scripts\pythonw.exe"

"%PY_EXE%" -c "import PySide6" 2>nul
if errorlevel 1 (
  echo Installing PySide6...
  pip install -r requirements.txt
)

set "RUN_CONSOLE=0"
if /I "%~1"=="--console" set "RUN_CONSOLE=1"

if "%RUN_CONSOLE%"=="1" (
  echo ===============================
  echo GhostRelay start (console mode)
  echo ===============================
  echo.
  "%PY_EXE%" launcher.py
  set "ERR=%ERRORLEVEL%"
  if not "%ERR%"=="0" (
    echo ERROR: GhostRelay failed to start.
    pause
  )
  exit /b %ERR%
)

start "" /min "%PYW_EXE%" launcher.py
exit /b 0
