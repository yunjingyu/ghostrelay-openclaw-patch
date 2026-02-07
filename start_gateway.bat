@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%") do set "SCRIPT_DIR=%%~fI"

set "PORT=18789"
set "BIND=loopback"
if not "%~1"=="" set "PORT=%~1"

set "OPENCLAW_STATE_DIR=%USERPROFILE%\.openclaw"
set "OPENCLAW_CONFIG_PATH=%OPENCLAW_STATE_DIR%\openclaw.json"

set "OPENCLAW_MAIN="
if exist "%SCRIPT_DIR%openclaw-main\openclaw.mjs" set "OPENCLAW_MAIN=%SCRIPT_DIR%openclaw-main"
if "%OPENCLAW_MAIN%"=="" if exist "%SCRIPT_DIR%..\openclaw-main\openclaw.mjs" set "OPENCLAW_MAIN=%SCRIPT_DIR%..\openclaw-main"

if "%OPENCLAW_MAIN%"=="" (
  if not "%OPENCLAW_CLI%"=="" (
    if exist "%OPENCLAW_CLI%" (
      for %%I in ("%OPENCLAW_CLI%") do set "OPENCLAW_MAIN=%%~dpI"
    )
  )
)

if "%OPENCLAW_MAIN%"=="" (
  echo [!] openclaw-main\openclaw.mjs not found.
  echo     Place openclaw-main next to ghostchat or set OPENCLAW_CLI.
  pause
  exit /b 1
)

for %%I in ("%OPENCLAW_MAIN%") do set "OPENCLAW_MAIN=%%~fI"
for %%I in ("%OPENCLAW_MAIN%\..") do set "PROJECT_ROOT=%%~fI"

if "%OPENCLAW_WORKSPACE%"=="" (
  if exist "%PROJECT_ROOT%\workspace" (
    set "OPENCLAW_WORKSPACE=%PROJECT_ROOT%\workspace"
  ) else (
    set "OPENCLAW_WORKSPACE=%SCRIPT_DIR%workspace"
  )
)
if not exist "%OPENCLAW_WORKSPACE%" mkdir "%OPENCLAW_WORKSPACE%" >nul 2>nul

set "DEFAULT_MODEL=%OPENCLAW_DEFAULT_MODEL%"
if "%DEFAULT_MODEL%"=="" set "DEFAULT_MODEL=google-vertex/gemini-2.0-flash"

set "VERTEX_SA_JSON=%SCRIPT_DIR%model\VertexGcp\black-alpha-486019-t6-bf161493c9d2.json"
if not exist "%VERTEX_SA_JSON%" set "VERTEX_SA_JSON=%PROJECT_ROOT%\model\VertexGcp\black-alpha-486019-t6-bf161493c9d2.json"

if /I not "%DEFAULT_MODEL:~0,7%"=="ollama/" (
  if "%GOOGLE_APPLICATION_CREDENTIALS%"=="" if exist "%VERTEX_SA_JSON%" set "GOOGLE_APPLICATION_CREDENTIALS=%VERTEX_SA_JSON%"
  if "%GOOGLE_CLOUD_PROJECT%"=="" set "GOOGLE_CLOUD_PROJECT=black-alpha-486019-t6"
  set "GCLOUD_PROJECT=%GOOGLE_CLOUD_PROJECT%"
  if "%GOOGLE_CLOUD_LOCATION%"=="" set "GOOGLE_CLOUD_LOCATION=us-central1"
)

echo ==========================================
echo      GhostRelay OpenClaw Gateway
echo ==========================================
echo Port: %PORT%
echo Config: %OPENCLAW_CONFIG_PATH%
echo Workspace: %OPENCLAW_WORKSPACE%
echo OpenClaw Main: %OPENCLAW_MAIN%
echo Default Model: %DEFAULT_MODEL%
if /I "%DEFAULT_MODEL:~0,7%"=="ollama/" (
  echo Ollama Base URL: %GHOSTRELAY_OLLAMA_BASE_URL%
) else (
  echo Vertex SA: %VERTEX_SA_JSON%
  echo Vertex: project=%GOOGLE_CLOUD_PROJECT% location=%GOOGLE_CLOUD_LOCATION%
)
echo.

if not exist "%OPENCLAW_MAIN%\dist\index.js" (
  echo [!] OpenClaw not built. Building...
  cd /d "%OPENCLAW_MAIN%"
  call pnpm install
  call pnpm build
  cd /d "%SCRIPT_DIR%"
  echo.
)

if not exist "%OPENCLAW_CONFIG_PATH%" (
  echo [!] No config found. Running onboard...
  cd /d "%OPENCLAW_MAIN%"
  call node openclaw.mjs onboard
  cd /d "%SCRIPT_DIR%"
  echo.
)

echo [*] Enforcing Local Mode config...
cd /d "%OPENCLAW_MAIN%"
call node openclaw.mjs config set gateway.mode local

if "%OPENCLAW_GATEWAY_TOKEN%"=="" (
  for /f %%T in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString()"') do set "OPENCLAW_GATEWAY_TOKEN=%%T"
)
echo [*] Ensuring gateway auth token is set
call node openclaw.mjs config set gateway.auth.mode token
call node openclaw.mjs config set gateway.auth.token "%OPENCLAW_GATEWAY_TOKEN%"

echo [*] Setting agent workspace: %OPENCLAW_WORKSPACE%
call node openclaw.mjs config set agents.defaults.workspace "%OPENCLAW_WORKSPACE%"

if /I "%DEFAULT_MODEL:~0,7%"=="ollama/" (
  if "%GHOSTRELAY_OLLAMA_API_KEY%"=="" set "GHOSTRELAY_OLLAMA_API_KEY=ollama-local"
  set "OLLAMA_API_KEY=%GHOSTRELAY_OLLAMA_API_KEY%"
  echo [*] Ollama auth env set (provider config is preserved)
  if not "%GHOSTRELAY_OLLAMA_BASE_URL%"=="" (
    if /I not "%GHOSTRELAY_OLLAMA_BASE_URL%"=="http://127.0.0.1:11434/v1" (
      echo [!] Custom Ollama Base URL is set: %GHOSTRELAY_OLLAMA_BASE_URL%
      echo     Ensure models.providers.ollama.baseUrl matches this endpoint.
    )
  )
)

echo [*] Setting default model: %DEFAULT_MODEL%
call node openclaw.mjs models set "%DEFAULT_MODEL%"

echo [*] Enabling OpenAI-compatible HTTP endpoint...
call node openclaw.mjs config set gateway.http.endpoints.chatCompletions.enabled true

echo [*] Starting Gateway...
echo     (Press Ctrl+C to stop)
echo.
call node openclaw.mjs gateway run --dev --allow-unconfigured --force --port %PORT% --bind %BIND%

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo [!] Gateway crashed or failed to start!
  echo     Error Level: %ERRORLEVEL%
)

echo.
echo ==========================================
echo   Gateway process ended.
echo ==========================================
pause
endlocal
