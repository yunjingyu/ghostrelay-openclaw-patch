Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ghostchatDir = (Resolve-Path (Join-Path $scriptDir "..")).Path
$port = 5177

$repoCandidates = @()
if ($env:OPENCLAW_WORKSPACE) {
  $repoCandidates += $env:OPENCLAW_WORKSPACE
}
$repoCandidates += $ghostchatDir
$repoCandidates += (Resolve-Path (Join-Path $ghostchatDir "..")).Path
$repoCandidates += (Get-Location).Path

$repoRoot = $null
foreach ($candidate in $repoCandidates) {
  if (-not $candidate) { continue }
  if (Test-Path (Join-Path $candidate "openclaw-main\openclaw.mjs")) {
    $repoRoot = (Resolve-Path $candidate).Path
    break
  }
}
if (-not $repoRoot) {
  $repoRoot = $ghostchatDir
}
$openclawMain = Join-Path $repoRoot "openclaw-main"

Write-Host "=========================================="
Write-Host "  OpenClaw Skills Dashboard"
Write-Host "=========================================="
Write-Host ""

# Kill existing server on 5177 if present
try {
  $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($conn) {
    Write-Host "[*] existing server PID $($conn.OwningProcess) stopping..."
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
  }
} catch {}

Start-Sleep -Seconds 1

# If still occupied, switch to 5178
try {
  $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($conn) {
    Write-Host "[*] 5177 still in use, switching to 5178."
    $port = 5178
  }
} catch {}

$dashboardUrl = "http://127.0.0.1:$port"

Write-Host "[1/3] Paths"
Write-Host "  Dashboard: $scriptDir"
Write-Host "  Repo root: $repoRoot"
Write-Host "  OpenClaw:  $openclawMain"
Write-Host ""

if (-not (Test-Path (Join-Path $scriptDir "server.mjs"))) {
  Write-Host "[ERR] server.mjs not found: $scriptDir"
  exit 1
}

# Env for this process
$workspacePath = if ($env:OPENCLAW_WORKSPACE) { $env:OPENCLAW_WORKSPACE } elseif (Test-Path (Join-Path $repoRoot "workspace")) { Join-Path $repoRoot "workspace" } else { Join-Path $ghostchatDir "workspace" }
if (-not (Test-Path $workspacePath)) {
  New-Item -ItemType Directory -Path $workspacePath -Force | Out-Null
}
$env:OPENCLAW_WORKSPACE = $workspacePath
if (Test-Path (Join-Path $openclawMain "openclaw.mjs")) {
  $env:OPENCLAW_CLI = (Join-Path $openclawMain "openclaw.mjs")
} else {
  Write-Host "[WARN] openclaw-main not found: $openclawMain"
}
$env:PORT = "$port"

# Ensure npm global bin is on PATH (so `clawhub` is discoverable)
try {
  $npmPrefix = & npm prefix -g 2>$null
  if ($LASTEXITCODE -eq 0 -and $npmPrefix) {
    $npmBin = Join-Path $npmPrefix "node_modules\.bin"
    if ($env:PATH -notlike "*$npmBin*") {
      $env:PATH = "$npmBin;$env:PATH"
    }
  }
} catch {}

# Also add user/global npm bin locations on Windows
try {
  if ($env:APPDATA) {
    $userNpmBin = Join-Path $env:APPDATA "npm"
    if (Test-Path $userNpmBin -and $env:PATH -notlike "*$userNpmBin*") {
      $env:PATH = "$userNpmBin;$env:PATH"
    }
  }
  if ($env:LOCALAPPDATA) {
    $localNpmBin = Join-Path $env:LOCALAPPDATA "npm"
    if (Test-Path $localNpmBin -and $env:PATH -notlike "*$localNpmBin*") {
      $env:PATH = "$localNpmBin;$env:PATH"
    }
  }
} catch {}

Write-Host "[2/3] Starting server on $dashboardUrl"
Write-Host ""

Start-Process $dashboardUrl | Out-Null

# Run node attached to this console (close console = stop server)
Set-Location $scriptDir
node server.mjs
