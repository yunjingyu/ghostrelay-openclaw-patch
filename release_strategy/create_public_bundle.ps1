param(
  [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$AllowlistPath = Join-Path $ScriptDir "PUBLISH_ALLOWLIST.txt"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDir = Join-Path $RepoRoot "ghostchat-public-bundle"
}

if (-not (Test-Path $AllowlistPath)) {
  throw "Allowlist not found: $AllowlistPath"
}

$Include = Get-Content -Path $AllowlistPath |
  ForEach-Object { $_.Trim() } |
  Where-Object { $_ -and -not $_.StartsWith("#") }

if (Test-Path $OutputDir) {
  Remove-Item -Recurse -Force $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($RelativePath in $Include) {
  $SourcePath = Join-Path $RepoRoot $RelativePath
  if (-not (Test-Path $SourcePath)) {
    Write-Warning "Missing file (skip): $RelativePath"
    continue
  }
  $TargetPath = Join-Path $OutputDir $RelativePath
  $TargetParent = Split-Path -Parent $TargetPath
  if (-not (Test-Path $TargetParent)) {
    New-Item -ItemType Directory -Force -Path $TargetParent | Out-Null
  }
  Copy-Item -Path $SourcePath -Destination $TargetPath -Force
}

$BundleGitIgnore = @"
settings.json
logs/
__pycache__/
.venv/
node_modules/
*.log
*.tmp
*.bak
*.pyc
*.gguf
*.safetensors
.env
.env.*
"@

$GitIgnorePath = Join-Path $OutputDir ".gitignore"
Set-Content -Path $GitIgnorePath -Value $BundleGitIgnore -Encoding UTF8

$ExcludedMustNotExist = @(
  "ghostchat/settings.json",
  "ghostchat/logs",
  "ghostchat/__pycache__"
)

foreach ($PathRel in $ExcludedMustNotExist) {
  $BundlePath = Join-Path $OutputDir $PathRel
  if (Test-Path $BundlePath) {
    throw "Excluded path found in bundle: $PathRel"
  }
}

Write-Host ""
Write-Host "Public bundle created:"
Write-Host "  $OutputDir"
Write-Host ""
Write-Host "Next:"
Write-Host "  1) Review bundle contents"
Write-Host "  2) Initialize/push to public repo"
Write-Host "  3) Keep openclaw-main as external dependency"
