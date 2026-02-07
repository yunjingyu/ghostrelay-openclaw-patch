# GhostRelay

Desktop chat/launcher patch package that runs alongside OpenClaw.

- Korean README: `ghostchat/README.ko.md`

## Status

- Version: `0.4.0-beta.1` (`ghostchat/VERSION`)
- Stage: `Beta`
- Date baseline: `2026-02-08`

## Validation Matrix

### Confirmed (executed)

- Core Python files compile with `python -m py_compile`
- Public bundle script works: `release_strategy/create_public_bundle.ps1`
- Bundle includes `LICENSE`/`NOTICE`/`.gitignore` and excludes private runtime files
- Private GitHub repo bootstrap path was validated (create + first push)

### Implemented in code (manual UI verification still needed)

- Launcher gateway controls (start/stop/restart)
- Ghost Chat main window + Companion window + tray management
- Profile-scoped workspace/config separation
- Ollama model manager (list/register/remove)
- Companion brain prepare/memory-sync flow with `BRAIN_SUMMARY.md`

### Not complete yet

- Fully automated multi-agent orchestration pipeline
- Long-session freeze/latency regression hardening
- Full E2E automation across UI flows

## Project Layout

```text
ghostchat/
  launcher.py
  ghost_chat.py
  ghost_chat_companion.py
  settings_store.py
  runtime_paths.py
  local_logger.py
  start.bat
  start_gateway.bat
  requirements.txt
  package.json
  VERSION
  CHANGELOG.md
  LICENSE
  NOTICE
  dashboard/
  release_strategy/
```

## Install

### Python

```powershell
pip install -r requirements.txt
```

### Node.js

Node.js 18+ is required for dashboard features.

## Run

### Start Launcher

```powershell
python launcher.py
```

or

```powershell
start.bat
```

- Default: minimized + `pythonw`
- Console mode: `start.bat --console`

### Start Gateway Script Directly

```powershell
start_gateway.bat
```

## Patch-Style Integration

GhostRelay is not a replacement for OpenClaw; it is a sidecar patch app.

```text
project-root/
  ghostchat/
  openclaw-main/
```

- Keep `openclaw-main` as external dependency
- Publish `ghostchat` as a separate repository/package

## Current Feature Scope

### Launcher

- Open skills dashboard
- Create/manage Ghost Chat windows
- Create/manage Companion windows
- Show gateway status and control buttons
- Open Ollama model manager dialog

### Ghost Chat

- Select/apply profile
- Conversation mode: `auto` / `chat` / `agent`
- Edit Persona/Memory/Session/Skills in side panels
- Local management commands (`/session`, `/skill`)

### Companion

- Select brain profile
- Brain prepare request on startup
- Memory sync request
- `workspace/.../memory/BRAIN_SUMMARY.md` write/update path

## Known Constraints

- If an Ollama model does not support tool-calling, `agent` mode may fail
- Some settings are safest after gateway restart
- UI responsiveness varies by model/context/hardware

## Collaboration Model (Main Protected, PR Open)

You can protect your `main` branch while allowing collaborators to work freely on feature branches.

Recommended GitHub settings:

1. Go to `Settings > Branches > Add rule` for `main`
2. Enable:
   - `Require a pull request before merging`
   - `Require approvals` (1+)
   - `Require status checks` (optional when CI is added)
   - `Restrict who can push to matching branches` (owner only)
3. Keep branch creation unrestricted so contributors can create branches/forks and open PRs

Result:

- Your `main` stays protected
- Others can branch/fork and submit PRs
- You stay as final merge gate

## Public Repository Policy

### Attribution

- License: `MIT`
- Copyright: Yun Jin Gyu (`akrnwkql@gmail.com`)
- Keep `LICENSE` and `NOTICE` in redistribution

### Privacy

Never commit:

- `settings.json`, `logs/`
- `workspace/`, `profiles/`, `memory/`
- `.env*`, `*credentials*.json`, `*service-account*.json`, `*.pem`, `*.key`
- `*.gguf`, `*.safetensors`, `*.onnx`

Safeguards:

- `.gitignore`
- release allowlist
- optional pre-commit hooks

## Build Public Bundle

```powershell
powershell -ExecutionPolicy Bypass -File .\ghostchat\release_strategy\create_public_bundle.ps1
```

Output:

- `ghostchat-public-bundle/ghostchat`

## Versioning (Temporary Policy)

- Format: `MAJOR.MINOR.PATCH[-label]`
- Current channel: `beta`
- Source of truth:
  - version: `ghostchat/VERSION`
  - changes: `ghostchat/CHANGELOG.md`

Temporary release rule:

- New feature: bump `MINOR`
- Bug fix: bump `PATCH`
- Large breaking architecture change: bump `MAJOR`

## License

- `MIT` (`ghostchat/LICENSE`)
- Free to use/modify/redistribute/commercialize
- Keep copyright/license notice
