# GhostRelay Public Release Strategy

This folder defines exactly what to publish so others can run GhostRelay quickly without your private data.

## What this gives you
- A strict file allowlist for public release.
- A clear exclude list for secrets/local runtime state.
- A script to generate a clean public bundle automatically.

## Required runtime for end users
- Python 3.10+
- Node.js 18+
- `openclaw-main` placed next to your GhostRelay folder (external dependency, not bundled)

Expected layout:

```text
project-root/
  ghostrelay/
  openclaw-main/
```

## Files in this folder
- `PUBLISH_ALLOWLIST.txt`: only files that should go public.
- `PUBLISH_EXCLUDE.txt`: extra guardrail patterns to avoid leaks.
- `create_public_bundle.ps1`: builds `ghostchat-public-bundle/` from allowlist.
- `precommit_blocklist.py`: blocks accidental commit of private/log/secret files.

## One-command bundle build
Run from the GhostRelay root (folder that contains `launcher.py`):

```powershell
powershell -ExecutionPolicy Bypass -File .\release_strategy\create_public_bundle.ps1
```

Optional custom output path:

```powershell
powershell -ExecutionPolicy Bypass -File .\release_strategy\create_public_bundle.ps1 -OutputDir .\dist\ghostrelay-public
```

## Publish checklist
1. Build bundle with `create_public_bundle.ps1`.
2. Confirm `settings.json` is not present.
3. Confirm `logs/` and `__pycache__/` are not present.
4. Confirm no credentials or model weights are included.
5. Confirm `LICENSE` and `NOTICE` are included in bundle.
6. Confirm `VERSION` and `CHANGELOG.md` are included in bundle.
7. If bilingual docs are used, confirm both `README.md` and `README.ko.md` are included.
8. Push bundle contents to public repository.

## Optional: pre-commit safety guard

Run these once in your public repo root:

```powershell
pip install pre-commit
pre-commit install
```

This uses `.pre-commit-config.yaml` and blocks commits containing private paths/secrets.
