# Contributing to GhostRelay

Thanks for contributing.

## Workflow

1. Create a branch from `main` (or use your fork).
2. Implement changes in small, reviewable commits.
3. Open a Pull Request to `main`.

## Branch Naming

Use one of:

- `feat/<short-name>`
- `fix/<short-name>`
- `docs/<short-name>`
- `refactor/<short-name>`

Examples:

- `feat/ollama-manager-ui`
- `fix/gateway-restart-timeout`

## Pull Request Checklist

- Keep scope focused and explain motivation
- Update docs if behavior changes
- Do not commit private data/logs/secrets/model files
- Confirm local checks pass (at least syntax and startup path)

## Main Branch Policy

- `main` is protected
- Changes are merged via PR
- At least one approval is required

## Security / Privacy

Never commit:

- `settings.json`, `logs/`
- `workspace/`, `profiles/`, `memory/`
- `.env*`, `*credentials*.json`, `*service-account*.json`
- `*.pem`, `*.key`, `*.gguf`, `*.safetensors`, `*.onnx`

Use:

- `.gitignore`
- `release_strategy/PUBLISH_ALLOWLIST.txt`
- optional pre-commit hooks
